import numpy as np
from scipy.stats import kurtosis, skew

# ============================================================
# 1. LOW-LEVEL SIGNAL GENERATORS
# ============================================================

def synth_accel(
    length,
    fs,
    base_freq=50,
    harmonics=(1, 2, 3),
    jitter=0.0,
    imbalance=0.0
):
    """
    Simulates a single accelerometer axis.
    - base_freq: rotation / shaft frequency
    - harmonics: mechanical harmonics
    - jitter: slow frequency drift
    - imbalance: low-frequency wobble
    """
    t = np.arange(length) / fs
    signal = np.zeros_like(t)

    for h in harmonics:
        amp = 1.0 / h
        freq = base_freq * h + jitter * np.sin(2 * np.pi * 0.1 * t)
        signal += amp * np.sin(2 * np.pi * freq * t)

    # imbalance / looseness
    signal += imbalance * np.sin(2 * np.pi * 5 * t)

    # sensor noise
    signal += 0.02 * np.random.randn(length)

    return signal.astype(np.float32)


def synth_audio(
    length,
    fs,
    base_freq=100,
    noise_level=0.02,
    fault=False
):
    """
    Simulates microphone signal.
    Fault adds broadband + bearing-like noise.
    """
    t = np.arange(length) / fs

    audio = 0.6 * np.sin(2 * np.pi * base_freq * t)
    audio += 0.3 * np.sin(2 * np.pi * (2 * base_freq) * t)
    audio += noise_level * np.random.randn(length)

    if fault:
        audio += 0.2 * np.sin(2 * np.pi * 300 * t)
        audio += 0.1 * np.random.randn(length)

    return audio.astype(np.float32)

# ============================================================
# 2. RAW SENSOR WINDOW (WHAT HARDWARE PRODUCES)
# ============================================================

def make_window(
    audio_fs=16000,
    accel_fs=2000,
    win_s=1.0,
    fault=False,
    drift=False,
    sensor_disagree=False
):
    """
    Produces one synchronized sensor window.
    This mirrors one inference cycle on hardware.
    """
    Na = int(audio_fs * win_s)
    Nacc = int(accel_fs * win_s)

    audio = synth_audio(Na, audio_fs, fault=fault)

    ax = synth_accel(Nacc, accel_fs, 40, jitter=0.2, imbalance=0.5 if drift else 0.0)
    ay = synth_accel(Nacc, accel_fs, 42, jitter=0.1, imbalance=0.3 if drift else 0.0)
    az = synth_accel(Nacc, accel_fs, 38, jitter=0.05, imbalance=0.2 if drift else 0.0)

    if sensor_disagree:
        ax2 = ax + 0.05 * np.random.randn(Nacc)
        ay2 = ay + 0.05 * np.random.randn(Nacc)
        az2 = az + 0.05 * np.random.randn(Nacc)
    else:
        ax2, ay2, az2 = ax.copy(), ay.copy(), az.copy()

    return {
        "audio": audio,
        "accel_lis": np.stack([ax, ay, az], axis=0),
        "accel_adx": np.stack([ax2, ay2, az2], axis=0),
        "label": int(fault)
    }

# ============================================================
# 3. FEATURE EXTRACTION (EDGE-COMPATIBLE)
# ============================================================

def compute_basic_features(window):
    """
    Extracts deterministic, explainable features.
    This logic will later run on ESP32 / Pi.
    """
    audio = window["audio"]
    acc_l = window["accel_lis"]
    acc_a = window["accel_adx"]

    feats = {}

    # Accelerometer per-axis stats
    for i, axis in enumerate(("x", "y", "z")):
        v = acc_l[i]
        feats[f"lis_mean_{axis}"] = v.mean()
        feats[f"lis_rms_{axis}"] = np.sqrt(np.mean(v**2))
        feats[f"lis_std_{axis}"] = v.std()
        feats[f"lis_kurt_{axis}"] = kurtosis(v)
        feats[f"lis_skew_{axis}"] = skew(v)

    # Magnitude
    mag = np.sqrt((acc_l**2).sum(axis=0))
    feats["lis_rms_mag"] = np.sqrt(np.mean(mag**2))

    # Audio
    feats["audio_rms"] = np.sqrt(np.mean(audio**2))
    feats["audio_zcr"] = np.mean(audio[:-1] * audio[1:] < 0)

    # Sensor disagreement (trust feature)
    rms_l = np.sqrt((acc_l**2).mean(axis=1))
    rms_a = np.sqrt((acc_a**2).mean(axis=1))
    feats["rms_disagreement"] = np.mean(np.abs(rms_l - rms_a))

    return feats
