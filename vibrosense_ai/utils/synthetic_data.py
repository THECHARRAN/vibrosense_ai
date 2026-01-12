import numpy as np
import torch

def generate_signal(length=16000, fault=False, drift=False):
    t = np.linspace(0, 1, length)

    # Base machine tone
    base = np.sin(2 * np.pi * 50 * t)

    # Harmonics
    harmonic = 0.3 * np.sin(2 * np.pi * 120 * t)

    signal = base + harmonic

    # Noise
    signal += 0.05 * np.random.randn(length)

    # Fault = amplitude + frequency instability
    if fault:
        signal *= np.linspace(1, 1.8, length)
        signal += 0.2 * np.sin(2 * np.pi * 300 * t)

    # Drift = slow degradation
    if drift:
        signal += 0.1 * np.sin(2 * np.pi * 5 * t)

    return signal.astype(np.float32)
def create_dataset(samples=1000, window=128, features=96):
    data = []

    for _ in range(samples):
        sig = generate_signal()
        x = np.random.randn(window, features) * 0.05
        x += sig[:window].reshape(-1, 1)
        data.append(x)

    return torch.tensor(data, dtype=torch.float32)
