import numpy as np
import torch
from torch.utils.data import DataLoader

# --- Data & features ---
from vibrosense_ai.utils.synthetic_data import make_window, compute_basic_features
from vibrosense_ai.features.vectorize import vectorize_features
from vibrosense_ai.features.normalize import FeatureNormalizer
from vibrosense_ai.data.sequence_builder import build_sequences
from vibrosense_ai.data.dataset import VibrosenseDataset

# --- Model & training ---
from vibrosense_ai.models.vibrosense_model import VibroSenseV2
from vibrosense_ai.training.trainer import Trainer

# ============================================================
# 1. FEATURE CONTRACT (MUST NEVER CHANGE ORDER)
# ============================================================

FEATURE_ORDER = [
    "lis_mean_x", "lis_rms_x", "lis_std_x", "lis_kurt_x", "lis_skew_x",
    "lis_mean_y", "lis_rms_y", "lis_std_y", "lis_kurt_y", "lis_skew_y",
    "lis_mean_z", "lis_rms_z", "lis_std_z", "lis_kurt_z", "lis_skew_z",
    "lis_rms_mag",
    "audio_rms", "audio_zcr",
    "rms_disagreement",
]

# ============================================================
# 2. GENERATE SYNTHETIC FEATURE VECTORS
# ============================================================

num_windows = 600
feature_vectors = []

for _ in range(num_windows):
    window = make_window(
        fault=False,
        drift=np.random.rand() < 0.2,
        sensor_disagree=np.random.rand() < 0.1
    )
    feats = compute_basic_features(window)
    vec = vectorize_features(feats, FEATURE_ORDER)
    feature_vectors.append(vec)
print("Feature keys:", sorted(feats.keys()))
feature_vectors = np.array(feature_vectors, dtype=np.float32)
print("Feature vectors:", feature_vectors.shape)
print("One feature vector example:", feature_vectors[0])
print("Feature dim:", feature_vectors.shape[1])


# ============================================================
# 3. NORMALIZATION (TRAINING-TIME)
# ============================================================

normalizer = FeatureNormalizer()
feature_vectors_norm = normalizer.fit_transform(feature_vectors)

# ============================================================
# 4. TEMPORAL SEQUENCES
# ============================================================

T = 64  # number of windows (~32s history)
X_seq = build_sequences(feature_vectors_norm, window_size=T)

print("Sequence tensor:", X_seq.shape)

# ============================================================
# 5. TREND TARGETS (SANITY TARGET)
# ============================================================

trend_targets = np.zeros((X_seq.shape[0], 5), dtype=np.float32)

# ============================================================
# 6. DATASET & LOADER
# ============================================================

dataset = VibrosenseDataset(X_seq, trend_targets)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

# ============================================================
# 7. MODEL + TRAINER
# ============================================================

feature_dim = X_seq.shape[-1]
model = VibroSenseV2(feature_dim)

device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = Trainer(model, loader, device)

# ============================================================
# 8. TRAIN
# ============================================================

num_epochs = 20

for epoch in range(num_epochs):
    metrics = trainer.train_epoch(epoch)
    print(f"Epoch {epoch}:", metrics)
