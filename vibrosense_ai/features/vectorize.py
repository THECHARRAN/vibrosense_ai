import numpy as np

def vectorize_features(feats, feature_order):
    """
    Converts a feature dictionary into a fixed-order numeric vector.

    feats: dict[str, float]
    feature_order: list[str]

    Returns:
        np.ndarray shape [F]
    """
    vector = []

    for key in feature_order:
        value = feats.get(key, 0.0)

        # Defensive programming
        if not np.isfinite(value):
            value = 0.0

        vector.append(float(value))

    return np.array(vector, dtype=np.float32)
