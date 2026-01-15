import numpy as np

def build_sequences(X, window_size, stride=1):
    """
    Builds sliding-window sequences from feature vectors.

    X: np.ndarray [N, F]
    window_size: int (T)
    stride: int (default 1)

    Returns:
        np.ndarray [N_seq, T, F]
    """
    X = np.asarray(X)
    N, F = X.shape

    sequences = []

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        seq = X[start:end]
        sequences.append(seq)

    if len(sequences) == 0:
        return np.empty((0, window_size, F), dtype=np.float32)

    return np.stack(sequences).astype(np.float32)
