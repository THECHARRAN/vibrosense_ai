import numpy as np
class FeatureNormalizer:
    def __init__(self, eps=1e-8, ema_alpha=0.01):
        self.mean = None
        self.std = None
        self.eps = eps
        self.ema_alpha = ema_alpha

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + self.eps

    def transform(self, X):
        return (X - self.mean) / self.std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
