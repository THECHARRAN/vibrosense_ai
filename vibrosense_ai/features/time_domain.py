import numpy as np
from scipy.stats import kurtosis, skew
def time_features(signal: np.ndarray):
    eps=1e-8
    feats={}
    feats["mean"]=signal.mean()
    feats["rms"]=np.sqrt(npmean(signal**2))
    feats["var"]=signal.var()
    feats["std"]=signal.std()
    feats["ptp"]=signal.ptp()
    feats["crest_factor"]=np.max(np.abs(signal))/(feats["rms"]+eps)
    feats["zcr"]=np.mean(signal[:-1]*signal[1:]<0)
    feats["kurtosis"]=kurtosis(signal)
    feats["skewness"]=skew(signal)
    return feats

