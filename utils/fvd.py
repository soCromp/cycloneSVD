from typing import List, Tuple, Optional, Callable
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean))


def _temporal_resample(video, T):
    # duplicates frames at evenly-spaced intervals to reach T frames
    cur_T = video.shape[1]
    if cur_T == T or T is None:
        return video 
    
    idx = np.linspace(0, cur_T-1, num=T).round().astype(int)
    return video[:, idx]


class PCAEncoder:
    def __init__(self, n_components=256, whiten=False):
        self.scaler = StandardScaler(with_mean=True, with_std=False)
        self.pca = PCA(n_components=n_components, whiten=whiten)
        self.fitted = False 
        self.target_T = None 
        
    def fit(self, videos, target_T=None):
        # videos should be a numpy matrix (N, T, C, H, W)
        assert videos.ndim == 4 or videos.ndim == 5, \
            "Input videos should be a numpy matrix (N, T, C, H, W) or (N, T, H, W)" 
        self.target_T = target_T 
        videos = _temporal_resample(videos, target_T)
        videos_flat = videos.reshape(videos.shape[0], -1)
        X = self.scaler.fit_transform(videos_flat)
        self.pca.fit(X)
        self.fitted = True 
        return self 
    
    def encode(self, videos):
        assert self.fitted, "PCAEncoder not fitted yet"
        assert videos.ndim == 4 or videos.ndim == 5, \
            "Input videos should be a numpy matrix (N, T, C, H, W) or (N, T, H, W)" 
        videos = _temporal_resample(videos, self.target_T)
        videos_flat = videos.reshape(videos.shape[0], -1)
        X = self.scaler.transform(videos_flat)
        X_pca = self.pca.transform(X)
        return X_pca
    
    
def compute_fvd(videos_real, videos_synth, target_T=None, 
                n_components=256, encoder=None):
    """videos should be numpy matrices (N, T, C, H, W) or (N, T, H, W).
    Can pass pre-fitted encoder"""
    assert videos_real.ndim == videos_synth.ndim
    assert videos_real.ndim == 4 or videos_real.ndim == 5
    
    if encoder is None:
        encoder = PCAEncoder(n_components=n_components, whiten=True)
        encoder.fit(videos_real, target_T=target_T)
    
    feats_real = encoder.encode(videos_real)
    feats_synth= encoder.encode(videos_synth)
    
    mu_real = feats_real.mean(axis=0)
    cov_real = np.cov(feats_real, rowvar=False)
    mu_synth = feats_synth.mean(axis=0)
    cov_synth = np.cov(feats_synth, rowvar=False)
    return _frechet_distance(mu_real, cov_real, mu_synth, cov_synth), encoder
        