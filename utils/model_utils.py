import numpy as np
from typing import Tuple

def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float=0.2,
        train_size: float=None,
        random_state: int=42
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    assert 0 < test_size < 1 or (train_size and 0 < train_size < 1), \
    "You must provide a valid value for train_size or test_size: between 0 and 1"

    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    if test_size:
        split_ratio = int(len(indices) * (1 - test_size))
    elif train_size:
        split_ratio = int(len(indices) * train_size)
    else:
        raise ValueError("Either train_size or test_size need to be not None. Both cannot be None.")

    train_idx, test_idx = indices[:split_ratio], indices[split_ratio:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def normalize(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        max_v = np.max(X)
        min_v = np.min(X)
        if max_v == min_v:
            return np.zeros_like(X)
        return ((X - min_v)/(max_v - min_v))
    
    max_v = np.max(X, axis=0)
    min_v = np.min(X, axis=0)
    range_v = max_v - min_v
    range_v[range_v == 0] = 1 # to avoid 0

    return ((X - min_v) / range_v)