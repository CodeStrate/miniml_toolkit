# raw implementation of StandardScaler from sklearn
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.stdev = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.stdev = X.std(axis=0)
        self.stdev[self.stdev == 0] = 1 # avoid 0s
        return self # <-- basically stores the values in object like it happens with sklearn
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.stdev
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)