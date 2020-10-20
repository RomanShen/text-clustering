import numpy as np


class KMeans:
    def __init__(self, ct_indices=[1, 4, 6]):
        self.ct_indices = ct_indices

    def fit(self, X):
        pass

    def closest_ct(self, X):
        cts = X[self.ct_indices]
        dist = np.sqrt(((X - cts[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(dist, axis=0)

    def move_ct(self, X):
        closest = self.closest_ct(X)
        cts = X[self.ct_indices]
        return np.array([X[closest==k].mean(axis=0) for k in range(cts.shape[0])])



