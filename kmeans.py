import numpy as np


class KMeans:
    def __init__(self, X, ct_indices=[1, 4, 6]):
        self.cts = X[ct_indices]
        self.X = X

    def fit(self, steps=4):
        self.move_ct()
        self.move_ct()

    def closest_ct(self):
        dist = np.sqrt(((self.X - self.cts[:, np.newaxis])**2).sum(axis=2))
        print(dist[:, 5])
        return np.argmin(dist, axis=0)

    def move_ct(self):
        closest = self.closest_ct()
        self.cts = np.array([self.X[closest==k].mean(axis=0) for k in range(self.cts.shape[0])])



