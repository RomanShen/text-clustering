import numpy as np
import matplotlib.pyplot as plt

from pca import pca


class KMeans:
    def __init__(self, X, ct_indices=[1, 4, 6]):
        self.cts = X[ct_indices]
        self.X = X
        self.clusters = {}
        self.fig, self.ax = plt.subplots()

    def fit(self, iter_times=5):
        plt.ion()
        for i in range(iter_times):
            cls = self.closest_ct()
            self.clusters[i] = cls
            self.move_ct()
            self.ax.scatter(pca(self.X)[:, 0], pca(self.X)[:, 1], c=cls)
            plt.pause(2)
        plt.ioff()
        plt.show()
        print(self.clusters)

    def closest_ct(self):
        dist = np.sqrt(((self.X - self.cts[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(dist, axis=0)

    def move_ct(self):
        closest = self.closest_ct()
        self.cts = np.array([self.X[closest == k].mean(axis=0) for k in range(self.cts.shape[0])])





