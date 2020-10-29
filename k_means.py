from pca import pca

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.patches import Circle

import seaborn
seaborn.set()


class KMeans:
    def __init__(self, X, ct_indices=[1, 4, 6]):
    # def __init__(self, X, ct_indices=[1, 4, 7]):
    # def __init__(self, X, ct_indices=[2, 4, 7]):

        self.X = X
        self.cts = X[ct_indices]
        self.radius = np.zeros(self.cts.shape[0])
        self.ct_indices = np.zeros(self.X.shape[0])
        self.clusters = []
        self.fig, self.ax = plt.subplots()
        self.colors = ['#9ACD32', '#6B8E23', '#EE82EE']

    def fit(self, iter_times=4):
        for i in range(iter_times):
            self.ax.cla()
            self.ax.xaxis.set_major_locator(MultipleLocator(2.0))
            self.ax.yaxis.set_major_locator(MultipleLocator(2.0))

            self.calculate_radius()
            self.clusters.append(self.ct_indices)

            pca_X = pca(self.X)
            pca_cts = pca(self.cts)

            self.ax.set_title(f"KMeans: Step {i + 1}")

            self.ax.scatter(pca_X[:, 0], pca_X[:, 1], c=[self.colors[i] for i in self.ct_indices], s=80)
            for l in range(self.X.shape[0]):
                self.ax.text(pca_X[l, 0] + 0.08, pca_X[l, 1], l + 1, verticalalignment='center', horizontalalignment='left')

            self.ax.scatter(pca_cts[:, 0], pca_cts[:, 1], c='r', s=100)
            for l in range(self.cts.shape[0]):
                cir = Circle(xy=(pca_cts[l, 0], pca_cts[l, 1]), radius=self.radius[l], alpha=0.5)
                self.ax.add_patch(cir)

            plt.savefig(f"kmeans_{i}.png")
            plt.pause(3)

            self.new_cts()
        plt.ioff()
        plt.show()

        print(self.clusters)

    def calculate_radius(self):
        dist = np.sqrt(((self.X - self.cts[:, np.newaxis])**2).sum(axis=2))
        self.ct_indices = np.argmin(dist, axis=0)
        dist = dist.min(axis=0)
        tmp = []
        for i in range(self.cts.shape[0]):
            if np.sum(self.ct_indices == i) == 0:
                tmp.append(0)
            else:
                tmp.append(dist[self.ct_indices == i].max(axis=0))
        self.radius = np.array(tmp)

    def new_cts(self):
        tmp = []
        for i in range(self.cts.shape[0]):
            if np.sum(self.ct_indices == i) == 0:
                tmp.append(self.cts[i])
            else:
                tmp.append(self.X[self.ct_indices == i].mean(axis=0))
        self.cts = np.array(tmp)

