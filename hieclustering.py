import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

from pca import pca


class HierarchicalClustering:
    def __init__(self, X, classes=3):
        self.X = X
        self.classes = classes
        self.distance_matrix = pairwise_distances(self.X, metric='cosine')
        np.fill_diagonal(self.distance_matrix, np.inf)
        self.array = np.arange(self.distance_matrix.shape[0])
        self.clusters = []
        self.fig, self.ax = plt.subplots()

    def fit(self):
        plt.ion()
        self.clusters.append(list(self.array))
        while True:
            self.ax.scatter(pca(self.X)[:, 0], pca(self.X)[:, 1], c=self.array)
            plt.pause(1.5)
            self.find_cluster()
            if np.unique(self.clusters[-1]).shape[0] <= self.classes:
                break
        self.ax.scatter(pca(self.X)[:, 0], pca(self.X)[:, 1], c=self.array)
        plt.pause(1.5)
        plt.ioff()
        plt.show()
        print(self.clusters)

    def find_cluster(self):
        row_index = -1
        col_index = -1
        min_val = np.inf
        for i in range(0, self.distance_matrix.shape[0]):
            for j in range(0, self.distance_matrix.shape[1]):
                if self.distance_matrix[i][j] <= min_val:
                    min_val = self.distance_matrix[i][j]
                    row_index = i
                    col_index = j
        for i in range(0, self.distance_matrix.shape[0]):
            if i != col_index:
                temp = min(self.distance_matrix[col_index][i], self.distance_matrix[row_index][i])
                self.distance_matrix[col_index][i] = temp
                self.distance_matrix[i][col_index] = temp

        for i in range(0, self.distance_matrix.shape[0]):
            self.distance_matrix[row_index][i] = np.inf
            self.distance_matrix[i][row_index] = np.inf

        minimum = min(row_index, col_index)
        maximum = max(row_index, col_index)
        for n in range(len(self.array)):
            if self.array[n] == maximum:
                self.array[n] = minimum
        self.clusters.append(list(self.array))
