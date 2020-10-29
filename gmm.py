from sklearn.datasets.samples_generator import make_blobs
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import seaborn
seaborn.set()


class GMM:

    def __init__(self, X, n_clusters):
        # initialize models
        self.X = X
        self.n_examples, self.n_features = self.X.shape
        self.n_clusters = n_clusters
        self.mean = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(self.n_clusters, self.n_features))
        self.cov = np.vstack([np.identity(self.n_features)[np.newaxis, :] for _ in range(self.n_clusters)])
        self.p = np.full((n_clusters,), 1.0 / n_clusters)

        # initialize plot
        self.fig, (self.ax_likelihood, self.ax_ellipse) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.tight_layout(pad=4)
        self.fig.suptitle("Expectation Maximization")

        self.ax_likelihood.set_title("Log Likelihood Change")
        self.ax_likelihood.set_xlabel("iteration")
        self.ax_likelihood.set_ylabel("Log Likelihood")

        self.ax_ellipse.set_title("Fitting Points")
        self.ax_ellipse.set_xlabel("x")
        self.ax_ellipse.set_ylabel("y")

    def fit(self, iter_times):
        plt.ion()
        for i in range(iter_times):
            # e-step
            p_cluster = np.zeros((self.n_examples, self.n_clusters))
            for c in range(self.n_clusters):
                var = np.sum((self.X - self.mean[c]).dot(np.linalg.inv(self.cov[c])) * (self.X - self.mean[c]), axis=1)
                p_cluster[:, c] = self.p[c] * np.exp(-var / 2) / np.sqrt(np.abs(2 * np.pi * np.linalg.det(self.cov[c])))
            normalized_p_cluster = p_cluster / (np.sum(p_cluster, axis=1).reshape((-1, 1)))

            # m-step
            self.p = np.sum(normalized_p_cluster, axis=0) / self.n_examples
            for c in range(self.n_clusters):
                self.mean[c] = np.average(self.X, axis=0, weights=normalized_p_cluster[:, c])
                self.cov[c] = np.cov(X.T, aweights=normalized_p_cluster[:, c])

            # predict
            y_pred = np.argmax(normalized_p_cluster, axis=1)

            # plot log_likelihood
            log_likelihood = np.sum(np.log(np.sum(p_cluster, axis=0)))
            self.ax_likelihood.plot(i, log_likelihood, 'ro', markersize=4)

            # plot points
            self.ax_ellipse.cla()
            self.ax_ellipse.scatter(self.X[:, 0], self.X[:, 1], c=y_pred, s=40, cmap='viridis', zorder=2)
            self.ax_ellipse.axis('equal')
            w_factor = 0.2 / self.p.max()
            for pos, covar, w in zip(self.mean, self.cov, self.p):
                self.draw_ellipse(pos, covar, alpha=w * w_factor)
            plt.pause(0.2)
        plt.ioff()
        plt.show()

    def draw_ellipse(self, pos, cov, **kwargs):
        if cov.shape == (2, 2):
            U, s, Vt = np.linalg.svd(cov)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(cov)

        for nsig in range(1, 4):
            self.ax_ellipse.add_patch(Ellipse(pos, nsig * width, nsig * height, angle, **kwargs))


if __name__ == '__main__':
    X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
    X = X[:, ::-1]
    gmm = GMM(X, 4)
    gmm.fit(50)
