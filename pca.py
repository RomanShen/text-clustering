import numpy as np


def normalize(X):
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    return (X - X_mean) / X_std


def pca(X, num_p=2):
    cov_mat = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
    index = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:, index]
    return np.matmul(X, eigen_vectors[:, :num_p])

