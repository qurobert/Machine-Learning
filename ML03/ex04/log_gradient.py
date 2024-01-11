import numpy as np
from ML03.ex00.sigmoid import sigmoid_


def log_gradient(x, y, theta):
    if x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None

    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))  # Add a column of ones for the bias term
    y_hat = sigmoid_(np.dot(x, theta))

    gradient = np.zeros((n + 1, 1))
    for j in range(n + 1):
        gradient[j] = np.sum((y_hat - y) * x[:, j].reshape(-1, 1)) / m

    return np.round(gradient, decimals=8)
