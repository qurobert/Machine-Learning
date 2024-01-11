import numpy as np
from ML03.ex00.sigmoid import sigmoid_


def vec_log_gradient(x, y, theta):
    if x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None

    m = x.shape[0]
    x = np.hstack((np.ones((m, 1)), x))  # Add a column of ones for the bias term
    y_hat = sigmoid_(np.dot(x, theta))

    gradient = (1 / m) * x.T.dot(y_hat - y)
    return np.round(gradient, decimals=8)
