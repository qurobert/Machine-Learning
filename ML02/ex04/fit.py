import numpy as np
from ML02.ex03.gradient import gradient


def fit_(x, y, theta, alpha, max_iter):
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(theta, np.ndarray) and isinstance(
            alpha, float) and isinstance(max_iter, int)):
        return None
    if x.shape[0] != y.shape[0] or (x.shape[1] + 1 != theta.shape[0]):
        return None

    for _ in range(max_iter):
        grad = gradient(x, y, theta)
        if grad is None:
            return None
        theta = theta - alpha * grad

    return theta
