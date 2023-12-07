import numpy as np


def loss_(y, y_hat):
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    m = len(y)
    return np.dot((y_hat - y).T, (y_hat - y)) / (2 * m)
