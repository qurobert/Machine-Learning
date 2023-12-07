import numpy as np


def gradient(x, y, theta):
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape[0] != y.shape[0]:
        return None
    if x.shape[1] == theta.shape[0]:
        theta = np.vstack([np.zeros((1, 1)), theta])
    if x.shape[1] + 1 != theta.shape[0]:
        return None
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(theta, np.ndarray)):
        return None

    m = x.shape[0]
    x0 = np.hstack((np.ones((m, 1)), x))  # Adding a column of 1's to x
    return (1 / m) * x0.T.dot(x0.dot(theta) - y)
