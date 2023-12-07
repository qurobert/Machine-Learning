import numpy as np


def predict_(x, theta):
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if not x.size or not theta.size:
        return None
    if x.shape[1] + 1 != theta.shape[0]:
        return None
    # Add a column of 1's to x
    x0 = np.hstack((np.ones((x.shape[0], 1)), x))
    # Perform matrix multiplication
    y_hat = x0.dot(theta)
    return y_hat
