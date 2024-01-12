import numpy as np


def reg_loss_(y, y_hat, theta, lambda_):
    if y.size == 0 or y_hat.size == 0 or theta.size == 0 or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    theta[0] = 0
    reg_term = lambda_ * np.dot(theta.T, theta).item()
    return (1 / (2 * m)) * (np.dot((y_hat - y).T, (y_hat - y)).item() + reg_term)
