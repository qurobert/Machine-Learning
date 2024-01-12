import numpy as np


def reg_log_loss_(y, y_hat, theta, lambda_):
    if y.size == 0 or y_hat.size == 0 or theta.size == 0 or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    theta[0] = 0
    reg_term = (lambda_ / (2 * m)) * np.dot(theta.T, theta).item()
    log_loss = - (1 / m) * (np.dot(y.T, np.log(y_hat)) + np.dot((1 - y).T, np.log(1 - y_hat))).item()
    return log_loss + reg_term
