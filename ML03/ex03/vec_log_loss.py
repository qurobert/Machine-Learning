import numpy as np


def vec_log_loss_(y, y_hat, eps=1e-15):
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None

    m = y.shape[0]
    ones = np.ones((m, 1))
    loss = - (1 / m) * np.sum(y * np.log(y_hat + eps) + (ones - y) * np.log(ones - y_hat + eps))
    return loss
