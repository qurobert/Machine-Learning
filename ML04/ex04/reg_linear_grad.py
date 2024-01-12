import numpy as np


def reg_linear_grad(y, x, theta, lambda_):
    if y.size == 0 or x.size == 0 or theta.size == 0 or y.shape[0] != x.shape[0]:
        return None
    m, n = x.shape
    x_augmented = np.hstack((np.ones((m, 1)), x))
    gradient = np.zeros((n + 1, 1))
    for j in range(n + 1):
        if j == 0:
            gradient[j] = np.sum(x_augmented[:, j].reshape(-1, 1) * (x_augmented.dot(theta) - y)) / m
        else:
            gradient[j] = (np.sum(x_augmented[:, j].reshape(-1, 1) * (x_augmented.dot(theta) - y)) / m) + (lambda_ * theta[j] / m)
    return gradient


def vec_reg_linear_grad(y, x, theta, lambda_):
    if y.size == 0 or x.size == 0 or theta.size == 0 or y.shape[0] != x.shape[0]:
        return None
    m, n = x.shape
    x_augmented = np.hstack((np.ones((m, 1)), x))
    theta_reg = np.copy(theta)
    theta_reg[0] = 0
    gradient = (1 / m) * x_augmented.T.dot(x_augmented.dot(theta) - y) + (lambda_ / m) * theta_reg
    return gradient

