import numpy as np


def iterative_l2(theta):
    if theta.size == 0:
        return None
    reg = 0.0
    for j in range(1, len(theta)):
        reg += theta[j] ** 2
    return reg


def l2(theta):
    if theta.size == 0:
        return None
    theta[0] = 0
    return np.dot(theta.T, theta).item()
