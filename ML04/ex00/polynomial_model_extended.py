import numpy as np


def add_polynomial_features(x, power):
    if x.size == 0:
        return None
    if power < 1:
        return None

    m, n = x.shape
    polynomial_features = np.ones((m, n * power))
    for i in range(1, power + 1):
        polynomial_features[:, (i - 1) * n: i * n] = np.power(x, i)

    return polynomial_features
