import numpy as np


def simple_predict(x, theta):
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if not x.size or not theta.size:
        return None
    if x.shape[1] + 1 != theta.shape[0]:
        return None
    y_predict = []
    for i in x:
        y_predict.append(theta[0] + np.dot(i, theta[1:]))
    return np.array(y_predict)
