import numpy as np
from ML03.ex00.sigmoid import sigmoid_

def logistic_predict_(x, theta):
    if x.size == 0 or theta.size == 0 or x.shape[1] + 1 != theta.shape[0]:
        return None

    # Add a column of ones to the input matrix x
    x = np.hstack((np.ones((x.shape[0], 1)), x))

    # Compute the predicted values using the sigmoid function
    return np.round(sigmoid_(np.dot(x, theta)),8)
