import numpy as np
from ML01.ex01.vec_gradient import simple_gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    theta = theta.astype(float)
    gradients = simple_gradient(x, y, theta)
    if gradients is None:
        return None
    for _ in range(max_iter):
        theta -= alpha * gradients
        gradients = simple_gradient(x, y, theta)
        if gradients is None:
            return None
    return np.round(theta, 8)
