import numpy as np


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
        The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.array, a vector of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """

    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        return
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x is None or y is None or theta is None or x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if theta.shape == (1, 2):
        theta = theta.reshape((2, 1))
    if (theta.ndim == 1 and theta.shape[0] != 2) or (theta.ndim == 2 and theta.shape != (2, 1)):
        return None
    if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1 and x.shape[0] != 1):
        return None
    if x.shape != y.shape:
        return None
    sum0 = 0
    sum1 = 0
    for i in range(x.size):
        sum0 += (theta[0] + theta[1] * x[i]) - y[i]
        sum1 += ((theta[0] + theta[1] * x[i]) - y[i]) * x[i]
    j0 = sum0 / x.size
    j1 = sum1 / x.size
    return np.round(np.array([j0, j1]).reshape((2, 1)), 8)
