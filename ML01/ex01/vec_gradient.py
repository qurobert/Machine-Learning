import numpy as np


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
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
    # Transform x to X_0 by adding a column of ones
    m = x.shape[0]
    X_0 = np.column_stack((np.ones((m, 1)), x.reshape(-1, 1)))

    # Compute the gradient
    gradient = (1 / m) * np.dot(X_0.T, np.dot(X_0, theta) - y.reshape(-1, 1))

    return np.round(gradient, 8)
