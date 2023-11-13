import numpy as np
from ex03.tools import add_intercept


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1.
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x and/or theta are not numpy.ndarray.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    if x is None or theta is None or x.size == 0 or theta.size == 0:
        return None
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if theta.shape == (1, 2):
        theta = theta.reshape((2, 1))
    if (theta.ndim == 1 and theta.shape[0] != 2) or (theta.ndim == 2 and theta.shape != (2, 1)):
        return None
    if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1 and x.shape[0] != 1):
        return None
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number):
        return

    # Add a column of ones to x
    X = add_intercept(x)

    # Perform matrix multiplication to get predictions
    y_hat = np.dot(X, theta)
    return y_hat
