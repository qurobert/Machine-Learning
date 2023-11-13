import numpy as np


def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1.
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return
    if x is None or theta is None or x.size == 0 or theta.size == 0:
        return None
    if theta.shape == (1, 2):
        theta = theta.reshape((2, 1))
    if (theta.ndim == 1 and theta.shape[0] != 2) or (theta.ndim == 2 and theta.shape != (2, 1) and theta.shape != (1, 2)):
        return None
    if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1 and x.shape[0] != 1):
        return None
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number):
        return
    return np.array(theta[0] + theta[1] * x)
