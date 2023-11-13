import numpy as np


def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        Raises:
            This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return
    if y.size == 0 or y_hat.size == 0 or y is None or y_hat is None:
        return None
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(-1, 1)
    if y.shape != y_hat.shape:
        return None
    J_elem = (y_hat - y) ** 2
    if J_elem is None:
        return
    return float(np.mean(J_elem) / 2)