import numpy as np


def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x is None or x.size == 0:
        return None
    if x.ndim > 2:
        return
    if not np.issubdtype(x.dtype, np.number):
        return
    # Get the number of rows (m)
    m = x.shape[0]

    # If x is a 1D array, reshape it into a 2D array
    if len(x.shape) == 1:
        x = x.reshape((m, 1))

    # Create a column of 1's of shape (m, 1)
    ones = np.ones((m, 1))

    # Horizontally stack the 1's and x
    return np.hstack((ones, x))
