import numpy as np


def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    return np.round((x - np.mean(x)) / np.std(x), 8)
