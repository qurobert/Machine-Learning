import numpy as np


def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape((-1, 1))
    if y.shape != y_hat.shape:
        return None
    return np.mean((y - y_hat) ** 2)


def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    mse = mse_(y, y_hat)
    if mse is None:
        return None
    return np.sqrt(mse)


def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape((-1, 1))
    if y.shape != y_hat.shape:
        return None
    return np.mean(np.abs(y - y_hat))


def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape((-1, 1))
    if y.shape != y_hat.shape:
        return None
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    return 1 - (ss_res / ss_tot)
