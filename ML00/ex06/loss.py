import numpy as np


def loss_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be a numpy.array, a vector.
        y_hat: has to be a numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    if y.size == 0 or y_hat.size == 0 or y is None or y_hat is None:
        return None
    J_elem = (y_hat - y)**2
    return J_elem

def loss_(y, y_hat):
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be a numpy.array, a vector.
        y_hat: has to be a numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0 or y is None or y_hat is None:
        return None
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(-1, 1)
    if y.shape != y_hat.shape:
        return None
    J_elem = loss_elem_(y, y_hat)
    if J_elem is None:
        return

    J_value = float(np.mean(J_elem) / 2)
    return J_value
