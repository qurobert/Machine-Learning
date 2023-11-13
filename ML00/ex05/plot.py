import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be a numpy.array, a vector of dimension m * 1.
        y: has to be a numpy.array, a vector of dimension m * 1.
        theta: has to be a numpy.array, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """

    if x is None or y is None or theta is None:
        return
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return
    if x.shape[0] != y.shape[0]:
        return
    if (theta.ndim == 1 and theta.shape[0] != 2) or (theta.ndim == 2 and theta.shape != (2, 1) and theta.shape != (1, 2)):
        return
    if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1 and x.shape[0] != 1):
        return

    prediction = theta[0] + theta[1] * x

    plt.scatter(x, y, c='blue', label='Data points')

    plt.plot(x, prediction, c='red', label='Prediction line')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data and Prediction Line')

    plt.legend()

    plt.show()
