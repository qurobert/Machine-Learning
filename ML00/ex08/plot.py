import matplotlib.pyplot as plt
import numpy as np
from ex07.vec_loss import loss_
from ex04.prediction import predict_


def plot_with_loss(x, y, theta):
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

    y_hat = predict_(x, theta)
    loss = loss_(y, y_hat) * 2

    # Create the plot
    plt.figure()
    plt.scatter(x, y, label='Data points')
    plt.plot(x, y_hat, color='red', label='Prediction')

    # Add loss to the title
    plt.title(f"Loss (J) = {loss:.6f}")

    if x.ndim == 1:
        x = x.reshape((-1, 1))
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape((-1, 1))
    # Add lines for the loss
    for xi, yi, yhati in zip(x, y, y_hat):
        plt.plot([xi, xi], [yi, yhati], color='red', linestyle='--')

    # Show legends and labels
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.show()
