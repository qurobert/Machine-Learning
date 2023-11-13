import numpy as np


class MyLinearRegression:
    """
    Description:
    My personal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def fit_(self, x, y):
        if not isinstance(self.alpha, float) or not isinstance(self.max_iter, int):
            return None
        self.thetas = self.thetas.astype(float)
        gradients = self.simple_gradient(x, y)
        if gradients is None:
            return None
        for _ in range(self.max_iter):
            self.thetas -= self.alpha * gradients
            gradients = self.simple_gradient(x, y)
            if gradients is None:
                return None
        self.thetas = np.round(self.thetas, 8)
        return self.thetas

    def predict_(self, x):
        if x is None or x.size == 0 or self.thetas is None or self.thetas.size == 0:
            return None
        if self.thetas.shape == (1, 2):
            self.thetas = self.thetas.reshape((2, 1))
        if (self.thetas.ndim == 1 and self.thetas.shape[0] != 2) or (self.thetas.ndim == 2 and self.thetas.shape != (2, 1)):
            return None
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1 and x.shape[0] != 1):
            return None
        x = np.column_stack((np.ones((x.shape[0], 1)), x.reshape(-1, 1)))
        return np.dot(x, self.thetas)

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.shape != y_hat.shape:
            return None
        if y.size == 0 or y_hat.size == 0 or y is None or y_hat is None:
            return None
        return np.round((y_hat - y) ** 2, 8)

    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.shape != y_hat.shape:
            return None
        if y.size == 0 or y_hat.size == 0 or y is None or y_hat is None:
            return None
        J_elem = (y_hat - y) ** 2
        return np.mean(J_elem) / 2

    def simple_gradient(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(self.thetas, np.ndarray):
            return None
        if x is None or y is None or self.thetas is None or x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None
        if self.thetas.shape == (1, 2):
            self.thetas = self.thetas.reshape((2, 1))
        if (self.thetas.ndim == 1 and self.thetas.shape[0] != 2) or (self.thetas.ndim == 2 and self.thetas.shape != (2, 1)):
            return None
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1 and x.shape[0] != 1):
            return None
        if x.shape != y.shape:
            return None
        m = x.shape[0]
        X_0 = np.column_stack((np.ones((m, 1)), x.reshape(-1, 1)))
        gradient = (1 / m) * np.dot(X_0.T, np.dot(X_0, self.thetas) - y.reshape(-1, 1))
        return np.round(gradient, 8)
