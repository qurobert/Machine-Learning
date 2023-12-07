import numpy as np

class MyLinearRegression:
    """
    Description:
    My personal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape(-1, 1)

    def fit_(self, x, y):
        if not isinstance(self.alpha, float) or not isinstance(self.max_iter, int):
            return None
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            return None
        if x.shape[0] != y.shape[0]:
            return None

        for _ in range(self.max_iter):
            # Calculate gradient
            gradient = self.gradient(x, y)
            if gradient is None:
                return None

            # Update thetas
            self.thetas -= self.alpha * gradient

        return self.thetas

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            return None
        x_augmented = np.hstack((np.ones((x.shape[0], 1)), x))
        return np.dot(x_augmented, self.thetas)

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

    def gradient(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        m = x.shape[0]
        x_augmented = np.hstack((np.ones((m, 1)), x))
        gradient = (1 / m) * np.dot(x_augmented.T, np.dot(x_augmented, self.thetas) - y.reshape(-1, 1))
        return gradient