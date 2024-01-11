import numpy as np


class MyLogisticRegression:
    def __init__(self, theta, alpha=0.000001, max_iter=1000000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape(-1, 1)

    def sigmoid_(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if self.theta.shape[0] != x.shape[1] + 1:
            return None
        x_augmented = np.hstack((np.ones((x.shape[0], 1)), x))
        return self.sigmoid_(np.dot(x_augmented, self.theta))

    def loss_elem_(self, y, y_hat):
        eps = 1e-15  # To prevent log(0)
        return - (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    def loss_(self, y, y_hat):
        loss_elements = self.loss_elem_(y, y_hat)
        return np.sum(loss_elements) / len(y)

    def gradient(self, x, y):
        m = x.shape[0]
        x_augmented = np.hstack((np.ones((m, 1)), x))
        y_hat = self.predict_(x)
        gradient = (1 / m) * np.dot(x_augmented.T, y_hat - y)
        return gradient

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2:
            return None
        if x.shape[0] != y.shape[0]:
            return None

        for _ in range(self.max_iter):
            gradient = self.gradient(x, y)
            if gradient is None:
                return None
            self.theta -= self.alpha * gradient

        return self.theta

    def data_spliter(x, y, proportion):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size == 0 or y.size == 0:
            return None
        if x.ndim != 2 or y.ndim != 2:
            return None
        if x.shape[0] != y.shape[0]:
            return None
        if not isinstance(proportion, float):
            return None
        if proportion < 0 or proportion > 1:
            return None

        # Combine x and y to shuffle them together
        combined = np.hstack((x, y))
        np.random.shuffle(combined)

        # Split the combined array back into x and y components
        split_idx = int(combined.shape[0] * proportion)
        x_train = combined[:split_idx, :-y.shape[1]]
        x_test = combined[split_idx:, :-y.shape[1]]
        y_train = combined[:split_idx, -y.shape[1]:]
        y_test = combined[split_idx:, -y.shape[1]:]

        return x_train, x_test, y_train, y_test
