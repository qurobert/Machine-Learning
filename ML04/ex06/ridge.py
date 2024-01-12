import numpy as np


class MyRidge:

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape(-1, 1)
        self.lambda_ = lambda_

    def get_params_(self):
        return self.thetas

    def set_params_(self, thetas):
        self.thetas = np.array(thetas).reshape(-1, 1)

    def predict_(self, x):
        x_augmented = np.hstack((np.ones((x.shape[0], 1)), x))
        return x_augmented.dot(self.thetas)

    def loss_elem_(self, y, y_hat):
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        m = y.shape[0]
        return ((1 / (2 * m)) * np.sum(self.loss_elem_(y, y_hat))
                + (self.lambda_ / (2 * m)) * np.sum(self.thetas[1:] ** 2))

    def gradient_(self, x, y):
        m, n = x.shape
        x_augmented = np.hstack((np.ones((m, 1)), x))
        y_hat = self.predict_(x)
        theta_reg = np.copy(self.thetas)
        theta_reg[0] = 0
        return (1 / m) * x_augmented.T.dot(y_hat - y) + (self.lambda_ / m) * theta_reg

    def fit_(self, x, y):
        for _ in range(self.max_iter):
            gradient = self.gradient_(x, y)
            self.thetas -= self.alpha * gradient
        return self.thetas
