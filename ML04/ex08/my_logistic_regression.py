import numpy as np


class MyLogisticRegression:
    supported_penalities = ['l2', 'none']

    def __init__(self, theta, alpha=0.000001, max_iter=1000000, penalty='l2', lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape(-1, 1)
        self.penalty = penalty if penalty in self.supported_penalities else 'none'
        self.lambda_ = lambda_ if self.penalty == 'l2' else 0

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

    def reg_log_loss_(self, y, y_hat):
        if y.size == 0 or y_hat.size == 0 or self.theta.size == 0 or y.shape != y_hat.shape:
            return None
        m = y.shape[0]
        self.theta[0] = 0
        reg_term = (self.lambda_ / (2 * m)) * np.dot(self.theta.T, self.theta).item()
        log_loss = - (1 / m) * (np.dot(y.T, np.log(y_hat)) + np.dot((1 - y).T, np.log(1 - y_hat))).item()
        return log_loss + reg_term

    def vec_reg_logistic_grad(self, y, x):
        if y.size == 0 or x.size == 0 or self.theta.size == 0 or y.shape[0] != x.shape[0]:
            return None
        m, n = x.shape
        x_augmented = np.hstack((np.ones((m, 1)), x))
        theta_reg = np.copy(self.theta)
        theta_reg[0] = 0
        gradient = (1 / m) * x_augmented.T.dot(self.sigmoid_(x_augmented.dot(self.theta)) - y) + (self.lambda_ / m) * theta_reg
        return gradient

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2:
            return None
        if x.shape[0] != y.shape[0]:
            return None

        for _ in range(self.max_iter):
            if self.penalty == 'l2':
                gradient = self.vec_reg_logistic_grad(y, x)
            else:
                gradient = self.gradient(x, y)
            if gradient is None:
                return None
            self.theta -= self.alpha * gradient

        return self.theta

    @staticmethod
    def data_spliter(x, y, train_size, val_size, test_size):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays.")
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be two-dimensional arrays.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows.")
        if not (0 <= train_size <= 1) or not (0 <= val_size <= 1) or not (0 <= test_size <= 1):
            raise ValueError("train_size, val_size, and test_size must be between 0 and 1.")
        if train_size + val_size + test_size != 1:
            raise ValueError("The sum of train_size, val_size, and test_size must be 1.")

        # Shuffle the data
        combined = np.hstack((x, y))
        np.random.shuffle(combined)

        # Calculate split indices
        train_end = int(combined.shape[0] * train_size)
        val_end = train_end + int(combined.shape[0] * val_size)

        # Split the data
        x_train = combined[:train_end, :-y.shape[1]]
        y_train = combined[:train_end, -y.shape[1]:]
        x_val = combined[train_end:val_end, :-y.shape[1]]
        y_val = combined[train_end:val_end, -y.shape[1]:]
        x_test = combined[val_end:, :-y.shape[1]]
        y_test = combined[val_end:, -y.shape[1]:]

        return x_train, x_val, x_test, y_train, y_val, y_test
