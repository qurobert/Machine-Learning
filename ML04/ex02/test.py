import unittest
import numpy as np
from linear_loss_reg import reg_loss_


class TestRegularizedLoss(unittest.TestCase):

    def test_regularized_loss_example1(self):
        y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
        y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
        theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
        self.assertAlmostEqual(reg_loss_(y, y_hat, theta, 0.5), 0.8503571428571429)

    def test_regularized_loss_example2(self):
        y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
        y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
        theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
        self.assertAlmostEqual(reg_loss_(y, y_hat, theta, 0.05), 0.5511071428571429)

    def test_regularized_loss_example3(self):
        y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
        y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
        theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
        self.assertAlmostEqual(reg_loss_(y, y_hat, theta, 0.9), 1.116357142857143)


if __name__ == '__main__':
    unittest.main()
