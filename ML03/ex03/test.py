import numpy as np
import unittest
from vec_log_loss import vec_log_loss_
from ML03.ex01.log_pred import logistic_predict_


class TestLogLoss(unittest.TestCase):
    def test1(self):
        y1 = np.array([1]).reshape((-1, 1))
        x1 = np.array([4]).reshape((-1, 1))
        theta1 = np.array([[2], [0.5]])
        y_hat1 = logistic_predict_(x1, theta1)
        self.assertAlmostEqual(vec_log_loss_(y1, y_hat1), 0.01814992791780973)

    def test2(self):
        y2 = np.array([[1], [0], [1], [0], [1]])
        x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
        theta2 = np.array([[2], [0.5]])
        y_hat2 = logistic_predict_(x2, theta2)
        self.assertAlmostEqual(vec_log_loss_(y2, y_hat2), 2.4825011602474483, places=5)

    def test3(self):
        y3 = np.array([[0], [1], [1]])
        x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
        theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
        y_hat3 = logistic_predict_(x3, theta3)
        self.assertAlmostEqual(vec_log_loss_(y3, y_hat3), 2.9938533108607053, places=5)
