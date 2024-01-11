import numpy as np
import unittest
from log_pred import logistic_predict_


class TestLogPred(unittest.TestCase):
    def test1(self):
        x = np.array([4]).reshape((-1, 1))
        theta = np.array([[2], [0.5]])
        self.assertEqual(logistic_predict_(x, theta), np.array(0.98201379))

    def test2(self):
        x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
        theta2 = np.array([[2], [0.5]])
        self.assertEqual(logistic_predict_(x2, theta2).tolist(), [[0.98201379], [0.99624161], [0.97340301], [0.99875204], [0.90720705]])

    def test3(self):
        x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
        theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
        self.assertEqual(logistic_predict_(x3, theta3).tolist(), [[0.03916572], [0.00045262], [0.2890505]])




