import numpy as np
import unittest
from ex04.prediction import predict_
from loss import loss_elem_, loss_


class TestLossFunction(unittest.TestCase):
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

    def test_example_1(self):
        self.assertTrue(np.array_equal(loss_elem_(self.y1, self.y_hat1), [[0.], [1], [4], [9], [16]]))

    def test_example_2(self):
        self.assertEqual(loss_(self.y1, self.y_hat1), 3.0)

    def test_example_3(self):
        self.assertEqual(loss_(self.y2, self.y_hat2), 2.142857142857143)

    def test_example_4(self):
        self.assertEqual(loss_(self.y2, self.y2), 0.0)
