import numpy as np
import unittest
from prediction import predict_


class TestPrediction(unittest.TestCase):
    x = np.arange(1, 13).reshape((4, -1))

    def test1(self):
        theta1 = np.array([5, 0, 0, 0]).reshape(-1, 1)
        self.assertTrue(np.array_equal(predict_(self.x, theta1), np.array([[5.], [5.], [5.], [5.]])))

    def test2(self):
        theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
        self.assertTrue(np.array_equal(predict_(self.x, theta2), np.array([[1.], [4.], [7.], [10.]])))

    def test3(self):
        theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
        self.assertTrue(np.allclose(predict_(self.x, theta3), np.array([[9.64], [24.28], [38.92], [53.56]]), atol=1e-8))

    def test4(self):
        theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
        self.assertTrue(np.allclose(predict_(self.x, theta4), np.array([[12.5], [32.], [51.5], [71.]]), atol=1e-8))
