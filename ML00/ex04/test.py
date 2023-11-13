import numpy as np
import unittest
from prediction import predict_


class TestPrediction(unittest.TestCase):
    x = np.arange(1, 6)

    def test_example_1(self):
        theta1 = np.array([[5], [0]])
        self.assertTrue(np.array_equal(predict_(self.x, theta1), np.array([[5.], [5.], [5.], [5.], [5.]])))

    def test_example_2(self):
        theta2 = np.array([[0], [1]])
        self.assertTrue(np.array_equal(predict_(self.x, theta2), np.array([[1.], [2.], [3.], [4.], [5.]])))

    def test_example_3(self):
        theta3 = np.array([[5], [3]])
        self.assertTrue(np.array_equal(predict_(self.x, theta3), np.array([[8.], [11.], [14.], [17.], [20.]])))

    def test_example_4(self):
        theta4 = np.array([[-3], [1]])
        self.assertTrue(np.array_equal(predict_(self.x, theta4), np.array([[-2.], [-1.], [0.], [1.], [2.]])))
