import numpy as np
import unittest
from vec_gradient import simple_gradient


class TestVecGradient(unittest.TestCase):
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    def test_empty(self):
        self.assertEqual(simple_gradient(self.x, self.y, np.array([])), None)

    def test_bad_value(self):
        self.assertEqual(simple_gradient(self.x, self.y, np.array(['a', 0])), None)

    def test_bad_shape_theta(self):
        self.assertEqual(simple_gradient(self.x, self.y, np.array([[0, 1], [2, 3]])), None)

    def test_example_1(self):
        theta1 = np.array([2, 0.7]).reshape((-1, 1))
        self.assertEqual(simple_gradient(self.x, self.y, theta1).tolist(), [[-19.0342574], [-586.66875564]])

    def test_example_2(self):
        theta2 = np.array([1, -0.4]).reshape((-1, 1))
        self.assertEqual(simple_gradient(self.x, self.y, theta2).tolist(), [[-57.86823748], [-2230.12297889]])

