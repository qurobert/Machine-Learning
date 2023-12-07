import numpy as np
import unittest
from gradient import gradient


class TestGradient(unittest.TestCase):
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    theta1 = np.array([3, 0.5, -6]).reshape((-1, 1))

    def test_shape(self):
        self.assertTrue(np.allclose(gradient(self.x, self.y, self.theta1), np.array([[-33.71428571], [-37.35714286], [183.14285714], [-393.]])))

    def test_shape2(self):
        theta2 = np.array([0, 0, 0]).reshape((-1, 1))
        self.assertTrue(np.allclose(gradient(self.x, self.y, theta2), np.array([[-0.71428571], [0.85714286], [23.28571429], [-26.42857143]])))

