import unittest
import numpy as np
from plot import plot


class TestPlot(unittest.TestCase):
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

    def test_empty(self):
        plot(None, None, None)

    def test_incorrect_shape(self):
        plot(np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8]))
        plot(np.array([1, 2, 3]), np.array([4, 5]), np.array([7, 8, 9]))
        plot(np.array([1, 2]), np.array([4, 5, 6]), np.array([7, 8, 9]))

    def test_example_1(self):
        theta1 = np.array([[4.5], [-0.2]])
        plot(self.x, self.y, theta1)

    def test_example_2(self):
        theta2 = np.array([[-1.5], [2]])
        plot(self.x, self.y, theta2)

    def test_example_3(self):
        theta3 = np.array([[3], [0.3]])
        plot(self.x, self.y, theta3)
