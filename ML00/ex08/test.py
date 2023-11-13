import numpy as np
from plot import plot_with_loss
import unittest


class TestPlot(unittest.TestCase):
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

    def test_example_1(self):
        theta1 = np.array([[18, -1]])
        plot_with_loss(self.x, self.y, theta1)

    def test_example_2(self):
        theta2 = np.array([[14], [0]])
        plot_with_loss(self.x, self.y, theta2)

    def test_example_3(self):
        theta3 = np.array([12, 0.8])
        plot_with_loss(self.x, self.y, theta3)
