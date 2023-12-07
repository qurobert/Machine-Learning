import numpy as np
import unittest
from fit import fit_
from ML02.ex01.prediction import predict_


class TestFit(unittest.TestCase):
    x = np.array([[1.76405235, 0.40015721, 0.97873798],
                  [2.2408932, 1.86755799, -0.97727788],
                  [0.95008842, -0.15135721, -0.10321885],
                  [0.4105985, 0.14404357, 1.45427351]])
    y = np.array([[5.41808292],
                  [-1.67894503],
                  [2.31674486],
                  [6.4240762]])
    theta = np.array([[-5], [3], [-2], [0]])
    theta_original = np.array([[2.],
                               [0.5],
                               [-1.],
                               [3.]])

    x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta2 = np.array([[42.], [1.], [1.], [1.]])

    def test(self):
        self.assertTrue(
            np.allclose(fit_(self.x, self.y, self.theta, alpha=0.0005, max_iter=1500000), self.theta_original))

    def test1(self):
        self.assertTrue(np.allclose(fit_(self.x2, self.y2, self.theta2, alpha=0.0005, max_iter=1500000),
                                    np.array([[41.99], [0.97], [0.77], [-1.20]]), atol=1e-2))

