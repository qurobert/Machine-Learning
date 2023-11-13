import numpy as np
import unittest
from fit import fit_


class TestFit(unittest.TestCase):
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000).tolist()

    def test_example_0(self):
        self.assertEqual(self.theta1, [[1.40709365], [1.1150909]])

    def test_example_1(self):
        m = self.x.shape[0]
        X_0 = np.column_stack((np.ones((m, 1)), self.x.reshape(-1, 1)))
        predict = np.round(np.dot(X_0, self.theta1), 8)
        expected = [[15.3408728], [25.38243697], [36.59126492], [55.95130097], [65.53471499]]
        for predict, expected in zip(predict.tolist(), expected):
            self.assertAlmostEqual(predict[0], expected[0])
