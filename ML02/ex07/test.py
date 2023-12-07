import numpy as np
import unittest
from ML02.ex07.polynomial_model import add_polynomial_features


class TestPolynomialModel(unittest.TestCase):
    x = np.arange(1, 6).reshape(-1, 1)

    def test_polynomial_model(self):
        self.assertTrue(np.array_equal(add_polynomial_features(self.x, 3),
                                       np.array([[1., 1., 1.],
                                                 [2., 4., 8.],
                                                 [3., 9., 27.],
                                                 [4., 16., 64.],
                                                 [5., 25., 125.]])))

    def test_polynomial_model_2(self):
        self.assertTrue(np.array_equal(add_polynomial_features(self.x, 6),
                                       np.array([[1, 1, 1, 1, 1, 1],
                                                 [2, 4, 8, 16, 32, 64],
                                                 [3, 9, 27, 81, 243, 729],
                                                 [4, 16, 64, 256, 1024, 4096],
                                                 [5, 25, 125, 625, 3125, 15625]])))
