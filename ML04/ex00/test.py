import numpy as np
import unittest
from polynomial_model_extended import add_polynomial_features


class TestAddPolynomialFeatures(unittest.TestCase):

    def test1(self):
        x = np.arange(1, 11).reshape(5, 2)
        expected_output = np.array([[1, 2, 1, 4, 1, 8],
                                    [3, 4, 9, 16, 27, 64],
                                    [5, 6, 25, 36, 125, 216],
                                    [7, 8, 49, 64, 343, 512],
                                    [9, 10, 81, 100, 729, 1000]])
        np.testing.assert_array_almost_equal(add_polynomial_features(x, 3), expected_output)

    def test2(self):
        x = np.arange(1, 11).reshape(5, 2)
        expected_output = np.array([[1, 2, 1, 4, 1, 8, 1, 16],
                                    [3, 4, 9, 16, 27, 64, 81, 256],
                                    [5, 6, 25, 36, 125, 216, 625, 1296],
                                    [7, 8, 49, 64, 343, 512, 2401, 4096],
                                    [9, 10, 81, 100, 729, 1000, 6561, 10000]])
        np.testing.assert_array_almost_equal(add_polynomial_features(x, 4), expected_output)


if __name__ == '__main__':
    unittest.main()
