import unittest
import numpy as np
from reg_linear_grad import reg_linear_grad, vec_reg_linear_grad


class TestRegularizedLinearGradient(unittest.TestCase):

    def setUp(self):
        self.x = np.array([
            [-6, -7, -9],
            [13, -2, 14],
            [-7, 14, -1],
            [-8, -4, 6],
            [-5, -9, 6],
            [1, -5, 11],
            [9, -11, 8]])
        self.y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
        self.theta = np.array([[7.01], [3], [10.5], [-6]])

    def test_example_1_1(self):
        result = reg_linear_grad(self.y, self.x, self.theta, 1)
        expected = np.array([[-60.99], [-195.64714286], [863.46571429], [-644.52142857]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_example_1_2(self):
        result = vec_reg_linear_grad(self.y, self.x, self.theta, 1)
        expected = np.array([[-60.99], [-195.64714286], [863.46571429], [-644.52142857]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_example_2_1(self):
        result = reg_linear_grad(self.y, self.x, self.theta, 0.5)
        expected = np.array([[-60.99], [-195.86142857], [862.71571429], [-644.09285714]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_example_2_2(self):
        result = vec_reg_linear_grad(self.y, self.x, self.theta, 0.5)
        expected = np.array([[-60.99], [-195.86142857], [862.71571429], [-644.09285714]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_example_3_1(self):
        result = reg_linear_grad(self.y, self.x, self.theta, 0.0)
        expected = np.array([[-60.99], [-196.07571429], [861.96571429], [-643.66428571]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_example_3_2(self):
        result = vec_reg_linear_grad(self.y, self.x, self.theta, 0.0)
        expected = np.array([[-60.99], [-196.07571429], [861.96571429], [-643.66428571]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


if __name__ == '__main__':
    unittest.main()
