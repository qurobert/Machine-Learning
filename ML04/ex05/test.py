import unittest
import numpy as np
from reg_logistic_grad import reg_logistic_grad, vec_reg_logistic_grad


class TestRegularizedLogisticGradient(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[0, 2, 3, 4],
                           [2, 4, 5, 5],
                           [1, 3, 2, 7]])
        self.y = np.array([[0], [1], [1]])
        self.theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    def test_example_1_1(self):
        result = reg_logistic_grad(self.y, self.x, self.theta, 1)
        expected = np.array([[-0.55711039],
                             [-1.40334809],
                             [-1.91756886],
                             [-2.56737958],
                             [-3.03924017]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        result = vec_reg_logistic_grad(self.y, self.x, self.theta, 1)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_example_2_1(self):
        result = reg_logistic_grad(self.y, self.x, self.theta, 0.5)
        expected = np.array([[-0.55711039],
                             [-1.15334809],
                             [-1.96756886],
                             [-2.33404624],
                             [-3.15590684]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        result = vec_reg_logistic_grad(self.y, self.x, self.theta, 0.5)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_example_3_1(self):
        result = reg_logistic_grad(self.y, self.x, self.theta, 0.0)
        expected = np.array([[-0.55711039],
                             [-0.90334809],
                             [-2.01756886],
                             [-2.10071291],
                             [-3.27257351]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        result = vec_reg_logistic_grad(self.y, self.x, self.theta, 0.0)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


if __name__ == '__main__':
    unittest.main()
