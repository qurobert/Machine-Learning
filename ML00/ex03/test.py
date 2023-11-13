import numpy as np
import unittest
from tools import add_intercept


class TestIntercept(unittest.TestCase):

    def test_empty_arg(self):
        self.assertEqual(add_intercept(np.array([])), None)
        self.assertEqual(add_intercept(None), None)

    def test_example_1(self):
        x = np.arange(1, 6)
        self.assertTrue(np.array_equal(add_intercept(x), [[1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.]]))

    def test_example_2(self):
        x = np.arange(1, 10).reshape((3, 3))
        self.assertTrue(np.array_equal(add_intercept(x), [[1., 1., 2., 3.], [1., 4., 5., 6.], [1., 7., 8., 9.]]))

    def test_non_numpy_array(self):
        self.assertEqual(add_intercept([1, 2, 3]), None)

    def test_none_input(self):
        self.assertEqual(add_intercept(None), None)

    def test_high_dimensional_input(self):
        self.assertEqual(add_intercept(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])), None)

