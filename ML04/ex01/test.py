import unittest
import numpy as np
from l2_reg import iterative_l2, l2


class TestL2Regularization(unittest.TestCase):

    def test_iterative_l2_example1(self):
        x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
        self.assertAlmostEqual(iterative_l2(x), 911.0)

    def test_l2_example2(self):
        x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
        self.assertAlmostEqual(l2(x), 911.0)

    def test_iterative_l2_example3(self):
        y = np.array([3, 0.5, -6]).reshape((-1, 1))
        self.assertAlmostEqual(iterative_l2(y), 36.25)

    def test_l2_example4(self):
        y = np.array([3, 0.5, -6]).reshape((-1, 1))
        self.assertAlmostEqual(l2(y), 36.25)


if __name__ == '__main__':
    unittest.main()
