import unittest
import numpy as np
from vec_loss import loss_


class TestVecLoss(unittest.TestCase):
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    def test_example_1(self):
        self.assertEqual(loss_(self.Y, self.X), 2.142857142857143)

    def test_example_2(self):
        self.assertEqual(loss_(self.Y, self.Y), 0.0)
