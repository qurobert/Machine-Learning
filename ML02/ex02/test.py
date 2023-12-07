import numpy as np
import unittest
from loss import loss_


class TestLoss(unittest.TestCase):
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    def test1(self):
        self.assertTrue(np.allclose(loss_(self.X, self.Y), 2.142857142857143))

    def test2(self):
        self.assertEqual(loss_(self.X, self.X), 0)
