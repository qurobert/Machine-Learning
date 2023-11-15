import numpy as np
import unittest
from minmax import minmax


class TestMinMax(unittest.TestCase):
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    def test1(self):
        self.assertEqual(minmax(self.X).flatten().tolist(), ([0.58333333, 1., 0.33333333, 0.77777778, 0.91666667,
                                                              0.66666667, 0.]))

    def test2(self):
        self.assertEqual(minmax(self.Y).flatten().tolist(), ([0.63636364, 1., 0.18181818, 0.72727273, 0.93939394,
                                                              0.6969697, 0.]))
