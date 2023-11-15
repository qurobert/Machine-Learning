import unittest

import numpy as np
from z_score import zscore


class TestZScore(unittest.TestCase):
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    def test1(self):
        self.assertEqual(zscore(self.X).tolist(),
                         ([-0.08620324, 1.2068453, -0.86203236, 0.51721942, 0.94823559,
                           0.17240647, -1.89647119]))

    def test2(self):
        self.assertEqual(zscore(self.Y).flatten().tolist(),
                         ([0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659,
                           0.28795027, -1.72770165]))
