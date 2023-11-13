import unittest
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
from other_losses import mse_, rmse_, mae_, r2score_


class TestOtherLosses(unittest.TestCase):
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    def test_mse_vs_sklearn(self):
        self.assertEqual(mse_(self.x, self.y), mean_squared_error(self.x, self.y))

    def test_rmse_vs_sklearn(self):
        self.assertEqual(rmse_(self.x, self.y), sqrt(mean_squared_error(self.x, self.y)))

    def test_mae_vs_sklearn(self):
        self.assertEqual(mae_(self.x, self.y), mean_absolute_error(self.x, self.y))

    def test_r2score_vs_sklearn(self):
        self.assertEqual(r2_score(self.x, self.y), r2score_(self.x, self.y))
