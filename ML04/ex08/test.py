import unittest
import numpy as np
from my_logistic_regression import MyLogisticRegression as mylogr


class TestMyLogisticRegression(unittest.TestCase):
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    def test_model_with_l2_penalty(self):
        model = mylogr(self.theta, lambda_=5.0)
        self.assertEqual(model.penalty, 'l2')
        self.assertEqual(model.lambda_, 5.0)

    def test_model_with_no_penalty(self):
        model = mylogr(self.theta, penalty=None)
        self.assertEqual(model.penalty, 'none')
        self.assertEqual(model.lambda_, 0.0)

    def test_model_with_no_penalty_and_lambda(self):
        model = mylogr(self.theta, penalty=None, lambda_=2.0)
        self.assertEqual(model.penalty, 'none')
        self.assertEqual(model.lambda_, 0.0)


if __name__ == '__main__':
    unittest.main()
