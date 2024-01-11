import numpy as np
import unittest
from my_logistic_regression import MyLogisticRegression as MyLR


class BaseTestMyLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shared data for all tests
        cls.X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
        cls.Y = np.array([[1], [0], [1]])
        cls.thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])

    def setUp(self):
        # New instance of MyLR for each test
        self.mylr = MyLR(self.thetas)


class TestMyLogisticRegressionPredict(BaseTestMyLogisticRegression):
    def test_predict_initial(self):
        prediction = self.mylr.predict_(self.X)
        expected_output = np.array([[0.99930437], [1.], [1.]])
        self.assertTrue(np.allclose(prediction, expected_output, atol=1e-2))


class TestMyLogisticRegressionLoss(BaseTestMyLogisticRegression):
    def test_loss_initial(self):
        loss = self.mylr.loss_(self.Y, self.mylr.predict_(self.X))
        expected_output = 11.513157421577004
        self.assertAlmostEqual(loss, expected_output, places=2)


class TestMyLogisticRegressionFit(BaseTestMyLogisticRegression):
    def test_fit_and_predict(self):
        print(self.mylr.theta)
        self.mylr.fit_(self.X, self.Y)
        prediction = self.mylr.predict_(self.X)
        expected_theta = np.array([[2.11826435], [0.10154334], [6.43942899], [-5.10817488], [0.6212541]])
        expected_prediction = np.array([[0.57606717], [0.68599807], [0.06562156]])
        self.assertTrue(np.allclose(self.mylr.theta, expected_theta, atol=1e-2))
        self.assertTrue(np.allclose(prediction, expected_prediction, atol=1e-2))


class TestMyLogisticRegressionLossAfterFit(BaseTestMyLogisticRegression):
    def test_loss_after_fit(self):
        self.mylr.fit_(self.X, self.Y)
        loss = self.mylr.loss_(self.Y, self.mylr.predict_(self.X))
        expected_output = 1.4779126923052268
        self.assertAlmostEqual(loss, expected_output, places=2)


if __name__ == '__main__':
    unittest.main()
