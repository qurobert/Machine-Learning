import numpy as np
import unittest
from mylinearregression import MyLinearRegression as MyLR


class BaseTestMyLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shared data for all tests
        cls.X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
        cls.Y = np.array([[23.], [48.], [218.]])

    def setUp(self):
        # New instance of MyLR for each test
        self.mylr = MyLR([[1.], [1.], [1.], [1.], [1]])


class TestMyLinearRegressionInitial(BaseTestMyLinearRegression):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.y_hat = np.array([[8.], [48.], [323.]])

    def test_predict_initial(self):
        self.assertTrue(np.array_equal(self.mylr.predict_(self.X).tolist(), self.y_hat))

    def test_cost_elem_(self):
        self.assertEqual(self.mylr.loss_elem_(self.Y, self.y_hat).tolist(), ([[225.], [0.], [11025.]]))

    def test_cost_(self):
        self.assertEqual(self.mylr.loss_(self.Y, self.y_hat), 1875.0)


class TestMyLinearRegressionAfterFit(BaseTestMyLinearRegression):
    def setUp(self):
        super().setUp()
        self.mylr.alpha = 1.6e-4
        self.mylr.max_iter = 200000
        self.mylr.fit_(self.X, self.Y)

    def test_fit_(self):
        self.assertTrue(np.allclose(self.mylr.thetas, np.array([[18.188], [2.767], [-0.374], [1.392], [0.017]])
                                    , atol=1e-2))

    def test_predict_after_fit(self):
        y_hat2 = self.mylr.predict_(self.X)
        self.assertTrue(np.allclose(y_hat2, np.array([[23.417], [47.489], [218.065]]), atol=1e-2))

    def test_loss_elem_after_fit(self):
        y_hat2 = self.mylr.predict_(self.X)
        loss_elem = self.mylr.loss_elem_(self.Y, y_hat2)
        self.assertTrue(np.allclose(loss_elem, np.array(([[0.174], [0.260], [0.004]])), atol=1e-2))

    def test_loss_after_fit(self):
        y_hat2 = self.mylr.predict_(self.X)
        loss = self.mylr.loss_(self.Y, y_hat2)
        self.assertTrue(np.allclose(loss, 0.0732, atol=1e-2))