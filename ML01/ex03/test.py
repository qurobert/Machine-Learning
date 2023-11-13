import numpy as np
from my_linear_regression import MyLinearRegression as MyLR
import unittest


class TestMyLinearRegression(unittest.TestCase):
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLR(np.array([[2], [0.7]]))
    lr2 = MyLR(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    y_hat = lr1.predict_(x)
    y_hat2 = lr2.predict_(x)

    def test_example00_(self):
        self.assertEqual(self.y_hat.tolist(), [[10.74695094], [17.05055804], [24.08691674], [36.24020866], [42.25621131]])

    def test_example01_(self):
        self.assertEqual(self.lr1.loss_elem_(self.y, self.y_hat).tolist(), [[710.45867381], [364.68645485], [469.96221651], [108.97553412], [299.37111101]])
    #
    def test_example02_(self):
        self.assertAlmostEqual(self.lr1.loss_(self.y, self.y_hat), 195.34539903032385)

    def test_example10_(self):
        self.assertEqual(self.lr2.thetas.tolist(), [[1.40709365], [1.1150909]])

    def test_example11_(self):
        np.testing.assert_almost_equal(self.y_hat2, np.array([[15.3408728], [25.38243697], [36.59126492], [55.95130097], [65.53471499]]), decimal=7, err_msg='', verbose=True)

    def test_example12_(self):
        np.testing.assert_almost_equal(self.lr2.loss_elem_(self.y, self.y_hat2).tolist(), [[486.66604863], [115.88278416], [84.16711596], [85.96919719], [35.71448348]], decimal=6, err_msg='', verbose=True)

    def test_example13_(self):
        np.testing.assert_almost_equal(self.lr2.loss_(self.y, self.y_hat2), 80.83996294128525, decimal=7, err_msg='', verbose=True)
