import numpy as np
import unittest
from sigmoid import sigmoid_


class TestSigmoid(unittest.TestCase):

    def test1(self):
        x = np.array([[-4]])
        self.assertEqual(sigmoid_(x), 0.01798620996209156)

    def test2(self):
        x = np.array([[2]])
        self.assertEqual(sigmoid_(x), 0.8807970779778823)

    def test3(self):
        x = np.array([[-4], [2], [0]])
        self.assertEqual(sigmoid_(x).tolist(), [[0.01798620996209156], [0.8807970779778823], [0.5]])





