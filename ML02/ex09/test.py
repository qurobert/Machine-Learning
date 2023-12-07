import numpy as np
from ML02.ex09.data_spliter import data_spliter
import unittest


class TestDataSpliter(unittest.TestCase):
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    x2 = np.array([[1, 42],
                   [300, 10],
                   [59, 1],
                   [300, 59],
                   [10, 42]])
    y2 = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

    def test_data_spliter_1(self):
        split_array = data_spliter(self.x1, self.y, 0.8)
        exemple_array = (np.array([1, 59, 42, 300]), np.array([10]), np.array([0, 0, 1, 0]), np.array([1]))
        self.assertTrue(split_array[0].__len__() == exemple_array[0].__len__())
        self.assertTrue(split_array[1].__len__() == exemple_array[1].__len__())
        self.assertTrue(split_array[2].__len__() == exemple_array[2].__len__())
        self.assertTrue(split_array[3].__len__() == exemple_array[3].__len__())

    def test_data_spliter_2(self):
        split_array = data_spliter(self.x1, self.y, 0.5)
        exemple_array = (np.array([59, 10]), np.array([1, 300, 42]), np.array([0, 1]), np.array([0, 0, 1]))
        self.assertTrue(split_array[0].__len__() == exemple_array[0].__len__())
        self.assertTrue(split_array[1].__len__() == exemple_array[1].__len__())
        self.assertTrue(split_array[2].__len__() == exemple_array[2].__len__())
        self.assertTrue(split_array[3].__len__() == exemple_array[3].__len__())

    def test_data_spliter_3(self):
        split_array = data_spliter(self.x2, self.y2, 0.8)
        exemple_array = (np.array(([[10, 42],
                                    [300, 59],
                                    [59, 1],
                                    [300, 10]])), np.array(([[1, 42]])), np.array([0, 1, 0, 1]), np.array([0]))
        self.assertTrue(split_array[0].__len__() == exemple_array[0].__len__())
        self.assertTrue(split_array[1].__len__() == exemple_array[1].__len__())
        self.assertTrue(split_array[2].__len__() == exemple_array[2].__len__())
        self.assertTrue(split_array[3].__len__() == exemple_array[3].__len__())

    def test_data_spliter_4(self):
        split_array = data_spliter(self.x2, self.y2, 0.5)
        exemple_array = (np.array([[59, 1],
                                   [10, 42]]), np.array([[300, 10],
                                                         [300, 59],
                                                         [1, 42]]), np.array([0, 0]), np.array([1, 1, 0]))
        self.assertTrue(split_array[0].__len__() == exemple_array[0].__len__())
        self.assertTrue(split_array[1].__len__() == exemple_array[1].__len__())
        self.assertTrue(split_array[2].__len__() == exemple_array[2].__len__())
        self.assertTrue(split_array[3].__len__() == exemple_array[3].__len__())
