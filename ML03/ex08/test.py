import unittest
from other_metrics import *
from sklearn import metrics as sk


class TestOtherMetricsExample1(unittest.TestCase):
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

    def testAccuracy(self):
        self.assertEqual(accuracy_score_(self.y, self.y_hat), sk.accuracy_score(self.y, self.y_hat))

    def testPrecision(self):
        self.assertEqual(precision_score_(self.y, self.y_hat), sk.precision_score(self.y, self.y_hat))

    def testRecall(self):
        self.assertEqual(recall_score_(self.y, self.y_hat), sk.recall_score(self.y, self.y_hat))

    def testF1Score(self):
        self.assertEqual(f1_score_(self.y, self.y_hat), sk.f1_score(self.y, self.y_hat))


class TestOtherMetricsExample2(unittest.TestCase):
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

    def testAccuracy(self):
        self.assertEqual(accuracy_score_(self.y, self.y_hat), sk.accuracy_score(self.y, self.y_hat))

    def testPrecision(self):
        self.assertEqual(precision_score_(self.y, self.y_hat, pos_label='dog'), sk.precision_score(self.y, self.y_hat, pos_label='dog'))

    def testRecall(self):
        self.assertEqual(recall_score_(self.y, self.y_hat, pos_label='dog'), sk.recall_score(self.y, self.y_hat, pos_label='dog'))

    def testF1Score(self):
        self.assertEqual(f1_score_(self.y, self.y_hat, pos_label='dog'), sk.f1_score(self.y, self.y_hat, pos_label='dog'))


class TestOtherMetricsExample3(unittest.TestCase):
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

    def testPrecision(self):
        self.assertEqual(precision_score_(self.y, self.y_hat, pos_label='norminet'),
                         sk.precision_score(self.y, self.y_hat, pos_label='norminet'))

    def testRecall(self):
        self.assertEqual(recall_score_(self.y, self.y_hat, pos_label='norminet'),
                         sk.recall_score(self.y, self.y_hat, pos_label='norminet'))

    def testF1Score(self):
        self.assertEqual(f1_score_(self.y, self.y_hat, pos_label='norminet'),
                         sk.f1_score(self.y, self.y_hat, pos_label='norminet'))