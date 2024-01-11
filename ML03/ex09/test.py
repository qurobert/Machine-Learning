import unittest
import numpy as np
from confusion_matrix import confusion_matrix_  # Make sure to import your implementation
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


class TestConfusionMatrix(unittest.TestCase):
    def test_confusion_matrix_example1(self):
        y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird']).reshape(-1, 1)
        y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet']).reshape(-1, 1)
        self.assertTrue(np.array_equal(confusion_matrix_(y, y_hat), sk_confusion_matrix(y, y_hat)))

    def test_confusion_matrix_example2(self):
        y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird']).reshape(-1, 1)
        y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet']).reshape(-1, 1)
        self.assertTrue(np.array_equal(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']),
                                       sk_confusion_matrix(y, y_hat, labels=['dog', 'norminet'])))

    def test_confusion_matrix_df_option1(self):
        y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird']).reshape(-1, 1)
        y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet']).reshape(-1, 1)
        result = confusion_matrix_(y, y_hat, df_option=True)
        expected = sk_confusion_matrix(y, y_hat)
        self.assertTrue(np.array_equal(result.to_numpy(), expected))

    def test_confusion_matrix_df_option2(self):
        y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird']).reshape(-1, 1)
        y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet']).reshape(-1, 1)
        result = confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True)
        expected = sk_confusion_matrix(y, y_hat, labels=['bird', 'dog'])
        self.assertTrue(np.array_equal(result.to_numpy(), expected))

if __name__ == '__main__':
    unittest.main()
