from TinyStatistician import TinyStatistician
import numpy as np
import unittest


class TestTinyStatistician(unittest.TestCase):
    x = [1, 42, 300, 10, 59]
    y = np.array(x)
    def test_validate_requirements(self):
        self.assertTrue(TinyStatistician.validate_requirements(self.x))
        self.assertTrue(TinyStatistician.validate_requirements(self.y))
        self.assertFalse(TinyStatistician.validate_requirements([]))
        self.assertFalse(TinyStatistician.validate_requirements([1, 2, "a"]))
        self.assertFalse(TinyStatistician.validate_requirements([1, 2, [1, 2]]))
        self.assertFalse(TinyStatistician.validate_requirements([1, 2, (1, 2)]))
        self.assertFalse(TinyStatistician.validate_requirements([1, 2, {"a": 1, "b": 2}]))

    def test_mean(self):

        self.assertEqual(TinyStatistician.mean(self.x), np.mean(self.x))
        self.assertEqual(TinyStatistician.mean(self.y), np.mean(self.y))
        self.assertIsNone(TinyStatistician.mean([]))

    def test_median(self):
        self.assertEqual(TinyStatistician.median(self.x), np.median(self.x))
        self.assertEqual(TinyStatistician.median(self.y), np.median(self.y))
        self.assertIsNone(TinyStatistician.median([]))

    def test_quartile(self):
        self.assertAlmostEqual(TinyStatistician.quartile(self.x)[0], np.percentile(self.x, 25))
        self.assertAlmostEqual(TinyStatistician.quartile(self.y)[0], np.percentile(self.y, 25))
        self.assertAlmostEqual(TinyStatistician.quartile(self.x)[1], np.percentile(self.x, 75))
        self.assertAlmostEqual(TinyStatistician.quartile(self.y)[1], np.percentile(self.y, 75))
        self.assertIsNone(TinyStatistician.quartile([]))

    def test_percentile(self):
        self.assertAlmostEqual(TinyStatistician.percentile(self.x, 10), np.percentile(self.x, 10))
        self.assertAlmostEqual(TinyStatistician.percentile(self.y, 10), np.percentile(self.y, 10))
        self.assertAlmostEqual(TinyStatistician.percentile(self.x, 15), np.percentile(self.x, 15))
        self.assertAlmostEqual(TinyStatistician.percentile(self.y, 15), np.percentile(self.y, 15))
        self.assertIsNone(TinyStatistician.percentile([], 25))
        self.assertIsNone(TinyStatistician.percentile(self.x, 101))
        self.assertIsNone(TinyStatistician.percentile(self.x, -1))

    def test_var(self):
        # In order to calculate the sample variable we need to set ddof=1 with numpy
        self.assertEqual(TinyStatistician.var(self.x), np.var(self.x, ddof=1))
        self.assertEqual(TinyStatistician.var(self.y), np.var(self.y, ddof=1))
        self.assertIsNone(TinyStatistician.var([]))

    def test_std(self):
        self.assertEqual(TinyStatistician.std(self.x), np.std(self.x, ddof=1))
        self.assertEqual(TinyStatistician.std(self.y), np.std(self.y, ddof=1))
        self.assertIsNone(TinyStatistician.std([]))

if __name__ == "__main__":
    unittest.main()
