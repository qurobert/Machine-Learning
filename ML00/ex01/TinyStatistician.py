from ctypes import Union
import numpy as np
import math
from math import sqrt

class TinyStatistician:

    @staticmethod
    def validate_requirements(x):
        if not isinstance(x, (list, np.ndarray)) or len(x) == 0:
            return False
        for i in x:
            if not isinstance(i, (int, float, np.int32, np.int64,  np.float32, np.float64)):
                return None
        return True

    @staticmethod
    def mean(x):
        if not TinyStatistician.validate_requirements(x):
            return None
        total_sum = 0.0
        for i in x:
            total_sum += i
        return float(total_sum / len(x))

    @staticmethod
    def median(x):
        if not TinyStatistician.validate_requirements(x):
            return None
        x = sorted(x)
        if len(x) % 2 == 0:
            return float(TinyStatistician.mean([x[len(x) // 2 - 1], x[len(x) // 2]]))
        else:
            return float(x[len(x) // 2])

    @staticmethod
    def quartile(x):
        if not TinyStatistician.validate_requirements(x):
            return None
        x = sorted(x)
        first_quartile = float(x[len(x) // 4])
        third_quartile = float(x[len(x) // 4 * 3])
        return [first_quartile, third_quartile]

    @staticmethod
    def percentile(x, p):
        if not TinyStatistician.validate_requirements(x):
            return None
        x = sorted(x)
        if p < 0 or p > 100:
            return None
        rank = (p / 100) * (len(x) - 1)
        if rank.is_integer():
            return float(x[rank])
        else:
            return float(x[int(rank)] + float(rank % 1) * (x[int(rank) + 1] - x[int(rank)]))

    @staticmethod
    def var(x):
        if not TinyStatistician.validate_requirements(x):
            return None
        mean = TinyStatistician.mean(x)
        total_sum = 0.0
        for i in x:
            total_sum += (i - mean) ** 2
        return float(total_sum / (len(x) - 1))

    @staticmethod
    def std(x):
        if not TinyStatistician.validate_requirements(x):
            return None
        return float(sqrt(TinyStatistician.var(x)))
