import numpy as np
import pandas as pd

from implementation.base_file_reader import FileReader
from implementation.constants import *


class FeatureExtractor(FileReader):
    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate_derivatives(values):
        timestamp = range(values.size)
        dv = {DERIVATIVE_Y: np.diff(values) / np.diff(timestamp),
              DERIVATIVE_X: np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}
        return np.insert(dv[DERIVATIVE_Y], 0, 0, axis=0)

    # @staticmethod
    # def calculate_autocorrelation(values, window_size=20):
    #     auto_correlations = []
    #     for value in values:
    #         res = np.array(np.corrcoef(np.array([value[:-window_size], value[window_size:]])))
    #         print(res.shape)
    #         auto_correlations.append(res)
    #
    #     return auto_correlations

    @staticmethod
    def calculate_rolling_mean(values, window_size=20):
        res = values.rolling(window_size).mean()
        res[:window_size] = 0
        return res

    @staticmethod
    def calculate_rolling_sum(values, window_size=20):
        res = values.rolling(window_size).sum()
        res[:window_size] = 0
        return res

    @staticmethod
    def calculate_rolling_variance(values, window_size=20):
        res = values.rolling(window_size).var()
        res[:window_size] = 0
        return res

    @staticmethod
    def calculate_rolling_skewness(values, window_size=20):
        res = values.rolling(window_size).skew()
        res[:window_size] = 0
        return res

    @staticmethod
    def calculate_rolling_kurtosis(values, window_size=20):
        res = values.rolling(window_size).kurt()
        res[:window_size] = 0
        return res
