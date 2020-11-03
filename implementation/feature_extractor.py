import numpy as np
import pandas as pd

from implementation.base_file_reader import FileReader
from implementation.constants import *


class FeatureExtractor(FileReader):
    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate_derivatives(values):
        derivatives = []
        for value in values:
            timestamp = range(value.size)
            dv = {DERIVATIVE_Y: np.diff(value) / np.diff(timestamp),
                  DERIVATIVE_X: np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}
            derivatives.append(np.insert(dv[DERIVATIVE_Y], 0, 0, axis=0))
        return derivatives

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
        variances = []
        for value in values:
            res = value.rolling(window_size).mean()
            res[:window_size] = 0
            variances.append(res)
        return variances

    @staticmethod
    def calculate_rolling_sum(values, window_size=20):
        variances = []
        for value in values:
            res = value.rolling(window_size).sum()
            res[:window_size] = 0
            variances.append(res)
        return variances

    @staticmethod
    def calculate_rolling_variance(values, window_size=20):
        variances = []
        for value in values:
            res = value.rolling(window_size).var()
            res[:window_size] = 0
            variances.append(res)
        return variances

    @staticmethod
    def calculate_rolling_skewness(values, window_size=20):
        skewness = []
        for value in values:
            res = value.rolling(window_size).skew()
            res[:window_size] = 0
            skewness.append(res)
        return skewness

    @staticmethod
    def calculate_rolling_kurtosis(values, window_size=20):
        kurtosis = []
        for value in values:
            res = value.rolling(window_size).kurt()
            res[:window_size] = 0
            kurtosis.append(res)
        return kurtosis
