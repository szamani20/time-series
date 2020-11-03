import numpy as np

from implementation.base_file_reader import FileReader
from implementation.constants import *


class FeatureExtractor(FileReader):
    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate_derivatives(dataframes):
        derivatives = []
        for df in dataframes:
            values = df[VALUE_COLUMN]
            timestamp = df[TIME_COLUMN]
            dv = {DERIVATIVE_Y: np.diff(values) / np.diff(timestamp),
                  DERIVATIVE_X: np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}
            derivatives.append(dv)
        return derivatives

    @staticmethod
    def calculate_rolling_autocorrelation(dataframes, window_size=20):
        auto_correlations = []
        for df in dataframes:
            values = df[VALUE_COLUMN]
            res = values.rolling(window_size).corr()
            res[:window_size] = 0
            auto_correlations.append(res)

        return auto_correlations

    @staticmethod
    def calculate_rolling_mean(dataframes, window_size=20):
        variances = []
        for df in dataframes:
            values = df[VALUE_COLUMN]
            res = values.rolling(window_size).mean()
            res[:window_size] = 0
            variances.append(res)
        return variances

    @staticmethod
    def calculate_rolling_sum(dataframes, window_size=20):
        variances = []
        for df in dataframes:
            values = df[VALUE_COLUMN]
            res = values.rolling(window_size).sum()
            res[:window_size] = 0
            variances.append(res)
        return variances

    @staticmethod
    def calculate_rolling_variance(dataframes, window_size=20):
        variances = []
        for df in dataframes:
            values = df[VALUE_COLUMN]
            res = values.rolling(window_size).var()
            res[:window_size] = 0
            variances.append(res)
        return variances

    @staticmethod
    def calculate_rolling_skewness(dataframes, window_size=20):
        skewness = []
        for df in dataframes:
            values = df[VALUE_COLUMN]
            res = values.rolling(window_size).skew()
            res[:window_size] = 0
            skewness.append(res)
        return skewness

    @staticmethod
    def calculate_rolling_kurtosis(dataframes, window_size=20):
        kurtosis = []
        for df in dataframes:
            values = df[VALUE_COLUMN]
            res = values.rolling(window_size).kurt()
            res[:window_size] = 0
            kurtosis.append(res)
        return kurtosis
