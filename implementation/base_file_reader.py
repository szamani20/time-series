import os
import pandas as pd

from implementation.constants import *
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler


class FileReader:
    def __init__(self):
        pass

    @staticmethod
    def read_yahoo_data(file_number=1):
        dataframe = pd.read_csv(os.path.join(DATASET_PATH, YAHOO_SUBPATH, YAHOO_BASE_FILENAME.format(file_number)))
        dataframe.rename(columns={'timestamp': TIME_COLUMN}, inplace=True)
        return dataframe

    @staticmethod
    def read_multiple_yahoo_data(file_numbers):
        dataframes = []
        for fn in file_numbers:
            dataframes.append(
                pd.read_csv(os.path.join(DATASET_PATH, YAHOO_SUBPATH, YAHOO_BASE_FILENAME.format(fn))).rename(
                    columns={'timestamp': TIME_COLUMN}))
        return dataframes

    @staticmethod
    def read_power_data(supply=True):
        if supply:
            dataframe = pd.read_csv(os.path.join(DATASET_PATH, CONCEPTS_SUBPATH,
                                                 POWER_PATH, POWER_SUPPLY)).rename(columns={'hour': TIME_COLUMN,
                                                                                            'supply': VALUE_COLUMN})
        else:
            dataframe = pd.read_csv(os.path.join(DATASET_PATH, CONCEPTS_SUBPATH,
                                                 POWER_PATH, POWER_TRANSFORM)).rename(columns={'hour': TIME_COLUMN,
                                                                                               'transform': VALUE_COLUMN})

        return dataframe

    @staticmethod
    def read_light_data(file_number=1):
        dataframe = pd.read_csv(
            os.path.join(DATASET_PATH, CONCEPTS_SUBPATH, LIGHT_PATH, LIGHT_BASE_FILENAME.format(file_number))).rename(
            columns={'timestamp': TIME_COLUMN,
                     'light': VALUE_COLUMN})
        return dataframe

    @staticmethod
    def normalize_values(values, normalization_range=(0, 100)):
        scaler = MinMaxScaler(feature_range=normalization_range)
        scaler.fit(values)
        normalized = scaler.transform(values)
        return normalized

    @staticmethod
    def normalize_time_series(datasets):
        for df in datasets:
            normalized_values = FileReader.normalize_values(values=df[VALUE_COLUMN].values.reshape(-1, 1))
            df[VALUE_COLUMN] = normalized_values
        return datasets

    @staticmethod
    def merge_time_series(datasets):
        df = pd.concat(datasets)
        df[TIME_COLUMN] = np.arange(df.shape[0])
        df.reset_index(drop=True, inplace=True)
        return df

    def inject_single_anomaly(self, df, rate=0.01, max_anomaly_size=3):
        return self.inject_anomaly([df], rate=rate, max_anomaly_size=max_anomaly_size)[0]

    @staticmethod
    def inject_anomaly(dataframes, rate=.01, max_anomaly_size=3):
        window_size = int(1 / rate)

        for df in dataframes:
            is_anomaly = [0] * df.shape[0]
            # not having an anomaly in the first 5 windows
            exception_window_size = int((WINDOW_EXCEPTION_RATE * df.shape[0]) / window_size)
            anomaly_counts = int(df.shape[0] * rate) - exception_window_size
            for i in range(exception_window_size, anomaly_counts + exception_window_size):
                max_window_value = df[VALUE_COLUMN][i * window_size: (i + 1) * window_size].max()
                anomaly_values = [round(max_window_value * 1.1), round(max_window_value * 1.2),
                                  round(max_window_value * 1.3), round(max_window_value * 1.4)]
                anomaly_size = random.randint(1, max_anomaly_size)
                anomaly_start_index = i * window_size + random.randint(1, window_size - anomaly_size - 1)
                anomaly_end_index = anomaly_start_index + anomaly_size
                is_anomaly[anomaly_start_index: anomaly_end_index] = [1] * anomaly_size
                df.iloc[anomaly_start_index: anomaly_end_index, 1] = random.choices(anomaly_values,
                                                                                    k=anomaly_size)
            df['is_anomaly'] = is_anomaly

        return dataframes

        # for i in range(len(dataframes)):
        #     df = dataframes[i]
        #     is_anomaly = [0] * df.shape[0]
        #     max_value = df[VALUE_COLUMN].max()
        #     anomaly_values = [round(max_value * 0.8), round(max_value * 0.9),
        #                       round(max_value * 1), round(max_value * 1.1), round(max_value * 1.2)]
        #     anomaly_counts = int(df.shape[0] * rate)
        #     for i in range(anomaly_counts):
        #         anomaly_size = random.randint(1, max_anomaly_size)
        #         anomaly_start_index = min(i * anomaly_counts + random.randint(1, anomaly_counts),
        #                                   df.shape[0] - max_anomaly_size)
        #         anomaly_end_index = anomaly_start_index + anomaly_size
        #
        #         is_anomaly[anomaly_start_index: anomaly_end_index] = [1] * anomaly_size
        #         df.iloc[anomaly_start_index: anomaly_end_index, 1] = random.choices(anomaly_values,
        #                                                                             k=anomaly_size)
        #
        #     df['is_anomaly'] = is_anomaly
        #
        # return dataframes
