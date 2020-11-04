import os
import pandas as pd

from implementation.constants import *
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random


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
    def inject_anomaly(dataframes, rate=.05, max_anomaly_size=3):
        for df in dataframes:
            is_anomaly = [0] * df.shape[0]
            max_value = df[VALUE_COLUMN].max()
            anomaly_values = [round(max_value * 2), round(max_value * 2.5),
                              round(max_value * 3), round(max_value * 3.5), round(max_value * 4)]
            anomaly_counts = int(df.shape[0] * rate)
            for i in range(anomaly_counts):
                anomaly_size = random.randint(1, max_anomaly_size)
                anomaly_start_index = min(i * anomaly_counts + random.randint(1, anomaly_counts),
                                          df.shape[0] - max_anomaly_size)
                anomaly_end_index = anomaly_start_index + anomaly_size

                is_anomaly[anomaly_start_index: anomaly_end_index] = [1] * anomaly_size
                df.iloc[anomaly_start_index: anomaly_end_index, 1] = random.choices(anomaly_values,
                                                                                    k=anomaly_size)

            df['is_anomaly'] = is_anomaly

        return dataframes
