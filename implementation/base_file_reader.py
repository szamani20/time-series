import os
import pandas as pd
from implementation.constants import *
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


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
            os.path.join(DATASET_PATH, CONCEPTS_SUBPATH, LIGHT_PATH, LIGHT_BASE_FILENAME.format(file_number)))
        dataframe.rename(columns={'light': VALUE_COLUMN})
        return dataframe
