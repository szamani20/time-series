import os
import pandas as pd

from implementation.base_file_reader import FileReader
from implementation.constants import *
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


class FileVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def visualize_data_with_trace(dataframes, title='', trace_name='is_anomaly'):
        for df in dataframes:
            anomalies = df[df[trace_name] == 1]
            anomalies = anomalies.index.tolist()

            fig = go.Figure(layout=go.Layout(
                title=go.layout.Title(text=title)
            ))

            fig.add_trace(go.Line(x=df[TIME_COLUMN], y=df[VALUE_COLUMN]))
            fig.add_trace(go.Scatter(x=[i for i in anomalies], y=[df[VALUE_COLUMN][i] for i in anomalies],
                                     marker=dict(color='crimson', size=6),
                                     mode='markers'))
            fig.show()

    @staticmethod
    def visualize_data_without_trace(dataframes, title=''):
        for df in dataframes:
            fig = go.Figure(layout=go.Layout(
                title=go.layout.Title(text=title)
            ))
            fig.add_trace(go.Line(x=df[TIME_COLUMN], y=df[VALUE_COLUMN]))
            fig.show()
