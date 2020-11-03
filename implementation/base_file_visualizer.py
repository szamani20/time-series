import os
import pandas as pd
from implementation.constants import *
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


class FileVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def visualize_data_with_trace(dataframes, trace_name='is_anomaly'):
        for df in dataframes:
            anomalies = df[df[trace_name] == 1]
            anomalies = anomalies.index.tolist()

            fig = go.Figure()
            fig.add_trace(go.Line(x=df[TIME_COLUMN], y=df[VALUE_COLUMN]))
            fig.add_trace(go.Scatter(x=[i + 1 for i in anomalies], y=[df[VALUE_COLUMN][i] for i in anomalies],
                                     marker=dict(color='crimson', size=6),
                                     mode='markers'))
            fig.show()

    @staticmethod
    def visualize_data_without_trace(dataframes):
        for df in dataframes:
            fig = go.Figure()
            fig.add_trace(go.Line(x=df[TIME_COLUMN], y=df[VALUE_COLUMN]))
            fig.show()
