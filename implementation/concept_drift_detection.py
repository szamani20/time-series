from pprint import pprint

from implementation.base_file_reader import FileReader
from implementation.base_file_visualizer import FileVisualizer
import pandas as pd
import numpy as np
from scipy import stats
from implementation.constants import *
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


class CDDetection:
    def __init__(self):
        self.base_window_size = 20
        self.sliding_window_coe = 10
        self.drift_window_size = 10
        self.min_stable_concept_length = 400
        self.maximum_drift_variation = 0.3
        self.max_len = max(self.base_window_size, self.drift_window_size, self.min_stable_concept_length)
        self.abrupt_time_threshold = 20
        self.abrupt_value_threshold = 15

    def read_df(self, num):
        fr = FileReader()
        fv = FileVisualizer()

        df = fr.read_light_data(num)

        # fv.visualize_data_without_trace([df])
        # print(df.head(100))

        mean = np.mean(df['value'])
        std = np.std(df['value']) / 2
        drift_std = np.std(df['value']) / 3
        ccv = stats.variation(df['value'], nan_policy='omit')
        print(mean, std, drift_std, ccv)

        return df, std, drift_std

    @staticmethod
    def calculate_df_stats(df):
        return np.std(df['value']) / 2, np.std(df['value']) / 3

    def identify_stable_concepts(self, df, epsilon, drift_epsilon):
        current_concept_start = self.base_window_size
        i = current_concept_start + self.min_stable_concept_length
        previous_concept_end = current_concept_start - 1
        stable_concept = True
        concept_start_end = [[0, previous_concept_end], [current_concept_start, -1]]
        while i < df.shape[0] - 2 * self.max_len:
            if stable_concept:
                i = current_concept_start + self.min_stable_concept_length
                ew = df.iloc[current_concept_start - self.base_window_size:i, 1].expanding().mean().tolist()
                stable_concept_value = ew[2 * self.base_window_size]
            while stable_concept and i < df.shape[0] - 2 * self.max_len:
                mean_so_far = (ew[-1] * len(ew) + df.iloc[i, 1]) / (len(ew) + 1)
                ew.append(mean_so_far)
                if abs(mean_so_far - stable_concept_value) > epsilon:
                    previous_concept_end = i
                    current_concept_start = -1
                    stable_concept = False
                    concept_start_end[-1][1] = previous_concept_end
                    concept_start_end.append([-1, -1])
                i += 1
                if len(ew) >= self.sliding_window_coe * self.base_window_size:
                    for k in range(1, len(ew)):
                        ew[k] = ((k + 1) * ew[k] - ew[0]) / k
                    ew.pop(0)

            if i >= df.shape[0] - 4 * self.max_len:
                print('HERE', i)
                break

            if not stable_concept:
                ew = df.iloc[
                     previous_concept_end - self.drift_window_size:i + self.min_stable_concept_length + 1,
                     1].expanding().mean().tolist()
                j = self.drift_window_size + 1
            while not stable_concept and i < df.shape[0] - 2 * self.max_len:
                mean_so_far = ew[j]
                min_mean = min(ew[j + 1:])
                max_mean = max(ew[j + 1:])
                mean_to_append = (ew[-1] * len(ew) + df.iloc[i + self.min_stable_concept_length + 1, 1]) / (len(ew) + 1)
                ew.append(mean_to_append)
                variation = stats.variation(df.iloc[i: i + self.min_stable_concept_length, 1], nan_policy='omit')
                if abs(mean_so_far - min_mean) < drift_epsilon and \
                        abs(mean_so_far - max_mean) < drift_epsilon \
                        and variation < self.maximum_drift_variation:
                    previous_concept_end = -1
                    current_concept_start = i
                    stable_concept = True
                    concept_start_end[-1][0] = current_concept_start
                i += 1
                j += 1

        if concept_start_end[-1][0] == -1 or concept_start_end[-1][1] == -1:
            concept_start_end.pop()

        # print(len(concept_start_end))
        # pprint(concept_start_end)

        return concept_start_end

    def visualize_concepts(self, df, concept_start_end):
        fig = go.Figure(layout=go.Layout(
            title=go.layout.Title(text='')
        ))
        concepts_start = np.array(concept_start_end)[:, 0].flatten()
        concepts_end = np.array(concept_start_end)[:, 1].flatten()
        fig.add_trace(go.Line(x=df['time'], y=df['value']))
        fig.add_trace(go.Scatter(x=[i for i in concepts_start], y=[df['value'][i] for i in concepts_start],
                                 marker=dict(color='green', size=10),
                                 name='Start',
                                 mode='markers'))
        fig.add_trace(go.Scatter(x=[i for i in concepts_end], y=[df['value'][i] for i in concepts_end],
                                 marker=dict(color='red', size=10),
                                 name='End',
                                 mode='markers'))
        fig.show()

    @staticmethod
    # CSE = concept_start_end
    def add_drift_column(df, cse):
        concept_column = [1] * df.shape[0]
        for se in cse:
            concept_column[se[0]: se[1]] = [0] * (se[1] - se[0])
        df[DRIFT_COLUMN] = concept_column
        return df

    def identify_drifts(self):
        pass


if __name__ == '__main__':
    cdd = CDDetection()
    df, std, drift_std = cdd.read_df(2)
    concept_start_end = cdd.identify_stable_concepts(df, std, drift_std)
    cdd.add_drift_column(df, concept_start_end)
    # print(df.head(5000))
    cdd.visualize_concepts(df, concept_start_end)
