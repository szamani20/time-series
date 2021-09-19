import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class Visualization:
    def __init__(self):
        pass

    def read_yahoo_data(self, base_path='dataset/Yahoo/', base_filename='real_{}.csv', file_number=1):
        dataframe = pd.read_csv(os.path.join(base_path, base_filename.format(file_number)))
        return dataframe

    def read_multiple_yahoo_data(self, filenumbers, base_path='dataset/Yahoo/', base_filename='real_{}.csv'):
        dataframes = []
        for fn in filenumbers:
            dataframes.append(pd.read_csv(os.path.join(base_path, base_filename.format(fn))))
        return dataframes


dataframe = pd.read_csv('dataset/concepts/sensor/light/1.csv')

values = dataframe['value']
anomalies = dataframe[dataframe['is_anomaly'] == 1]
anomalies = anomalies.index.tolist()

timestamp = dataframe['timestamp']
# derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
#                'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}

# fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# fig.show()

fig = go.Figure()
fig.add_trace(go.Line(x=dataframe['timestamp'], y=dataframe['value']))
fig.add_trace(go.Scatter(x=[i + 1 for i in anomalies], y=[dataframe['value'][i] for i in anomalies],
                         marker=dict(color="crimson", size=6),
                         mode="markers"))
fig.show()

##########################################################

# dataframe = pd.read_csv('dataset/concepts/powers/anomaly/power_supply_anomaly_dense.csv')
# # dataframe = dataframe.iloc[:5000, ]
# values = dataframe['supply']
# timestamp = dataframe['hour']
#
# derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
#                'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}
#
# # fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# # fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Line(x=dataframe['hour'], y=dataframe['supply']))
# fig.show()

##########################################################

# dataframe = pd.read_csv('dataset/concepts/sensor/light_anomaly_dense/58.csv')
# # dataframe = dataframe.iloc[:5000, ]
# values = dataframe['light']
# timestamp = dataframe['time']
# anomalies = dataframe[dataframe['is_anomaly'] == 1]
# anomalies = anomalies.index.tolist()
#
# # derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
# #                'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}
#
# # fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# # fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Line(x=dataframe['time'], y=dataframe['light']))
#
# fig.add_trace(go.Scatter(x=[i + 1 for i in anomalies], y=[dataframe['light'][i] for i in anomalies],
#                          marker=dict(color="crimson", size=6),
#                          mode="markers"))
# fig.show()
