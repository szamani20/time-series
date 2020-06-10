import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# dataframe = pd.read_csv('dataset/A1Benchmark/real_40.csv')
# dataframe = pd.read_csv('dataset/A1Benchmark/real_67.csv')
# dataframe = pd.read_csv('dataset/A1Benchmark/real_66.csv')
# dataframe = pd.read_csv('dataset/A1Benchmark/real_65.csv')
# dataframe = pd.read_csv('dataset/A1Benchmark/real_62.csv')
# dataframe = pd.read_csv('dataset/A1Benchmark/real_61.csv')
# dataframe = pd.read_csv('dataset/A1Benchmark/real_59.csv')
# dataframe = pd.read_csv('dataset/A1Benchmark/real_5.csv')
# # dataframe = pd.read_csv('dataset/A1Benchmark/real_1.csv')
#
#
# values = dataframe['value']
# anomalies = dataframe[dataframe['is_anomaly'] == 1]
# anomalies = anomalies.index.tolist()
#
# timestamp = dataframe['timestamp']
# # derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
# #                'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}
#
# # fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# # fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Line(x=dataframe['timestamp'], y=dataframe['value']))
# fig.add_trace(go.Scatter(x=[i + 1 for i in anomalies], y=[dataframe['value'][i] for i in anomalies],
#                          marker=dict(color="crimson", size=6),
#                          mode="markers"))
# fig.show()

##########################################################

# dataframe = pd.read_csv('dataset/concepts/powers/anomaly/power_supply_anomaly_dense.csv')
# # dataframe = dataframe.iloc[:5000, ]
# values = dataframe['supply']
# timestamp = dataframe['hour']
#
# # derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
# #                'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}
#
# # fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# # fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Line(x=dataframe['hour'], y=dataframe['supply']))
# fig.show()

##########################################################

dataframe = pd.read_csv('dataset/concepts/sensor/light_anomaly_dense/58.csv')
# dataframe = dataframe.iloc[:5000, ]
values = dataframe['light']
timestamp = dataframe['time']
anomalies = dataframe[dataframe['is_anomaly'] == 1]
anomalies = anomalies.index.tolist()

# derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
#                'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}

# fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# fig.show()

fig = go.Figure()
fig.add_trace(go.Line(x=dataframe['time'], y=dataframe['light']))

fig.add_trace(go.Scatter(x=[i + 1 for i in anomalies], y=[dataframe['light'][i] for i in anomalies],
                         marker=dict(color="crimson", size=6),
                         mode="markers"))
fig.show()
