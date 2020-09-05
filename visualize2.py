import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# dataframe = pd.read_csv('dataset/concepts/powers/power_supply.csv')
#
# values = dataframe['value']
# anomalies = dataframe[dataframe['anomaly_point'] == 1]
# anomalies = anomalies.index.tolist()
#
# timestamp = dataframe.index.tolist()
#
# derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
#                'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 - 0.5).astype(int)}
# fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Line(y=dataframe['value']))
# fig.add_trace(go.Scatter(x=[i + 1 for i in anomalies], y=[dataframe['value'][i] for i in anomalies],
#                          marker=dict(color="crimson", size=6),
#                          mode="markers"))
# fig.show()


########################################################################################

dataframe = pd.read_csv('/home/szamani/PycharmProjects/anomaly-detection/dataset/concepts/sensor/light_anomaly/1.csv')
# dataframe = dataframe.iloc[:5000, ]
values = dataframe['light']
timestamp = dataframe['time']

# derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
#                'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}

# fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# fig.show()

fig = go.Figure()
fig.add_trace(go.Line(x=dataframe['time'], y=dataframe['light']))
fig.show()
