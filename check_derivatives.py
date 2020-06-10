import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

dataframe = pd.read_csv('dataset/A1Benchmark/real_42.csv')

values = dataframe['value']
anomalies = dataframe[dataframe['is_anomaly'] == 1]
anomalies = anomalies.index.tolist()

timestamp = dataframe['timestamp']
derivatives = {'y_p': np.diff(values) / np.diff(timestamp),
               'x_p': np.array((np.array(timestamp)[:-1] + np.array(timestamp)[1:]) / 2 + 0.5).astype(int)}
derivatives = pd.DataFrame(derivatives)

# print(dataframe.iloc[:20])
print(derivatives.iloc[1380:, ])

# fig = px.line(pd.DataFrame(data=derivatives), x='x_p', y='y_p')
# fig.show()
