import pandas as pd
import os
import random

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

##########################################experimental######################################

# base_path = 'dataset/concepts/powers'
# anomaly_path = 'dataset/concepts/powers/anomaly'
# power_supply = 'power_supply.csv'
# power_supply_anomaly = 'power_supply_anomaly_dense.csv'
# power_transform = 'power_transform.csv'
# power_transform_anomaly = 'power_transform_anomaly_dense.csv'
#
# df = pd.read_csv(os.path.join(base_path, power_transform), index_col=['hour'])
# is_anomaly = [0] * df.shape[0]
# max_value = df['transform'].max()
# anomaly_values = [round(max_value * 2), round(max_value * 2.5),
#                   round(max_value * 3), round(max_value * 3.5), round(max_value * 4)]
#
# for i in range(int(df.shape[0] / 300)):
#     anomaly_position = min(i * 300 + random.randint(1, 300), df.shape[0] - 1)
#     anomaly_size = random.randint(1, 5)
#     collective_next_position = min(anomaly_position + anomaly_size, df.shape[0] - 1)
#     is_anomaly[anomaly_position: collective_next_position] = anomaly_size * [1]
#     df.iloc[anomaly_position: collective_next_position, 0] = random.choices(anomaly_values, k=anomaly_size)
#
# df['is_anomaly'] = is_anomaly
# df.to_csv(os.path.join(anomaly_path, power_transform_anomaly))

######################################################################################

# base_path = 'dataset/concepts/sensor/light'
# # base_anomaly_path = 'dataset/concepts/sensor/light_anomaly'
# base_anomaly_dense_path = 'dataset/concepts/sensor/light_anomaly_dense'
# for root, dirs, files in os.walk(base_path):
#     for file in files:
#         df = pd.read_csv(os.path.join(base_path, file), index_col=['time'])
#         is_anomaly = [0] * df.shape[0]
#         max_value = df['light'].max()
#         anomaly_values = [round(max_value * 2), round(max_value * 2.5),
#                           round(max_value * 3), round(max_value * 3.5), round(max_value * 4)]
#
#         for i in range(int(df.shape[0] / 300)):
#             anomaly_position = min(i * 300 + random.randint(1, 300), df.shape[0] - 1)
#             anomaly_size = random.randint(1, 5)
#             collective_next_position = min(anomaly_position + anomaly_size, df.shape[0] - 1)
#             is_anomaly[anomaly_position: collective_next_position] = anomaly_size * [1]
#             df.iloc[anomaly_position: collective_next_position, 0] = random.choices(anomaly_values, k=anomaly_size)
#
#         df['is_anomaly'] = is_anomaly
#         df.to_csv(os.path.join(base_anomaly_dense_path, file))


##############################################################################################

base_path = '../dataset/concepts/sensor/light'
# base_anomaly_path = 'dataset/concepts/sensor/light_anomaly'
# base_anomaly_dense_path = 'dataset/concepts/sensor/light_anomaly_dense'
base_anomaly_skewed_path = 'dataset/concepts/sensor/light_anomaly_skewed'
for root, dirs, files in os.walk(base_path):
    for file in files:
        df = pd.read_csv(os.path.join(base_path, file), index_col=['time'])
        is_anomaly = [0] * df.shape[0]
        max_value = df['light'].max()
        anomaly_values = [round(max_value * 2), round(max_value * 2.5),
                          round(max_value * 3), round(max_value * 3.5), round(max_value * 4)]

        dense_part = int(df.shape[0] * .1)

        for i in range(int(dense_part / 100)):
            anomaly_position = min(i * 100 + random.randint(1, 100), dense_part - 1)
            anomaly_size = random.randint(1, 5)
            collective_next_position = min(anomaly_position + anomaly_size, df.shape[0] - 1)
            is_anomaly[anomaly_position: collective_next_position] = anomaly_size * [1]
            df.iloc[anomaly_position: collective_next_position, 0] = random.choices(anomaly_values, k=anomaly_size)

        df['is_anomaly'] = is_anomaly
        df.to_csv(os.path.join(base_anomaly_skewed_path, file))
