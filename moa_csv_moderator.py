from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skmultiflow.drift_detection import kswin

from implementation.concept_drift_detection import CDDetection

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

df = pd.read_csv('dataset/concepts/synthetic/inc.txt',
                 usecols=[4],
                 header=None,
                 # nrows=50,
                 names=['value'])

df.index.set_names(['time'], inplace=True)
df['value'] = df['value'] * 100
df = pd.concat([df]*100)
df.reset_index(drop=True, inplace=True)
df.index.set_names(['time'], inplace=True)
# print(df.shape)
# print(df.head(2000))

# abrupt_drift = [0] * df.shape[0]
# for i in range(1, int(df.shape[0] / 25)):
#     abrupt_drift[i * 25] = 1
# df['drift'] = abrupt_drift
# print(df.head())

# df.to_csv('dataset/concepts/synthetic/inc.csv')
df.reset_index(inplace=True)
print(df.head())


cdd = CDDetection()
mean = np.mean(df['value'])
std = np.std(df['value']) / 3
drift_std = np.std(df['value']) / 4
print(std, drift_std)
concept_start_end = cdd.identify_stable_concepts(df, std, drift_std)
cdd.add_drift_column(df, concept_start_end)
# print(df.head(5000))
cdd.visualize_concepts(df, concept_start_end)

# ks = kswin.KSWIN()
# ll = df['value'].tolist()
# det = []
# for i in range(len(ll)):
#     ks.add_element(ll[i])
#     if ks.detected_change():
#         det.append(i)
#
# print(det)
# print(len(det))
# drift_start_end = [[i * 25 - 3, i * 25 + 3] for i in range(1, int(df.shape[0] / 25))]
# tp = 0
# fp = 0
# fn = 0
# ot = 0
# print(drift_start_end)
# print(len(drift_start_end))
# for d in det:
#     for se in drift_start_end:
#         if d in range(se[0], se[1]):
#             tp += 1
#             fn += 1
#             break
#     else:
#         fp += 1
#
# fn = len(drift_start_end) - fn
# print(tp, fp, fn)
# if fn < 0:
#     fn = 0
# print(tp / (tp + .5 * (fp + fn)))
