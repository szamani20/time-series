import pandas as pd
from matplotlib import pyplot as plt, axis


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


sensor_dataframe = pd.read_csv('dataset/concepts/sensor/sensor.csv',
                               usecols=['light', 'class'],
                               dtype={
                                   'rcdminutes': int
                               })

classes = sensor_dataframe['class'].unique()

for i, cls in enumerate(classes):
    cls_df = sensor_dataframe[sensor_dataframe['class'] == cls].reset_index(drop=True).drop(labels=['class'], axis=1)

    cls_df.to_csv('dataset/concepts/sensor/light/{}.csv'.format(cls), index_label='time')
    # print(cls_df)
    #
    # cls_df['light'].plot()
    # if i == 0:
    #     break

# plt.show()
