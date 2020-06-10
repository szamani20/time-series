import pandas as pd

power_dataframe = pd.read_csv('dataset/concepts/powers/powersupply.csv')

hour_continuous = [i for i in range(power_dataframe.shape[0])]
power_dataframe['hour'] = hour_continuous
power_dataframe.set_index('hour', drop=['hour'], inplace=True)

power_supply_dataframe = power_dataframe.drop(labels=['transform'], axis=1)
power_transform_dataframe = power_dataframe.drop(labels=['supply'], axis=1)

power_supply_dataframe.to_csv('dataset/concepts/powers/{}.csv'.format('power_supply'))
power_transform_dataframe.to_csv('dataset/concepts/powers/{}.csv'.format('power_transform'))
