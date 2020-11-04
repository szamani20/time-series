from implementation.base_file_reader import FileReader
from implementation.base_file_visualizer import FileVisualizer
from implementation.constants import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

from implementation.feature_extractor import FeatureExtractor

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=sys.maxsize)


class AnomalyClassification(FileReader, FileVisualizer):
    def __init__(self):
        super().__init__()

    def train_knn(self, datasets, k=10):
        # yahoo_data = self.read_multiple_yahoo_data(list(range(1, YAHOO_SIZE + 1)))
        fe = FeatureExtractor()
        features = {
            'der': fe.calculate_derivatives,
            'mean': fe.calculate_rolling_mean,
            'kur': fe.calculate_rolling_kurtosis,
            'ske': fe.calculate_rolling_skewness,
            'sum': fe.calculate_rolling_sum,
            'var': fe.calculate_rolling_variance,
            # 'ac': fe.calculate_autocorrelation,
        }

        f1_scores = []
        i = 0
        for yd in datasets:
            i += 1
            values = yd[VALUE_COLUMN]
            for f in features.items():
                res = f[1](values)
                if np.isnan(res).any():
                    # print('Feature {} has nan'.format(f))
                    continue
                yd[f[0]] = res

            # yd = yd.iloc[20:]

            X = yd.drop(columns=[ANOMALY_COLUMN, TIME_COLUMN])

            if X.isnull().values.any():
                print('NaN', i)
                f1_scores.append(0)
                continue

            y = yd[ANOMALY_COLUMN].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred))

        self.visualize_data_without_trace([pd.DataFrame(data=enumerate(f1_scores),
                                                        columns=[TIME_COLUMN, VALUE_COLUMN])])


ac = AnomalyClassification()
ac.train_knn(ac.read_multiple_yahoo_data(list(range(1, YAHOO_SIZE + 1))))

light_dataframes = []
for i in range(1, LIGHT_SIZE + 1):
    try:
        df = ac.read_light_data(i)
        light_dataframes.append(df)
    except Exception:
        pass
# ac.visualize_data_without_trace(light_dataframes[:2])
ac.inject_anomaly(light_dataframes)
# ac.visualize_data_with_trace(light_dataframes[:2])
ac.train_knn(light_dataframes)
