from implementation.base_file_reader import FileReader
from implementation.base_file_visualizer import FileVisualizer
from implementation.constants import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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

    @staticmethod
    def extract_features():
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
        return features

    def train_model(self, model, datasets, title='', k=10, history=True):
        features = self.extract_features()

        f1_scores = []
        i = 0
        for ds in datasets:
            i += 1
            values = ds[VALUE_COLUMN]
            for f in features.items():
                res, past_res = f[1](values)

                # past_res is the shifted version of res. So no need to check
                if np.isnan(res).any():
                    # print('Feature {} has nan'.format(f))
                    continue
                ds[f[0]] = res
                if history and past_res is not None:
                    ds[f[0] + '_past'] = past_res

            X = ds.drop(columns=[ANOMALY_COLUMN, TIME_COLUMN])

            if X.isnull().values.any():
                print('NaN', i)
                f1_scores.append(0)
                continue

            y = ds[ANOMALY_COLUMN].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred))

        self.visualize_data_without_trace([pd.DataFrame(data=enumerate(f1_scores),
                                                        columns=[TIME_COLUMN, VALUE_COLUMN])], title=title)


ac = AnomalyClassification()

##################################

# yd = ac.read_multiple_yahoo_data(list(range(1, YAHOO_SIZE + 1)))
# ac.normalize_time_series(yd)
# # ac.visualize_data_with_trace(yd[59:60])
# single_merged_df = ac.merge_time_series(yd)
# # ac.train_model(KNeighborsClassifier(n_neighbors=10), yd, title='F1-Score, KNN, Yahoo')
# # ac.visualize_data_with_trace([single_merged_df])
# ac.train_model(KNeighborsClassifier(n_neighbors=10),
#                [single_merged_df],
#                title='F1-Score, KNN, Yahoo Normalized and merged, considering 200 history',
#                history=True)
# ac.train_model(MLPClassifier(),
#                [single_merged_df],
#                title='F1-Score, FFNN, Yahoo Normalized and merged, considering 200 history',
#                history=True)

##################################

light_dataframes = []
for i in range(1, LIGHT_SIZE - 45):
    try:
        df = ac.read_light_data(i)
        light_dataframes.append(df)
    except Exception:
        pass
# ac.visualize_data_without_trace(light_dataframes[2:5])
ac.inject_anomaly(light_dataframes)
ac.visualize_data_with_trace(light_dataframes[2:3])
ac.train_model(KNeighborsClassifier(n_neighbors=10),
               light_dataframes,
               title='F1-Score, KNN-10, Light, not considering history',
               history=False)
ac.train_model(MLPClassifier(),
               light_dataframes,
               title='F1-Score, FFNN, Light, not considering history',
               history=False)
ac.train_model(SVC(),
               light_dataframes,
               title='F1-Score, SVM, Light, not considering history',
               history=False)
ac.train_model(RandomForestClassifier(),
               light_dataframes,
               title='F1-Score, RF, Light, not considering history',
               history=False)

ac.train_model(KNeighborsClassifier(n_neighbors=10),
               light_dataframes,
               title='F1-Score, KNN-10, Light, considering 50 history',
               history=True)
ac.train_model(MLPClassifier(),
               light_dataframes,
               title='F1-Score, FFNN, Light, considering 50 history',
               history=True)
ac.train_model(SVC(),
               light_dataframes,
               title='F1-Score, SVM, Light, considering 50 history',
               history=True)
ac.train_model(RandomForestClassifier(),
               light_dataframes,
               title='F1-Score, RF, Light, considering 50 history',
               history=True)

##################################
