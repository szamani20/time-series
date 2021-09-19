from implementation.base_file_reader import FileReader
from implementation.base_file_visualizer import FileVisualizer
from implementation.concept_drift_detection import CDDetection
from implementation.constants import *
import pandas as pd
from scipy import stats
import numpy as np
from tqdm import tqdm
import sys
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
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
            'ac': fe.calculate_autocorrelation,
        }
        return features

    def train_model(self, model, datasets, title='', drift=False, test_size=0.2):
        features = self.extract_features()

        f1_scores = []
        i = 0
        for ds in datasets:
            i += 1
            values = ds[VALUE_COLUMN]
            for f in features.items():
                res, _ = f[1](values)

                # past_res is the shifted version of res. So no need to check
                if np.isnan(res).any():
                    # print('Feature {} has nan'.format(f))
                    continue
                ds[f[0]] = res

            if not drift and DRIFT_COLUMN in ds.columns.tolist():
                ds.drop([DRIFT_COLUMN], axis=1, inplace=True)

            # print(ds.head(1))

            X = ds.drop(columns=[ANOMALY_COLUMN, TIME_COLUMN])

            if X.isnull().values.any():
                print('NaN', i)
                # f1_scores.append(0)
                continue

            y = ds[ANOMALY_COLUMN].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            corr = ds.corr()
            sns.heatmap(corr,
                        xticklabels=corr.columns.values,
                        yticklabels=corr.columns.values)
            plt.show()

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Check for error
            continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)

        # return None
        print('AVG. F1-Score: ', np.mean(f1_scores))
        self.visualize_data_without_trace([pd.DataFrame(data=enumerate(f1_scores),
                                                        columns=[TIME_COLUMN, VALUE_COLUMN])],
                                          title=title + str(np.mean(f1_scores)))


ac = AnomalyClassification()
cdd = CDDetection()

##################################

yahoo_datasets = ac.read_multiple_yahoo_data(list(range(1, YAHOO_SIZE + 1)))
for yd in yahoo_datasets:
    try:
        std, drift_std = cdd.calculate_df_stats(yd)
        concept_start_end = cdd.identify_stable_concepts(yd, std, drift_std)
        cdd.add_drift_column(yd, concept_start_end)
    except Exception as e:
        print(e)

# Partial features

# ac.train_model(KNeighborsClassifier(n_neighbors=10),
#                yahoo_datasets[:1],
#                title='F1-Score, KNN, Yahoo, Considering CD ', drift=True)
# ac.train_model(KNeighborsClassifier(n_neighbors=10),
#                yahoo_datasets[:1],
#                title='F1-Score, KNN, Yahoo, Not Considering CD ', drift=False)

# ac.train_model(RandomForestClassifier(),
#                yahoo_datasets,
#                title='F1-Score, RF, Yahoo, Not Considering CD ', drift=True)
# ac.train_model(RandomForestClassifier(),
#                yahoo_datasets,
#                title='F1-Score, RF, Yahoo, Considering CD ', drift=False)

# ac.train_model(SVC(),
#                yahoo_datasets,
#                title='F1-Score, SVC, Yahoo, Considering CD ', drift=True)
# ac.train_model(SVC(),
#                yahoo_datasets,
#                title='F1-Score, SVC, Yahoo, Not Considering CD ', drift=False)

ac.train_model(MLPClassifier(),
               yahoo_datasets,
               title='F1-Score, FFNN, Yahoo, Considering CD ', drift=True)
ac.train_model(MLPClassifier(),
               yahoo_datasets,
               title='F1-Score, FFNN, Yahoo, Not Considering CD ', drift=False)


##################################


light_dataframes = []
for i in range(1, 15):
    try:
        df = ac.read_light_data(i)
        df = ac.inject_single_anomaly(df)
        std, drift_std = cdd.calculate_df_stats(df)
        concept_start_end = cdd.identify_stable_concepts(df, std, drift_std)
        cdd.add_drift_column(df, concept_start_end)
        light_dataframes.append(df)
    except Exception as e:
        print(e)

print(len(light_dataframes))
print(light_dataframes[0].head(5000))

ac.visualize_data_without_trace(light_dataframes[0:1])
ac.visualize_data_with_trace(light_dataframes[0:1])

# ac.train_model(KNeighborsClassifier(n_neighbors=10),
#                light_dataframes, drift=True,
#                title='F1-Score, KNN-10, Light, Considering CD ')
# ac.train_model(KNeighborsClassifier(n_neighbors=10),
#                light_dataframes, drift=False,
#                title='F1-Score, KNN-10, Light, Not Considering CD ')

# ac.train_model(RandomForestClassifier(),
#                light_dataframes, drift=True,
#                title='F1-Score, RF, Light, Considering CD ')
# ac.train_model(RandomForestClassifier(),
#                light_dataframes, drift=False,
#                title='F1-Score, RF, Light, Not Considering CD ')

# ac.train_model(MLPClassifier(),
#                light_dataframes, drift=True,
#                title='F1-Score, FFNN, Light, Considering CD ')
# ac.train_model(MLPClassifier(),
#                light_dataframes, drift=False,
#                title='F1-Score, FFNN, Light, Not Considering CD ')

ac.train_model(SVC(),
               light_dataframes, drift=True,
               title='F1-Score, SVM, Light, Considering CD ')
ac.train_model(SVC(),
               light_dataframes, drift=False,
               title='F1-Score, SVM, Light, Not Considering CD ')

##################################
