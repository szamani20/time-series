from implementation.base_file_reader import FileReader
from implementation.base_file_visualizer import FileVisualizer
from implementation.constants import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from implementation.feature_extractor import FeatureExtractor

pd.set_option('display.expand_frame_repr', False)


class AnomalyClassification(FileReader, FileVisualizer):
    def __init__(self):
        super().__init__()

    def train_knn(self):
        data = self.read_yahoo_data(file_number=40)
        values = data[VALUE_COLUMN]
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
        for f in features.items():
            res = f[1]([values])
            data[f[0]] = res[0]

        self.visualize_data_with_trace(dataframes=[data])
        data = data.iloc[20:]
        X = data.drop(columns=[ANOMALY_COLUMN, TIME_COLUMN])
        y = data[ANOMALY_COLUMN].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        print(X_train)

        knn = KNeighborsClassifier(n_neighbors=20)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print(y_pred)
        print(y_test)
        print(knn.score(X_test, y_test))
        # tn, fp, fn, tp
        print(confusion_matrix(y_test, y_pred).ravel())


ac = AnomalyClassification()
ac.train_knn()
