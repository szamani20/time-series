This repository contains the implementation and experiments I
developed for my research project.

### datasets

The datasets I used are located in the `dataset/` directory. There are two
subdirectories there.

One is `Yahoo/` which contains 67 Yahoo! datasets used in my research.

The other is `concepts/` which contains three other subdirectories,
`powers/`, `sensor`/, and `synthetic/`. The `powers/` subdirectory contains the
power supply dataset in both `arff` and `csv` formats.

The `sensor/light/` subdirectory contains the 55 light datasets I used in
my research.

### Baseline comparison study implementation (INN)

The implementation for the baseline study I used as a comparison basis is located
at the `paper-implementaion/` directory. The Python implementation of the INN paper
is located in the `error_detection.py` file. There is also a Jupyter notebook file
`error-detection.ipynb` which served as a rapid experimental version to
quickly perform multiple experiments without having to run the python file from the
beginning.

### Concept Drift Detection

The main algorithm of the research, the concept drift detection is located at
the `implementaion/` directory and specifically in the `concept_drift_detection.py`
file.

### Helper Files

In the same directory, `implementation/`, there are some helper functions and
base classes:

`base_file_reader.py` which helps with reading files and saving to them
if required.

`base_file_visualizer.py` which serves a similar purpose for visualizations.

`constants.py` which contains constants used in the concept drift detection
and anomaly detection.

`feature_extractor.py` which contains implementation for extracting statistical
features from the time series.

### Anomaly Detection

In the same directory, `imlementation/` lies the code for anomaly detection.
The implementation for the anomaly detection which is the last step of the research
is located in the `anomaly_classification_model.py` file.
It contains the code to use statistical features and CD features with four
classifiers I used to model anomaly detection. `SVM`, `KNN`, `Random Forest`, 
and `Multi-Layer Perceptron Neural Network` model. `MLP` was chosen based on its
performance.

### One Time Use or Experimental Use Codes

There are multiple files in the `experimental-codes` directory that only served
as experimental codes in the earlier stage of the development or are only
required to be executed once to prepare or transform datasets into the
right format (e.g. arff to csv converter).

### Notebook Experiments and Dataset Stats

Jupyter notebook are rapid tools to perform quick experiments without the hassles
of the ordinary Python files. The dataset statistics and Exploratory Data Analyses
(EDA) are located in the `notebooks/` directory. EDA graphs and showcases for
concept drift are located in this directory. They are not needed to perform the
anomaly detection or concept drift detection.
