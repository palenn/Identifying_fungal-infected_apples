# coding=gbk
import numpy as np
import csv
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
np.set_printoptions(suppress=True)
# data path
path = r"D:/PycharmProjects/pythonProject1/苹果数据/Data/eliminate_anomalous_data/all_features_rejected_outliers.csv"
data = pd.read_table(path, ',')
data = np.array(data)
labels = data[:, 0].astype(np.int)
data = np.delete(data, 0, axis=1)
data = preprocessing.scale(data)
lda = p = LinearDiscriminantAnalysis(n_components=3)    # n_components:set data dimension after dimension reduction
data = lda.fit_transform(data, labels)  # LDA data dimensionality reduction
data = np.hstack(labels, data)
paths = "all_fearures_LDA.csv"  # storage data path
with open(paths, 'w', newline="") as csvwrite:
    w = csv.writer(csvwrite)
    w.writerows(data)
