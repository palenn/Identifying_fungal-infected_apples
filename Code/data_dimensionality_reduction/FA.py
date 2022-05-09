# coding=gbk
import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing
np.set_printoptions(suppress=True)
# data path
path = r"D:/PycharmProjects/pythonProject1/苹果数据/Data/eliminate_anomalous_data/all_features_rejected_outliers.csv"
data = pd.read_table(path, ',')
data = np.array(data)
labels = data[:, 0].astype(np.int)
data = np.delete(data, 0, axis=1)
data = preprocessing.scale(data)
norms = np.linalg.norm(data, axis=1)
for i in range(data.shape[1]):
    data[:, i] = data[:, i] / norms
transformer = FactorAnalysis(n_components=3, random_state=0)    # n_components:set data dimension after dimension reduction
data = transformer.fit_transform(data)  # FA data dimensionality reduction
data = np.hstack(labels, data)
paths = "all_fearures_FA.csv"  # storage data path
with open(paths, 'w', newline="") as csvwrite:
    w = csv.writer(csvwrite)
    w.writerows(data)
