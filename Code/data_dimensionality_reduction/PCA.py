# coding=gbk
import numpy as np
import csv
import tensorflow as tf
import pandas as pd
np.set_printoptions(suppress=True)
from sklearn.decomposition import PCA
# data path
path = r"D:/PycharmProjects/pythonProject1/苹果数据/Data/eliminate_anomalous_data/all_features_rejected_outliers.csv"
data = pd.read_table(path, ',')
data = np.array(data)
labels = data[:, 0].astype(np.int)
labels = tf.expand_dims(labels, axis=1)
data = np.delete(data, 0, axis=1)
norms = np.linalg.norm(data, axis=1)
for i in range(data.shape[1]):
    data[:, i] = data[:, i] / norms
pca = PCA(n_components=3)   # n_components:set data dimension after dimension reduction
data =pca.fit_transform(data)   # PCA data dimensionality reduction
data = np.hstack((labels, data))
paths = "all_fearures_PCA.csv"  # storage data path
with open(paths, 'w', newline="") as csvwrite:
    w = csv.writer(csvwrite)
    w.writerows(data)