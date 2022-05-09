# coding=gbk
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.svm import SVC
np.set_printoptions(suppress=True)
# data path
path = r"D:/PycharmProjects/pythonProject1/苹果数据/Data/dimensionality_reduction_data/all_fearures_lda.csv"
data = pd.read_table(path, sep=',')
data = np.array(data)
labels = data[:, 0].astype(np.int)      # samples labels
data = np.delete(data, 0, axis=1)       # samples features
# shuffle labels and data randomly
np.random.seed(12)
np.random.shuffle(data)
np.random.seed(12)
np.random.shuffle(labels)
svm = SVC()
scores = []
# 10-fold cross-validation
for train_index, test_index in KFold(10).split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        svm.fit(x_train, y_train)
        res = svm.score(x_test, y_test)
        scores.append(res)  # Accuracy after 10-fold cross-validation
print(np.mean(scores), np.std(scores))  # np.mean(scores), np.std(scores): average accuracy and standard deviation of accuracy
