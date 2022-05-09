# coding=gbk
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, datasets, optimizers, metrics
from tensorflow.keras.layers import Dense, Dropout
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
Y = np.array(tf.one_hot(labels, 4)).astype(np.int)

# BPNN model
def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
scores = []
# 10-fold cross-validation
for train_index, test_index in KFold(10).split(data):
        print(test_index)
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model = build_model()
        model.fit(x_train, y_train, batch_size=4, epochs=120, verbose=0)
        res = model.evaluate(x_test, y_test)[1]
        scores.append(res)  # Accuracy after 10-fold cross-validation
print(np.mean(scores), np.std(scores))  # np.mean(scores), np.std(scores): average accuracy and standard deviation of accuracy

