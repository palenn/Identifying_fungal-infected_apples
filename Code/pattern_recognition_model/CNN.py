# coding=gbk
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, metrics
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
np.set_printoptions(suppress=True)
# data path
path = r"D:/PycharmProjects/pythonProject1/苹果数据/Data/eliminate_anomalous_data/all_features_rejected_outliers.csv"
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
print(data.shape)
def cnn_model():
    model = Sequential()
    model.add(Convolution1D(512, 6))
    model.add(Activation('relu'))
    model.add(Convolution1D(256, 6))
    model.add(Activation('relu'))
    model.add(Convolution1D(32, 3))
    model.add(Activation('relu'))
    model.add(Convolution1D(8, 3))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(16, 'relu'))
    model.add(Dense(8, 'relu'))
    model.add(Dense(4, 'softmax'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0006),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

scores = []
# 10-fold cross-validation
for train_index, test_index in KFold(10).split(data):
        print("test index: ", test_index)
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        x_test = tf.expand_dims(x_test, axis=-1)
        x_train = tf.expand_dims(x_train, axis=-1)
        model = cnn_model()
        model.fit(x_train, y_train, batch_size=4, epochs=60, verbose=0)
        res = model.evaluate(x_test, y_test)[1]
        scores.append(res)  # Accuracy after 10-fold cross-validation
print(np.mean(scores), np.std(scores))  # np.mean(scores), np.std(scores): average accuracy and standard deviation of accuracy