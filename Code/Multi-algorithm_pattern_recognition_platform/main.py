# -*- coding:utf-8 -*-
# pyinstaller stack_main.py --noconsole --hidden-import sklearn.neighbors._partition_nodes --hidden-import sklearn.utils._typedefs
from UI import Ui_Form
from GWO import GWO_BPNN
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import export_text
np.set_printoptions(threshold=np.inf)
import time
from sklearn.neighbors import KNeighborsClassifier
from PyQt5.Qt import QWidget, QApplication, QFileDialog, QErrorMessage, QIcon
from sklearn.metrics import f1_score
from sklearn import svm
from SSA import SSA_BPNN
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
np.set_printoptions(suppress=True)
import tensorflow as tf
from PSO import PSO_BPNN
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, Sequential, losses, optimizers, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

class Main(QWidget, Ui_Form):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)

    # knn page
    def knn_show(self):
        self.stackedWidget.setCurrentIndex(1)

    # cnn page
    def cnn_show(self):
        self.stackedWidget.setCurrentIndex(2)

    # svm page
    def svm_show(self):
        self.stackedWidget.setCurrentIndex(3)

    # rf page
    def rf_show(self):
        self.stackedWidget.setCurrentIndex(4)

    # bpnn page
    def bpnn_show(self):
        self.stackedWidget.setCurrentIndex(5)

    # pso_bpnn page
    def pso_bpnn_show(self):
        self.stackedWidget.setCurrentIndex(6)

    # gwo_bpnn page
    def gwo_bpnn_show(self):
        self.stackedWidget.setCurrentIndex(7)

    def ssa_bpnn_show(self):
        self.stackedWidget.setCurrentIndex(8)

    # mian page
    def exit_main(self):
        self.stackedWidget.setCurrentIndex(0)

    # popup when error occurs
    def error_show(self, text):
        '''
        :param text: error content
        '''
        error = QErrorMessage(self)
        error.setWindowTitle("Error message")
        error.showMessage(text)

    # Select training sample data
    def cnn_trainfile_chose_show(self):
        path = QFileDialog.getOpenFileName(self, "choose train file", "./", "ALL(*.csv *.xls *.xlsx)")
        self.cnn_train_file_path.setText(path[0])

    # training CNN model
    def cnn_train(self):
        try:
            k1 = int(self.k_1.text())
            k2 = int(self.k_2.text())
            k3 = int(self.k_3.text())
            k4 = int(self.k_4.text())
            s_1 = int(self.s_1.text())
            s_2 = int(self.s_2.text())
            s_3 = int(self.s_3.text())
            s_4 = int(self.s_4.text())
            lr = float(self.lr.text())
            interation = int(self.interation_num.text())
            file_name = self.cnn_train_file_path.text()
            data = pd.read_table(file_name, sep=",")
            data = np.array(data)
            y = data[:, 0].astype(np.int)
            output_dim = len(set(y))
            print(output_dim)
            if output_dim > 2:
                loss_f = 'categorical_crossentropy'
            else:
                loss_f = 'binary_crossentropy'
            data = np.delete(data, 0, axis=1)
            np.random.seed(12)
            np.random.shuffle(data)
            np.random.seed(12)
            np.random.shuffle(y)
            Y = np.array(tf.one_hot(y, output_dim)).astype(np.int)
            if data.shape[1] + 4 - s_4 - s_3 - s_2 - s_1 > 1:
                data = np.array(data)
                begin = time.perf_counter()
                self.model_cnn = Sequential()
                self.model_cnn.add(Convolution1D(k1, s_1))
                self.model_cnn.add(Activation('relu'))
                self.model_cnn.add(Convolution1D(k2, s_2))
                self.model_cnn.add(Activation('relu'))
                self.model_cnn.add(Convolution1D(k3, s_3))
                self.model_cnn.add(Activation('relu'))
                self.model_cnn.add(Convolution1D(k4, s_4))
                self.model_cnn.add(Activation('relu'))
                self.model_cnn.add(Flatten())
                self.model_cnn.add(Dense(16, 'relu'))
                self.model_cnn.add(Dense(8, 'relu'))
                self.model_cnn.add(Dense(output_dim, 'softmax'))
                self.model_cnn.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss_f,
                                  metrics=['accuracy'])
                scores = []
                for train_index, test_index in KFold(10).split(data):

                    x_train, x_test = data[train_index], data[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]
                    x_test = tf.expand_dims(x_test, axis=-1)
                    x_train = tf.expand_dims(x_train, axis=-1)
                    model = self.model_cnn
                    model.fit(x_train, y_train, batch_size=4, epochs=interation, verbose=0)
                    res = model.evaluate(x_test, y_test)[1]
                    scores.append(res)
                res = round(np.mean(scores) * 100, 2)
                sd = round(np.std(scores), 3)
                self.cnn_train_res.setText(str(res) + "%")
                self.cnn_train_sd.setText(str(sd))
                over = time.perf_counter()
                times = round(over-begin, 2)
                self.cnn_train_time.setText(str(times)+"s")

            else:
                error_cnn1 = QErrorMessage(self)
                error_cnn1.setWindowTitle("Error message")
                error_cnn1.showMessage("The convolution kernel is too large, "
                                       "please reduce the convolution kernel")
        except:
            error_cnn = QErrorMessage(self)
            error_cnn.setWindowTitle("Error message")
            error_cnn.showMessage("The data is incorrect, please recheck the data format.")

    # Selecting testing samples
    def cnn_testfile_chose_show(self):
        path = QFileDialog.getOpenFileNames(self, "choose testing files", "./", "ALL(*.csv)")[0]
        paths = ""
        for i in range(len(path)):
            if i == len(path)-1:
                paths += path[i]
            else:
                paths = paths+path[i]+","
        self.cnn_test_file_path.setText(paths)

    # CNN model test samples
    def cnn_test(self):
        try:
            if self.cnn_train_res.text() == "":
                error1 = QErrorMessage(self)
                error1.setWindowTitle("Error message")
                error1.showMessage("The sample is not trained, please train the sample first")
            else:
                path = self.cnn_test_file_path.text()
                if path == "":
                    error2 = QErrorMessage(self)
                    error2.setWindowTitle("Error message")
                    error2.showMessage("please select testing files")
                else:
                    scores = ""
                    p = self.cnn_test_file_path.text().split(",")
                    for j in range(len(p)):
                        path_cnn = p[j]
                        print(path_cnn)
                        data = pd.read_table(path_cnn, sep=",", header=None)
                        pre_data = np.array(data)[0]
                        pre_data = tf.expand_dims(pre_data, -1)
                        pre_data = tf.transpose(pre_data)
                        print(pre_data)
                        res = self.model_cnn.predict(pre_data)
                        index = np.argmax(res) + 1
                        scores = scores+str(index)+" "

                    self.cnn_res_show.setText("The objects respectively belong to\n"+" (No."+scores +")category in the labels")
        except:
            error3 = QErrorMessage(self)
            error3.setWindowTitle("Error message")
            error3.showMessage("The sample data is wrong.")

    # Select training file for RF model
    def rf_trainfile_show(self):
        path = QFileDialog.getOpenFileName(self, "选择一个.csv文件", "./", "ALL(*.csv)")
        self.rf_trainfile_path.setText(path[0])

    # training RF model
    def rf_train(self):
        try:
            if self.rf_trainfile_path.text() == "":
                self.error_show("selecting training data, please.")
            else:
                f_names = []
                fea_dim = int(self.rf_features.text())
                for i in range(1, fea_dim+1):
                    f_names.append("fea"+str(i))
                fea_chose = self.rf_fea_chose.currentText()
                fea_split = self.rf_fea_split.currentText()
                data = pd.read_table(self.rf_trainfile_path.text(), sep=',')
                data = np.array(data)
                y = data[:, 0].astype(np.int)
                if fea_dim >= len(set(y)):
                    self.error_show("The dimension of feature parameters should be less than the"
                                    " number of sample types, please reset")
                else:
                    begin = time.perf_counter()
                    data = np.delete(data, 0, axis=1)
                    data = np.array(data)
                    self.lda_rf = LinearDiscriminantAnalysis(n_components=fea_dim)
                    data = self.lda_rf.fit_transform(data, y)
                    data = np.array(data)
                    self.decisition_tree_classifiter = DecisionTreeClassifier(criterion=fea_chose,
                                                                             splitter=fea_split)
                    np.random.seed(12)
                    np.random.shuffle(data)
                    np.random.seed(12)
                    np.random.shuffle(y)
                    scores = []
                    for train_index, test_index in KFold(10).split(data):
                        x_train, x_test = data[train_index], data[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        self.decisition_tree_classifiter.fit(x_train, y_train)
                        res = self.decisition_tree_classifiter.score(x_test, y_test)
                        scores.append(res)

                    res = round(np.mean(scores) * 100, 2)
                    sd = round(np.std(scores), 3)
                    over = time.perf_counter()
                    times = round(over-begin, 2)
                    self.rf_train_acc.setText(str(res) + "%")
                    self.rf_train_time.setText(str(times)+"s")
                    self.rf_train_sd.setText(str(sd))

        except:
            self.error_show("Data is wrong")

    # selecting testing files for RF model
    def rf_testfile_show(self):
        path = QFileDialog.getOpenFileNames(self, "choose testing files", "./", "ALL(*.csv)")[0]
        paths = ""
        for i in range(len(path)):
            if i == len(path) - 1:
                paths += path[i]
            else:
                paths = paths + path[i] + ","
        self.rf_testfile_path.setText(paths)

    # testing samples by RF model
    def rf_test(self):
        if self.rf_testfile_path.text() == "":
            self.error_show("Please select the sample data to be tested.")
        else:
            if self.rf_train_acc.text() == "":
                self.error_show("The sample is not trained, please train the sample first.")
            else:
                try:
                    paths = self.rf_testfile_path.text().split(",")
                    scores = ""
                    for i in range(len(paths)):
                        path = paths[i]
                        data = pd.read_table(path, sep=",", header=None)
                        data = np.array(data)[0]
                        pre_data = self.lda_rf.transform([data])
                        res = self.decisition_tree_classifiter.predict(pre_data)[0]+1
                        scores = scores+str(res)+" "
                    self.rf_res_show.setText("The objects respectively belong to\n"+" (No."+scores +")category in the labels")
                except:
                    self.error_show("sample data is wrong")

    # select training file for SVM model
    def svm_trainfile_lab(self):
        path = QFileDialog.getOpenFileName(self, "./", "请选择一个.csv文件", "ALL(*.csv)")
        self.svm_trainfile_path.setText(path[0])

    # training SVM model
    def svm_train(self):
        if self.svm_trainfile_path.text() == "":
            self.error_show("Please select training sample data.")
        else:
            try:
                fea_dim = int(self.svm_fea.text())
                cefa = float(self.svm_cefa.text())
                decision = self.svm_decision.currentText()
                kernel = self.svm_kernel.currentText()
                data = pd.read_table(self.svm_trainfile_path.text(), sep=',')
                data = np.array(data)
                y = data[:, 0].astype(np.int)
                data = np.delete(data, 0, axis=1)
                # data = preprocessing.scale(data)
                data = np.array(data)
                if fea_dim >= len(set(y)):
                    self.error_show("The dimension of feature parameters should be "
                                    "less than the number of sample types, please reset.")
                else:
                    begin = time.perf_counter()
                    self.lda_svm = LinearDiscriminantAnalysis(n_components=fea_dim)
                    data = self.lda_svm.fit_transform(data, y)
                    data = np.array(data)

                    self.predictor = svm.SVC(gamma='scale', C=cefa, decision_function_shape=decision, kernel=kernel)
                    np.random.seed(12)
                    np.random.shuffle(data)
                    np.random.seed(12)
                    np.random.shuffle(y)
                    scores = []
                    for train_index, test_index in KFold(10).split(data):
                        x_train, x_test = data[train_index], data[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        self.predictor.fit(x_train, y_train)
                        result = self.predictor.predict(x_test)
                        res = f1_score(result, y_test, average='micro')
                        scores.append(res)
                    res = round(np.mean(scores) * 100, 2)
                    sd = round(np.std(scores), 3)
                    over = time.perf_counter()
                    times = round(over-begin, 2)
                    self.svm_train_res.setText(str(res) + "%")
                    self.svm_train_time.setText(str(times)+"s")
                    self.svm_train_sd.setText(str(sd))
            except:
                self.error_show("sample data is wrong")

    # Select testing files for SVM model
    def svm_testfile_lab(self):
        path = QFileDialog.getOpenFileNames(self, "./", "choose testing files", "ALL(*.csv)")[0]
        paths = ""
        for i in range(len(path)):
            if i == len(path) - 1:
                paths += path[i]
            else:
                paths = paths + path[i] + ","
        self.svm_testfile_path.setText(paths)

    # testing samples by SVM model
    def svm_test(self):
        if self.svm_testfile_path.text() == "":
            self.error_show("Please select the sample data to be tested.")
        else:
            if self.svm_train_res.text() == "":
                self.error_show("The sample is not trained, please train the sample first.")
            else:
                try:
                    scores = ""
                    paths = self.svm_testfile_path.text().split(",")
                    for i in range(len(paths)):
                        path = paths[i]
                        data = pd.read_table(path, sep=",", header=None)
                        data = np.array(data)
                        pre_data = self.lda_svm.transform(data)
                        res = self.predictor.predict(pre_data)[0]+1
                        scores = scores+str(res)+" "
                    self.svm_test_res.setText("The objects respectively belong to\n"+" (No."+scores +")category in the labels")
                except:
                    self.error_show("sample data is wrong")

    # Select training file for BPNN model
    def bpnn_trainfile_chose(self):
        path = QFileDialog.getOpenFileName(self, "./", "choose training file", "ALL(*.csv)")
        self.bpnn_trainfile_path.setText(path[0])

    # training BPNN model
    def bpnn_train(self):
        if self.bpnn_trainfile_path.text() == "":
            self.error_show("Please select training sample data")
        else:
            try:
                one = int(self.bpnn_num1.text())
                two = int(self.bpnn_num2.text())
                three = int(self.bpnn_num3.text())
                print(one)
                four = int(self.bpnn_num4.text())
                cr = float(self.bpnn_cr.text())
                lr = float(self.bpnn_lr.text())
                interation = int(self.bpnn_interation.text())
                fea_dim = int(self.bpnn_fea_dim.text())
                print(fea_dim)
                data = pd.read_table(self.bpnn_trainfile_path.text(), sep=',')
                data = np.array(data)
                y = data[:, 0].astype(np.int)
                output_dim = len(set(y))
                if output_dim > 2:
                    loss_f = 'categorical_crossentropy'
                else:
                    loss_f = 'binary_crossentropy'
                data = np.delete(data, 0, axis=1)
                if fea_dim >= output_dim:
                    self.error_show("The feature parameter dimension "
                                    "should be smaller than the data label type, please reset.")
                else:
                    self.lda_bpnn = LinearDiscriminantAnalysis(n_components=fea_dim)
                    data = self.lda_bpnn.fit_transform(data, y)
                    data = np.array(data)
                    begin = time.perf_counter()
                    np.random.seed(12)
                    np.random.shuffle(data)
                    np.random.seed(12)
                    np.random.shuffle(y)
                    Y = np.array(tf.one_hot(y, output_dim)).astype(np.int)
                    self.bpnn_model = Sequential()
                    self.bpnn_model.add(Dense(one, input_dim=fea_dim, activation='relu'))
                    self.bpnn_model.add(layers.BatchNormalization())
                    self.bpnn_model.add(Dense(two, activation='relu'))
                    self.bpnn_model.add(layers.BatchNormalization())
                    self.bpnn_model.add(Dense(three, activation='relu'))
                    self.bpnn_model.add(layers.BatchNormalization())
                    self.bpnn_model.add(Dense(four, activation='relu'))
                    self.bpnn_model.add(layers.BatchNormalization())
                    self.bpnn_model.add(Dense(output_dim, activation='softmax'))
                    self.bpnn_model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss_f,
                                  metrics=['accuracy'])
                    scores = []
                    for train_index, test_index in KFold(10).split(data):
                        print(test_index)
                        x_train, x_test = data[train_index], data[test_index]
                        y_train, y_test = Y[train_index], Y[test_index]
                        model = self.bpnn_model
                        model.fit(x_train, y_train, batch_size=4, epochs=interation, verbose=0)
                        res = model.evaluate(x_test, y_test)[1]
                        scores.append(res)
                    accu = round(np.mean(scores) * 100, 2)
                    sd = round(np.std(scores), 3)
                    over = time.perf_counter()
                    times = round(over-begin, 2)
                    self.bpnn_train_res.setText(str(accu) + '%')
                    self.bpnn_train_time.setText(str(times)+"s")
                    self.bpnn_train_sd.setText(str(sd))

            except:
                self.error_show("sample data is wrong")

    # selecting test filse for BPNN model
    def bpnn_testfile_chose(self):
        path = QFileDialog.getOpenFileNames(self, "./", "choose testing files", "ALL(*.csv)")[0]
        paths = ""
        for i in range(len(path)):
            if i == len(path) - 1:
                paths += path[i]
            else:
                paths = paths + path[i] + ","
        self.bpnn_testfile_path.setText(paths)

    # testing samples by BPNN model
    def bpnn_test(self):
        if self.bpnn_train_res.text() == "":
            self.error_show("Please train the sample first.")
        else:
            if self.bpnn_testfile_path.text() == "":
                self.error_show("Please select the sample data to be tested.")
            else:
                try:
                    scores = ""
                    paths = self.bpnn_testfile_path.text().split(",")
                    for i in range(len(paths)):
                        path = paths[i]
                        data = pd.read_table(path, sep=',', header=None)
                        data = np.array(data)
                        pre_data = self.lda_bpnn.transform(data)
                        res = self.bpnn_model.predict(pre_data)
                        res = np.argmax(res)+1
                        scores = scores+str(res)+" "
                    self.bpnn_test_res.setText("The objects respectively belong to\n"+" (No."+scores +")category in the labels")
                except:
                    self.error_show("sample data is wrong")

    # selecting training file for KNN model
    def knn_trainfile_chose(self):
        path = QFileDialog.getOpenFileName(self, "./", "chose training file", "ALL(*.csv)")
        self.knn_trainfile_path.setText(path[0])

    # selecting testing files for KNN model
    def knn_testfile_chose(self):
        path = QFileDialog.getOpenFileNames(self, "./", "chose testing files", "ALL(*.csv)")[0]
        p = ""
        for i in range(len(path)):
            if i == len(path)-1:
                p = p+path[i]
            else:
                p = p + path[i] + ","
        self.knn_testfile_path.setText(p)

    # training KNN model
    def knn_train(self):
        if self.knn_trainfile_path.text() == "":
            self.error_show("Please select training sample data.")
        else:
            try:
                fea_dim = int(self.knn_fea.text())
                leaf_size = int(self.knn_leafsize.text())
                n_neig = int(self.knn_neig.text())
                algo = self.knn_algo.currentText()
                data = pd.read_table(self.knn_trainfile_path.text(), sep=',')
                data = np.array(data)
                y = data[:, 0].astype(np.int)
                out_dim = len(set(y))
                data = np.delete(data, 0, axis=1)
                if fea_dim >= out_dim:
                    self.error_show("The dimension of the feature parameter should be l"
                                    "ess than the number of sample types, please reset it. ")
                else:
                    begin = time.perf_counter()
                    self.lda_knn = LinearDiscriminantAnalysis(n_components=fea_dim)
                    data = self.lda_knn.fit_transform(data, y)
                    data = np.array(data)
                    self.knn = KNeighborsClassifier(n_neighbors=n_neig, algorithm=algo, leaf_size=leaf_size)
                    np.random.seed(12)
                    np.random.shuffle(data)
                    np.random.seed(12)
                    np.random.shuffle(y)
                    scores = []
                    for train_index, test_index in KFold(10).split(data):
                        x_train, x_test = data[train_index], data[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        self.knn.fit(x_train, y_train)
                        res = self.knn.score(x_test, y_test)
                        scores.append(res)
                        print("dsdsfdsfsdf")

                    res = round(np.mean(scores) * 100, 2)
                    sd = round(np.std(scores), 3)
                    over = time.perf_counter()
                    times = round(over-begin, 2)
                    self.knn_train_res.setText(str(res)+"%")
                    self.knn_train_time.setText(str(times)+"s")
                    self.knn_train_sd.setText(str(sd))
            except:
                self.error_show("sample data is  wrong")

    # testing samples by KNN model
    def knn_test(self):
        if self.knn_testfile_path.text() == "":
            self.error_show("Please select the sample data to be tested.")
        else:
            if self.knn_train_res.text() == "":
                self.error_show("The sample is not trained, please train the sample first.")
            else:
                try:
                    l = ""
                    path = self.knn_testfile_path.text().split(",")
                    for i in range(len(path)):
                        print(path[i])
                        data = pd.read_table(path[i], sep=",", header=None)
                        data = np.array(data)
                        print(data.shape)
                        pre_data = self.lda_knn.transform(data)
                        res = self.knn.predict(pre_data)[0]+1
                        l = l+str(res)+" "
                    self.knn_train_res.setWordWrap(True)
                    self.knn_test_res.setText("The objects respectively belong to "+"\n"+"No.("+ l +") category in the labels")
                except:
                    self.error_show("sample data is wrong")

    # selecting training file for PSO_BPNN model
    def pso_trainfile_chose(self):
        path = QFileDialog.getOpenFileName(self, "./", "chose training file", "ALL(*.csv)")
        self.pso_trainfile_path.setText(path[0])

    # selecting testing files for PSO_BPNN model
    def pso_testfile_chose(self):
        path = QFileDialog.getOpenFileNames(self, "./", "chose testing files", "ALL(*.csv)")[0]
        paths = ""
        for i in range(len(path)):
            if i == len(path) - 1:
                paths += path[i]
            else:
                paths = paths + path[i] + ","
        self.pso_testfile_path.setText(paths)

    # training PSO_BPNN model
    def pso_train(self):
        if self.pso_trainfile_path.text() == "":
            self.error_show("Please select training sample data.")
        else:
            try:
                fea_dim = int(self.pso_fea.text())
                c1 = float(self.pso_lr1.text())
                c2 = float(self.pso_lr2.text())
                interation = int(self.pso_interation.text())
                num = int(self.pso_num.text())
                pso_interation = int(self.pso_bpnn_interation.text())
                data = pd.read_table(self.pso_trainfile_path.text(), sep=',')
                data = np.array(data)
                y = data[:, 0].astype(np.int)
                out_dim = len(set(y))
                data = np.delete(data, 0, axis=1)
                if fea_dim >= out_dim:
                    self.error_show("The dimension of the feature parameter "
                                    "should be less than the number of sample types, please reset it.")
                else:
                    self.lda_pso = LinearDiscriminantAnalysis(n_components=fea_dim)
                    data = self.lda_pso.fit_transform(data, y)
                    data = np.array(data)

                    x_train, x_test, y_train, y_test = train_test_split(
                        data, y,
                        test_size=0.32
                    )
                    y_train = tf.one_hot(y_train, out_dim)
                    y_test = tf.one_hot(y_test, out_dim)
                    begin = time.perf_counter()
                    self.pso_bpnn = PSO_BPNN(fea_dim, out_dim, pso_interation)
                    res = self.pso_bpnn.pso(x_train, y_train, x_test, y_test, c1, c2, interation, num)
                    np.random.seed(12)
                    np.random.shuffle(data)
                    np.random.seed(12)
                    np.random.shuffle(y)
                    Y = np.array(tf.one_hot(y, out_dim)).astype(np.int)
                    scores = []

                    for train_index, test_index in KFold(10).split(data):
                        print(test_index)
                        x_train, x_test = data[train_index], data[test_index]
                        y_train, y_test = Y[train_index], Y[test_index]
                        model = self.pso_bpnn.pso_model
                        model.fit(x_train, y_train, batch_size=4, epochs=interation, verbose=0)
                        res = model.evaluate(x_test, y_test)[1]
                        scores.append(res)
                    res = round(np.mean(scores) * 100, 2)
                    sd = round(np.std(scores), 3)
                    over = time.perf_counter()
                    times = round(over-begin, 2)
                    self.pso_train_res.setText(str(res)+"%")
                    self.pso_train_time.setText(str(times)+"s")
                    self.pso_train_sd.setText(str(sd))

            except:
                self.error_show("sample data is wrong")

    # testing samples by PSO_BPNN
    def pso_test(self):
        if self.pso_train_res.text() == "":
            self.error_show("Please train the sample first.")
        else:
            if self.pso_testfile_path.text() == "":
                self.error_show("Please select the sample data to be tested.")
            else:
                try:
                    scores = ""
                    paths = self.pso_testfile_path.text().split(",")
                    for i in range(len(paths)):
                        path = paths[i]
                        data = pd.read_table(path, sep=',', header=None)
                        data = np.array(data)
                        pre_data = self.lda_pso.transform(data)
                        res = self.pso_bpnn.pso_model.predict([pre_data])
                        res = np.argmax(res) + 1
                        scores = scores+str(res)+" "
                    self.pso_test_res.setText("The objects respectively belong to\n"+" (No."+scores +")category in the labels")
                except:
                    self.error_show("sample data is wrong")

    # selecting training file for GWO_BPNN
    def gwo_trainfile_chose(self):
        path = QFileDialog.getOpenFileName(self, "./", "chose training file", "ALL(*.csv)")
        self.gwo_trainfile_path.setText(path[0])

    # selecting testing file for GWO_BPNN
    def gwo_testfile_chose(self):
        path = QFileDialog.getOpenFileNames(self, "./", "chose testing files", "ALL(*.csv)")[0]
        paths = ""
        for i in range(len(path)):
            if i == len(path) - 1:
                paths += path[i]
            else:
                paths = paths + path[i] + ","
        self.gwo_testfile_path.setText(paths)

    # training GWO_BPNN model
    def gwo_train(self):
        if self.gwo_trainfile_path.text() == "":
            self.error_show("Please select training sample data.")
        else:
            try:
                fea_dim = int(self.gwo_fea.text())
                dowm = int(self.gwo_down.text())
                up = int(self.gwo_up.text())
                interation = int(self.gwo_interation.text())
                num = int(self.gwo_num.text())
                bp_interation = int(self.gwo_interation_2.text())
                data = pd.read_table(self.gwo_trainfile_path.text(), sep=',')
                data = np.array(data)
                y = data[:, 0].astype(np.int)
                out_dim = len(set(y))
                data = np.delete(data, 0, axis=1)
                if fea_dim >= out_dim:
                    self.error_show("The dimension of the feature parameter "
                                    "should be less than the number of sample types, please reset it.")
                else:
                    self.lda_gwo = LinearDiscriminantAnalysis(n_components=fea_dim)
                    data = self.lda_gwo.fit_transform(data, y)
                    data = np.array(data)
                    x_train, x_test, y_train, y_test = train_test_split(
                        data, y,
                        test_size=0.32
                    )
                    y_train = tf.one_hot(y_train, out_dim)
                    y_test = tf.one_hot(y_test, out_dim)
                    begin = time.perf_counter()
                    self.gwo = GWO_BPNN(fea_dim, out_dim, bp_interation)
                    res = self.gwo.GWO(dowm, up, 4, num, interation, x_train, y_train, x_test, y_test)
                    np.random.seed(12)
                    np.random.shuffle(data)
                    np.random.seed(12)
                    np.random.shuffle(y)
                    Y = np.array(tf.one_hot(y, out_dim)).astype(np.int)
                    scores = []

                    for train_index, test_index in KFold(10).split(data):
                        print(test_index)
                        x_train, x_test = data[train_index], data[test_index]
                        y_train, y_test = Y[train_index], Y[test_index]
                        model = self.gwo.gwo_model
                        model.fit(x_train, y_train, batch_size=4, epochs=interation, verbose=0)
                        res = model.evaluate(x_test, y_test)[1]
                        scores.append(res)
                    res = round(np.mean(scores) * 100, 2)
                    sd = round(np.std(scores), 2)
                    over = time.perf_counter()
                    times = round(over-begin, 3)
                    self.gwo_train_res.setText(str(res)+"%")
                    self.gwo_train_time.setText(str(times)+"s")
                    self.gwo_train_sd.setText(str(sd))

            except:
                self.error_show("sample data is wrong")

    # testing samples by GWO_BPNN
    def gwo_test(self):
        if self.gwo_train_res.text() == "":
            self.error_show("Please train the sample first.")
        else:
            if self.gwo_testfile_path.text() == "":
                self.error_show("Please select the sample data to be tested.")
            else:
                try:
                    scores = ""
                    paths = self.gwo_testfile_path.text().split(",")
                    for i in range(len(paths)):
                        path = paths[i]
                        data = pd.read_table(path, sep=',', header=None)
                        data = np.array(data)
                        pre_data = self.lda_gwo.transform(data)
                        res = self.gwo.gwo_model.predict([pre_data])
                        res = np.argmax(res) + 1
                        scores = scores+str(res)+" "
                    self.gwo_test_res.setText("The objects respectively belong to\n"+" (No."+scores +")category in the labels")
                except:
                    self.error_show("sample data is wrong")

    # selecting training file for SSA_BPNN
    def ssa_trainfile_chose(self):
        path = QFileDialog.getOpenFileName(self, "chose training file", "./", "ALL(*.csv *.xls *.xlsx)")
        self.ssa_trainfile_path.setText(path[0])

    # selecting testing files for SSA_BPNN
    def ssa_testfile_chose(self):
        path = QFileDialog.getOpenFileNames(self, "chose testing files", "./", "ALL(*.csv *.xls *.xlsx)")[0]
        paths = ""
        for i in range(len(path)):
            if i == len(path) - 1:
                paths += path[i]
            else:
                paths = paths + path[i] + ","
        self.ssa_testfile_path.setText(paths)

     # training SSA_BPNN
    def ssa_train(self):
        if self.ssa_trainfile_path.text() == "":
            self.error_show("Please select training sample data.")
        else:
            try:
                fea_dim = int(self.gwo_fea.text())
                dowm = int(self.ssa_low.text())
                up = int(self.ssa_up.text())
                interation = int(self.ssa_interation.text())
                bp_interation = int(self.ssa_bpnn_interation.text())
                num = int(self.ssa_num.text())
                data = pd.read_table(self.ssa_trainfile_path.text(), sep=',')
                data = np.array(data)
                y = data[:, 0].astype(np.int)
                out_dim = len(set(y))
                data = np.delete(data, 0, axis=1)
                if fea_dim >= out_dim:
                    self.error_show("The dimension of the feature parameter should be "
                                    "less than the number of sample types, please reset it.")
                else:
                    self.lda_ssa = LinearDiscriminantAnalysis(n_components=fea_dim)
                    data = self.lda_ssa.fit_transform(data, y)
                    data = np.array(data)
                    x_train, x_test, y_train, y_test = train_test_split(
                        data, y,
                        test_size=0.2
                    )
                    begin = time.perf_counter()
                    y_train = tf.one_hot(y_train, out_dim)
                    y_test = tf.one_hot(y_test, out_dim)
                    self.ssa = SSA_BPNN(fea_dim, out_dim, bp_interation)
                    res = self.ssa.SSA(num, interation, dowm, up, 4, x_train, y_train, x_test, y_test)
                    np.random.seed(12)
                    np.random.shuffle(data)
                    np.random.seed(12)
                    np.random.shuffle(y)
                    Y = np.array(tf.one_hot(y, out_dim)).astype(np.int)
                    scores = []

                    for train_index, test_index in KFold(10).split(data):
                        print(test_index)
                        x_train, x_test = data[train_index], data[test_index]
                        y_train, y_test = Y[train_index], Y[test_index]
                        model = self.ssa.ssa_model
                        model.fit(x_train, y_train, batch_size=4, epochs=interation, verbose=0)
                        res = model.evaluate(x_test, y_test)[1]
                        scores.append(res)
                    res = round(np.mean(scores) * 100, 2)
                    sd = round(np.std(scores), 3)
                    over = time.perf_counter()
                    times = round(over-begin, 2)
                    self.ssa_train_time.setText(str(times)+"s")
                    self.ssa_train_res.setText(str(res) + "%")
                    self.ssa_train_sd.setText(str(sd))

            except:
                self.error_show("sample data is wrong")

    # testing samples by SSA_BPNN
    def ssa_test(self):
        if self.ssa_train_res.text() == "":
            self.error_show("Please train the sample first.")
        else:
            if self.ssa_testfile_path.text() == "":
                self.error_show("Please select the sample data to be tested.")
            else:
                try:
                    scores = ""
                    paths = self.ssa_testfile_path.text().split(",")
                    for i in range(len(paths)):
                        path = paths[i]
                        data = pd.read_table(path, sep=',', header=None)
                        data = np.array(data)
                        pre_data = self.lda_ssa.transform(data)
                        res = self.ssa.ssa_model.predict([pre_data])
                        res = np.argmax(res) + 1
                        scores = scores+str(res)+" "
                    self.ssa_test_res.setText("The objects respectively belong to\n"+" (No."+scores +")category in the labels")
                except:
                    self.error_show("sample data is wrong")



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main = Main()
    main.setWindowTitle("Multi-algorithm Pattern Recognition Platform")
    main.setWindowIcon(QIcon("logo.ico"))
    main.show()
    sys.exit(app.exec_())