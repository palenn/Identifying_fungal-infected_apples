# coding=gbk
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import numpy
import pandas as pd
import random
from tensorflow.keras import layers, Sequential, losses,  optimizers, metrics
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
np.set_printoptions(suppress=True)

class GWO_bpnn():
    def __init__(self, fea, out_dim, interation):
        '''
        :param fea: features dimension
        :param out_dim: output dimension
        :param interation: bpnn model interation number
        '''
        self.fea = fea
        self.out_dim = out_dim
        self.fitness = 0
        self.interation = interation

    def fun(self, X, x_train, y_train, x_test, y_test):
        '''
        :param X: hidden layer neurons number parameter
        :param x_train: train data
        :param y_train: train data labels
        :param x_test: test data
        :param y_test: test data labels
        :return:
        '''
        w1 = int(X[0])
        w2 = int(X[1])
        w3 = int(X[2])
        w4 = int(X[3])
        model = Sequential()
        model.add(Dense(w1, input_dim=self.fea, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(w2, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(w3, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(w4, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(self.out_dim, activation='softmax'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, validation_split=0.22, epochs=self.interation, batch_size=6)
        loss_and_metric = model.evaluate(x_test, y_test, batch_size=2)
        if loss_and_metric[1]>self.fitness:
            self.fitness = loss_and_metric[1]
            self.gwo_model = model
        return 1 - loss_and_metric[1]

    def GWO(self, lb, ub, dim, SearchAgents_no, Max_iter, x_train, y_train, x_test, y_test):
        '''
        :param lb: lower bound of search boundary
        :param ub: upper bound of search boundary
        :param dim: features dimension
        :param SearchAgents_no: number of wolves
        :param Max_iter: number of interation
        :param x_train: train data
        :param y_train: train data labels
        :param x_test: test data
        :param y_test: test data labels
        :return: max fitness value
        '''

        # initialize alpha, beta, and delta_pos position
        Alpha_pos = numpy.zeros(dim)
        Alpha_score = float("inf")
        Beta_pos = numpy.zeros(dim)
        Beta_score = float("inf")
        Delta_pos = numpy.zeros(dim)
        Delta_score = float("inf")
        # list列表类型
        if not isinstance(lb, list):  # 作用：来判断一个对象是否是一个已知的类型。 其第一个参数（object）为对象，第二个参数（type）为类型名，若对象的类型与参数二的类型相同则返回True
            lb = [lb] * dim  # 生成[100，100，.....100]30个
        if not isinstance(ub, list):
            ub = [ub] * dim

        # Initialize the positions of search agents
        Positions = numpy.zeros((SearchAgents_no, dim))
        for i in range(dim):
            Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        Convergence_curve = numpy.zeros(Max_iter)

        # finding optimal parameters by iteration
        for l in range(0, Max_iter):
            for i in range(0, SearchAgents_no):


                for j in range(dim):  # 30
                    Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])
                # calculate the fitness function value
                fitness = self.fun(Positions[i, :], x_train, y_train, x_test, y_test)  # 把某行数据带入函数计算
                if fitness < Alpha_score:
                    Alpha_score = fitness  # Update alpha
                    Alpha_pos = Positions[i, :].copy()

                if (fitness > Alpha_score and fitness < Beta_score):
                    Beta_score = fitness  # Update beta
                    Beta_pos = Positions[i, :].copy()

                if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                    Delta_score = fitness  # Update delta
                    Delta_pos = Positions[i, :].copy()

            a = 2 - l * ((2) / Max_iter);

            for i in range(0, SearchAgents_no):
                for j in range(0, dim):
                    r1 = random.random()
                    r2 = random.random()

                    A1 = 2 * a * r1 - a;
                    C1 = 2 * r2;
                    D_alpha = abs(C1 * Alpha_pos[j] - Positions[
                        i, j]); [j]
                    X1 = Alpha_pos[j] - A1 * D_alpha;

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a;
                    C2 = 2 * r2;

                    D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                    X2 = Beta_pos[j] - A2 * D_beta;

                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a;
                    C3 = 2 * r2;

                    D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                    X3 = Delta_pos[j] - A3 * D_delta;

                    Positions[i, j] = (X1 + X2 + X3) / 3
            Convergence_curve[l] = Alpha_score;
        return 1-Alpha_score


if __name__ == '__main__':
    # data path
    path = r"D:/PycharmProjects/pythonProject1/苹果数据/Data/dimensionality_reduction_data/all_fearures_lda.csv"
    data = pd.read_table(path, sep=',')
    data = np.array(data)
    labels = data[:, 0].astype(np.int)  # samples labels
    data = np.delete(data, 0, axis=1)  # samples features
    # shuffle labels and data randomly
    np.random.seed(12)
    np.random.shuffle(data)
    np.random.seed(12)
    np.random.shuffle(labels)
    labels = tf.one_hot(labels, 4)
    train_data = data[0:100, :]
    train_labels = labels[0:100, :]
    test_data = data[100:, :]
    test_labels = labels[100:, :]
    gwo_bpnn = GWO_bpnn(3, 4, 120)
    gwo_bpnn.GWO(15, 100, 10, 150, 3, train_data, train_labels, test_data, test_labels)
    labels = np.array(labels)
    scores = []
    for train_index, test_index in KFold(10).split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model = gwo_bpnn.gwo_model
        model.fit(x_train, y_train, batch_size=4, epochs=120, verbose=0)
        res = model.evaluate(x_test, y_test)[1]
        scores.append(res)
    print(np.mean(scores), np.std(scores))   # np.mean(scores), np.std(scores): average accuracy and standard deviation of accuracy


