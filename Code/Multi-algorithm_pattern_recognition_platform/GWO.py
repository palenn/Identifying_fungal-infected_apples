# -*- coding:utf-8 -*-
import numpy as np
import numpy
import tensorflow as tf
import random
from tensorflow.keras import layers, Sequential, losses,  optimizers, metrics
from tensorflow.keras.layers import Dense
np.set_printoptions(suppress=True)

class GWO_BPNN():
    def __init__(self, fea, out_dim, interation):
        '''
        :param fea: features dimension
        :param out_dim: output dimension
        :param interation: bpnn model interation number
        '''
        super(GWO_BPNN, self).__init__()
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
        history = model.fit(x_train, y_train, validation_split=0.22, epochs=self.interation, batch_size=6)
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
        Alpha_pos = numpy.zeros(dim)
        Alpha_score = float("inf")
        Beta_pos = numpy.zeros(dim)
        Beta_score = float("inf")

        Delta_pos = numpy.zeros(dim)
        Delta_score = float("inf")
        if not isinstance(lb, list):
            lb = [lb] * dim
        if not isinstance(ub, list):
            ub = [ub] * dim

        # Initialize the positions of search agents
        Positions = numpy.zeros((SearchAgents_no, dim))
        for i in range(dim):
            Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        Convergence_curve = numpy.zeros(Max_iter)

        for l in range(0, Max_iter):
            for i in range(0, SearchAgents_no):
                for j in range(dim):  # 30
                    Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])
                fitness = self.fun(Positions[i, :], x_train, y_train, x_test, y_test)
                if fitness < Alpha_score:
                    Alpha_score = fitness  # update alpha
                    Alpha_pos = Positions[i, :].copy()

                if (fitness > Alpha_score and fitness < Beta_score):
                    Beta_score = fitness  # update beta
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

                    D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);
                    X1 = Alpha_pos[j] - A1 * D_alpha;

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a;  #
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
