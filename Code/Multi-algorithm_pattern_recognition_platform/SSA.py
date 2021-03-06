# -*- coding:utf-8 -*-
import numpy as np
from tensorflow.keras import layers, Sequential, losses,  optimizers, metrics
from tensorflow.keras.layers import Dense

class SSA_BPNN():
    def __init__(self, fea_dim, outdim, interation):
        '''
        :param fea: features dimension
        :param out_dim: output dimension
        :param interation: bpnn model interation number
        '''
        self.fea = fea_dim
        self.outdim = outdim
        self.fitness = 0
        self.interation = interation

    def fun(self, X, x_train, y_train, x_test, y_test):
        '''
        :param X: hidden layer neurons number parameter
        :param x_train: train data
        :param y_train: train data labels
        :param x_test: test data
        :param y_test: test data labels
        :return: max fitness value
        '''
        l = []
        X = X.astype(np.int)
        w1 = X[0]
        w2 = X[1]
        w3 = X[2]
        w4 = X[3]
        model = Sequential()
        model.add(Dense(w1, input_dim=self.fea, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(w2, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(w3, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(w4, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(self.outdim, activation='softmax'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, validation_split=0.20, epochs=self.interation, batch_size=6)
        loss_and_metric = model.evaluate(x_test, y_test, batch_size=4)
        if self.fitness < loss_and_metric[1]:
            self.fitness = loss_and_metric[1]
            print(self.fitness)
            print("8"*12)
            self.ssa_model = model
        l.append(loss_and_metric[1] * 100)
        return max(l)

    def Bounds(self, s, Lb, Ub):
        '''
        :param s: current position of
        :param Lb: lower bound of search boundary
        :param Ub: upper bound of search boundary
        :return: updated position
        '''
        temp = s
        for i in range(len(s)):
            if temp[i] < Lb[0, i]:
                temp[i] = Lb[0, i]
            elif temp[i] > Ub[0, i]:
                temp[i] = Ub[0, i]

        return temp

    def SSA(self, pop, interation, c, d, dim, x_train, y_train, x_test, y_test):
        '''
        :param pop: sparrow population
        :param interation: iterate number
        :param c: lower bound value of search boundary
        :param d: upper bound value of search boundary
        :param dim: features dimension
        :param x_train: train data
        :param y_train: train labels
        :param x_test: test data
        :param y_test: test labels
        :return: global fitness valuetest:
        :return: global best fitness
        '''
        P_percent = 0.2
        fitness_val_list = []
        pNum = round(pop * P_percent)
        lb = c * np.ones((1, dim))
        ub = d * np.ones((1, dim))
        X = [[128, 64, 32, 16], [132, 66, 30, 20], [120, 66, 38, 22], [160, 80, 40, 20], [256, 128, 64, 32]]
        X = np.array(X)
        fit = np.zeros((pop, 1))
        for i in range(pop):
            fit[i, 0] = self.fun(X[i, :], x_train, y_train, x_test, y_test)
        pFit = fit
        pX = X
        fMin = np.max(fit[:, 0])
        bestI = np.argmax(fit[:, 0])
        bestX = X[bestI, :]
        fitness_val_list.append(fMin)
        Convergence_curve = np.zeros((1, interation))
        for t in range(interation):
            sortIndex = np.argsort(-(pFit.T))
            fmax = np.min(pFit[:, 0])
            B = np.argmin(pFit[:, 0])
            worse = X[B, :]

            r2 = np.random.rand(1)
            if r2 < 0.8:
                for i in range(pNum):
                    r1 = np.random.rand(1)
                    X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] * np.exp(-(i) / (r1 * interation))
                    X[sortIndex[0, i], :] = self.Bounds(X[sortIndex[0, i], :], lb, ub)
                    fit[sortIndex[0, i], 0] = self.fun(X[sortIndex[0, i], :], x_train, y_train, x_test, y_test)  # ?????????????????????
            elif r2 >= 0.8:
                for i in range(pNum):
                    Q = np.random.rand(1)
                    X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] + Q * np.ones((1, dim))
                    X[sortIndex[0, i], :] = self.Bounds(X[sortIndex[0, i], :], lb, ub)
                    fit[sortIndex[0, i], 0] = self.fun(X[sortIndex[0, i], :], x_train, y_train, x_test, y_test)
            bestII = np.argmax(fit[:, 0])

            bestXX = X[bestII, :]

            for ii in range(pop - pNum):
                i = ii + pNum
                A = np.floor(np.random.rand(1, dim) * 2) * 2 - 1
                if i > pop / 2:
                    Q = np.random.rand(1)
                    X[sortIndex[0, i], :] = Q * np.exp(worse - pX[sortIndex[0, i], :] / np.square(i))
                else:
                    X[sortIndex[0, i], :] = bestXX + np.dot(np.abs(pX[sortIndex[0, i], :] - bestXX),
                                                            1 / (A.T * np.dot(A, A.T))) * np.ones((1, dim))
                X[sortIndex[0, i], :] = self.Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = self.fun(X[sortIndex[0, i], :], x_train, y_train, x_test, y_test)

            arrc = np.arange(len(sortIndex[0, :]))

            c = np.random.permutation(arrc)
            b = sortIndex[0, c[0:20]]
            for j in range(len(b)):
                if pFit[sortIndex[0, b[j]], 0] > fMin:
                    X[sortIndex[0, b[j]], :] = bestX + np.random.rand(1, dim) * np.abs(
                        pX[sortIndex[0, b[j]], :] - bestX)
                else:
                    X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :] + (2 * np.random.rand(1) - 1) * np.abs(
                        pX[sortIndex[0, b[j]], :] - worse) / (pFit[sortIndex[0, b[j]]] - fmax + 10 ** (-50))
                X[sortIndex[0, b[j]], :] = self.Bounds(X[sortIndex[0, b[j]], :], lb, ub)
                fit[sortIndex[0, b[j]], 0] = self.fun(X[sortIndex[0, b[j]]], x_train, y_train, x_test, y_test)
            for i in range(pop):

                if fit[i, 0] > pFit[i, 0]:
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]
                if pFit[i, 0] > fMin:
                    fMin = pFit[i, 0]
                    bestX = pX[i, :]
                    print(bestX)
            Convergence_curve[0, t] = fMin
            fitness_val_list.append(fMin)
        return fMin
