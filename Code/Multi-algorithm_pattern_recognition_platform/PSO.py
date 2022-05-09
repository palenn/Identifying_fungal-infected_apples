# -*- coding:utf-8 -*-
import numpy as np
from tensorflow.keras import layers, Sequential, losses, datasets, optimizers, metrics
from tensorflow.keras.layers import Dense, Dropout
np.set_printoptions(suppress=True)
class PSO_BPNN():
    def __init__(self, fea_dim, out_dim, interation):
        '''
        :param fea_dim: features dimension
        :param out_dim: output dimension
        :param interation: interation number
        '''
        super(PSO_BPNN, self).__init__()
        self.g_fitness = 0
        self.fea_dim = fea_dim
        self.out_dim = out_dim
        self.interation = interation

    def bpnn(self, X, x_train, y_train, x_test, y_test):
        '''
        :param X: hidden layer neurons number parameter
        :param x_train: train data
        :param y_train: train data labels
        :param x_test: test data
        :param y_test: test data labels
        :return: max fitness value
        '''
        metrics = []
        X = X.astype(np.int)
        for i in range(X.shape[0]):
            w1 = X[i, 0]
            w2 = X[i, 1]
            w3 = X[i, 2]
            w4 = X[i, 3]
            model = Sequential()
            model.add(Dense(w1, input_dim=self.fea_dim, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(Dense(w2, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(Dense(w3, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(Dense(w4, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(Dense(self.out_dim, activation='softmax'))
            model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(x_train, y_train, validation_split=0.22, epochs=self.interation, batch_size=2)
            loss_and_metric = model.evaluate(x_test, y_test, batch_size=4)
            print("accuracy is:", loss_and_metric[1])
            if loss_and_metric[1]>self.g_fitness:
                self.pso_model = model
            metrics.append(loss_and_metric[1])
        return np.array(metrics)

    def velocity_update(self, V, X, pbest, gbest, c1, c2, w, max_val):
        '''
        :param V: current speed
        :param X: curent position
        :param pbest: local best fitness
        :param gbest: global best fitness
        :param c1: learn rate1
        :param c2: learn rate2
        :param w: inertia weight
        :param max_val: max speed value
        retuen : updated speed
        '''
        size = X.shape[0]
        r1 = np.random.random((size, 1))
        r2 = np.random.random((size, 1))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V[V < -max_val] = -max_val
        V[V > -max_val] = max_val
        b = V.astype(np.int)
        return b

    def position_update(self, X, V):
        '''
        :param X: current position
        :param V: current speed
        :return: updated position
        '''
        P = V + X
        a = P.astype(np.int)
        return a

    def pso(self, x_train, y_train, x_test, y_test, C1, C2, interation, num):
        '''
        :param x_train: train data
        :param y_train: train data labels
        :param x_test: test data
        :param y_test: test data labels
        :param C1: learn rate1
        :param C2: learn rate2
        :param interation: interation number
        :param num: partical number
        :return: best fitness
        '''
        w = 1
        c1 = C1
        c2 = C2
        dim = 4
        size = num
        iter_num = interation
        max_val = 0.5
        fitness_val_list = []
        X = np.random.randint(15, 170, size=(size, dim))
        V = np.random.uniform(-0.5, 0.5, size=(size, dim))

        p_fitness = self.bpnn(X, x_train, y_train, x_test, y_test)
        self.g_fitness = p_fitness.max()
        fitness_val_list.append(self.g_fitness)
        pbest = X
        gbest = X[p_fitness.argmax()]
        for i in range(1, iter_num):
            V = self.velocity_update(V, X, pbest, gbest, c1, c2, w, max_val)
            X = self.position_update(X, V)
            p_fitness2 = self.bpnn(X, x_train, y_train, x_test, y_test)
            g_fitness2 = p_fitness2.max()
            if g_fitness2 == 1.0:
                break
            for j in range(size):
                if p_fitness[j] < p_fitness2[j]:
                    pbest[j] = X[j]
                    p_fitness[j] = p_fitness2[j]
            if self.g_fitness < g_fitness2:
                gbest = X[p_fitness2.argmax()]
                self.g_fitness = g_fitness2
            fitness_val_list.append(self.g_fitness)
            i += 1
        return fitness_val_list[-1]

