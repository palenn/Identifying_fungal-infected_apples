# coding=gbk
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, Sequential, losses, datasets, optimizers, metrics
from tensorflow.keras.layers import Dense, Dropout
np.set_printoptions(suppress=True)
class PSO():
    def __init__(self, fea_dim, out_dim, interation):
        '''
        :param fea_dim: features dimension
        :param out_dim: output dimension
        :param interation: interation number
        '''
        super(PSO, self).__init__()
        self.g_fitness = 0  # global optimum
        self.fea_dim = fea_dim
        self.out_dim = out_dim
        self.interation = interation

    # BPNN model
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

    # speed update
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
        :return: updated position
        '''
        size = X.shape[0]
        r1 = np.random.random((size, 1))
        r2 = np.random.random((size, 1))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V[V < -max_val] = -max_val
        V[V > -max_val] = max_val
        b = V.astype(np.int)
        return b

    # position update
    def position_update(self, X, V):
        P = V + X
        a = P.astype(np.int)
        return a

    # Particle Swarm Optimization
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
        :return: best fitness value
        '''
        w = 1
        c1 = C1
        c2 = C2
        dim = 4
        size = num
        iter_num = interation
        max_val = 0.5
        fitness_val_list = []

        # initialize particle position of population
        X = np.random.randint(15, 170, size=(size, dim))
        # initialize particle speed of population
        V = np.random.uniform(-0.5, 0.5, size=(size, dim))
        p_fitness = self.bpnn(X, x_train, y_train, x_test, y_test)

        print("p_fitness", p_fitness)
        self.g_fitness = p_fitness.max()
        print("g_fitness", self.g_fitness)
        fitness_val_list.append(self.g_fitness)
        print("fitness_list", fitness_val_list)
        pbest = X
        print("pbest", pbest)
        gbest = X[p_fitness.argmax()]
        print("gbest", gbest)
        # iterate update
        for i in range(1, iter_num):
            V = self.velocity_update(V, X, pbest, gbest, c1, c2, w, max_val)
            X = self.position_update(X, V)
            p_fitness2 = self.bpnn(X, x_train, y_train, x_test, y_test)
            g_fitness2 = p_fitness2.max()
            if g_fitness2 == 1.0:
                break
            # iterate the best position of each particle
            for j in range(size):
                if p_fitness[j] < p_fitness2[j]:
                    pbest[j] = X[j]
                    p_fitness[j] = p_fitness2[j]
                # iterate the best speed of each particle
            if self.g_fitness < g_fitness2:
                gbest = X[p_fitness2.argmax()]
                self.g_fitness = g_fitness2
            fitness_val_list.append(self.g_fitness)
            i += 1
        print("Best fitness：%.5f" % fitness_val_list[-1])
        print("Best parameters w1=%d, w2=%d, w3=%d, w4=%d" % (gbest[0], gbest[1], gbest[2], gbest[3]))
        return fitness_val_list[-1]

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
    pso_bpnn = PSO(3, 4, 120)
    pso_bpnn.pso(train_data, train_labels, test_data, test_labels, 0.8, 0.6, 150, 15)
    labels = np.array(labels)
    scores = []
    for train_index, test_index in KFold(10).split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model = pso_bpnn.pso_model
        model.fit(x_train, y_train, batch_size=4, epochs=120, verbose=0)
        res = model.evaluate(x_test, y_test)[1]
        scores.append(res)
    print(np.mean(scores), np.std(scores))   # np.mean(scores), np.std(scores): average accuracy and standard deviation of accuracy



