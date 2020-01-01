import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy


class LinearRegression:

    def __init__(self):
        self.w_vec = np.array([0,0])

    # return: ndarray
    def read_csv(self, file):
        arr = pd.read_csv(file)
        arr = np.array(arr)
        return arr

    # return: vector
    def train(self, data_training):
        x_11 = len(data_training)
        x_12 = sum(data_training[:, 0])
        x_21 = copy.copy(x_12)
        x_22 = sum(np.array([i ** 2 for i in data_training[:, 0]]))
        X = np.array([[x_11, x_12], [x_21, x_22]])
        a = np.array([sum(data_training[:, 1]), sum([l[0] * l[1] for l in data_training])])
        self.w_vec = np.linalg.solve(X, a)
        return self.w_vec

    # return: ndarray
    def plot(self, w_vec, data_training):
        x = data_training[:, 0]
        # print(x)
        y = [w_vec[0] + w_vec[1] * i for i in x]
        plt.plot(x, y)
        plt.plot(x, data_training[:, 1], 'o')
        # plt.show()
        plt.savefig('/tmp/view.png')

    def do_train(self):
        arr = self.read_csv('/tmp/sample.csv')
        w_vec = self.train(arr)
        self.plot(w_vec, arr)

    def predict(self, x):
        return self.w_vec[0] + self.w_vec[1] * x


if __name__ == '__main__':
    # p = LinearRegression()
    # data_training = p.read_csv('./sample.csv')
    # print(data_training)
    # w_vec = p.train(data_training)
    # print(w_vec)
    # p.plot(w_vec, data_training)
    lr = LinearRegression()
    lr.do_train()
    print(lr.predict(5))

