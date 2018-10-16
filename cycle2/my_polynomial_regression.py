import numpy as np
import collections


class MyPolynomialRegression:

    def __init__(self):
        self.X = []
        self.w = []
        self.degree = 1

    def fit(self, x, t, degree=1, ridge_param=0.0):
        self.degree = degree
        self.X = np.array(list(map(lambda i: (
            list(map(lambda j: i ** j, range(0, degree + 1, 1)))
        ), x)), dtype=float)
        right_side = np.dot(self.X.T, t)
        left_side = np.dot(self.X.T, self.X) + (ridge_param * np.identity(degree + 1) if ridge_param > 0 else 0)
        self.w = np.linalg.solve(left_side, right_side)

    def __predict_at_once(self, test):
        return sum(list(map(
            lambda i: self.w[i] * test ** i, range(0, self.degree + 1, 1)
        )))

    def predict(self, test):
        if isinstance(test, collections.Iterable):
            arr = test
        else:
            arr = [test]
        return list(map(lambda a: self.__predict_at_once(a), arr))
