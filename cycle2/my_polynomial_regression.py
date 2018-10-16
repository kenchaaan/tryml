import numpy as np


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
        left_side = np.dot(self.X.T, self.X) + ridge_param * np.identity(degree + 1)
        self.w = np.linalg.solve(left_side, right_side)

    def predict(self, arr):
        return list(map(
            lambda a: sum(list(map(
                lambda i: self.w[i] * a ** i, range(0, self.degree + 1, 1)
            ))), arr))
