import math
import numpy.linalg as lin
import numpy as np


class MyPerceptron:

    def __init__(self):
        self.w = []

    def fit(self, test_data, test_label, itr=1, nu=0.1, random_state=1):
        np.random.seed(random_state)
        self.w = np.random.rand(len(test_data.columns) + 1)
        for times in range(itr):
            test_data_3d = []
            for x in test_data.values.tolist():
                x.append(1.0)
                test_data_3d.append(x)
            for i in range(len(test_data_3d)):
                i_np = np.array(test_data_3d[i])
                if np.dot(i_np, self.w) * test_label.loc[i] < 0:
                    self.w = self.w + test_label.loc[i] * nu * i_np

    def predict(self, data):
        test_data_3d = []
        for x in data.values.tolist():
            x.append(1.0)
            test_data_3d.append(x)
        result = list(map(lambda j: np.dot(np.array(j), self.w) / abs(np.dot(np.array(j), self.w)), test_data_3d))
        return result
