import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MakeSamples:

    def make_sample(self, a, b):
        sample = np.array([[i, a * i + b + np.random.randn()] for i in np.arange(0, 10, 0.1)])
        return sample

    def to_csv(self):
        arr = self.make_sample(2, 3)
        df = pd.DataFrame(arr)
        df.to_csv('static/sample.csv', index=None)


if __name__ == '__main__':
    ms = MakeSamples()
    ms.to_csv()
