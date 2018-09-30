import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def choose_from_stdn(sigma):
    return sigma * np.random.randn()


def extract_data(seeds):
    sigma_ = 0.3
    arr = []
    for x in seeds:
        y = np.sin(2 * math.pi * x) + choose_from_stdn(sigma_)
        arr.append(y)
    return [seeds, arr]


def plot_data(data):
    ar_ = extract_data(data)
    plt.plot(ar_[0], ar_[1], "o")
    plt.show()


def out_csv(data, file):
    ar_ = extract_data(data)
    with open(file, "w") as f:
        x_ = pd.DataFrame({'index': ar_[0], 'output': ar_[1]}).sample(frac=1)
        x_.to_csv(file, header=False, index=None)


def calculate_n_generation(initial_x, n_range, h):
    arr = [initial_x]
    x = initial_x
    if not (0 < initial_x < 1) or not (0 < h < 1):
        pass
    for i in n_range:
        if i == 0:
            continue
        x += x * (1 - x) - h
        if x <= 0:
            arr.append(0)
        else:
            arr.append(x)
    return arr


# x = np.arange(0, 1, 0.01)
# out_csv(data=x, file='sin.csv')
