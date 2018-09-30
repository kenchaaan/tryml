import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd


def show_normal_dist():
    # show the graph of the density function of normal dist.
    x = np.arange(-4, 4, 0.1)
    y = st.norm.pdf(x, loc=0, scale=1)
    plt.plot(x, y)
    plt.show()


def solve_2dim_equations(a, b, c, d, i, j):
    # solve simultaneous equation on 2 dim.
    mat = np.array([[a, b], [c, d]])
    vec = np.array([i, j])
    return np.linalg.solve(mat, vec)


def out_normal_data_to_csv(filepath, num, mu, sigma):
    # out data according to normal distribution to "filepath"
    x = pd.DataFrame(columns=['height'])
    x['height'] = sigma * np.random.randn(num) + mu
    with open(filepath, 'w') as outfile:
        x.to_csv(outfile, header=False)


def plot_data_from_csv(filepath):
    # plot data from csv
    with open(filepath, 'r') as f:
        x = pd.read_csv(f, header=None, names=('index', 'height'))
    plt.plot(x['height'].values, "o")
    plt.show()
