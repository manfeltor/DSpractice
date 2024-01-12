import pandas as pd
import quandl, datetime
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import random

# hm = how many data points // var = variance // step = x multiplicator // base = y=0 correction // correlation = slope
def create_ds(hm, var, step, base=0):
    val = step + base
    ys = []

    for i in range(hm):
        y = val + random.randrange(-var, var)
        ys.append(y)
        val += step

    xs = [i for i in range(hm)]
    return np.array([xs, ys])

# xs, ys = create_ds(5, 1, 1)
# plt.scatter(xs, ys)
# plt.show()

def create_ds(hm, var, step, base=0, correlation=True):
    val = step + base
    ys = []

    for i in range(hm):
        y = val + random.randrange(-var, var)
        ys.append(y)
        val += step
    
    xs = [i for i in range(hm)]
        
    return np.array(xs, ys)


def generate_test_ds(hm, var, m, stepx, basex = 0, y_zero = 0):
    
    xs = [(i * stepx) + basex for i in range(hm)]
    ys = [(i*m) + y_zero + random.randrange(-var, var) for i in xs]
    return np.array([xs, ys])

xs, ys = generate_test_ds(5, 1, 3, 2)
plt.plot(xs, ys)
print(xs, ys)
plt.show()
