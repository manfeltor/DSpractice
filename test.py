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

def generate_test_ds(hm, var, m, stepx, basex = 0, y_zero = 0):
    
    xs = [(i * stepx) + basex for i in range(hm)]
    ys = [(i*m) + y_zero + random.randrange(-var, var) for i in xs]
    return np.array([xs, ys])

predict = 0.2

xs, ys = generate_test_ds(100, 20, 2, 1)

df = pd.DataFrame({'X': xs, 'Y': ys})
df['label'] = df['Y'].shift(-int(len(df)*predict))
df_pred = df[df.isna().any(axis=1)]
df = df.dropna()
print(df)

 x = np.array(df.drop(columns='label'))
y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
scr = clf.score(x_test, y_test)
print(scr)
