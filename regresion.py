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
def create_ds(hm, var, step, base=0, correlation=True):
    val = step + base
    ys = []

    for i in range(hm):
        y = val + random.randrange(-var, var)
        ys.append(y)
        val += step
    
    xs = [i for i in range(hm)]
        
    return np.array(xs, ys)

style.use('ggplot')

df = pd.read_excel('gls.xlsx')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', 'Date']]
df.set_index('Date', inplace=True)
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'])*100
df['CHANGE_PCT'] = ((df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'])*100

df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'CHANGE_PCT']]

forecats_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = 10
df['label'] = df[forecats_col].shift(-forecast_out)
# df.dropna(inplace=True)

X = np.array(df.drop(columns='label'))

X = preprocessing.scale(X)
X_Lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

# print(len(X))
# print(len(X))
# print(len(X_Lately))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)

open_pickle = open('linearpk.pickle', 'rb')
clf = pickle.load(open_pickle)

# with open('linearpk.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# acc = clf.score(X_test, y_test)

# print("Coefficients:", clf.coef_)
# print(acc)

forecast_set = clf.predict(X_Lately)

# print(forecast_set)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
# last_unix = pd.to_datetime(last_date)

# print(type(last_date))

# print(last_date)
# print(last_unix)
one_day = pd.Timedelta(days=1)
next_day = last_date + one_day
# next_unix = last_unix + one_day

for i in forecast_set:
    next_date = next_day
    next_day += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df.plot(y=['Adj. Close', 'forecast'])

plt.show()