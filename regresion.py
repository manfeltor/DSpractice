import pandas as pd
import quandl, datetime
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = pd.read_csv('WIKI_GOOGL.csv')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

# print("Coefficients:", clf.coef_)
# print(acc)

forecast_set = clf.predict(X_Lately)

# print(forecast_set)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = pd.to_datetime(last_date)

# print(last_date)
# print(last_unix)
one_day = pd.Timedelta(days=1)
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    print(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]