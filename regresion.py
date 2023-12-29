import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('WIKI_GOOGL.csv')
df.describe
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'])*100
df['CHANGE_PCT'] = ((df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'])*100

df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'CHANGE_PCT']]

forecats_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = 10
df['label'] = df['Adj. Close'].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(columns='label'))
y = np.array(df['label'])

X = preprocessing.scale(X)

print(len(X), len(y))
print(len(df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

print(acc)
