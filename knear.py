import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors

file_path_data =  "cluster/breast-cancer-wisconsin.data"

col_names = [
    'Sample code number',
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion', 
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
    'Class'
    ]

df = pd.read_csv(file_path_data, header=None, names=col_names)
df.replace('?', -99999, inplace=True)
df.drop('Sample code number', axis=1, inplace=True)

X = np.array(df.drop('Class', axis=1))
Y = np.array(df['Class'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# print(len(df))
# print(len(X_train), len(X_test))
# print(len(Y_train), len(Y_test))

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, Y_train)
scr = clf.score(X_test, Y_test)

# print("score: ", scr)

samp_data = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,1,1,2,3,2,1]])

samp_data_reshaped = samp_data.reshape(2, -1)

pred = clf.predict(samp_data_reshaped)
print(pred)
print(len(samp_data_reshaped))