from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None

df = pd.read_csv('fish_train.csv')

X_train = df[df.columns[2:]]
X_train[['Width', 'Height', 'Length1', 'Length2', 'Length3']] = X_train[['Width', 'Height', 'Length1', 'Length2', 'Length3']] ** 3
y_train = df[['Weight']]

df_res = pd.read_csv('fish_reserved.csv')

X_test = df_res[df_res.columns[1:]]
X_test[['Width', 'Height', 'Length1', 'Length2', 'Length3']] = X_test[['Width', 'Height', 'Length1', 'Length2', 'Length3']] ** 3

pca = PCA(n_components=1)
pca.fit(X_train[['Length1', 'Length2', 'Length3']])
X_train['Lengths'] = pca.transform(X_train[['Length1', 'Length2', 'Length3']])
X_test['Lengths'] = pca.transform(X_test[['Length1', 'Length2', 'Length3']])

X_train = X_train.join(df.Species)
X_test = X_test.join(df_res.Species)

X_train = pd.concat([X_train.drop(['Species'], axis=1), pd.get_dummies(X_train['Species'], drop_first=True)], axis=1)
X_test = pd.concat([X_test.drop(['Species'], axis=1), pd.get_dummies(X_test['Species'], drop_first=True)], axis=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_lst = [float(*_) for _ in np.ndarray.tolist(y_pred)]
print(y_pred_lst)
