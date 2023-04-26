import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("adult_data_train.csv", na_values=['?'])
df = df.drop(["education", "marital-status"], axis=1)

df = df.fillna(df.mode().iloc[0])
X_train = df.drop('label', axis=1)
y_train = df['label']

X_train = pd.get_dummies(X_train, drop_first=True)
X_train = X_train.drop('native-country_Holand-Netherlands', axis=1)

df_res = pd.read_csv("adult_data_reserved.csv", na_values=['?'])
df_res = df_res.drop(["education", "marital-status"], axis=1)

df_res = df_res.fillna(df_res.mode().iloc[0])
X_test = pd.get_dummies(df_res, drop_first=True)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(np.ndarray.tolist(y_pred))