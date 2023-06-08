import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

df = pd.read_csv("adult_data_train.csv", na_values=['?'])
df = df.drop(["education", "marital-status"], axis=1)
# print(df.info())

num_df = df.select_dtypes(include=['int', 'float'])
X_train, X_test, y_train, y_test = train_test_split(num_df.drop('label', axis=1), num_df['label'], test_size=0.2, random_state=11, stratify=num_df['label'])

print(X_train['fnlwgt'].mean())

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f1_score(y_test, y_pred))

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
print(X_train.fnlwgt.mean())

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f1_score(y_test, y_pred))

n_rows_with_na = (df.isna().sum(axis=1) > 0).sum()
print(n_rows_with_na)

df = df.dropna()
df = pd.get_dummies(df, drop_first=True)
print(len(df.columns))

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred)
print(f1)