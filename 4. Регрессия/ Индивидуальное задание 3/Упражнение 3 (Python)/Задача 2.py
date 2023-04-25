from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd

df = pd.read_csv('fish_train.csv')
X_train, X_test, y_train, y_test = train_test_split(df[df.columns[2:]], df[['Weight']],
                                                    test_size=0.2, random_state=29,
                                                    stratify=df[["Species"]])
print(X_train.Width.mean())
reg = LinearRegression().fit(X_train, y_train)
y = reg.predict(X_test)
r2 = r2_score(y_test, y)
print(r2)

corr = X_train.corr(method='pearson')
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
print(df.columns)
X = df[df.columns[2:5]]
pca = PCA(svd_solver='full', n_components=1).fit(X)
evr_cum = np.cumsum(pca.explained_variance_ratio_)
print(evr_cum[0])

X_lengths = pd.DataFrame(PCA(svd_solver='full', n_components=1).fit_transform(X[X.columns[:3]]),
                         columns=['Lengths'])
df_copy1 = df.drop(df.columns[2:5], axis=1)
df_copy1.insert(2, 'Lengths', X_lengths['Lengths'])
# print(df.head())
X_train, X_test, y_train, y_test = train_test_split(df_copy1[df_copy1.columns[2:]], df_copy1[['Weight']],
                                                    test_size=0.2, random_state=29,
                                                    stratify=df_copy1[["Species"]])
reg = LinearRegression().fit(X_train, y_train)
y = reg.predict(X_test)
r2 = r2_score(y_test, y)
print(r2)
# sns.scatterplot(x=df_copy.Width, y=df_copy.Weight)
# plt.show()
X_train[['Width', 'Height', "Lengths"]] = X_train[['Width', 'Height', "Lengths"]] ** 3
X_test[['Width', 'Height', "Lengths"]] = X_test[['Width', 'Height', "Lengths"]] ** 3
print(X_train.Width.mean())
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)

X_train_cat = X_train.join(df_copy1.Species)
X_test_cat = X_test.join(df_copy1.Species)
X_train_encoded = pd.get_dummies(X_train_cat, columns=['Species'])
X_test_encoded = pd.get_dummies(X_test_cat, columns=['Species'])
reg = LinearRegression().fit(X_train_encoded, y_train)
y_pred = reg.predict(X_test_encoded)
r2 = r2_score(y_test, y_pred)
print(r2)

X_train_encoded = pd.get_dummies(X_train_cat, columns=['Species'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test_cat, columns=['Species'], drop_first=True)
reg = LinearRegression().fit(X_train_encoded, y_train)
y_pred = reg.predict(X_test_encoded)
r2 = r2_score(y_test, y_pred)
print(r2)