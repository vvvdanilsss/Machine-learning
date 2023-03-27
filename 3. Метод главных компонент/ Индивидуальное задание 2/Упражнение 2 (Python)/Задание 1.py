import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

"""
В предложенном файле находится набор синтетических данных. Данные описывают
60 объектов, каждый из которых обладает 10 признаками. Ваша задача, используя
метод главных компонент, перейти к новым координатам и найти следующие параметры.
"""
df = pd.read_csv('94_16.csv', header=None)
X = PCA(svd_solver='full', n_components=2).fit_transform(df)
X_df = pd.DataFrame(X, columns=['x', 'y'])

"""
Введите координату первого объекта относительно первой главной компоненты.
Введите координату первого объекта относительно второй главной компоненты.
"""
print(X_df.iloc[0])
sns.scatterplot(data=X_df, x='x', y='y')

"""
Какое количество групп объектов можно выделить, если использовать только первые
две главных компоненты?
"""
plt.show()
X_max = PCA(svd_solver='full', n_components=10).fit(df)
evr_cum = np.cumsum(X_max.explained_variance_ratio_)

"""
Введите долю объясненной дисперсии при использовании первых двух главных компонент.
Какое минимальное количество главных компонент необходимо использовать, чтобы доля
объясненной дисперсии превышала 0.85
"""
print(np.cumsum(evr_cum[1]))
for _ in range(len(evr_cum)):
    if evr_cum[_] > 0.85:
        print(_+1)
        break

