from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

"""
Перед вами результаты наблюдений длительности нахождения человека в очереди в
зависимости от количества людей в этой очереди.
"""
df = pd.read_csv('tab.csv', index_col='id')

"""
Обучите модель линейной регрессии для прогнозирования и введите указанные параметры.
"""
reg = LinearRegression().fit(df[['X']], df[['Y']])

"""
Определите выборочное среднее X:
Определите выборочное среднее Y:
"""
print(df.X.mean(), df.Y.mean())

"""
Найдите коэффициент Q1:
Найдите коэффициент Q0:
"""
print(reg.coef_[0][0], reg.intercept_[0])

"""
Оцените точность модели, вычислив R^2 статистику:
"""
y = reg.predict(df[['X']])
r2 = r2_score(df.Y, y)
print(r2)