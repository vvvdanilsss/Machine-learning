import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
Для прохода на новогодний корпоратив в ИТМО при входе нужно отгадать «логотип
мероприятия». Для получения изображения логотипа необходимо по первым десяти
главным компонентам восстановить исходное изображение (в качестве пригласительных
рассылались матрица счётов и матрица весов первых десяти ГК).
"""
df_loadings = pd.read_csv('X_loadings_792.csv', header=None, delimiter=';').T.to_numpy()
df_reduced = pd.read_csv('X_reduced_792.csv', header=None, delimiter=';').to_numpy()
df = np.dot(df_reduced, df_loadings)
plt.imshow(df, cmap='gray')
plt.show()