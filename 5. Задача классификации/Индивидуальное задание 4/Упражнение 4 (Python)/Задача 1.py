import pandas as pd
from scipy.spatial import distance
import numpy as np
import io

data = """id,X,Y,Class
1,30,96,0
2,21,73,0
3,17,32,0
4,99,28,1
5,67,51,0
6,50,54,1
7,27,84,0
8,18,31,1
9,11,34,0
10,46,91,1"""
df = pd.read_csv(io.StringIO(data), sep=",", index_col='id')
x = np.array([54, 68])
df['Euclidean'] = [distance.euclidean(obj, x) for index, obj in df.iloc[:,:2].iterrows()]
print(df.Euclidean.min())
print(df.sort_values('Euclidean').head(3))
df['Manhattan'] = [distance.cityblock(obj, x) for index, obj in df.iloc[:,:2].iterrows()]
print(df.Manhattan.min())
print(df.sort_values('Manhattan').head(3))
