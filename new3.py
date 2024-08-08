import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names = header)

array = data.values
x = array[:, 0:8]
y = array[:, 8]
print(x.shape, y.shape)

#scaler = MinMaxScaler(feature_range=(0, 1))
#rescaled_x= scaler.fit_transform(x)
#print(rescaled_x)
#scaler = StandardScaler()
#rescaled_x = scaler.fit_transform(x)
#print(rescaled_x)

scaler = Binarizer(threshold=0.5)
rescaled_x=scaler.fit_transform(x)
print(rescaled_x)





