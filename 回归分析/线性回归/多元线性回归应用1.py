from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd

Boston = load_boston()
X = Boston.data
Y = Boston.target
model = LinearRegression().fit(X,Y)
print("回归系数：",model.coef_)
print("截距",model.intercept_)
