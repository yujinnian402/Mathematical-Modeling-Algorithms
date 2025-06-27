import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 2], [2, 1], [3, 3], [4, 5], [5, 4]])
Y = np.array([2.3, 2.9, 3.6, 4.8, 5.3])

model = LinearRegression()
model.fit(X, Y)
print('回归系数:', model.coef_)
print('截距:', model.intercept_)
