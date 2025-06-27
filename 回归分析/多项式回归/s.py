import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures # 从sklearn导入多项式特征构造工具
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']

data = pd.read_csv('poly_reg.csv')
X = data[['x']]
y = data['y']

poly = PolynomialFeatures(degree=3) # 实例化多项式特征生成器,degree = 3表示构造3次多项式
X_poly = poly.fit_transform(X) # 对原始特征X做变换:生成多项式特征矩阵

model = LinearRegression()
model.fit(X_poly, y) # 用多项式特征X_poly和标签y训练模型,学习特征与标签的线性关系   fit过程就是找一组系数,让预测值与真实值的误差（如均方误差）最小

print("回归系数:", model.coef_)
print("截距:", model.intercept_)

y_pred = model.predict(X_poly)

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='原始数据')
plt.plot(X, y_pred, color='orange', label='三阶多项式回归拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.title('多项式回归')
plt.legend()
plt.grid()
plt.show()
