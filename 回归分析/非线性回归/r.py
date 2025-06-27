import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei']

def exp_func(x, a, b):
    return a * np.exp(b * x)

# 读取数据
data = pd.read_csv('poly_reg.csv')
X = data['x']
y = data['y']

# 非线性拟合
popt, pcov = curve_fit(exp_func, X, y, p0=(1, 0.1))
    # 使用scipy.optimize.curve_fit函数(内部已封装好算法来寻找最优参数),把指数函数exp_func拟合到数据点(X,y)上
    # p0 = (1.0.1)给出参数的初始猜测值:a 1,   b 0,1(注意在非线性拟合中初始猜测值p0十分关键)
    # popt是一个数组,包含了拟合得到的最优参数值([a,b])  pcov是参数的协方差矩阵(用于估算参数的标准差以及他们之间的相关性)
a, b = popt
print("指数模型参数: a=", a, ", b=", b)

y_pred = exp_func(X, a, b)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='原始数据')
plt.plot(X, y_pred, color='red', label='指数模型拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.title('非线性回归（指数型）')
plt.legend()
plt.grid()
plt.show()
