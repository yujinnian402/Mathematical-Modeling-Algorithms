import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge # 从sklearn库导入岭回归模型

plt.rcParams['font.sans-serif'] = ['SimHei']

data = pd.read_csv('ridge_lasso.csv')
X = data[['x1', 'x2', 'x3']] # 从 data 中取出 'x1'、'x2'、'x3' 三列，作为特征（二维 DataFrame）(X固定是二维DataFrame,即使多列特征)
y = data['y'] # 从 data 中取出 'y' 列，作为标签（一维 Series）

ridge = Ridge(alpha=10.0)
    # 实例化岭回归模型，设置正则化系数 alpha=10.0(岭回归核心参数:值越大,正则化越强(对系数的惩罚越大,模型越"保守",避免过拟合);值越小,正则化越弱(接近普通线性回归))
ridge.fit(X, y) # 用特征 X 和标签 y 训练模型，学习特征与标签的线性关系（带正则化）
    # fit 过程：模型会找一组系数 w（对应 x1、x2、x3）和截距 b，让预测值与真实值的误差（加正则化项）最小。
print("岭回归系数:", ridge.coef_) # # 输出模型的系数 [w1, w2, w3]（对应 x1、x2、x3 的权重）
print("截距:", ridge.intercept_) # # 输出截距项 b（模型公式：y = w1*x1 + w2*x2 + w3*x3 + b）

y_pred = ridge.predict(X)

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(y)), y, 'o', label='原始数据')
plt.plot(np.arange(len(y)), y_pred, '-', label='岭回归预测')
plt.xlabel('样本编号')
plt.ylabel('y')
plt.title('岭回归')
plt.legend()
plt.grid()
plt.show()
