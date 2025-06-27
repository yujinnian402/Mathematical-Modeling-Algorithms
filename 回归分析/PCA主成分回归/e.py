import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression # 主成分分析法 + 线性回归 = 主成分回归

plt.rcParams['font.sans-serif'] = ['SimHei']

data = pd.read_csv('ridge_lasso.csv')
X = data[['x1', 'x2', 'x3']]
y = data['y']

# 主成分分析，取前2个主成分
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
    # 使用fit_transform方法对特征数据X进行主成分分析,先拟合(fit)数据找到主成分.再将数据转换(transform)到主成分空间,得到降维后的特征数据X——pca(numpy数组)
model = LinearRegression()
model.fit(X_pca, y)
print("PCR回归系数:", model.coef_)
print("PCR截距:", model.intercept_)

y_pred = model.predict(X_pca)

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(y)), y, 'o', label='原始数据') # o表示用圆点样式绘制
plt.plot(np.arange(len(y)), y_pred, '-', label='主成分回归预测') # -表示用折线样式绘制
plt.xlabel('样本编号')
plt.ylabel('y')
plt.title('主成分回归（PCA回归）')
plt.legend()
plt.grid()
plt.show()
