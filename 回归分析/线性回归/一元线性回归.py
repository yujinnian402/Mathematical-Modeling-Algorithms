import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # 使用reshape函数将前面的一维数组转化为二维数组(列向量),-1表示自动计算该维度的长度，
# 注意只有一行是一维数组,有多行，即使只有一列也是二维数组
Y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

plt.scatter(X, Y, label='真实值')
plt.plot(X, Y_pred, color='red', label='拟合线')
plt.legend()
plt.title('一元线性回归示例')
plt.show()

print('斜率：', model.coef_[0])
print('截距：', model.intercept_)
