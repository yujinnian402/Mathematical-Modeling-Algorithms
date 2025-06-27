# -*- coding: utf-8 -*-
"""
灰色评价法——生态环境质量多指标评价
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei",  "sans-serif"]

data = pd.read_csv('case3_gray_environment.csv')
X = data.iloc[:, 1:].values.astype(float)
ref = X.max(axis=0) # 按列取最大值
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) # 极差标准化
rho = 0.5 # 分辨系数(控制关联系数的分辨程度)
m, n = X.shape # 获取numpy二维数组X的信息,解包后将行数赋值给m,列数赋值给n
delta = np.abs(X_norm - ref / ref.max())
delta_min = delta.min() # 计算
delta_max = delta.max()
xi = (delta_min + rho * delta_max) / (delta + rho * delta_max) # 计算关联系数
r = xi.mean(axis=1) # 计算关联度
res = pd.DataFrame({'城市': data['城市'], '关联度': r})
print(res.sort_values('关联度', ascending=False))
plt.bar(res['城市'], res['关联度'])
plt.title('灰色评价-环境质量综合排序')
plt.ylabel('灰色关联度')
plt.show()
