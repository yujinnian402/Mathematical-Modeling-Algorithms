# -*- coding: utf-8 -*-
"""
TOPSIS 供应商优选
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei",  "sans-serif"]

data = pd.read_csv('case2_topsis_suppliers.csv')
weights = np.array([0.3, 0.2, 0.3, 0.2])
X = data.iloc[:, 1:].values.astype(float)
# 价格/交付周期为成本型，产品质量/服务为效益型
for j in range(X.shape[1]): # 遍历X的各个列,即各个评价指标
    if j in [0,1]:
        X[:, j] = X[:, j].min() / X[:, j]  # 成本型归一化反向
    else:
        X[:, j] = X[:, j] / np.sqrt((X[:, j] ** 2).sum()) # 针对效益性型指标
V = X * weights # 将标准化后的指标数据X与权重weights对应相乘,得到加权后的矩阵V
A_pos = V.max(axis=0) # 正理想解
A_neg = V.min(axis=0) # 负理想解
D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1)) # 计算每个供应商(每行)到正理想解的欧氏距离
D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
C = D_neg / (D_pos + D_neg) # 计算贴合度C
res = pd.DataFrame({'供应商': data['供应商'], '得分': C})
print(res.sort_values('得分', ascending=False)) # 按得分列降序排序
plt.bar(res['供应商'], res['得分'])
plt.title('TOPSIS供应商综合得分')
plt.ylabel('得分')
plt.show()
