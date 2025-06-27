# -*- coding: utf-8 -*-
"""
AHP 层次分析法——公司选址决策
流程：读取csv数据，构造判断矩阵，计算权重，一致性检验，输出权重与排名
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei",  "sans-serif"]

# 1. 读取数据
data = pd.read_csv('case1_ahp_company.csv')

# 2. 构造判断矩阵（假定为专家法两两比较得到，可以实际根据问卷/调研获取）
A = np.array([
    [1, 3, 2, 0.5],
    [1/3, 1, 0.5, 1/3],
    [0.5, 2, 1, 1/3],
    [2, 3, 3, 1]
])

# 3. 求最大特征值和特征向量，得到权重
w, v = np.linalg.eig(A) # 计算特征值和特征向量,返回包含特征值的数组(复数)w和特征向量矩阵v
max_idx = np.argmax(w.real) # 查照特征值最大值索引
max_eigval = w[max_idx].real # 取特征值最大值的值
weight = v[:, max_idx].real # 从矩阵v所有行(:)中提取第max_idx列(最大特征值对应的特征向量)
weight = weight / weight.sum() # 归一化得到权重向量

# 4. 一致性检验
n = A.shape[0] # 获取矩阵行数
CI = (max_eigval - n) / (n - 1) # 一致性指标CI计算(CI为0表示完全一致,值越大不一致性越强)
RI_list = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45] # n=4时RI=0.90
CR = CI / RI_list[n] # 一致性比率CR计算与判断(CR<0.1一致性可接受,判断矩阵合理  CR>=0.1需重新调整判断矩阵)
print(f"一致性比率CR={CR:.4f},{'通过' if CR<0.1 else '不通过'}")

# 5. 综合得分与排序
ind_score = data.iloc[:, 1:].values @ weight
    # iloc[:,1:]提取所有行(:)和从第二行到最后一行(1:)的数据   .values将DataFrame转换为numpy数组  @是矩阵乘法,计算加权和
result = pd.DataFrame({'城市': data['城市'], '综合得分': ind_score}) # 将字典形式的数据组合成表格形式(键为列名,值为列数据)
city_score = result.groupby('城市').mean().sort_values('综合得分', ascending=False)
    # 按城市名称分组(处理可能重复的数据),后计算每组的平均值,按得分降序排列
print(city_score)

# 6. 可视化结果
plt.bar(city_score.index, city_score['综合得分']) # 绘制柱状图
plt.title('AHP公司选址方案得分')
plt.ylabel('综合得分')
plt.show()