# -*- coding: utf-8 -*-
"""
多目标非线性工艺参数优化（加权法）
数据文件：process_data.csv
优化目标：能耗最小、产量最大、污染最小（归一化加权）
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize # 用于执行优化参数
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# 1. 读取数据
data = pd.read_csv("process_data.csv") # data会变成一个DataFrame(表格型数据结构)

# 2. 目标函数归一化（极小化全部目标）
def normalize(series, minimize=True):
    arr = series.values
    if minimize:
        return (arr - arr.min()) / (arr.max() - arr.min()) # 对要极小化的目标(能耗,污染)做归一化,将其值映射到[0,1]区间,值越小代表越优
    else:
        # 若目标是最大化（如产量），转为极小化目标
        return (arr.max() - arr) / (arr.max() - arr.min()) # 对要最大化的目标(产量)做归一化,转为极小化,值越小代表原产量越大,越符合最大化需求

data['能耗归一'] = normalize(data['能耗(kWh)'], True)
data['产量归一'] = normalize(data['产量(kg/h)'], False)
data['污染归一'] = normalize(data['污染指数'], True) # 最终得到归一化后的能耗归一,产量归一,污染归一列,存入data中

# 3. 加权法目标函数
def weighted_obj(x, w=(0.4, 0.4, 0.2)): #
    idx = int(x[0])  # 简化为“选最优工艺参数行”
    val = w[0]*data['能耗归一'][idx] + w[1]*data['产量归一'][idx] + w[2]*data['污染归一'][idx]
    return val

# 4. 优化：在所有参数行中找到最优方案
res = minimize(weighted_obj, x0=[0], bounds=[(0, len(data)-1)], method='Powell') # bounds限制x的范围,指定使用Powell优化算法

opt_idx = int(res.x[0]) # 得到优化结果res包含最优解等信息,res.x[0]是最优索引
opt_row = data.iloc[opt_idx] # 取出整数opt_idx对应行数据opt_row

print("最优方案：\n", opt_row)

# 5. 帕累托前沿可视化
plt.figure(figsize=(8,6))
plt.scatter(data['能耗(kWh)'], data['产量(kg/h)'], c=data['污染指数'], cmap='cool', s=60)
    # 用污染指数作为颜色映射(cmap='cool'指定颜色映射方案),点的大小s=60
plt.colorbar(label='污染指数') # 添加颜色条,并设置颜色条标签为污染指数
plt.xlabel('能耗(kWh)')
plt.ylabel('产量(kg/h)')
plt.title('能耗-产量-污染 帕累托前沿')
plt.scatter([opt_row['能耗(kWh)']], [opt_row['产量(kg/h)']], color='red', marker='*', s=180, label='最优方案')
plt.legend()
plt.tight_layout() # 自动优化布局,避免元素重叠
plt.show()