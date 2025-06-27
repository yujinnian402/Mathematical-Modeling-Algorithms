# -*- coding: utf-8 -*-
"""
理想点法：先找到每个目标的单独最优点，再选“距离理想点最近”的可行解
"""
import pandas as pd
import numpy as np

data = pd.read_csv("process_data.csv")

# 各目标最优（极小/极大）
best_energy = data['能耗(kWh)'].min()
best_yield = data['产量(kg/h)'].max()
best_pollution = data['污染指数'].min() # 寻找各目标单独最优值

# 计算每个点到理想点的距离
norm = lambda s: (s - s.min()) / (s.max() - s.min()) # 定义匿名函数用于统一量纲
data['d2ideal'] = ((norm(data['能耗(kWh)']) - 0)**2 + (norm(data['产量(kg/h)']) - 1)**2 + (norm(data['污染指数']) - 0)**2)**0.5
     # 对每一列做归一化,之后分别计算与对应理想差值的平方,结果存入data新增的d2ideal列,代表每条数据到理想点的距离
opt_idx = data['d2ideal'].idxmin() # 调用DataFrame的idxmin方法,找到d2ideal列中值最小的行索引,存入opt_idx
print("理想点法最优方案：\n", data.loc[opt_idx]) # 使用loc方法取出行数据