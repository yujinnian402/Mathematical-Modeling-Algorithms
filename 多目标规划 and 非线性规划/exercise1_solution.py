# -*- coding: utf-8 -*-
"""
练习1：二目标最优化，加权法
"""
import pandas as pd
from scipy.optimize import minimize

data = pd.read_csv("exercise1_data.csv")
# 简化为在所有行中选一行
w = [0.5, 0.5]
obj = lambda x: w[0]*data['目标1'][int(x[0])] + w[1]*data['目标2'][int(x[0])]
res = minimize(obj, x0=[0], bounds=[(0, len(data)-1)], method='Powell')
print("自测1最优解：\n", data.iloc[int(res.x[0])])