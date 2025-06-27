# -*- coding: utf-8 -*-
"""
练习2：ε-约束法
"""
import pandas as pd
from scipy.optimize import minimize

data = pd.read_csv("exercise2_data.csv")
# 目标1极小化，目标2不超过15
obj = lambda x: data['目标1'][int(x[0])]
constraint = {'type':'ineq', 'fun': lambda x: 15 - data['目标2'][int(x[0])]}  # 目标2<=15
res = minimize(obj, x0=[0], bounds=[(0, len(data)-1)], constraints=[constraint], method='Powell')
print("自测2最优解：\n", data.iloc[int(res.x[0])])