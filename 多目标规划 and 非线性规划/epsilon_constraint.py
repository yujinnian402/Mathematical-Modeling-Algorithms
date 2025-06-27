# -*- coding: utf-8 -*-
"""
ε-约束法：将其中一个目标作为目标，其余转为约束
案例：工艺参数中能耗最小，产量不低于某阈值
"""
import pandas as pd
from scipy.optimize import minimize

data = pd.read_csv("process_data.csv")

# 目标函数：能耗最小
obj = lambda x: data['能耗(kWh)'][int(x[0])] # x[0]是优化求解的变量,转为整数后作为行索引,从data中提取对应行的耗能值

# 约束：产量不低于120（可调ε阈值）
constraint = {'type':'ineq', 'fun': lambda x: data['产量(kg/h)'][int(x[0])] - 120}

 # 'type': 'ineq'：表示不等式约束（inequality 的缩写 ），要求约束函数结果 ≥ 0。
 # 'fun'：约束函数，计算 “产量 - 120”。若结果 ≥ 0，说明产量满足 ≥ 120 的要求；否则不满足。
 # 本质：用 ε- 约束法，把 “产量” 从目标转为约束条件（要求产量 ≥ 120）。

res = minimize(obj, x0=[0], bounds=[(0, len(data)-1)], constraints=[constraint], method='SLSQP')
   # x[0] = 0代表初始猜测值:从第0行开始搜索 优化算法:SLSQP(适用于边界约束、等式约束和不等式约束的非线性优化问题) 约束条件:产量>=120
opt_idx = int(res.x[0])
print("ε-约束法最优方案：\n", data.iloc[opt_idx])