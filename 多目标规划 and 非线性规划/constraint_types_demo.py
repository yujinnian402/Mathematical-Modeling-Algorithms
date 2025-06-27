# -*- coding: utf-8 -*-
"""
约束类型演示：等式约束、不等式约束、边界
"""
from scipy.optimize import minimize, Bounds

# 目标：min x^2  s.t. x>=1, x<=4, x=2
obj = lambda x: x[0]**2

# 不等式约束 x>=1
con_ineq = {'type':'ineq', 'fun': lambda x: x[0]-1} # ineq不等式约束
# 等式约束 x=2
con_eq = {'type':'eq', 'fun': lambda x: x[0]-2} # eq 等式约束
# 边界约束 x<=4
bounds = Bounds([None], [4])

res = minimize(obj, [0], constraints=[con_ineq, con_eq], bounds=bounds)
print('约束类型综合最优：', res.x)