# -*- coding: utf-8 -*-
"""
变量类型对比：连续、整数、二元变量建模方法
"""
from scipy.optimize import minimize, Bounds
import numpy as np

# 目标：min (x1-3)^2 + (x2-2)^2
# 连续变量
def obj_cont(x):
    return (x[0]-3)**2 + (x[1]-2)**2
res1 = minimize(obj_cont, [0,0], bounds=Bounds([0,0],[5,5]))
print('连续变量最优：', res1.x)

# 整数变量（暴力枚举）
xs = np.arange(0,6)
best = None
for x1 in xs:
    for x2 in xs:
        val = (x1-3)**2 + (x2-2)**2
        if best is None or val<best[0]:
            best = (val, (x1,x2))
print('整数变量最优：', best[1])

# 二元变量
minval, minx = None, None
for b1 in [0,1]:
    for b2 in [0,1]:
        val = (b1-1)**2 + (b2-1)**2
        if minval is None or val<minval:
            minval, minx = val, (b1,b2)
print('二元变量最优：', minx)