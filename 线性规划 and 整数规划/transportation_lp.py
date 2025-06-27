import numpy as np
import pandas as pd
from scipy.optimize import linprog

data = pd.read_csv('transportation_problem.csv', index_col=0).T.values # 将第0列作为索引列
cost = np.array(data).T # 成本矩阵表示从每个仓库到每个超市的单位运输成本
supply = [30, 40, 20]
demand = [20, 30, 30, 10]
c = cost.flatten() # 将成本矩阵cost展平为一维数组

A_eq = [] # 创建空列表,便于后续存储线性规划中的等式约束矩阵(系数矩阵)和等式右侧的常量值
b_eq = []

# 仓库供给约束
for i in range(3):
    a = np.zeros(12)  # 创建一个长度为12的一维零数组(3*4=12)
    a[i*4:(i+1)*4] = 1
    A_eq.append(a)
    b_eq.append(supply[i])
# 超市需求约束
for j in range(4): # 循环4次,因为有四个需求点
    a = np.zeros(12)
    a[j::4] = 1
    A_eq.append(a)
    b_eq.append(demand[j])

res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs') # method='highs'：指定求解线性规划问题的算法，
    # highs 是一种高效的求解器 。函数返回的结果存储在 res 变量中，包含最优解、目标函数值等信息 。
print("最小运输总费用:", res.fun)
print("最优方案:")
print(res.x.reshape((3,4)))


# a[i*4:(i+1)*4] = 1:
  # 对第 i 个供应点对应的决策变量位置赋值为 1 。因为每个供应点对应 4 个需求点的运输量（4 个决策变量 ），
  # 所以索引从 i*4 到 (i+1)*4 ，这样设置后，构建的行向量与决策变量相乘求和，就可以表示该供应点的总运输量（所有发往需求点的运输量之和 ）。

# a[j::4] = 1:
  # 对第 j 个需求点对应的决策变量位置赋值为 1 。由于每 4 个决策变量对应一个需求点（3 个供应点到该需求点的运输量 ），所以步长设为 4 ，
  # 索引从 j 开始，这样构建的行向量与决策变量相乘求和，能表示该需求点的总收货量（所有供应点发往该需求点的运输量之和 ）。
