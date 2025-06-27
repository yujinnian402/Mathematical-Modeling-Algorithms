import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum # LpProblem用于定义线性规划问题,LpMaxmize表示目标是最大化,
    # LpVariable用于创建决策变量，LpSum用于便捷地构建线性表达式的求和操作

# 读取CSV数据
data = pd.read_csv('factory_resource_allocation.csv')
profit = data['Profit'].tolist() # tolist方法将提取出的Profit列数据转化为python列表
material = data['Material'].tolist()
labor = data['Labor'].tolist()
machine = data['Machine'].tolist()
resource_limits = [100, 120, 90]

# 定义模型
model = LpProblem('Factory_Resource_Allocation', sense = LpMaximize) # 使用LpProblem类创建一个线性规划问题对象model
    # snese参数设置为LpProblem表明这个线性规划问题是要最大化目标函数

# 定义决策变量
x = [LpVariable(f'x_{i}', lowBound=0) for i in range(len(data))] # 用LpVariable创建一个决策变量，下限为0

# 目标函数
model += lpSum([profit[i] * x[i] for i in range(len(data))])

# 约束
model += lpSum([material[i] * x[i] for i in range(len(data))]) <= resource_limits[0]
model += lpSum([labor[i] * x[i] for i in range(len(data))]) <= resource_limits[1]
model += lpSum([machine[i] * x[i] for i in range(len(data))]) <= resource_limits[2]

# 求解
model.solve() # 调用model的solve方法，让pulp库去求解

# 输出
for i in range(len(data)):
    print(f'产品{data.Product[i]} 生产量 =', x[i].varValue)
print('最大利润 =', model.objective.value())
