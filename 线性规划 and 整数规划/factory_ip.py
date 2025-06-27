import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

# 读取CSV数据
data = pd.read_csv('factory_resource_allocation.csv')
profit = data['Profit'].tolist()
material = data['Material'].tolist()
labor = data['Labor'].tolist()
machine = data['Machine'].tolist()
resource_limits = [100, 120, 90]

# 定义模型
model = LpProblem('Factory_Resource_Allocation', LpMaximize)

# 定义决策变量
x = [LpVariable(f'x_{i}', lowBound=0, cat='Integer') for i in range(len(data))]

# 目标函数
model += lpSum([profit[i] * x[i] for i in range(len(data))])

# 约束
model += lpSum([material[i] * x[i] for i in range(len(data))]) <= resource_limits[0]
model += lpSum([labor[i] * x[i] for i in range(len(data))]) <= resource_limits[1]
model += lpSum([machine[i] * x[i] for i in range(len(data))]) <= resource_limits[2]

# 求解
model.solve()

# 输出
for i in range(len(data)):
    print(f'产品{data.Product[i]} 生产量 =', x[i].varValue)
print('最大利润 =', model.objective.value())
