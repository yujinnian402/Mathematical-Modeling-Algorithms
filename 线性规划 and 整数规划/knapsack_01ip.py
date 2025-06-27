import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary # LpBinary设置变量为二进制

# 读取CSV数据
data = pd.read_csv('knapsack_sample.csv')
values = data['Value'].tolist()
weights = data['Weight'].tolist()
capacity = 15

model = LpProblem('Knapsack_01',  sense = LpMaximize)
x = [LpVariable(f'x_{i}', cat=LpBinary) for i in range(len(data))]

model += lpSum([values[i]*x[i] for i in range(len(data))])
model += lpSum([weights[i]*x[i] for i in range(len(data))]) <= capacity

model.solve()
for i in range(len(data)):
    print(f'物品{data.Item[i]} 是否选中 =', x[i].varValue)
print('最大总价值 =', model.objective.value())
