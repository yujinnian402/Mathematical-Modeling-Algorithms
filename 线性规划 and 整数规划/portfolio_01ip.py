import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

data = pd.read_csv('portfolio_selection.csv')
investment = data['Investment'].tolist()
returns = data['ReturnRate'].tolist()
budget = 50  # 总资金上限（万元）

model = LpProblem('Portfolio_01', LpMaximize)
x = [LpVariable(f'x_{i}', cat=LpBinary) for i in range(len(data))]

model += lpSum([returns[i]*x[i] for i in range(len(data))])
model += lpSum([investment[i]*x[i] for i in range(len(data))]) <= budget

model.solve()
for i in range(len(data)):
    print(f'项目{data.Project[i]} 是否投资 =', x[i].varValue)
print('最大总收益率 =', model.objective.value())
