import pandas as pd

# 读取数据
df = pd.read_csv('knapsack_sample.csv')
weights = df['weight'].tolist()
values = df['value'].tolist()
n = len(weights) # 计算物品的数量
W = 8  # 总容量，可自行修改

# 初始化DP表，f[i][j]表示前i个物品容量j时的最大价值
f = [[0]*(W+1) for _ in range(n+1)] # 二维列表f作为动态规划的状态表,有n+1行,W+1列

for i in range(1, n+1):
    for j in range(W+1):
        if j >= weights[i-1]:
            # 决策：选或不选第i个物品，取最大价值
            f[i][j] = max(f[i-1][j], f[i-1][j-weights[i-1]] + values[i-1])
        else:
            f[i][j] = f[i-1][j]

print("最大价值：", f[n][W])
