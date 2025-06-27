import pandas as pd
import numpy as np

# 读取网格数据
grid = pd.read_csv('grid_path.csv', header=None).values
rows, cols = grid.shape

# 初始化DP表
dp = np.zeros((rows, cols), dtype=int)
dp[0, 0] = grid[0, 0]

# 填第一列 处理边界格子
for i in range(1, rows):
    dp[i, 0] = dp[i-1, 0] + grid[i, 0]
# 填第一行 处理边界格子
for j in range(1, cols):
    dp[0, j] = dp[0, j-1] + grid[0, j]

# 状态转移 处理非边界格子
for i in range(1, rows):
    for j in range(1, cols):
        dp[i, j] = min(dp[i-1, j], dp[i, j-1]) + grid[i, j]

print("最小路径和：", dp[rows-1, cols-1])
