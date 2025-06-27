import pandas as pd
from collections import defaultdict # python内置的默认字典,方便按组分类存储物品(键不存在时自动创建列表)

# 读取数据
df = pd.read_csv('group_knapsack.csv')
W = 5  # 总容量

# 分组数据
groups = defaultdict(list) # 创建一个默认值为列表的字典,键是组号,值是该组所有物品的(重量,价值)元组
for _, row in df.iterrows(): # 遍历df的每一行,iterrows为生成行迭代器
    groups[row['group']].append((row['weight'], row['value'])) # 按groups列的值(组号),把该行weight,value打包成元组,追加到groups
       # 对应的列表里

# 初始化DP表，f[j]表示容量j时的最大价值
f = [0] * (W+1)

for g in groups.values(): # 外层循环:组遍历
    # 每组只能选一个
    tmp = f[:] # 临时数组赋值:每次处理新组前，复制当前f数组到tmp。因为分组背包每组只能选一个，用tmp做 “临时状态”，
          # 避免同一组内多个物品互相干扰（保证每组选一个时，基于 “选当前组前的状态” 更新 ）。
    for weight, value in g: #组内遍历
        for j in range(weight, W+1): # 动态规划递推核心部分
            tmp[j] = max(tmp[j], f[j-weight]+value)
    f = tmp

print("分组背包最大价值：", f[W])
