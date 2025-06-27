import pandas as pd

df = pd.read_csv('merge_intervals.csv')
intervals = df.values.tolist()
# 按起点排序
intervals.sort(key=lambda x: x[0])

merged = []
for interval in intervals:
    if not merged or merged[-1][1] < interval[0]:
        merged.append(interval)
    else:
        merged[-1][1] = max(merged[-1][1], interval[1])

print("合并后区间：", merged)
