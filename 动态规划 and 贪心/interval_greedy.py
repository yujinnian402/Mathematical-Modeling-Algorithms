import pandas as pd

# 读取区间数据
df = pd.read_csv('interval_scheduling.csv')
intervals = df.values.tolist()
# 按“结束时间”升序排序
intervals.sort(key=lambda x: x[2])

res = [] # 用于存储最终选择的不重叠区间的名称(标识信息)
last_end = -float('inf') # 初始化last_end变量(设为负无穷),用于记录上一个被选中区间的结束时间，
for name, start, end in intervals: # 贪心选择区间(核心逻辑)
    # 若当前区间与前一个不重叠，则选
    if start >= last_end:
        res.append(name)
        last_end = end

print("可选最多不重叠区间数：", len(res))
print("选择的区间：", res)
