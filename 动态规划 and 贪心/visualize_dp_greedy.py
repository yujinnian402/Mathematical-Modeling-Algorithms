import matplotlib.pyplot as plt
import pandas as pd

# 以区间调度为例，画出区间选择
df = pd.read_csv('interval_scheduling.csv')
fig, ax = plt.subplots(figsize=(6,3))
for idx, row in df.iterrows():
    ax.plot([row['start'], row['end']], [idx, idx], 'o-', label=row['job'])
ax.set_xlabel('时间')
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['job'])
ax.set_title('区间调度示意图')
plt.legend()
plt.tight_layout()
plt.show()
