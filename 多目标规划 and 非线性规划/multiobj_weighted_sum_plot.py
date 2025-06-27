# -*- coding: utf-8 -*-
"""
工艺参数多目标方案可视化
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

data = pd.read_csv("process_data.csv")
plt.scatter(data['能耗(kWh)'], data['产量(kg/h)'], c=data['污染指数'], cmap='cool', s=60)
plt.xlabel('能耗(kWh)')
plt.ylabel('产量(kg/h)')
plt.colorbar(label='污染指数')
plt.title('能耗-产量-污染 多目标分布')
plt.tight_layout()
plt.show()