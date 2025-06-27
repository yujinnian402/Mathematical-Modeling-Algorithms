# -*- coding: utf-8 -*-
"""
多目标帕累托前沿可视化（适配工艺参数、路径规划、自测练习）
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

try:
    data = pd.read_csv("process_data.csv")
    x, y, c = data['能耗(kWh)'], data['产量(kg/h)'], data['污染指数']
    plt.xlabel('能耗(kWh)')
    plt.ylabel('产量(kg/h)')
    plt.scatter(x, y, c=c, cmap='viridis', s=70)
    plt.colorbar(label='污染指数')
    plt.title('工艺参数多目标帕累托前沿')
except:
    data = pd.read_csv("exercise3_data.csv")
    plt.scatter(data['距离'], data['能耗'], c='g', s=60)
    plt.xlabel('距离')
    plt.ylabel('能耗')
    plt.title('练习题多目标帕累托分析')
plt.tight_layout()
plt.show()