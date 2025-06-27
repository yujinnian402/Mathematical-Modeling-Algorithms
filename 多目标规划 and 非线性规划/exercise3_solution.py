# -*- coding: utf-8 -*-
"""
练习3：路径帕累托多目标分析
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("exercise3_data.csv")
plt.scatter(data['距离'], data['能耗'], c='g', s=60)
plt.xlabel('距离')
plt.ylabel('能耗')
plt.title('自测3 多目标帕累托分析')
plt.tight_layout()
plt.show()