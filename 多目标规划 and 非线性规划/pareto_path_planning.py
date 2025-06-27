# -*- coding: utf-8 -*-
"""
多目标路径规划 - 目标空间帕累托前沿 + 地图空间路径连线
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# 读取数据
paths = pd.read_csv("path_data.csv")
points = pd.read_csv("map_points.csv")

# ----------- 1. 帕累托前沿判定 -----------
costs = paths[['长度(km)', '能耗(kWh)']].values
pareto = []
for i, cost in enumerate(costs):
    if not any((other[0] <= cost[0] and other[1] < cost[1]) or (other[0] < cost[0] and other[1] <= cost[1]) for other in costs):
        pareto.append(i)

# ----------- 2. 地图空间 路径连线可视化 -----------
plt.figure(figsize=(9, 7))
colors = ['b', 'g', 'orange', 'purple', 'c', 'm', 'y']
for i, row in paths.iterrows():
    pts_idx = list(map(int, row['点序列'].split(',')))
    route = points.iloc[pts_idx][['x', 'y']].values
    # 路径连线
    if i in pareto:
        plt.plot(route[:, 0], route[:, 1], color='r', linewidth=3, alpha=0.8, label='帕累托最优' if i==pareto[0] else None)
    else:
        plt.plot(route[:, 0], route[:, 1], color=colors[i % len(colors)], linewidth=2, alpha=0.6, label=f"{row['路径编号']}" if i==0 else None)
    # 路径点
    plt.scatter(route[:, 0], route[:, 1], color=colors[i % len(colors)], s=60)
    # 起点/终点特殊高亮
    plt.scatter(route[0, 0], route[0, 1], color='k', s=80, marker='s', label='起点' if i==0 else "")
    plt.scatter(route[-1, 0], route[-1, 1], color='k', s=80, marker='*', label='终点' if i==0 else "")
    # 路径编号标注
    mid = route[len(route)//2]
    plt.text(mid[0], mid[1]+0.4, row['路径编号'], fontsize=12, color=colors[i % len(colors)], fontweight='bold')
# 地图点编号标注
for idx, pt in points.iterrows():
    plt.text(pt['x'], pt['y']-0.25, f"{idx}", fontsize=10, color='k')

plt.xlabel('x 坐标')
plt.ylabel('y 坐标')
plt.title('路径地图空间连线图（高亮帕累托最优路径）')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()

# ----------- 3. 目标空间帕累托前沿（原有） -----------
plt.figure(figsize=(8, 6))
plt.scatter(paths['长度(km)'], paths['能耗(kWh)'], c='b', label='所有路径')
plt.scatter(paths.iloc[pareto]['长度(km)'], paths.iloc[pareto]['能耗(kWh)'], c='r', label='帕累托最优')
for i, row in paths.iterrows():
    plt.text(row['长度(km)'], row['能耗(kWh)'], row['路径编号'])
plt.xlabel('路径总长度 (km)')
plt.ylabel('总能耗 (kWh)')
plt.legend()
plt.title('路径方案多目标帕累托前沿')
plt.tight_layout()
plt.show()
