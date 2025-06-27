import pandas as pd
import numpy as np

edges = pd.read_csv('graph_floyd.csv', header=None)
nodes = set(edges[0]) | set(edges[1]) # 取edges中第一列(起点列)和第二列(终点列)的并集
nodes = sorted(list(nodes)) # 将节点集合转化为列表并排序
idx = {node: i for i, node in enumerate(nodes)} # 创建一个字典,将每个节点映射到一个索引
n = len(nodes)
dist = np.full((n, n), np.inf) # 创建一个n*n的矩阵dist,初始值全部设为np.inf
for i in range(n):
    dist[i][i] = 0
for _, row in edges.iterrows():
    u, v, w = row[0], row[1], row[2]
    dist[idx[u]][idx[v]] = w # 二维数组表示矩阵
    dist[idx[v]][idx[u]] = w

for k in range(n): # floyd算法核心逻辑:三重循环.k代表中间节点,i代表起点,j代表终点  k中间节点从0开始一直到n-1(-1情形在之前初始化的时候已设立好)
    for i in range(n):
        for j in range(n):
            if dist[i][j] > dist[i][k] + dist[k][j]:
                dist[i][j] = dist[i][k] + dist[k][j]

for i in range(n):
    for j in range(n):
        print(f"最短路 {nodes[i]}->{nodes[j]}: {dist[i][j]}")
