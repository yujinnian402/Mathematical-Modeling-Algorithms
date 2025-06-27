import pandas as pd
import heapq # heapq模块提供堆(优先队列)相关的功能,在 Dijkstra 算法里用来高效获取距离最小的节点

# 读取边数据
edges = pd.read_csv('graph.csv')

# 提取所有节点（包括出边、入边的点）
all_nodes = set(edges['from']).union(set(edges['to'])) # set将函数数据转为集合,union方法将两个集合合并

# 构建邻接表
graph = {} # graph字典用于存储图的邻接表示.字典中的键是起点节点，对应的值是一个列表,列表中的元素是元组(终点,权重)
for i, row in edges.iterrows():
    graph.setdefault(row['from'], []).append((row['to'], row['weight'])) # 构建整个图的邻接表结构

# Dijkstra算法
def dijkstra(graph, start, all_nodes):
    # 初始化所有节点距离
    dist = {node: float('inf') for node in all_nodes}
    dist[start] = 0
    visited = set()
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap) # 从堆中弹出距离最小的节点u以及对应的当前距离d
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.get(u, []): # 遍历节点u的所有邻接节点v以及对应的边权重w
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return dist

# 示例调用
print(dijkstra(graph, 'A', all_nodes))
