import pandas as pd
from collections import deque, defaultdict

edges = pd.read_csv('network.csv')
graph = defaultdict(list)
capacity = defaultdict(lambda: defaultdict(int))
for i, row in edges.iterrows():
    u, v, cap = row['from'], row['to'], row['capacity']
    graph[u].append(v)
    graph[v].append(u)
    capacity[u][v] += cap

def bfs(s, t, parent):
    visited = set()
    queue = deque([s])
    visited.add(s)
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited and capacity[u][v] > 0:
                visited.add(v)
                parent[v] = u
                if v == t:
                    return True
                queue.append(v)
    return False

def edmonds_karp(s, t):
    flow = 0
    parent = {}
    while bfs(s, t, parent):
        v = t
        path_flow = float('inf')
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, capacity[u][v])
            v = u
        v = t
        while v != s:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = u
        flow += path_flow
    return flow

print(edmonds_karp('S', 'T'))
