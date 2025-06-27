
from collections import defaultdict

graph = defaultdict(dict)
graph['S']['A'] = 10
graph['S']['C'] = 10
graph['A']['B'] = 4
graph['A']['C'] = 2
graph['C']['D'] = 9
graph['D']['B'] = 6
graph['B']['T'] = 10
graph['D']['T'] = 10

def dfs(path, u, t, flow):
    if u == t:
        return flow
    for v in graph[u]:
        if (u, v) not in path and graph[u][v] > 0:
            min_cap = min(flow, graph[u][v])
            path.add((u, v))
            f = dfs(path, v, t, min_cap)
            if f > 0:
                graph[u][v] -= f
                graph[v].setdefault(u, 0)
                graph[v][u] += f
                return f
            path.remove((u, v))
    return 0

def ford_fulkerson(s, t):
    flow = 0
    while True:
        f = dfs(set(), s, t, float('inf'))
        if f == 0:
            break
        flow += f
    return flow

print("Ford-Fulkerson 最大流:", ford_fulkerson('S', 'T'))

