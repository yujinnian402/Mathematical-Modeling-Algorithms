from collections import deque # 导入双端队列用于BFS

class Dinic:
    def __init__(self, n):
        self.n = n # 节点数量
        self.graph = [[] for _ in range(n)] # 使用邻接表graph存储图结构
        self.level = [0] * n # Level数组记录每个节点的层次(用于分层图)
        self.ptr = [0] * n # ptr数组用于当前弧优化

    # 添加边的实现
    def add_edge(self, u, v, cap):
        self.graph[u].append([v, cap, len(self.graph[v])]) # 正向边:存储目标节点v,容量cap,反向边索引为v的邻接表长度
        self.graph[v].append([u, 0, len(self.graph[u]) - 1]) # 反向边:目标节点u,初始容量0,存储正向边在u的邻接表中的索引(用于快速更新残留容量)

    # BFS构建层次图(从源点开始,为每个节点分配层次)
    def bfs(self, s, t):
        self.level = [-1] * self.n # 初始化层次为-1(未访问)
        queue = deque([s]) # 初始化队列,源点s入队
        self.level[s] = 0 # 初始化源点层次为0
        while queue: # 队列非空时循环
            u = queue.popleft() # 取出队首节点
            for v, cap, _ in self.graph[u]: # 遍历u的所有临界边
                if cap > 0 and self.level[v] == -1:
                    self.level[v] = self.level[u] + 1 # 若边容量>0且v没有被访问,更新v的层次
                    queue.append(v)
        return self.level[t] != -1 # 返回是否能到达汇点

    # DFS寻找阻塞流
    def dfs(self, u, t, flow): # 从u到t的DFS,携带当前流量上限flow
        if u == t: # 若到达汇点,返回当前流量上限
            return flow
        while self.ptr[u] < len(self.graph[u]): # 遍历u的所有邻接边,使用ptr[u]作为起始索引(当前弧优化)
            v, cap, rev = self.graph[u][self.ptr[u]] # 获取当前边信息:目标节点v,容量cap,反向边索引rev
            if cap > 0 and self.level[v] == self.level[u] + 1: # 检查层次条件和容量条件
                pushed = self.dfs(v, t, min(flow, cap)) # 递归调用DFS,流量上限为当前边容量和原流量上限的较小值
                if pushed: # 若找到增广路径
                    self.graph[u][self.ptr[u]][1] -= pushed # 减少正向边容量
                    self.graph[v][rev][1] += pushed # 增加反向边容量
                    return pushed # 返回增广的流量
            self.ptr[u] += 1 # 当前边处理完毕,指针后移
        return 0 # 未找到增广路径,返回0

    def max_flow(self, s, t):
        flow = 0
        #重复构建层次图:直到无法从源点到达汇点
        while self.bfs(s, t):
            self.ptr = [0] * self.n #重置所有节点的当前弧指针
            #重复DFS寻找阻塞流,直到找不到增广路径
            while True:
                pushed = self.dfs(s, t, float('inf')) # 初始流量上限设为无穷大
                if pushed == 0: # 若找不到增广路径
                    break
                flow += pushed # 累加增广的流量
        return flow

# 示例图：0-S, 1-A, 2-B, 3-T
G = Dinic(4)
G.add_edge(0, 1, 10)
G.add_edge(0, 2, 10)
G.add_edge(1, 2, 2)
G.add_edge(1, 3, 4)
G.add_edge(2, 3, 10)
print("Dinic 最大流:", G.max_flow(0, 3))