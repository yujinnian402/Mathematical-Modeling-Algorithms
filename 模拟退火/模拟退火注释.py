import math  # 导入数学库，用于计算距离 (math.hypot) 和指数函数 (math.exp)
import random # 导入随机数库，用于生成随机数、打乱列表和随机选择

# 计算两点间距离
def distance(a, b):
    # 定义一个函数 distance，接收两个点 a 和 b 作为输入。
    # a 和 b 都是包含两个元素的元组或列表，代表二维坐标 (x, y)。
    # math.hypot(dx, dy) 计算 sqrt(dx*dx + dy*dy)，即两点间的欧几里得距离。
    return math.hypot(a[0] - b[0], a[1] - b[1])

# 计算整个路径长度
def total_distance(route, cities):
    # 定义一个函数 total_distance，计算给定路线的总长度。
    # route 是一个包含城市索引的列表，表示访问顺序。
    # cities 是一个包含所有城市坐标的列表。
    dist = 0 # 初始化总距离为 0
    # 遍历路线中的每个城市索引 i
    for i in range(len(route)):
        # 计算当前城市 route[i] 与下一个城市之间的距离。
        # route[(i + 1) % len(route)] 用于处理最后一个城市回到第一个城市的情况（取模运算实现循环）。
        # cities[index] 获取对应索引的城市坐标。
        dist += distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return dist # 返回计算出的总距离

# 随机交换两个城市，生成新解（邻域解）
def neighbor(route):
    # 定义一个函数 neighbor，用于在当前路线附近生成一个新路线（邻域解）。
    # route 是当前的路线列表。
    # random.sample(range(len(route)), 2) 从路线的索引范围 [0, len(route)-1] 中随机选择两个不重复的索引 a 和 b。
    a, b = random.sample(range(len(route)), 2)
    # 交换索引 a 和 b 对应的城市在路线中的位置。
    route[a], route[b] = route[b], route[a]
    return route # 返回修改后的新路线

# 模拟退火算法
def simulated_annealing(cities, T_initial=1000, T_min=1e-8, alpha=0.995, L=100):
    # 定义模拟退火主函数。
    # cities: 城市坐标列表。
    # T_initial: 初始温度，一个较高的值。
    # T_min: 终止温度，当温度低于此值时算法停止。
    # alpha: 降温系数 (0 < alpha < 1)，每次迭代后温度乘以该系数。
    # L: 内循环次数，即在每个温度下尝试生成新解的次数 (马尔可夫链长度)。
    n = len(cities) # 获取城市数量
    # 生成一个初始路线，包含从 0 到 n-1 的所有城市索引。
    current_route = list(range(n))
    # random.shuffle 将初始路线随机打乱，得到一个随机的起始解。
    random.shuffle(current_route)
    # 计算当前初始路线的总距离（成本）。
    current_cost = total_distance(current_route, cities)
    # 将当前温度 T 初始化为初始温度 T_initial。
    T = T_initial
    # 复制当前路线作为目前找到的最佳路线。
    best_route = list(current_route)
    # 将当前成本作为目前找到的最佳成本。
    best_cost = current_cost

    # 外循环：当温度 T 高于最小温度 T_min 时继续迭代。
    while T > T_min:
        # 内循环：在当前温度 T 下，执行 L 次迭代。
        for _ in range(L):
            # 生成一个邻域解（新路线）。注意：neighbor 函数会修改传入的列表，
            # 所以传入 list(current_route) 来创建一个副本，避免直接修改 current_route。
            new_route = neighbor(list(current_route))
            # 计算新路线的总距离（新成本）。
            new_cost = total_distance(new_route, cities)
            # 计算新成本与当前成本的差值 delta。
            delta = new_cost - current_cost

            # Metropolis 准则：判断是否接受新解。
            # 如果 delta < 0，表示新解更优（成本更低），则无条件接受。
            # 如果 delta >= 0，表示新解更差或相同，则以一定概率接受。
            # 这个概率是 math.exp(-delta / T)，温度 T 越高，接受差解的概率越大；
            # 随着 T 降低，接受差解的概率逐渐减小。random.random() 生成一个 [0, 1) 之间的随机数。
            if delta < 0 or random.random() < math.exp(-delta / T):
                # 如果接受新解，则更新当前路线和当前成本。
                current_route = new_route
                current_cost = new_cost
                # 如果当前成本优于历史最佳成本，则更新最佳路线和最佳成本。
                if current_cost < best_cost:
                    best_route = list(current_route) # 存储最佳路线的副本
                    best_cost = current_cost
        # 内循环结束后，降低温度。
        T *= alpha

    # 外循环结束（温度降至 T_min 以下），返回找到的最佳路线和对应的最短距离。
    return best_route, best_cost

# 示例：生成 10 个随机城市的坐标 (x, y 都在 0 到 100 之间)
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
# 调用模拟退火函数求解 TSP 问题。
best_route, best_cost = simulated_annealing(cities)

# 打印结果。
print("最优路线:", best_route) # 输出找到的最佳城市访问顺序（索引列表）。
print("最短距离:", best_cost) # 输出对应的最短路径长度。