import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 工时表
job_time = np.array([
    [3,2,7,6],
    [2,3,4,5],
    [4,5,3,2],
    [7,4,6,3],
    [6,7,2,4],
    [5,2,5,3],
    [4,5,3,6],
    [2,3,4,5]
])

pop = 50
tasks = 8
max_iter = 200
cross_rate = 0.7 # 交叉概率0.7
mutation_rate = 0.05 # 变异概率0.05
curve = [] # 记录每轮迭代的历史最优加工时间,用于收敛曲线可视化

def calc_makespan(seq): # 定义计算总加工时间的函数makespan(给定一个任务顺序seq)
    m = job_time.shape[1] # 机器数量(4台)
    endtime = np.zeros(m) # 初始化每台机器的完工时间为0
    for idx in seq: # 按顺序依次安排每个任务
        for j in range(m): # 每个任务在每台机器上加工
            endtime[j] = max(endtime[j], endtime[j-1] if j > 0 else 0) + job_time[idx, j] 
            # 计算第j台机器的完工时间要满足两重限制 
    return endtime[-1] # 返回最后一台机器的完工时间即为总加工时间

# 顺序交叉OX实现,适合排序问题
def ox(parent1, parent2):
    size = len(parent1)
    a, b = sorted(np.random.choice(range(size), 2, replace=False)) # 随机选取一个区间[a,b]
    child = [-1]*size # 初始化子代染色体,-1表示未填充
    # 复制parent1区间
    child[a:b+1] = parent1[a:b+1]
    # 从parent2依次补全
    fill = [x for x in parent2 if x not in child]
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill[j]
            j += 1
    return np.array(child, dtype=int)

# 初始化种群
popu = [np.random.permutation(tasks) for _ in range(pop)]
fitness = np.array([calc_makespan(p) for p in popu])

best_seq = None
best_fit = np.inf

for it in range(max_iter): # 外层循环
    # 选择（锦标赛）
    idx = np.random.randint(0, pop, (pop, 2)) # 生成(pop.2)的随机整数数组,每行是两个个体的索引
    sel = [popu[i[0]] if fitness[i[0]] < fitness[i[1]] else popu[i[1]] for i in idx]
    # 每次从两个个体中选出适应度更优的进入下一代(锦标赛选择),保证优胜劣汰
    # 交叉
    offspring = [] # 新一代个体
    for i in range(0, pop, 2):
        p1, p2 = sel[i].copy(), sel[i+1].copy()
        if np.random.rand() < cross_rate: # 按交叉概率决定是否进行OX交叉,否则直接复制父代
            child1 = ox(p1, p2)
            child2 = ox(p2, p1)
            offspring.extend([child1, child2])
        else:
            offspring.extend([p1, p2])
    # 变异
    for o in offspring:
        if np.random.rand() < mutation_rate:
            i1, i2 = np.random.choice(tasks, 2, replace=False)
            o[i1], o[i2] = o[i2], o[i1]
    popu = offspring
    fitness = np.array([calc_makespan(p) for p in popu])
    # 历史最优
    if fitness.min() < best_fit:
        best_fit = fitness.min()
        best_seq = popu[np.argmin(fitness)].copy()
    curve.append(best_fit)

print("最短总加工时间：", best_fit)
print("最优任务顺序：", best_seq+1)

# 可视化：收敛曲线
plt.figure(figsize=(7,4))
plt.plot(curve, lw=2)
plt.title("GA收敛曲线(最短总加工时间随迭代轮数变化)")
plt.xlabel("迭代轮数")
plt.ylabel("历史最优总加工时间")
plt.grid(True)
plt.tight_layout()
plt.show()
