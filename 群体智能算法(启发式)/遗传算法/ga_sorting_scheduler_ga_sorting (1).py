import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取订单数据，每行是一个分拣订单，含10件物品，目标是最优分拣路径（示意，实际业务可复杂化）
data = pd.read_csv('sorting_orders.csv')
num_orders = len(data)
item_cols = [f'item{i+1}' for i in range(10)]

# 假设每件物品有个坐标表
item_positions = {chr(65+i): (random.randint(0,50), random.randint(0,50)) for i in range(10)}

# 适应度函数：路径总距离

def calc_distance(order_items):
    x0, y0 = 0, 0  # 起点为分拣站
    total = 0
    for item in order_items:
        x, y = item_positions[item]
        total += np.sqrt((x-x0)**2 + (y-y0)**2)
        x0, y0 = x, y
    # 返回分拣站
    total += np.sqrt(x0**2 + y0**2)
    return total

# GA参数
pop_size = 50
max_gen = 100
cross_rate = 0.7
mutate_rate = 0.1

# 初始化种群（每个个体为物品的排列顺序）
def init_population():
    return [random.sample(item_cols, len(item_cols)) for _ in range(pop_size)]

def evaluate(pop, order_items):
    return np.array([calc_distance([order_items[col] for col in ind]) for ind in pop])

def select(pop, scores):
    idx = np.argsort(scores)
    return [pop[i] for i in idx[:pop_size//2]]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-2)
    child = parent1[:point]
    for item in parent2:
        if item not in child:
            child.append(item)
    return child

def mutate(ind):
    if random.random() < mutate_rate:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind

# 对每个订单分别优化
for order_idx, row in data.iterrows():
    order_items = row[item_cols]
    pop = init_population()
    best_dist_record = []
    best_ind = None
    best_score = float('inf')
    for gen in range(max_gen):
        scores = evaluate(pop, order_items)
        if scores.min() < best_score:
            best_score = scores.min()
            best_ind = pop[np.argmin(scores)]
        best_dist_record.append(best_score)
        # 选择
        parents = select(pop, scores)
        children = []
        while len(children) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            children.append(child)
        pop = children
    print(f"订单{row['order_id']}最优路径:{best_ind}, 总距离:{best_score:.2f}")
    # 每个订单都画一张收敛曲线
    plt.figure()
    plt.plot(best_dist_record)
    plt.xlabel('代数')
    plt.ylabel('路径总距离')
    plt.title(f'订单{row["order_id"]}分拣路径GA收敛')
    plt.show()
