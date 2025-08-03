import numpy as np
import random

# 假设有3台机器，5个工件，每个工件加工顺序不同，目标是使总用时最短（简单JobShop调度）
# job_table[i][j]表示第i个工件在第j台机器上的加工时间
job_table = np.array([
    [6, 2, 4],
    [3, 7, 5],
    [5, 4, 6],
    [8, 3, 2],
    [4, 6, 7]
])
num_jobs, num_machines = job_table.shape

# 个体编码为5个工件的加工顺序
pop_size = 40
max_gen = 80
mutate_rate = 0.12

def evaluate(ind):
    sequence = ind
    machine_time = np.zeros(num_machines)
    for job_idx in sequence:
        for m in range(num_machines):
            # 每台机器加工完该工件的结束时间 = 前一台机器加工完的时间和本机可用时间取最大+本机加工时间
            start = max(machine_time[m], machine_time[m-1] if m>0 else 0)
            machine_time[m] = start + job_table[job_idx, m]
    return machine_time[-1]

def init_pop():
    return [random.sample(range(num_jobs), num_jobs) for _ in range(pop_size)]

def select(pop, scores):
    idx = np.argsort(scores)
    return [pop[i] for i in idx[:pop_size//2]]

def crossover(p1, p2):
    cut = random.randint(1, num_jobs-2)
    child = p1[:cut]
    for job in p2:
        if job not in child:
            child.append(job)
    return child

def mutate(ind):
    if random.random() < mutate_rate:
        i, j = random.sample(range(num_jobs), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind

pop = init_pop()
best_makespan = float('inf')
best_ind = None
record = []
for gen in range(max_gen):
    scores = np.array([evaluate(ind) for ind in pop])
    if scores.min() < best_makespan:
        best_makespan = scores.min()
        best_ind = pop[np.argmin(scores)]
    record.append(best_makespan)
    parents = select(pop, scores)
    children = []
    while len(children) < pop_size:
        p1, p2 = random.sample(parents, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        children.append(child)
    pop = children
print("最优工件顺序:", best_ind, "最短总用时:", best_makespan)
import matplotlib.pyplot as plt
plt.plot(record)
plt.xlabel('代数')
plt.ylabel('最短总用时')
plt.title('GA优化JobShop收敛曲线')
plt.show()
