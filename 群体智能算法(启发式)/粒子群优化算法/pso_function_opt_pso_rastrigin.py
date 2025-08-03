import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取初始粒子参数（二维Rastrigin函数）
particles = pd.read_csv('rastrigin_params.csv')
num_particles = len(particles) # 粒子数量

# Rastrigin函数定义
def rastrigin(X):
    x, y = X[0], X[1]
    return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

# PSO参数
max_iter = 100
w = 0.8     # 惯性权重
c1 = 1.5    # 个体学习因子
c2 = 1.5    # 群体学习因子

# 初始化位置和速度
X = particles[['x_init','y_init']].values # 所有粒子的初始位置
V = np.random.uniform(-1, 1, (num_particles, 2)) # 所有粒子的初始速度,随机生成
P = X.copy()  # 个体历史最优位置
fitness = np.array([rastrigin(x) for x in X]) # 当前适应度
P_fitness = fitness.copy() # 个体历史最优适应度,初始化为当前值
G = P[P_fitness.argmin()]  # 全局最优
G_fitness = P_fitness.min()

# 记录收敛曲线
gbest_curve = []

for it in range(max_iter):
    r1 = np.random.rand(num_particles, 2)
    r2 = np.random.rand(num_particles, 2)
    # 更新速度
    V = w*V + c1*r1*(P-X) + c2*r2*(G-X)
    # 更新位置
    X = X + V
    # 更新适应度
    fitness = np.array([rastrigin(x) for x in X])
    # 更新个体历史最优和全局最优
    better_mask = fitness < P_fitness
    P[better_mask] = X[better_mask]
    P_fitness[better_mask] = fitness[better_mask]
    if P_fitness.min() < G_fitness:
        G = P[P_fitness.argmin()]
        G_fitness = P_fitness.min()
    gbest_curve.append(G_fitness)

print("PSO最优结果:", G, "最小函数值:", G_fitness)

# 可视化收敛曲线
plt.figure()
plt.plot(gbest_curve)
plt.xlabel('迭代次数')
plt.ylabel('最优函数值')
plt.title('PSO优化Rastrigin函数收敛曲线')
plt.grid()
plt.show()

# 可视化最终粒子分布和全局最优
plt.figure(figsize=(7,6))
plt.scatter(X[:,0], X[:,1], c='b', label='粒子') 
    # 用散点图画出所有粒子的最终位置(蓝色点) 注意不是每个粒子自身的历史最优位置
plt.scatter(G[0], G[1], c='r', marker='*', s=150, label='全局最优') # 用红色星形标记全局最优位置
plt.xlabel('x')
plt.ylabel('y')
plt.title('PSO最终粒子分布')
plt.legend()
plt.grid()
plt.show()
