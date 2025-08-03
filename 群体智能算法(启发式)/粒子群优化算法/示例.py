import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

# 能耗目标函数
rastrigin = lambda x1,x2: 20 + x1**2 + x2**2 - 10*(np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2))

# 粒子群参数
pop, dim = 30, 2 # 设置粒子数量30
max_iter = 100 # 最大迭代次数100
w, c1, c2 = 0.8, 1.5, 1.5 # w代表惯性权重,c1,c2代表个体和群体学习因子
bound = [-5, 5] # 搜索空间边界

X = np.random.uniform(bound[0], bound[1], (pop, dim)) # 初始化位置X,X在边界范围内均匀随机分布
V = np.random.uniform(-1, 1, (pop, dim)) 
    # 初始化速度V,速度在[-1,1]范围内均匀随机分布 形状均是(pop,dim),即每个粒子有2个坐标和2个速度分量
Pbest = X.copy() # 初始化每个粒子的历史最优位置为当前位置
Pbest_fit = rastrigin(X[:,0], X[:,1]) # 初始化每个粒子的历史最优适应度
Gbest = X[np.argmin(Pbest_fit)]
Gbest_fit = np.min(Pbest_fit)
curve = [] # 用于记录每轮迭代的全局最优适应度,后续用于收敛曲线可视化

for it in range(max_iter): # 外层循环,控制迭代次数
    # 计算适应度
    fit = rastrigin(X[:,0], X[:,1]) # 计算所有粒子的当前适应度(能耗值)(广播机制)
    # 更新个体最优
    better = fit < Pbest_fit 
    # 判断哪些粒子的当前适应度比历史最优还好(better是布尔数组,存储当前所有粒子的适应度)
    Pbest[better] = X[better]
    Pbest_fit[better] = fit[better] # 更新这些粒子的历史最优位置和适应度
    # 更新全局最优
    if Pbest_fit.min() < Gbest_fit:
        Gbest = Pbest[np.argmin(Pbest_fit)] # argmin函数返回数组中最小值的索引(位置)
        Gbest_fit = Pbest_fit.min()
    # 记录收敛曲线
    curve.append(Gbest_fit)
    # 速度和位置更新
    r1, r2 = np.random.rand(pop, dim), np.random.rand(pop, dim)
    V = w*V + c1*r1*(Pbest-X) + c2*r2*(Gbest-X)
    X = X + V
    # 越界处理
    X = np.clip(X, bound[0], bound[1])

print("最优能耗：", Gbest_fit)
print("最优参数：", Gbest)

# 可视化：收敛曲线
plt.figure(figsize=(7,4))
plt.plot(curve, lw=2)
plt.title("PSO收敛曲线(最小能耗随迭代变化)")
plt.xlabel("迭代轮数")
plt.ylabel("历史最优能耗")
plt.grid(True)
plt.tight_layout()
plt.show()

# 可视化：搜索轨迹分布
plt.figure(figsize=(6,6))
x = np.linspace(bound[0], bound[1], 200)
y = np.linspace(bound[0], bound[1], 200)
Xg, Yg = np.meshgrid(x, y) # 生成二维坐标网格,计算每个点的能耗值Z
Z = rastrigin(Xg, Yg)
plt.contourf(Xg, Yg, Z, levels=50, cmap='viridis') # 画出能耗函数的等高线分布(背景色表示能耗高低)
plt.colorbar(label="能耗")
plt.scatter(Gbest[0], Gbest[1], c='r', s=120, label="最优点") # 标出最优点(红色大点)
plt.title("PSO能耗优化搜索轨迹")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.tight_layout()
plt.show()