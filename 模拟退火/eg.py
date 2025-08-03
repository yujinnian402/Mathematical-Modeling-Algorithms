import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

# 目标函数
def func(x):
    return x * np.sin(10 * x) + 1.0

np.random.seed(42)  # 保证结果可复现(设置随机种子,保证每次运行结果一致)
T0 = 10             # 初始温度(控制接受劣解的概率)
T_min = 1e-4        # 终止温度
alpha = 0.98        # 降温系数(每轮温度乘以alpha,模拟缓慢降温)
n_iter = 5000       # 最大迭代步数(防止死循环)
x = np.random.uniform(0, 2)    # 初始点
f_curr = func(x)
f_best, x_best = f_curr, x
T = T0
trace = [f_curr]    # 历史最优值跟踪(用于记录每步的历史最优值,便于画收敛曲线)
x_trace = [x] # 用于记录每步的历史最优点

# 模拟退火主循环
for step in range(n_iter):
    delta = np.random.uniform(-0.1, 0.1)  # 邻域扰动
    x_new = x + delta
    x_new = np.clip(x_new, 0, 2)  # 保证x落在区间[0,2],越界就截断
    f_new = func(x_new)
    deltaE = f_new - f_curr
    # 判断是否接受新解
    if deltaE < 0 or np.random.rand() < np.exp(-deltaE / T): 
        # 接受新解的条件 np.random.rand()生成0~1之间的随机数
        x, f_curr = x_new, f_new
        if f_curr < f_best:
            x_best, f_best = x, f_curr
    trace.append(f_best)
    x_trace.append(x_best)
    T *= alpha
    if T < T_min:
        break
print(f'最优x: {x_best:.5f}, 最优f(x): {f_best:.5f}')

# 可视化
xx = np.linspace(0, 2, 500)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(xx, func(xx), label='f(x)')
plt.scatter([x_best], [f_best], color='r', label='最优点')
plt.title('目标函数及最优点')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.subplot(1,2,2)
plt.plot(trace)
plt.title('收敛曲线')
plt.xlabel('迭代步')
plt.ylabel('历史最优f(x)')
plt.tight_layout()
plt.show()
