import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

city_coord = np.array([
    [11,18], [7,16], [14,12], [9,8], [15,5],
    [3,10], [2,4], [8,3], [13,1], [6,1]
])
city_num = len(city_coord)
D = np.sqrt(((city_coord[:,None,:]-city_coord[None,:,:])**2).sum(-1)) 
    # 计算得到距离矩阵:采用Numpy的广播机制加至三维直接计算出距离矩阵避免双层循环

tau = np.ones((city_num, city_num)) 
    # 初始化信息素矩阵tau,所有城市对之间的信息素初值为1,用np.ones生成city_num*city_num的全1矩阵
alpha, beta, rho, Q = 1, 5, 0.1, 100 
    # rho代表信息素挥发率, Q代表信息素总量常数,alpha和beta分别代表信息素重要程度和距离启发因子
max_iter, ant_num = 50, city_num # 设置最大迭代次数为50,每次迭代蚂蚁数量等于城市数量
best_path, best_len = None, np.inf # 初始化最优路径和最短路径长度
curve = [] # 记录收敛曲线

for it in range(max_iter):
    paths, lens = [], []
    for ant in range(ant_num):
        unvisited = list(range(city_num))
        path = [np.random.choice(unvisited)]
        unvisited.remove(path[0])

        while unvisited:
            i = path[-1] # -1索引表示最后一个,即蚂蚁当前所在的城市
            probs = []
            for j in unvisited:
                prob = (tau[i,j]**alpha)*(1/D[i,j]**beta)
                probs.append(prob) # probs是所有未访问城市的转移概率列表
            probs = np.array(probs)
            probs /= probs.sum() # 每一种单独可选城市的概率计算公式
            next_city = np.random.choice(unvisited, p=probs) # 根据转移概率随机选择下一个城市
            path.append(next_city)
            unvisited.remove(next_city) # 更新路径和未访问列表
        length = sum(D[path[k], path[(k+1)%city_num]] for k in range(city_num)) # 取余运算是因为环形路径
        if length < best_len:
            best_len, best_path = length, path.copy() # 若本蚂蚁路径比历史最优段,则更新最优路径和长度,
        paths.append(path)
        lens.append(length) # 将本蚂蚁的路径和长度加入本轮所有蚂蚁的列表
    tau *= (1-rho)
    for a in range(ant_num):
        for k in range(city_num):
            i, j = paths[a][k], paths[a][(k+1)%city_num] # 当前城市和下一个城市(环形路径)
            tau[i,j] += Q/lens[a]
    curve.append(best_len)  # 记录每轮历史最优路径长度

print("最短路径长度:", best_len)
print("最优路径:", [p+1 for p in best_path])

# ================== 可视化部分 ==================
best_path_cycle = best_path + [best_path[0]]
plt.figure(figsize=(6,6))
plt.scatter(city_coord[:,0], city_coord[:,1], color='red', s=80)
for i, (x, y) in enumerate(city_coord):
    plt.text(x+0.2, y, str(i+1), fontsize=12)
plt.plot(city_coord[best_path_cycle,0], city_coord[best_path_cycle,1], '-o', color='blue', lw=2)
plt.title("ACO最优路径展示")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.show()

# 收敛曲线可视化
plt.figure(figsize=(7,4))
plt.plot(curve, marker='o', lw=2)
plt.title("ACO TSP收敛曲线(历史最优路径长度随迭代轮数变化)")
plt.xlabel("迭代轮数")
plt.ylabel("历史最优路径长度")
plt.grid(True)
plt.tight_layout()
plt.show()