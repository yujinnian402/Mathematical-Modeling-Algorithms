import numpy as np
import matplotlib.pyplot as plt

# 解决中文与负号乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def GM11(x0):
    x0 = np.array(x0) # 将输入的X0转化为numpy数组
    n = len(x0)
    # 一次累加生成
    x1 = x0.cumsum() # numpy数组的cumsum方法返回累加序列数组x1
    # 构造背景值序列
    z1 = 0.5 * (x1[1:] + x1[:-1]) # x1[ :1]意思是x1从索引1到末尾的子数组     x1[ :-1]意思是x1从开头到倒数第二个元素的子数组
    B = np.vstack((-z1, np.ones(n-1))).T # numpy数组的vstack方法将背景值序列取负与n-1个1的数组垂直堆叠，之后再经(T.)转置得到设计矩阵B
    Y = x0[1:] # 构建观测值向量矩阵Y
    # 最小二乘法求参数
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0] # 使用lstsq函数(最小二乘法求解线性方程组)，对B和Y求解得到参数a(发展系数：反应系统自身演化速度)
        # b(灰色作用量：相当于稳定项和外部驱动力)
    # 建立预测模型
    def f(k):
        return (x0[0] - b/a) * np.exp(-a*k) + b/a  # 时间响应函数
    # 还原预测值
    x1_hat = [f(i) for i in range(n+5)]  # 新预测5期(生成累加序列预测值)
    x0_hat = [x1_hat[0]] + [x1_hat[i]-x1_hat[i-1] for i in range(1, len(x1_hat))] # 差分还原得到原始序列预测值
    return np.array(x0_hat)

# 用法示例
data = [36.5, 38.2, 40.7, 43.1, 46.0, 49.1, 52.3, 55.9, 59.7, 63.7]
pred = GM11(data)

plt.plot(range(1, len(data)+1), data, 'o-', label='原始数据') # 'o-'表示数据点用圆点标记，点之间用实线连接
plt.plot(range(1, len(pred)+1), pred, 's--', label='GM(1,1)预测') # 's--'表示数据点用方形标记，点之间用虚线连接
plt.xlabel('期数')
plt.ylabel('数值')
plt.legend()
plt.title('GM(1,1)灰色预测')
plt.show()
