import numpy as np
import matplotlib.pyplot as plt

# 1. 构造样例数据
def generate_data(n=50):
    np.random.seed(0) # 设置好随机种子，确保后续的每次随机数生成一致，括号里可以是任何数，1，42等
    X = np.linspace(0, 10, n) # 用 linspace 在 [0, 10] 区间生成 n 个等距数值，作为自变量 X（模拟输入特征）
    Y = 2 * X + 1 + np.random.normal(0, 2, n)  # y = 2x + 1 + 噪声(均值0，标准差2，正态噪声分布，模拟真实数据的波动)
    return X, Y

# 2. 计算最优参数（最小二乘法）
def linear_regression(X, Y):
    n = len(X)
    X_mean = np.mean(X) #计算均值
    Y_mean = np.mean(Y)
    b1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2) # 按最小二乘法公式计算斜率b1
    b0 = Y_mean - b1 * X_mean # 计算截距b0
    return b0, b1

# 3. 可视化及拟合效果
def plot_fit(X, Y, b0, b1):
    plt.scatter(X, Y, label='Data') # label用于图例说明
    plt.plot(X, b0 + b1 * X, color='red', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend() # 显示图例
    plt.title('Simple Linear Regression')
    plt.show()

# 主流程
X, Y = generate_data()
b0, b1 = linear_regression(X, Y)
plot_fit(X, Y, b0, b1)
print(f'拟合结果: y = {b0:.2f} + {b1:.2f}x')