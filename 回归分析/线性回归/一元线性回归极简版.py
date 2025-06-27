# 数据
x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.0, 7.8, 10.2]
n = len(x)

x_mean = sum(x) / n
y_mean = sum(y) / n
b1 = sum((xi - x_mean)*(yi - y_mean) for xi, yi in zip(x, y)) / sum((xi - x_mean)**2 for xi in x) # 根据最小二乘法斜率计算公式
# zip(x.y)将x,y配对
b0 = y_mean - b1 * x_mean
print(f"拟合直线: y = {b0:.2f} + {b1:.2f}x")