import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 设置matplotlib支持中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']     # 中文用黑体
plt.rcParams['axes.unicode_minus'] = False       # 负号正常显示

# 1. 读取数据
df = pd.read_csv("linear_sample_1.csv") # 用pandas库的read_csv函数读取CSV文件，将其解析为DataFrame(表格型数据结构)
print(df.head()) # 调用DataFrame的head方法，默认显示前五行数据

# 2. 拆分X、y
X = df[["x"]].values # values属性转化为Numpy数组
y = df["y"].values

# 3. 建模
model = LinearRegression()
model.fit(X, y)

# 4. 输出参数
print("截距:", model.intercept_)
print("系数:", model.coef_)

# 5. 预测
print("当x=20时预测y值:", model.predict([[20]])[0])

# 6. 可视化
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='原始数据')
plt.plot(X, model.predict(X), color='red', label='拟合直线')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("线性回归示例：linear_sample_1.csv")
plt.show()
