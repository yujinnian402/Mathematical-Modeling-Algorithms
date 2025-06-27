import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("linear_with_outliers.csv")
X = df[["x"]].values
y = df["y"].values

model = LinearRegression()
model.fit(X, y)

print("截距:", model.intercept_)
print("系数:", model.coef_)
print("R²分数:", model.score(X, y))

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="原始数据（含异常点）")
plt.plot(X, model.predict(X), color='red', label="拟合直线")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("一元线性回归（含异常点）")
plt.show()