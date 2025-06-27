import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("house_price_sample.csv")
X = df[["area", "rooms", "floor"]].values
y = df["price"].values

model = LinearRegression()
model.fit(X, y)

print("截距:", model.intercept_)
print("系数:", model.coef_)
print("R²分数:", model.score(X, y))

# 可视化：面积-价格关系
plt.figure(figsize=(8,6))
plt.scatter(df["area"], y, color='blue', label="真实房价")
plt.scatter(df["area"], model.predict(X), color='red', s=15, label="预测房价")
plt.xlabel("面积")
plt.ylabel("房价")
plt.legend()
plt.title("房价预测（面积-房价投影）")
plt.show()