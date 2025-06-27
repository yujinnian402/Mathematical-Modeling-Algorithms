import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("multilinear_sample.csv")
X = df[["x1", "x2"]].values
y = df["y"].values

model = LinearRegression()
model.fit(X, y)

print("截距:", model.intercept_)
print("系数:", model.coef_)
print("R²分数:", model.score(X, y))

# 可视化（投影到x1和真实y维度）
plt.figure(figsize=(8,6))
plt.scatter(df["x1"], y, color='blue', label="真实y vs x1")
plt.scatter(df["x1"], model.predict(X), color='red', s=15, label="预测y vs x1")
plt.xlabel("x1")
plt.ylabel("y")
plt.legend()
plt.title("多元线性回归（2特征，投影x1-y）")
plt.show()