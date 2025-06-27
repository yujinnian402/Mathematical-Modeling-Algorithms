import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("student_score_sample.csv")
X = df[["self_study", "class_time", "sleep_hours", "family_income"]].values
y = df["score"].values

model = LinearRegression()
model.fit(X, y)

print("截距:", model.intercept_)
print("系数:", model.coef_)
print("R²分数:", model.score(X, y))

# 可视化：自学时长-分数
plt.figure(figsize=(8,6))
plt.scatter(df["self_study"], y, color='blue', label="真实分数")
plt.scatter(df["self_study"], model.predict(X), color='red', s=15, label="预测分数")
plt.xlabel("自学时长")
plt.ylabel("分数")
plt.legend()
plt.title("学生成绩预测（自学时长-分数投影）")
plt.show()