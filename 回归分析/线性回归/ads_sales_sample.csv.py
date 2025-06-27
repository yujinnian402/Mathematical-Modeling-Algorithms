import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("ads_sales_sample.csv")
X = df[["tv", "radio", "web"]].values # 通过.values将这三列作为特征,转换为Numpy数组,赋值给x(特征矩阵,每行是样本,每列是特征)
y = df["sales"].values # 选取sales列作为标签(因变量),转换为Numpy数组,赋值给y

model = LinearRegression()
model.fit(X, y)

print("截距:", model.intercept_) # 输出线性回归模型的截距(intercept_)属性
print("系数:", model.coef_) # 输出特征对应的系数(coef_)属性
print("R²分数:", model.score(X, y)) # 越接近1拟合效果越好

# 可视化：tv投入-销售额
plt.figure(figsize=(8,6))
plt.scatter(df["tv"], y, color='blue', label="真实销售")
plt.scatter(df["tv"], model.predict(X), color='red', s=15, label="预测销售") # 点的大小15
plt.xlabel("电视广告投入")
plt.ylabel("销售额")
plt.legend()
plt.title("广告投入与销售额（电视-销售投影）")
plt.show()