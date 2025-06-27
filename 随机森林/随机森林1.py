# 1. 导入库
import numpy as np
from sklearn.model_selection import train_test_split # 将数据划分为训练集和测试集
from sklearn.ensemble import RandomForestRegressor # 一种基于随机森林算法的回归模型
from sklearn.metrics import r2_score, mean_squared_error # metrics模块提供各种评估模型性能的指标
# r2_score即决定系数
# mean_squared_error是均方误差

# 2. 创建简单的模拟数据
# 假设我们想根据特征X预测数值y
# y = sin(X) + 一点噪声
np.random.seed(42) # 保证随机性可复现
X = np.sort(5 * np.random.rand(80, 1), axis=0) # 生成80个0-5之间的随机特征值
y = np.sin(X).ravel() + np.random.randn(80) * 0.1 # 计算对应的y值并加上噪声

# 3. 数据划分 (训练集和测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. 创建并训练随机森林回归模型
# n_estimators: 森林中树的数量
# random_state: 控制随机性，保证结果可复现
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
print("开始训练回归模型...")
rf_regressor.fit(X_train, y_train)
print("训练完成。")

# 5. 在测试集上进行预测
y_pred = rf_regressor.predict(X_test)

# 6. 评估模型性能
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # 计算均方误差

print("\n--- 模型评估 ---")
print(f"R² Score (拟合优度): {r2:.4f}") # R²越接近1越好
print(f"RMSE (均方根误差): {rmse:.4f}") # RMSE越小越好

# 7. 预测新数据点 (可选)
# 假设有一个新的X值，我们想预测对应的y
new_X = np.array([[2.5]]) # 注意需要是二维数组
predicted_y = rf_regressor.predict(new_X)
print(f"\n预测 X={new_X[0][0]} 时的 y 值: {predicted_y[0]:.4f}")
