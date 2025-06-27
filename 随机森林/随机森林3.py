# 1. 导入库
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# from joblib import dump, load # 如果需要保存/加载模型

# 2. 模拟或加载空间数据
# 假设我们有一个CSV文件 'spatial_data.csv' 包含 x, y, depth 列
# 这里我们先模拟生成一些数据
np.random.seed(42)
n_points = 500
x_coords = np.random.rand(n_points) * 7000 # 模拟x坐标
y_coords = np.random.rand(n_points) * 9000 # 模拟y坐标
# 模拟一个更复杂的地形：一个倾斜平面 + 几个高斯山峰/洼地 + 噪声
depth = (100 + 0.005 * x_coords + 0.002 * y_coords
         - 50 * np.exp(-((x_coords - 2000)**2 + (y_coords - 3000)**2) / (2 * 1000**2))
         + 40 * np.exp(-((x_coords - 5000)**2 + (y_coords - 6000)**2) / (2 * 1500**2))
         + np.random.randn(n_points) * 3)

# 创建DataFrame (模拟从文件读取)
df = pd.DataFrame({'x': x_coords, 'y': y_coords, 'depth': depth})
# print("模拟数据前5行:\n", df.head())

# 3. 准备 X 和 y
X = df[['x', 'y']].values
y = df['depth'].values

# 4. 数据划分 (可选，如果数据量足够大，用于评估拟合效果)
# 如果数据量少，有时会用全部数据来训练以获得最佳拟合
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练随机森林回归模型
# 可以尝试增加 n_estimators 看看效果
rf_fitter = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1,
                                  max_depth=20, # 限制树深度，防止过拟合
                                  min_samples_leaf=3) # 叶节点最小样本数，防止过拟合

print("\n开始训练空间数据拟合模型...")
rf_fitter.fit(X_train, y_train)
# 或者用全部数据训练: rf_fitter.fit(X, y)
print("训练完成。")

# 6. 评估拟合效果 (如果在划分的数据上训练)
y_pred = rf_fitter.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n模型在测试集上的 R² 分数: {r2:.4f}")
if r2 < 0.9:
    print("警告：模型拟合效果可能不足以精确预测任意点，考虑增加数据或调整参数。")

# 7. 应用：预测任意点的深度 (核心用法)
def get_predicted_depth(x_coord, y_coord, model):
    """使用训练好的模型预测给定坐标的深度"""
    # 输入需要是二维数组
    return model.predict([[x_coord, y_coord]])[0]

# 示例：预测几个点的深度
point1_x, point1_y = 2500, 3500
point2_x, point2_y = 6000, 7000

depth1 = get_predicted_depth(point1_x, point1_y, rf_fitter)
depth2 = get_predicted_depth(point2_x, point2_y, rf_fitter)

print(f"\n预测坐标 ({point1_x}, {point1_y}) 处的深度: {depth1:.2f} m")
print(f"预测坐标 ({point2_x}, {point2_y}) 处的深度: {depth2:.2f} m")

# B477论文就是用类似这样的函数，结合梯度计算，进行后续的路径规划
# 例如，他们可能还需要一个 get_predicted_gradient(x, y, model_gx, model_gy) 函数

