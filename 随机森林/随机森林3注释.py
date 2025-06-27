# -*- coding: utf-8 -*-
"""
空间数据插值与预测 - 随机森林回归建模
适用场景：地形建模、资源勘探等需要空间插值的领域
典型应用论文：B477空间路径规划算法相关研究
"""

# 1. 导入库 ----------------------------------------------------------------
import numpy as np               # 数值计算核心库
import pandas as pd              # 数据处理工具
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归算法
from sklearn.model_selection import train_test_split  # 数据集拆分工具
from sklearn.metrics import r2_score  # 回归模型评估指标
# from joblib import dump, load  # 模型持久化工具 (按需使用)

# 2. 创建模拟空间数据集 ------------------------------------------------------
# 生成虚拟空间数据：x坐标(m), y坐标(m), depth深度(m)
np.random.seed(42)  # 固定随机种子确保结果可重复
n_points = 500      # 总采样点数

# 随机生成坐标点（模拟覆盖7000m×9000m的区域）
x_coords = np.random.rand(n_points) * 7000
y_coords = np.random.rand(n_points) * 9000

# 定义复杂地形函数：倾斜基底 + 高斯起伏 + 白噪声
depth = (
    100                            # 基础海拔
    + 0.005 * x_coords             # x方向倾斜：东高西低 (每米增加0.005m)
    + 0.002 * y_coords             # y方向倾斜：北高南低 (每米增加0.002m)
    - 50 * np.exp(                 # 洼地中心 (坐标2000,3000)
        -((x_coords - 2000)**2 + (y_coords - 3000)**2) / (2 * 1000**2) # 使用高斯函数
    )
    + 40 * np.exp(                 # 山峰中心 (坐标5000,6000)
        -((x_coords - 5000)**2 + (y_coords - 6000)**2) / (2 * 1500**2) # 使用高斯函数
    )
    + np.random.randn(n_points) * 3  # 随机噪声（标准差3m）
)

# 将数据组织为DataFrame（模拟CSV文件读取结果）
df = pd.DataFrame({ # dataframe类似excel中的工作表
    'x': x_coords,
    'y': y_coords,
    'depth': depth
})

# 3. 数据集准备 ------------------------------------------------------------
# 特征矩阵（经度维度坐标）
X = df[['x', 'y']].values  # 提取为Numpy数组 (形状 [500,2])

# 目标变量（深度值）
y = df['depth'].values     # 一维数组 (长度500)

# 4. 训练集-测试集划分 ------------------------------------------------------
# test_size=0.2: 保留20%数据作为测试集（100个点）, 80%用于训练（400个点）
# random_state=42: 固定划分方式便于结果复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. 创建并训练随机森林回归模型 ----------------------------------------------
rf_fitter = RandomForestRegressor(
    n_estimators=200,   # 决策树数量（增加树量可提升稳定性但会增加计算量）
    random_state=42,    # 固定随机性保证结果可复现
    n_jobs=-1,          # 使用全部CPU核心加速训练（-1表示自动检测）
    max_depth=20,       # 单棵树的最大深度（抑制过拟合，增强泛化性）
    min_samples_leaf=3  # 叶节点所需最少样本数（值越大抗噪声能力越强）
)

print("\n开始训练空间插值模型...")
rf_fitter.fit(X_train, y_train)  # 执行模型训练
print("模型训练完成。")

# 6. 模型性能评估 ----------------------------------------------------------
y_pred = rf_fitter.predict(X_test)  # 在测试集上进行预测
r2 = r2_score(y_test, y_pred)      # 计算R²分数

print(f"\n模型评估结果:")
print(f"R²分数 = {r2:.4f}")  # 输出保留四位小数的评估结果
if r2 < 0.9:
    print("[建议] 如果R² < 0.9：\n" 
          "1. 尝试增加n_estimators（如300或500）\n" 
          "2. 收集更多训练样本\n" 
          "3. 检查数据噪声是否过大")

# 7. 核心应用：空间任意点深度预测函数 ----------------------------------------
def get_predicted_depth(x_coord: float, y_coord: float, model) -> float:
    """基于训练模型的深度预测函数
    参数：
        x_coord : x轴坐标（单位应与训练数据一致）
        y_coord : y轴坐标
        model   : 已训练好的回归模型
    返回：
        predicted_depth : 预测深度值（单位与训练时相同）
    """
    # 注意：Scikit-learn要求输入必须是二维数组
    # [[x_coord, y_coord]] 创建形状为 (1,2) 的输入数组
    return model.predict([[x_coord, y_coord]])[0]

# 示例预测点1：洼地中心附近（坐标2000,3000）的周边点
point1_x, point1_y = 2500, 3500

# 示例预测点2：山峰中心附近（坐标5000,6000）的随机点
point2_x, point2_y = 6000, 7000

# 执行预测并格式化输出结果
print(f"\n空间深度预测样例:")
print(f"坐标 ({point1_x}, {point1_y}) 处预测深度: {get_predicted_depth(point1_x, point1_y, rf_fitter):.2f} m")
print(f"坐标 ({point2_x}, {point2_y}) 处预测深度: {get_predicted_depth(point2_x, point2_y, rf_fitter):.2f} m")

# 8. 后续扩展思路（如B477论文应用） -----------------------------------------
# 可添加梯度计算函数（需要训练两个模型分别预测x,y方向的偏导）：
# def get_predicted_gradient(x, y, grad_x_model, grad_y_model):
#     grad_x = grad_x_model.predict([[x, y]])[0]
#     grad_y = grad_y_model.predict([[x, y]])[0]
#     return grad_x, grad_y

# 模型保存代码样例（使用joblib）：
# dump(rf_fitter, 'terrain_model.joblib')  # 保存模型
# loaded_model = load('terrain_model.joblib')  # 加载模型
