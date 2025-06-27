# 1. 导入必要的库
import numpy as np  # 用于数值计算
from sklearn.datasets import load_iris  # 加载内置的鸢尾花数据集
from sklearn.model_selection import train_test_split  # 用于划分数据集
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.metrics import accuracy_score, classification_report  # 导入评估指标

# --- 第2步: 加载数据集 ---
# 加载鸢尾花数据集
iris = load_iris()
# X 存储特征数据 (花萼长度、花萼宽度、花瓣长度、花瓣宽度)
X = iris.data
# y 存储目标变量 (鸢尾花的类别：0, 1, 2)
y = iris.target
# 获取特征名称和类别名称（可选，方便理解）
feature_names = iris.feature_names
target_names = iris.target_names

# 打印数据集基本信息（可选）
print("--- 数据集信息 ---")
print(f"特征名称: {feature_names}")
print(f"类别名称: {target_names}")
print(f"数据维度 (样本数, 特征数): {X.shape}")
print("------------------\n")

# --- 第3步: 数据划分 ---
# 将数据集划分为训练集（用于训练模型）和测试集（用于评估模型）
# test_size=0.3 表示将30%的数据作为测试集
# random_state=42 确保每次划分结果一致，便于复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 打印划分后的数据维度（可选）
print("--- 数据划分结果 ---")
print(f"训练集特征维度: {X_train.shape}")
print(f"测试集特征维度: {X_test.shape}")
print("--------------------\n")

# --- 第4步: 创建并训练随机森林模型 ---
# 创建随机森林分类器实例
# n_estimators=100: 森林中决策树的数量，可以调整
# random_state=42: 保证模型构建过程的随机性可复现
# n_jobs=-1: 使用所有可用的CPU核心进行并行计算（可选，能加速训练）
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 使用训练数据 (X_train, y_train) 训练模型
# fit() 方法是模型学习数据模式的过程
print("--- 开始训练模型 ---")
rf_classifier.fit(X_train, y_train)
print("--- 模型训练完成 ---\n")

# --- 第5步: 使用模型进行预测 ---
# 使用训练好的模型对测试集 (X_test) 进行预测
print("--- 开始预测 ---")
y_pred = rf_classifier.predict(X_test)
print("--- 预测完成 ---\n")
# y_pred 包含了模型对测试集样本类别的预测结果

# --- 第6步: 评估模型性能 ---
# 计算准确率 (Accuracy): 预测正确的样本数 / 总样本数
accuracy = accuracy_score(y_test, y_pred)

# 生成详细的分类报告，包含每个类别的精确率(Precision)、召回率(Recall)、F1分数(F1-score)以及支持数(Support)
report = classification_report(y_test, y_pred, target_names=target_names)

# 打印评估结果
print("======== 模型评估结果 ========")
print(f"准确率 (Accuracy): {accuracy:.4f}") # 保留4位小数
print("\n分类报告 (Classification Report):\n")
print(report)
print("============================")

# --- 第7步: 查看特征重要性 (可选) ---
# 随机森林可以评估每个特征对预测的贡献程度
importances = rf_classifier.feature_importances_
# 创建一个特征名称和重要性得分的对应关系
feature_importance_map = zip(feature_names, importances)
# 按重要性降序排序
sorted_feature_importance = sorted(feature_importance_map, key=lambda x: x[1], reverse=True)

print("\n======== 特征重要性 ========")
for feature, importance in sorted_feature_importance:
    print(f"特征: {feature}, 重要性: {importance:.4f}")
print("============================")
