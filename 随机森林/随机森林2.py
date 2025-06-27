# 1. 导入库
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. 加载鸢尾花数据集
iris = load_iris() # 返回一个Bunch对象（类似字典）--包含鸢尾花数据集的所有信息（特征数据，目标标签，特证名称，类别名称等）
X = iris.data # 特征数据（二维Numpy数组）形状为（150，4）（150个鸢尾花样本，每个样本有4个特征值）
y = iris.target # 目标标签（一维numpy数组）形状为（150，）
target_names = iris.target_names # 获取类别名称（属性）一维numpy数组['setosa','versicolor','virginica']

# 3. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 创建并训练随机森林分类模型
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
print("开始训练分类模型...")
rf_classifier.fit(X_train, y_train)
print("训练完成。")

# 5. 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)

# 6. 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names) # target_names是关键字参数

print("\n--- 模型评估 ---")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print("\n分类报告 (Classification Report):\n")
print(report)

# 7. 预测新样本类别 (可选)
# 假设有一个新的花的数据，我们想预测它的类别
# 特征顺序：花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]]) # 注意是二维数组
predicted_class_index = rf_classifier.predict(new_flower)[0]
predicted_class_name = target_names[predicted_class_index]
print(f"\n预测新样本 {new_flower[0]} 的类别为: {predicted_class_name} (索引 {predicted_class_index})")

