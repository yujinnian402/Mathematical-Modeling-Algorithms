import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(" ---数据集信息--- ")
print(f"特征名称:{feature_names}")
print(f"类别名称:{target_names}")
print(f"数据维度(样本数，特征数):{X.shape}")
print("--------------------\n")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

print("--- 数据划分结果 ---")
print(f"训练集特征维度:{X_train.shape}")
print(f"测试集特征维度:{X_test.shape}")
print("---------------\n")

rf_classifier = RandomForestClassifier(n_estimators = 100,random_state = 42,n_jobs=-1)
print("--- 开始训练模型 ---")
rf_classifier.fit(X_train,y_train)
print(" --- 模型训练完成 --- ")

print(" --- 开始预测 --- ")
y_pred = rf_classifier.predict(X_test)
print(" --- 预测完成 --- ")


accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred,target_names = target_names)


print("=======打印评估结果=======")
print(f"准确率(Accuracy:{accuracy:.4f}")
print("\n分类报告（Classification Report):\n")
print(report)
print("====================")

importances = rf_classifier.feature_importances_
feature_importance_map = zip(feature_names,importances)
sorted_feature_importance = sorted(feature_importance_map,key=lambda x:x[1],reverse=True)

print("\n====== 特征重要性 ======")
for feature,importance in sorted_feature_importance:
    print(f"特征:{feature},重要性:{importance:.4f}")
print("=============================")



