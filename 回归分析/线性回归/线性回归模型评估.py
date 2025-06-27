from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# y_true:真实值, y_pred:预测值
y_true = [3, 5, 2.5, 7]
y_pred = [2.8, 4.9, 2.6, 6.8]
print('R2:', r2_score(y_true, y_pred)) # 决定系数
print('MSE:', mean_squared_error(y_true, y_pred)) # 均方误差
print('MAE:', mean_absolute_error(y_true, y_pred)) # 平均绝对误差