from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # 导入均方误差MSE评估指标mean_squared_error

X, y = fetch_california_housing(return_X_y=True) # 设置返回格式为特征矩阵X(二维数组:样本数*特征数)和标签向量y(一维数组:样本数)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))