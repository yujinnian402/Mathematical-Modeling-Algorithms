import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]

# 1. 数据加载与可视化（示例数据）
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
ts = data['Passengers'] #从DataFrame类型的data里面提取出名为'passengers'的不同月份乘客数量的列
# 并将其赋值给变量ts（pandas的Series类型，是一种一维数组，会保留原DataFrame的索引）存储，
# data[['Passengers']]返回的是二维的DataFrame

plt.plot(ts)
plt.title('原始数据')
plt.show()

# 2. 拆分训练集/测试集
train = ts[:-12] #前n-12个为训练集
test = ts[-12:] #最后12个为测试集

# 3. 构建ARIMA模型并拟合
model = ARIMA(train, order=(1,1,1))  # (p,d,q)实际可调
model_fit = model.fit()

# 4. 预测
pred = model_fit.forecast(steps=12) # 此处步长与测试集长度保持一致，才能保证测试集的每个部分都有原来的真实值与新的预测值进行对比
print("预测值：", pred)

# 5. 可视化对比
plt.plot(range(len(ts)), ts, label='真实值') # x轴（range(len(ts)))注意range函数是从0到len(ts)，y轴(ts).
plt.plot(range(len(train), len(ts)), pred, label='预测值') # x轴的range函数是从len(train)到len(ts),y轴(pred).
plt.legend()
plt.title('ARIMA模型预测效果')
plt.show()
