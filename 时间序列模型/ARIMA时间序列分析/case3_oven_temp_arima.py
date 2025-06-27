import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei",  "sans-serif"]

# 数据读取
data = pd.read_csv('case3_oven_temp.csv', index_col='时间')

# 一阶差分（如需）
data_diff = data['中心温度'].diff().dropna()

# 建模（示例为(1,1,1)）
model = ARIMA(data['中心温度'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# 预测
forecast = model_fit.forecast(steps=5)
print('未来5步预测：')
print(forecast)

# 可视化
plt.plot(data.index, data['中心温度'], label='实际')
plt.plot(range(data.index[-1]+1, data.index[-1]+6), forecast, label='预测')
plt.legend()
plt.title('炉温曲线中心温度ARIMA预测')
plt.xlabel('时间')
plt.ylabel('中心温度')
plt.tight_layout()
plt.show()
