import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller # 从...模块导入adfuller函数，用于平稳性检验
from statsmodels.tsa.arima.model import ARIMA # 从...模块导入ARIMA类,用于构建ARIMA模型
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # 从...模块导入plot_acf和plot_pacf函数,用于绘制自相关图和偏自相关图

plt.rcParams["font.family"] = ["SimHei",  "sans-serif"]

# 1. 数据读取
data = pd.read_csv('case1_digital_economy.csv', parse_dates=['时间'], index_col='时间')
    # parse_dates参数将时间列解析为日期时间格式    index_col参数将时间列设置为DataFrame的索引

# 2. 平稳性检验
result = adfuller(data['成交量']) # 对成交量列进行ADF单位根检验,判断时间序列是否平稳
print('ADF检验p值:', result[1]) # 打印ADF检验的p值,该值小于0.05表示序列平稳

# 3. 差分
if result[1] > 0.05:
    data_diff = data['成交量'].diff().dropna() # 序列非平稳,对成交量列进行一阶差分   dropna()方法删除差分后产生的NaN值
else:
    data_diff = data['成交量']

# 4. 可视化自相关/偏自相关
plot_acf(data_diff)
plt.title('ACF')
plt.show()
plot_pacf(data_diff)
plt.title('PACF')
plt.show()

# 5. ARIMA建模（示例为(1,1,1)，实际据AIC定阶）
model = ARIMA(data['成交量'], order=(1,1,1))
model_fit = model.fit() # 拟合ARIMA序列
print(model_fit.summary()) # 打印模型拟合的摘要信息,包括模型参数,AIC,BIC等统计量

# 6. 预测
forecast = model_fit.forecast(steps=5)
print('未来5步预测：')
print(forecast)

# 7. 可视化
plt.plot(data.index, data['成交量'], label='历史') # 绘制历史成交量数据,x轴为时间索引,y轴为成交量值,设置图例标签为"历史"
plt.plot(pd.date_range(data.index[-1], periods=6, freq='5min')[1:], forecast, label='预测')
    # 生成未来时间索引,从最后一个历史时间点开始,生成6个时间点,频率为5分钟,取生成时间索引的第二个到最后一个(排除第一个,即历史最后一个时间点),然后绘制预测数据
plt.legend()
plt.title('数字经济板块成交量ARIMA预测')
plt.xlabel('时间')
plt.ylabel('成交量')
plt.tight_layout()
plt.show()
