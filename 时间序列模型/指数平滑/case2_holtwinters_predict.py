import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 从...库的时间序列分析模块tsa中导入...类,支持趋势trend和季节性seasonal的拟合
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # 过滤警告信息,忽略一些非关键警告

plt.rcParams["font.family"] = ["SimHei",  "sans-serif"]

# 数据读取
data = pd.read_csv('case2_holtwinters_product.csv', parse_dates=['日期']) # 将日期列解析为pandas时间类型(datatime)
data.set_index('日期', inplace=True) # 直接修改data,而非创建新对象

# 霍尔特-温特斯三次指数平滑（季节周期为12）
model = ExponentialSmoothing(data['产量'], trend='add', seasonal='add', seasonal_periods=12) # 趋势,季节波动均为加法叠加
fit = model.fit()
forecast = fit.forecast(steps=6)

# 可视化
plt.plot(data.index, data['产量'], label='历史产量')
plt.plot(pd.date_range(data.index[-1], periods=7, freq='ME')[1:], forecast, label='预测') # 频率为'月度'
plt.title('霍尔特-温特斯多季节指数平滑预测')
plt.xlabel('日期')
plt.ylabel('产量')
plt.legend()
plt.tight_layout()
plt.show()
