import pandas as pd
from statsmodels.tsa.api import VAR # 导入向量自回归VAR模型,用于构建多变量时间序列模型
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei",  "sans-serif"]

# 数据读取
data = pd.read_csv('case4_multivariate.csv', index_col='时间') # 指定以时间列作为DataFrame的索引,将数据转换为时间序列结构

# 建模
model = VAR(data)
results = model.fit(maxlags=2, ic='aic') # 设置最大滞后阶数为2，模型会尝试从滞后 1 到 2 中，依据选定准则选最优。

# 预测
forecast = results.forecast(data.values, steps=3)
print('未来3步预测：')
print(forecast)

# 可视化
plt.plot(data.index, data['A'], label='A-历史')
plt.plot(data.index, data['B'], label='B-历史')
plt.plot(range(data.index[-1]+1, data.index[-1]+4), forecast[:,0], label='A-预测')
plt.plot(range(data.index[-1]+1, data.index[-1]+4), forecast[:,1], label='B-预测')
plt.legend()
plt.title('多元时间序列VAR预测')
plt.tight_layout()
plt.show()
