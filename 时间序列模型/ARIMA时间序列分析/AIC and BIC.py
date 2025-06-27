import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
best_aic = float('inf')
best_order = None
for p in range(3):
    for d in range(2):
        for q in range(3):
            try:
                model = ARIMA(data['成交量'], order=(p,d,q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p,d,q)
            except:
                continue
print('AIC最小的模型:', best_order)
