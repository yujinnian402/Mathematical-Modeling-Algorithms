from pmdarima import auto_arima

stepwise_model = auto_arima(data['成交量'],
                            start_p=0, start_q=0,
                            max_p=3, max_q=3, m=1,
                            start_P=0, seasonal=False,
                            d=None, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
print(stepwise_model.summary())
