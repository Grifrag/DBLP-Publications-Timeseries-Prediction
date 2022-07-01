import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

year = pd.read_csv('Bima2.csv', index_col=[0], sep=';')
year.drop(2020, inplace=True)


def stationary(xronoseira):

    rolaver = xronoseira.rolling(window=10).mean()


    #Paragwgh twn average
    original = plt.plot(xronoseira, color='blue', label='Original')
    aver = plt.plot(rolaver, color='red', label='Moving Average')
    plt.legend(loc='best')
    plt.title('Moving Average')
    plt.show(block=False)

    #Dickey-Fuller :
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(xronoseira, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


stationary(year)
year_log = np.log(year)
moving_avg = year_log.rolling(window=10).mean()
plt.plot(year_log)
plt.plot(moving_avg, color='red')
year_log_moving_avg_diff = year_log - moving_avg
year_log_moving_avg_diff.dropna(inplace=True)
print(year_log_moving_avg_diff.head(10))
plt.show()
stationary(year_log_moving_avg_diff)
expwighted_avg = year_log.ewm(halflife=10).mean()
plt.plot(year_log)
plt.plot(expwighted_avg, color='red')
plt.show()
year_log_ewma_diff = year_log - expwighted_avg
stationary(year_log_ewma_diff)

year_log_diff = year_log - year_log.shift()
plt.plot(year_log_diff)
plt.show()
year_log_diff.dropna(inplace=True)
stationary(year_log_diff)

lag_acf = acf(year_log_diff, nlags=20)
lag_pacf = pacf(year_log_diff, nlags=20, method='ols')

#PLOT
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(year_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(year_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

#PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(year_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(year_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

model = ARIMA(year_log, order=(2, 1, 0))
results_AR = model.fit(disp=-1,transparams=False)
plt.plot(year_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
#plt.title('RSS: %.4f' % sum((results_AR.fittedvalues - year_log_diff) ** 2))
plt.show()

model = ARIMA(year_log, order=(0, 1, 2))
results_MA = model.fit(disp=-1,transparams=False)
plt.plot(year_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-year_log_diff)**2))
plt.show()

model = ARIMA(year_log, order=(0, 1, 0))
results_ARIMA = model.fit(disp=-1,transparams=False)
plt.plot(year_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-year_log_diff)**2))
plt.show()

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(year_log.iloc[0], index=year_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
print(predictions_ARIMA_log.tail())

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(year,label='Original')
plt.plot(predictions_ARIMA,label='Prediction')
#plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-year)**2)/len(year)))
plt.show()
plt.legend(loc='upper left')
print(predictions_ARIMA)

results_ARIMA.plot_predict(1,300)
plt.show()