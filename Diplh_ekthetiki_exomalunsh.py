import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

year = pd.read_csv('Bima2.csv',skiprows=0,nrows=80, index_col=[0], sep=';')
#year.drop(2020, inplace=True)
print(year.tail())
optimal_alpha = None
optimal_gamma = None
best_mse=None
db = year.iloc[:,:].values.astype('float32')
mean_results_for_all_possible_alpha_gamma_values = np.zeros((9,9))
for gamma in range(0,9):
    for alpha in range(0,9):
        pt = db[0][0]
        bt = db [1][0] - db[0][0]
        mean_for_alpha_gamma = np.zeros(len(db))
        mean_for_alpha_gamma[0] = np.power(db[0][0] - pt, 2)
        for i in range(1,len(db)):
            temp_pt = ((alpha +1)*0.1)*db[i][0]+(1-((alpha + 1)*0.1))*(pt + bt)
            bt = ((gamma +1)* 0.1)*(temp_pt - pt) + (1 -((gamma + 1)* 0.1))*bt
            pt = temp_pt
            mean_for_alpha_gamma[i] = np.power(db[i][0]-pt,2)
            mean_results_for_all_possible_alpha_gamma_values[gamma][alpha] = np.mean(mean_for_alpha_gamma)
            optimal_gamma,optimal_alpha = np.unravel_index(np.argmin(mean_results_for_all_possible_alpha_gamma_values),np.shape(mean_results_for_all_possible_alpha_gamma_values))
optimal_alpha = (optimal_alpha +1)*0.1
optimal_gamma = (optimal_gamma +1)*0.1
best_mse = np.min(mean_results_for_all_possible_alpha_gamma_values)
print("Best MSE = %s"%best_mse)
print("Optimal alpha = %s"%optimal_alpha)
print("Optimal gamma = %s"%optimal_gamma)

pt = db[0][0]
bt = db[1][0] - db[0][0]
for i in range(1,len(db)):
    tempt_pt = optimal_alpha*db[i][0]+(1-optimal_alpha)*(pt + bt)
    bt = optimal_gamma * (temp_pt - pt) + (1 - optimal_gamma)*bt
    pt = temp_pt
print("P_t = %s" % pt)
print("b_t= %s" % bt)
print("Next obesrevation = %s"%(pt+(1*bt)))
forecast = np.zeros(len(db)+1)
pt = db[0][0]
bt = db[1][0]- db[0][0]
forecast[0] = pt
for i in range(1,len(db)):
    temp_pt = optimal_alpha * db[i][0] + (1-optimal_alpha) * (pt + bt)
    bt = optimal_gamma *(tempt_pt - pt) + (1 - optimal_gamma ) * bt
    pt = temp_pt
    forecast[i]=pt
forecast[-1] = pt+(1*bt)
plt.plot(db[:,0],label = 'real data')
plt.plot(forecast, label='forecast')
plt.legend()
plt.show()

'''
print(year.describe())
year_mean = year.rolling(window=10).mean()
plt.plot(year_mean, label="Rolling mean trend")
year_base = pd.concat([year,year.shift(1)],axis=1)
year_base.columns = ['Actual_Publications','Forecast_Publications']
year_base.dropna(inplace=True)
print(year_base)
year_error = mean_squared_error(year_base.Actual_Publications,year_base.Forecast_Publications)
print(np.sqrt(year_error))'''







