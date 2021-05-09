import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
import statsmodels.api as sm
from numpy import linalg as LA
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA,ARIMAResults,ARMA,ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ADF test function
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
       print('\t%s: %.3f' % (key, value))
# ACF function
def auto_cor_cal(y, t=0):
    y_diff = y - np.mean(y)
    if t:
        return np.sum(y_diff[t:] * y_diff[:-t]) \
               / np.sum(np.square(y_diff))
    else:
        return 1

# GPAC function
def gpac_cal (r, row, col):
    def calc_phi(r, j, k):

        b = [[0 for _ in range(k)] for _ in range(k)]
        for m in range(k):
            for n in range(k):
                b[m][n] = r[abs(j + m - n)]

        a = [[e for e in row] for row in b]  # b.copy()
        for i in range(k):
            a[i][-1] = r[j + 1 + i]

        # to numpy
        a_np = np.array(a)
        b_np = np.array(b)
        return np.linalg.det(a_np) / np.linalg.det(b_np)

    retval = [[0 for _ in range(1, col)] for _ in range(row)] # all zero

    for j in range(row):
        for k in range(1, col):
            retval[j][k - 1] = calc_phi(r, j, k)

    result = np.array(retval)

    df = pd.DataFrame(result, index=np.arange(row), columns=np.arange(1, col)) # table
    print(df)
    return df

# import dataset
data = pd.read_excel('AirQualityUCI.xlsx',parse_dates=[["Date", "Time"]],index_col=[0])

# Plot of the dependent variable vs time
y = data[["CO(GT)"]]
plt.figure()#fig, ax = plt.subplots()
plt.plot(y)
plt.title("CO(GT) vs Time")
plt.ylabel("CO mg/m^3")
plt.xlabel("Time")
plt.legend(["CO(GT)"])
plt.show()

# ACF/PACF pf the dependent variables
lags = 100
acf = sm.tsa.stattools.acf(y, nlags = lags)
pacf = sm.tsa.stattools.pacf(y, nlags = lags)

plt.figure(figsize=(8,8))
plt.subplot(211)
plot_acf(y,ax=plt.gca(),lags =lags)
plt.ylabel("Magnitude")
plt.subplot(212)
plot_pacf(y,ax=plt.gca(),lags =lags)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.show()

# Correlation Matrix with seaborn heatmap
cor = data.corr()

fig, ax = plt.subplots(figsize=(10,10))
ax =sns.heatmap(cor, vmin = -1, vmax = 1, annot=True,
                center=0,cmap=sns.diverging_palette(20,220,n=200), square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right')
plt.title("Correlation Matrix of Air Quality Dataset")
plt.show()

# Split the dataset into train set 80% and test set 20%
y_train, y_test = train_test_split(y, shuffle= False, test_size=0.2)

# Stationarity check ----------------------------------------------- Stationarity check
ADF_Cal(y)

# Plot mean and variance versus time
CO_mean = []
CO_var = []

for i in range(1, len(data)+1):
    CO_mean.append(data.head(i)["CO(GT)"].mean())
    CO_var.append(data.head(i)["CO(GT)"].var())

data["COmean"] = CO_mean
data["COvar"] = CO_var

plt.figure(figsize=(8,8))
plt.subplot(211)
plt.plot(data.COmean)
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Rolling Mean and Rolling Variance of Orginal data')
plt.legend(["Rolling Mean"])

plt.subplot(212)
plt.plot(data.COvar)
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(["Rolling Variance"])
plt.show()

#================================
# y with first differencing
#================================
CO_1 = y.diff(1)
CO_1 = CO_1[1:]
data['CO_1'] = CO_1

CO_1_mean = []
CO_1_var = []

for i in range(1, len(data)+1):
    CO_1_mean.append(data.head(i).CO_1.mean())
    CO_1_var.append(data.head(i).CO_1.var())

data["CO_1_mean"] = CO_1_mean
data["CO_1_var"] = CO_1_var

plt.figure(figsize=(8,8))
plt.subplot(211)
plt.plot(data.CO_1_mean)
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Plot of Rolling mean and Rolling Variance with Differencing ')
plt.legend(["Rolling Mean"])

plt.subplot(212)
plt.plot(data.CO_1_var)
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(["Rolling Variance"])
plt.show()

ADF_Cal(CO_1)

# ACF/PACF pf the differenced y target
lags = 100

acf = sm.tsa.stattools.acf(CO_1, nlags = lags)
pacf = sm.tsa.stattools.pacf(CO_1, nlags = lags)

plt.figure(figsize=(8,8))
plt.subplot(211)
plot_acf(CO_1,ax=plt.gca(),lags =lags)
plt.ylabel("Magnitude")
plt.subplot(212)
plot_pacf(CO_1,ax=plt.gca(),lags =lags)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.show()

#================================
# y with second differencing
#================================
y = data[["CO_1"]]
CO_1_24 = y.diff(24)
CO_1_24 = CO_1_24[24:]
CO_1_24 = CO_1_24[1:]
data['CO_1_24'] = CO_1_24

CO_1_24_mean = []
CO_1_24_var = []

for i in range(1, len(data)+1):
    CO_1_24_mean.append(data.head(i).CO_1.mean())
    CO_1_24_var.append(data.head(i).CO_1.var())

data["CO_1_24_mean"] = CO_1_24_mean
data["CO_1_24_var"] = CO_1_24_var

plt.figure(figsize=(8,8))
plt.subplot(211)
plt.plot(data.CO_1_24_mean)
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Plot of Rolling mean and Rolling Variance with Differencing ')
plt.legend(["Rolling Mean"])

plt.subplot(212)
plt.plot(data.CO_1_24_var)
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(["Rolling Variance"])
plt.show()

ADF_Cal(CO_1_24)

# ACF/PACF of the differenced y target

acf = sm.tsa.stattools.acf(CO_1_24, nlags = lags)
pacf = sm.tsa.stattools.pacf(CO_1_24, nlags = lags)

plt.figure(figsize=(8,8))
plt.subplot(211)
plot_acf(CO_1_24,ax=plt.gca(),lags =lags)
plt.ylabel("Magnitude")
plt.subplot(212)
plot_pacf(CO_1_24,ax=plt.gca(),lags =lags)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.show()

# Time Series Decomposition ------------------------------------------ Decomposition
CO = data["CO(GT)"]
CO = pd.Series(np.array(data["CO(GT)"]),
                 index = pd.date_range('2004-03-10 18:00:00', periods =len(CO), freq='h'),
                 name='Air Quality (CO) Decomposition Plot' )
STL = STL(CO)
res = STL.fit()

T = res.trend
S = res.seasonal
R = res.resid

fig = res.plot()
fig.set_size_inches(8, 6)
plt.show()

#Calculate seasonally adjusted data and plot it vs the original
adjusted_seasonal = CO - S
detrend = CO - T

plt.figure(figsize=(8,6))
plt.plot(CO, label= 'Original set')
plt.plot(detrend, label='Detrended')
plt.title("Original vs Detrended Data")
plt.xlabel("Date")
plt.ylabel("CO mg/m^3")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(CO, label= 'Original set')
plt.plot(adjusted_seasonal, label='Adjusted seasonal')
plt.title("Original vs Seasonally Adjusted")
plt.xlabel("Date")
plt.ylabel("CO mg/m^3")
plt.legend()
plt.show()

# # Find the strength of trend and strength of seasonality
R = np.array(R)
S = np.array(S)
T = np.array(T)
Ft = np.max([0,1 - np.var(R)/np.var(T+R)])
Fs = np.max([0,1 - np.var(R)/np.var(S+R)])
print("The strength of trend for this dataset is =", Ft)
print("The strength of seasonality for this dataset is =", Fs)

# Feature selection -------------------------------------------------Feature selection
X = data[["PT08.S1(CO)","C6H6(GT)", "PT08.S2(NMHC)","NOx(GT)","PT08.S3(NOx)",
          "NO2(GT)","PT08.S4(NO2)","PT08.S5(O3)","T","RH","AH"]]
X = sm.add_constant(X)
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())
X = X_train.values
print("The condition number for X original is =", LA.cond(X))
H = np.matmul(X.T,X)
s,d,v = np.linalg.svd(H)
print("SingularValues d original is =", d)

#================
# remove constant 0.949
#================
X = data[["PT08.S1(CO)","C6H6(GT)", "PT08.S2(NMHC)","NOx(GT)","PT08.S3(NOx)",
          "NO2(GT)","PT08.S4(NO2)","PT08.S5(O3)","T","RH","AH"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())

#================
# remove AH 0.949
#================
X = data[["PT08.S1(CO)","C6H6(GT)", "PT08.S2(NMHC)","NOx(GT)","PT08.S3(NOx)",
          "NO2(GT)","PT08.S4(NO2)","PT08.S5(O3)","T","RH"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())

#================
# remove C6H6(GT) 0.946
#================
X = data[["PT08.S1(CO)","PT08.S2(NMHC)","NOx(GT)","PT08.S3(NOx)",
          "NO2(GT)","PT08.S4(NO2)","PT08.S5(O3)","T","RH"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())

#=====================
# remove PT08.S2(NMHC) 0.946
#=====================
X = data[["PT08.S1(CO)","NOx(GT)","PT08.S3(NOx)",
          "NO2(GT)","PT08.S4(NO2)","PT08.S5(O3)","T","RH"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())

#=====================
# remove T 0.938
#=====================
X = data[["PT08.S1(CO)","NOx(GT)","PT08.S3(NOx)",
          "NO2(GT)","PT08.S4(NO2)","PT08.S5(O3)","RH"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())

#=====================
# remove PT08.S5(O3) 0.938
#=====================
X = data[["PT08.S1(CO)","NOx(GT)","PT08.S3(NOx)",
          "NO2(GT)","PT08.S4(NO2)","RH"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())

#=====================
# remove RH 0.936
#=====================
X = data[["PT08.S1(CO)","NOx(GT)","PT08.S3(NOx)",
          "NO2(GT)","PT08.S4(NO2)"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())

#=====================
# remove NO2(GT) 0.934
#=====================
X = data[["PT08.S1(CO)","NOx(GT)","PT08.S3(NOx)","PT08.S4(NO2)"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())

#=========================
# remove PT08.S1(CO) 0.933
#=========================
X = data[[ "NOx(GT)","PT08.S3(NOx)","PT08.S4(NO2)"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
print(model.summary())

X = X_train.values
print("The condition number for X after feature elimination is =", LA.cond(X))
H = np.matmul(X.T,X)
s,d,v = np.linalg.svd(H)
print("SingularValues d after feature elimination is =", d)

# Develop the multiple linear regression model------------------------------Muti Linear
Muli_linear_FE = np.ndarray.flatten(Y_test.values) - np.array(test_predictions) # residuals of forecast
MSE_Muli_linear = round(np.square(Muli_linear_FE).mean(),2)

plt.figure()
plt.plot(Y_train)
plt.plot(Y_test)
plt.plot(train_predictions)
plt.plot(test_predictions)
plt.title("Multiple Linger Regression Model with MSE = {}".format(MSE_Muli_linear))
plt.xlabel("Time")
plt.ylabel("CO mg/m^3")
plt.legend(['Train Dataset','Test Dataest','Prediction','Forecast'])
plt.show()

#t-test
print("The p-value of t-test is: ",model.pvalues)
#f-test
print("The p-value of f-test is: ",model.f_pvalue)

# AIC,BIC,RMSE,R-squared and Adjusted R-squared
Muli_linear_PE = np.ndarray.flatten(Y_train.values) - np.array(train_predictions) # residuals of prediction
Muli_linear_FE = np.ndarray.flatten(Y_test.values) - np.array(test_predictions) # residuals of forecast
MSE_Muli_linear = round(np.square(Muli_linear_FE).mean(),2)

print("The AIC value of the model is :",model.aic)
print("The BIC value of the model is :",model.bic)
print("The MSE value of residual is :",MSE_Muli_linear)
print("The RMSE value of residual is :",np.sqrt(MSE_Muli_linear))
print("The R-squared value of the model is: ",model.rsquared)
print("The Adjusted R-squared value of the model is: ",model.rsquared_adj)

plt.figure()
plt.plot(Y_train.index,Muli_linear_PE)
plt.plot(Y_test.index,Muli_linear_FE)
plt.title("Multiple Linger Regression Model Prediction Error vs Forecast Error")
plt.xlabel("Time")
plt.ylabel("CO mg/m^3")
plt.legend(['Prediction Error','Forecast Error'])
plt.show()

# ACF of residuals
lags = 30
ry = []
for t in range(lags):
    ry.append(auto_cor_cal(Muli_linear_FE, t))
Ry_Muli_linear_FE = ry[::-1][:-1] + ry
x = np.linspace(-lags, lags, 2*lags-1)
m = 1.96/np.sqrt(len(Muli_linear_FE))
plt.stem(x, Ry_Muli_linear_FE)
plt.axhspan(-m,m,alpha = .1, color = 'black')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("ACF of Residuals - Multiple Linger Regression Model")
plt.show()

# Q value
Q_Muli_linear_FE = len(Muli_linear_FE)*sum(np.square(Ry_Muli_linear_FE[lags:]))
print("The Q-value is :",Q_Muli_linear_FE)

# Variance and mean of the residuals
Var_Muli_linear_PE = Muli_linear_PE.var()
Mean_Muli_linear_PE = Muli_linear_PE.mean()

Var_Muli_linear_FE = Muli_linear_FE.var()
Mean_Muli_linear_FE = Muli_linear_FE.mean()

print("The variance of the predict error is :",Var_Muli_linear_PE)
print("The mean of the predict error is :",Mean_Muli_linear_PE)

print("The variance of the forecast error is :",Var_Muli_linear_FE)
print("The mean of the forecast error is :",Mean_Muli_linear_FE)

# 14 Base-models
# -------------Average Method----------------------------------------------------
lags = 30
ytrain = y_train.values
ytest = y_test.values
hstep = []
for i in range(len(ytest)):
    hstep.append(np.mean(ytrain))
hstep_np = np.array(hstep)                              #forecast
Average_FE = ytest.flatten()-hstep_np                   #forecast error
MSE_Average = round(np.square(Average_FE).mean(),2)     #forecast MSE

RMSE_Average = np.sqrt(MSE_Average)
Average_FE_mean = Average_FE.mean()
Average_FE_var = Average_FE.var()

print("The mean of error of Average Model is :", Average_FE_mean)
print("The variance of Average of Naive Model is :", Average_FE_var)
print("The MSE of Average Model is :", MSE_Average)
print("The RMSE of Average Model is :",RMSE_Average)

plt.plot(y_train,label= "Train Data")
plt.plot(y_test,label= "Test Data")
plt.plot(y_test.index,hstep_np,label= "Average Method prediction")
plt.legend(loc='upper left')
plt.title("Average Method with MSE = {}".format(MSE_Average))
plt.xlabel("Date")
plt.ylabel("CO mg/m^3")
plt.show()

ry = []
for t in range(lags):
    ry.append(auto_cor_cal(Average_FE, t))
Average_Ry = ry[::-1][:-1] + ry                          #forecast error ACF

Average_Q = len(Average_FE)*np.sum(np.square(ry[1:]))    # Q
print("The Q value of Average Model is:",Average_Q)

x = np.linspace(-lags, lags, 2*lags-1)
m = 1.96/np.sqrt(len(Average_FE))
plt.stem(x, Average_Ry)
plt.axhspan(-m,m,alpha = .1, color = 'black')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("ACF of Residuals - Average Model")
plt.show()

#-----------Naive Method------------------------------------------------------------
hstepNaive = []
for i in range(len(ytest)):
    hstepNaive.append(ytrain[-1])
Naive_hstep_np = np.array(hstepNaive)                  #forecast
Naive_FE = ytest.flatten()-Naive_hstep_np.flatten()            #forecast error
MSE_Naive = round(np.square(Naive_FE).mean(),2)     #forecast MSE

RMSE_Naive = np.sqrt(MSE_Naive)
Naive_FE_mean = Naive_FE.mean()
Naive_FE_var = Naive_FE.var()

print("The mean of error of Naive Model is :", Naive_FE_mean)
print("The variance of error of Naive Model is :", Naive_FE_var)
print("The MSE of Naive Model is :", MSE_Naive)
print("The RMSE of Naive Model is :",RMSE_Naive)

plt.plot(y_train,label= "Train Data")
plt.plot(y_test,label= "Test Data")
plt.plot(y_test.index,Naive_hstep_np,label= "Naive Method prediction")
plt.legend(loc='upper left')
plt.title("Naive Method with MSE = {}".format(MSE_Naive))
plt.xlabel("Date")
plt.ylabel("CO mg/m^3")
plt.show()

ry = []
for t in range(lags):
    ry.append(auto_cor_cal(Naive_FE, t))
Naive_Ry = ry[::-1][:-1] + ry                          #forecast error ACF

Naive_Q = len(Naive_FE)*np.sum(np.square(ry[1:]))    # Q
print("The Q value of Naive Model is:",Naive_Q)

x = np.linspace(-lags, lags, 2*lags-1)
m = 1.96/np.sqrt(len(Naive_FE))
plt.stem(x, Naive_Ry)
plt.axhspan(-m,m,alpha = .1, color = 'black')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("ACF of Residuals - Naive Model")
plt.show()

#-----------Drift Method------------------------------------------------------------
hstepDrift = []
for i in range(len(ytrain)+1,len(data)+1):
    slope = (ytrain[-1] - ytrain[0]) / (len(ytrain) - 1)
    hstepDrift.append(slope * i + ytrain[0] - slope )
Drift_hstep_np = np.array(hstepDrift)                   #forecast
Drift_FE = ytest.flatten()-Drift_hstep_np.flatten()             #forecast error
MSE_Drift = round(np.square(Drift_FE).mean(),2)     #forecast MSE
RMSE_Drift = np.sqrt(MSE_Drift)
Drift_FE_mean = Drift_FE.mean()
Drift_FE_var = Drift_FE.var()

print("The mean of error of Drift Model is :", Drift_FE_mean)
print("The variance of error of Drift Model is :", Drift_FE_var)
print("The MSE of Drift Model is :", MSE_Drift)
print("The RMSE of Drift Model is :",RMSE_Drift)

plt.plot(y_train,label= "Train Data")
plt.plot(y_test,label= "Test Data")
plt.plot(y_test.index,Drift_hstep_np,label= "Drift Method prediction")
plt.legend(loc='upper left')
plt.title("Drift Method with MSE = {}".format(MSE_Drift))
plt.xlabel("Date")
plt.ylabel("CO mg/m^3")
plt.show()

ry = []
for t in range(lags):
    ry.append(auto_cor_cal(Drift_FE, t))
Drift_Ry = ry[::-1][:-1] + ry                          #forecast error ACF

Drift_Q = len(Drift_FE)*np.sum(np.square(ry[1:]))    # Q
print("The Q value of Drift Model is:",Drift_Q)

x = np.linspace(-lags, lags, 2*lags-1)
m = 1.96/np.sqrt(len(Drift_FE))
plt.stem(x, Drift_Ry)
plt.axhspan(-m,m,alpha = .1, color = 'black')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("ACF of Residuals - Drift Model")
plt.show()

#-----------SES Method------------------------------------------------------------
yhat = []
for i in range(len(ytrain)-1):
    yhat.append(ytrain[i])
    yhat[i] = 0.5*ytrain[i]+0.5*yhat[i-1]

hstepSES = []
for i in range(len(ytest)):
    hstepSES.append(yhat[-1])
SES_hstep_np = np.array(hstepSES)                          #forecast
SES_FE = ytest.flatten()- SES_hstep_np.flatten()                #forecast error
MSE_SES = round(np.square(SES_FE).mean(),2)           #forecast MSE
RMSE_SES = np.sqrt(MSE_SES)
SES_FE_mean = SES_FE.mean()
SES_FE_var = SES_FE.var()

print("The mean of error of SES Model is :", SES_FE_mean)
print("The variance of error of SES Model is :", SES_FE_var)
print("The MSE of SES Model is :", MSE_SES)
print("The RMSE of SES Model is :",RMSE_SES)

plt.plot(y_train,label= "Train Data")
plt.plot(y_test,label= "Test Data")
plt.plot(y_test.index, SES_hstep_np,label= "SES Method prediction")
plt.legend(loc='upper left')
plt.title("SES Method with MSE = {}".format(MSE_SES))
plt.xlabel("Date")
plt.ylabel("CO mg/m^3")
plt.show()

ry = []
for t in range(lags):
    ry.append(auto_cor_cal(SES_FE, t))
SES_Ry = ry[::-1][:-1] + ry                          #forecast error ACF

SES_Q = len(SES_FE)*np.sum(np.square(ry[1:]))    # Q
print("The Q value of SES Model is:",SES_Q)

x = np.linspace(-lags, lags, 2*lags-1)
m = 1.96/np.sqrt(len(SES_FE))
plt.stem(x, SES_Ry)
plt.axhspan(-m,m,alpha = .1, color = 'black')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("ACF of Residuals - SES Model")
plt.show()

# ---------------Holt-Winters method-------------------------------------------------
holtt = ets.ExponentialSmoothing(y_train,trend='add',damped_trend=False,seasonal='add',seasonal_periods=24).fit()
holtf = holtt.forecast(steps=len(y_test))                                     # please try seasonal_periods= 12
HoltWinter_hstep_np =np.array(holtf)                                       #forecast
HoltWinter_FE = np.array(y_test.values).flatten() - HoltWinter_hstep_np    #forecast error
MSE_HoltWinter = round(np.square(HoltWinter_FE).mean(),2)                  #forecast MSE

RMSE_HoltWinter = np.sqrt(MSE_HoltWinter)
HoltWinter_FE_mean = HoltWinter_FE.mean()
HoltWinter_FE_var = HoltWinter_FE.var()

print("The mean of error of HoltWinter Model is :", HoltWinter_FE_mean)
print("The variance of error of HoltWinter Model is :", HoltWinter_FE_var)
print("The MSE of HoltWinter Model is :", MSE_HoltWinter)
print("The RMSE of HoltWinter Model is :",RMSE_HoltWinter)

lags = 30
ry = []
for t in range(lags):
    ry.append(auto_cor_cal(HoltWinter_FE, t))
HoltWinter_Ry = ry[::-1][:-1] + ry                          #forecast error ACF

HoltWinter_Q = len(HoltWinter_FE)*np.sum(np.square(ry[1:]))    # Q
print("The Q value of HoltWinter Model is:",HoltWinter_Q)

plt.plot(y_train,label= "Train Data")
plt.plot(y_test,label= "Test Data")
plt.plot(y_test.index,HoltWinter_hstep_np,label= "Holt Winter Method prediction")
plt.legend(loc='upper left')
plt.title("Holt Winter Method with MSE = {}".format(MSE_HoltWinter))
plt.xlabel("Date")
plt.ylabel("CO mg/m^3")
plt.show()

x = np.linspace(-lags, lags, 2*lags-1)
m = 1.96/np.sqrt(len(HoltWinter_FE))
plt.stem(x, HoltWinter_Ry)
plt.axhspan(-m,m,alpha = .1, color = 'black')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("ACF of Residuals - HoltWinter Model")
plt.show()

# 11, ARMA and ARIMA and SARIMA model order determination--------------------------ARMA
y = CO_1_24.values # differenced data

ry = []
for t in range(lags):
    ry.append(auto_cor_cal(y, t))

# Display GPAC table for k=8, j=8, use seaborn
table = gpac_cal(ry,10,10)
fig, ax = plt.subplots(figsize=(10,10))
ax =sns.heatmap(table, vmin = -1, vmax = 1, annot=True,
                center=0,cmap=sns.diverging_palette(20,220,n=200), square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right')
plt.title("Generalized Partial Autocorrelation(GPAC) Table")
plt.show()


# 12 Estimate ARMA model parameters using Levenberg Marquardt algorithm
y = data["CO(GT)"].values # orignal data
y_train, y_test = train_test_split(y, shuffle= False, test_size=0.2)
# na = 4
# nb = 4
na = 1
nb = 6

model = sm.tsa.ARMA(y_train,(na,nb)).fit(trend='nc',disp=0)

print(model.summary())

for i in range(na):
    print("The AR estimated coefficient a{}".format(i), "is:", model.params[i])
for i in range(nb):
    print("The MA estimated coefficient b{}".format(i), "is:", model.params[i+na])
for i in range(na):
    print("The confidence interval for estimated coefficient a{}".format(i), "is:", model.conf_int()[i])
for i in range(nb):
    print("The confidence interval for estimated coefficient b{}".format(i), "is:", model.conf_int()[i+na])

print("The standard deviation of the parameter estimates is: ",model.params.std())

# 13 Diagnostic Analysis

# Chi-sqyare test
model_hat = model.predict(start=0, end=len(y_train)-1)
model_h_hat = model.predict(start=len(y_train)-1, end=len(data))

plt.figure()
plt.plot(y, label = "True data")
plt.plot(model_hat, label = "Predicted data")
plt.plot(np.arange(7484-1,9357),model_h_hat, label = "Forecast data")
plt.xlabel("Samples")
plt.ylabel("CO mg/m^3")
plt.legend()
plt.title(" ARMA (1,6) Prediction and Forecast ")
plt.show()

lags = 30
ARMA_PE = y_train-model_hat
ARMA_FE = y_test-model_h_hat[2:]   # forecast error
MSE_ARMA = round(np.square(ARMA_FE).mean(),2)
print("The MSE of ARMA (1,6) is : ",MSE_ARMA)
print("The variance of prediction error is : ",ARMA_PE.var())
print("The variance of forecast error is :",ARMA_FE.var())
print("The variance of prediction vs forecast error is : ",ARMA_PE.var() /ARMA_FE.var())
ry = []
for t in range(lags):
    ry.append(auto_cor_cal(ARMA_PE, t))
ARMA_Ry = ry[::-1][:-1] + ry                          #prediction error ACF

x = np.linspace(-lags, lags, 2*lags-1)
m = 1.96/np.sqrt(len(ARMA_PE))
plt.stem(x, ARMA_Ry)
plt.axhspan(-m,m,alpha = .1, color = 'black')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("ACF of Residuals - ARMA(1,6)")
plt.show()

Q = len(y_train)*np.sum(np.square(ry[1:]))
print("The Q value of ARMA (1,6) is : ",Q)
DOF = lags - na - nb
alfa = 0.01
chi_critical = chi2.ppf(1-alfa, DOF)
if Q< chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")

print("The corvariance of estimated parameters is : ",model.cov_params())

# # SARIMA model ------------------------------------------------------- SARIMA model
# y = CO_1_24.values # differenced data
# # Finding best orders
# Sarimax_model = auto_arima(data["CO(GT)"],
#                        start_P=1,
#                        start_q=1,
#                        max_p=3,
#                        max_q=3,
#                        m=24,
#                        seasonal=True,
#                        d=None,
#                        D=1,
#                        trace=True,
#                        error_action='ignore',
#                        suppress_warnings=True,
#                        stepwise=True)
# Sarimax_model.summary()

# ARIMA(1,0,0)(2,1,0)[24]
y = data[["CO(GT)"]]
y_train, y_test = train_test_split(y, shuffle= False, test_size=0.2)
model = SARIMAX(y_train,order=(1, 0, 0),
              seasonal_order=(2, 1, 0, 24),
              enforce_stationarity=False,
              enforce_invertibility=False)
results = model.fit()

forecast = results.predict(start = len(y_train),
                           end=len(data),
                           typ='levels')

plt.figure()
plt.plot(y_train, label = "Train")
plt.plot(y_test, label = "Train")
plt.plot(y_test.index,forecast[:-1], label = "Forecast data")
plt.xlabel("Time")
plt.ylabel("CO mg/m^3")
plt.legend()
plt.title(" SARIMAX (1,0,0)(2,1,0)(24) and Forecast ")
plt.show()

SARIMAX_FE = y_test.values.flatten() - forecast[:-1]     # forecast error
MSE_SARIMAX = round(np.square(SARIMAX_FE).mean(),2)
print("The MSE of SARIMAX (1,0,0)(2,1,0)(24) is : ",MSE_SARIMAX)
print("The variance of forecast error is : ",SARIMAX_FE.var())
print("The mean of forecast error is : ",SARIMAX_FE.mean())

lags = 30
ry = []
for t in range(lags):
    ry.append(auto_cor_cal(SARIMAX_FE, t))
SARIMAX_Ry = ry[::-1][:-1] + ry                          #forecast error ACF

x = np.linspace(-lags, lags, 2*lags-1)
m = 1.96/np.sqrt(len(SARIMAX_FE))
plt.stem(x, SARIMAX_Ry)
plt.axhspan(-m,m,alpha = .1, color = 'black')
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.title("ACF of Residuals - SARIMAX (1,0,0)(2,1,0)(24)")
plt.show()

Q = len(y_test)*np.sum(np.square(ry[1:]))
print("The Q value of SARIMAX (1,0,0)(2,1,0)(24) is : ",Q)

# H-step Ahead Prediction
X = data[[ "NOx(GT)","PT08.S3(NOx)","PT08.S4(NO2)"]]
Y = data[["CO(GT)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
model = sm.OLS(Y_train, X_train).fit()
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

plt.figure(figsize=(12,6))
plt.plot(Y_test)
plt.plot(test_predictions)
plt.title("Multiple Linger Regression Model - CO(GT) Forecost")
plt.xlabel("Time")
plt.ylabel("CO mg/m^3")
plt.legend(['Test Dataest','H-step Prediction'])
plt.show()

