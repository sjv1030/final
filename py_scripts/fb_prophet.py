#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Building a MonteCarlo model with FBProphet underneath. The variables for the scenarios will be those identified by Silverio. The predicted values will be generated from an ARIMA model with drift. 

Variables: trade weighted dollar
Prospective: quadratic interpolated rig and production data, 10-day volatility

"""
import fbprophet
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
import datetime
import numpy as np
import quandl
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from sklearn import preprocessing
#local mongo
#client=MongoClient('localhost',81)
#db=client['commodities']

#mlab
client=MongoClient('mongodb://team_nyc:persyy@ds229450.mlab.com:29450/commodities',serverSelectionTimeoutMS=3000)
db=client.commodities

#query ng data from mongo
values=db['values']
test=values.find({'ng_val':{'$exists':True}},{'_id':0,'day_timestamp':1,'ng_val':1})

ng_daily_df=pd.DataFrame(list(test))
ng_daily_df.tail()
#sort the dataframe
ng_daily_df.sort_values('day_timestamp',axis=0,ascending=True,inplace=True)

'''Variable creation and analysis - ARMA, regression'''
#calculate the volatility regressors; type: close-to-close volatility
#method 1
temp=ng_daily_df['ng_val'].pct_change().values+1
#method 2
#shifted=ng_daily_df['ng_val'].shift(1)
#ratio=ng_daily_df['ng_val']/shifted
#log conversion... would get an approximation to this just by calling percent_change() pandas method
ng_daily_df['log_vec']=np.log(temp)
ng_daily_df['roll_10']=ng_daily_df['log_vec'].rolling(9).mean()
test=(ng_daily_df['log_vec']-ng_daily_df['roll_10'])**2
gy=test.rolling(9).sum()
#close-to-close volatility array
yu=np.sqrt((254.5/8)*gy)
#on any given rolling 20-day period, stratify/bin the observations in 1.09488-point ranges
np.amax(yu)
yu[0:30]
good=yu[~np.isnan(yu)]
def rolling_apply(ser,window):
    bins=np.array([0.0,1.09,2.18,3.27,5.5])
    i=np.arange(ser.shape[0]+1-window)
    #results=np.zeros(ser.shape[0])
    results=[]
    for g in i:
        #results[g+window-1]=np.digitize(ser[g:window+g],bins)
        results.append(np.digitize(ser[g:window+g],bins))
    return(results)
    
test=rolling_apply(good,30)

def dic_counters(lista):
    master=[]
    for i in lista:
        #counter={1:0,2:0,3:0,4:0,5:0}    
        master.append(np.bincount(i))
    return(master)
        
gh=dic_counters(test)
gh[-20:]

#loop through each element in the list and augment a dictionary of counters for each bin

#volatility is only positive an is not an ARMA process, unless it is scaled/standardized... nonetheless a regression can be run on volatility to determine some levels for scenario testing, and perhaps fit a probability distribution
plt.close()
plt.plot(ng_daily_df.index.values[-400:],yu[-400:])
plt.title('rolling 9-day volatility')
plt.show()

plot_pacf(yu[-200:])
ad=adfuller(yu[-900:])
ad
plot_acf(yu[-900:],lags=20)

#scaling
yu_good=yu[~np.isnan(yu).values].values
prep=yu_good.reshape(-1,1)
yu_robust=preprocessing.robust_scale(prep)
np.amax(yu_robust)
#plot result
plt.close()
plt.plot(ng_daily_df.index.values[-5360:],yu_robust)
plt.title('rolling 9-day volatility')
plt.show()

# Variable #2: trade-weighted dollar
twd_m = quandl.get('FRED/DTWEXM') 
#twd detrending... with a quadratic regression:
#inputs will be a sequence representation of the time series date values

#the statsmodels method... preferable because it provides summary stats on the model unlike sklearn's regression function
X = np.array([i for i in range(0, twd_m.shape[0])])
x_sqrd = X**2
#a cubic model
x_cubd = X**3
test=pd.Series(X)
test1=pd.Series(x_sqrd)
test2=pd.Series(x_cubd)
inputs=pd.concat([test,test1,test2],axis=1)
y = twd_m.values
x = sm.add_constant(inputs)
x_train = x.iloc[:-100,]
x_test = x.iloc[-100:,]
y_train = y[:-100]
y_test = y[-100:]
# create model; fit onto training data
ols_modelmm = sm.OLS(y_train,x_train) 
ols_fitmm = ols_modelmm.fit()

# Print OLS summary
print(ols_fitmm.summary())
#prediction
predmm = ols_fitmm.predict(x_test)

#plot residuals
plt.close
residmm = ols_fitmm.resid
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sm.qqplot(residmm, fit=True, line='45',ax=ax)
ax.set_title('Q-Q Plot of Residuals')
plt.show()

#plot the fit
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(y_train, label='Actual')
ax.plot(ols_fitmm.fittedvalues, label='Fitted', marker='o', linestyle='dashed')
ax.legend()
ax.set_title('Model Fit on M/M%')
plt.show()

# Plot Actual and forecasted values of testing set... appears like a straight line but it's part of a three-order model
plt.close()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(twd_m.index.values[-100:],y_test, label='Actual', linestyle='dashdot')
ax.plot(twd_m.index.values[-100:],predmm, label='Forecast', lw=3)
ax.legend()
ax.set_title('Forecasted Values')
plt.show()


#remove the trend from the series.. over the span of the entire time series... so we will fit over the entire series
x.shape #11452
y.shape #11452
ols_modelmm_large = sm.OLS(y,x) #11452
ols_fitmm_large = ols_modelmm_large.fit()
fits_large=ols_fitmm_large.fittedvalues #11452
residmm = ols_fitmm_large.resid #11452
resids=y-fits_large.values.reshape(len(fits_large),1)
#compare... yes, equivalent
resids[0:5]
residmm[0:5]

#fit an ARMA model on the residuals... need to check for unit root with Augmented Dickey Fuller test
#which trend removal/stationary device works better? apply a Dickey-Fuller or ADF test
type(resids)
adfuller(residmm)
#check autocorrelation at this outset
plt.close()
fig=plt.figure()
ax=fig.add_subplot(111)
plot_acf(resids,ax=ax,lags=20)
plt.show()

plt.close()
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('A suggestion of strong mean reversion')
plot_pacf(resids,ax=ax,lags=20)
plt.show()

#construct three models

'''
BIC = np.zeros(7)
for p in range(7):
    mod = ARMA(model_diff2, order=(p,0)) #first placeholder is for AR parameter and the second is for MA parameter
    res = mod.fit()
# Save BIC for AR(p)    
    BIC[p] = res.bic 
#here is the AR1 model
mod=ARMA(model_diff2,order=(1,0))    
res=mod.fit() #returns an ARMAResults class
res.bic
res.arparams #coefficient, which is very close to the PACF at t-1
print(res.summary())
'''
#AR1
mod=ARMA(resids, order=(1,0))
res=mod.fit()
#check the BIC
res.bic
#this is basically a random walk and cannot be predicted
res.arparams


#AR1,MA1
mod2=ARMA(resids, order=(1,1))
res2=mod2.fit()
res2.bic
res2.maparams
#construct an ARMA model with drift


#regress ng on twd in FBProphet















#can calculate a rolling mean as an input

#the dependent variable field is to labeled 'y'; datetime field labeled 'ds'
#instantiate the prophet object
ts_prophet=fbprophet.Prophet(changepoint_prior_scale=0.15, interval_width=0.95)
fb_version=ng_daily_df.rename(columns={'day_timestamp':'ds','ng_val':'y'})
#ensure that we only select data where there is a volatility measure
fb_version1=fb_version.loc[~np.isnan(yu),:]
#add regressor... optimally this would be standardized
good=yu[~np.isnan(yu)]
fb_version1['volatility']=pd.Series(good)
ts_prophet.add_regressor('volatility')
train=fb_version1.iloc[-500:-35,]
test=fb_version1.iloc[-35:,]
#fit the model
ts_prophet.fit(train)
#just the date output... this serves as input to the predict() function
ts_forecast=ts_prophet.make_future_dataframe(periods=35,freq='D'
)
#don't know if I'm passing the appropriate arguments here and above
forecast_data=ts_prophet.predict(test)
#plot the prediction

#plt.title('FB Prophet forecast on only price')
ts_prophet.plot(forecast_data,xlabel='Date', ylabel='LNG Price')
plt.close()
plt.plot(forecast_data['ds'],forecast_data['yhat'],'-',color='red')
plt.plot(ng_daily_df['day_timestamp'][-35:],ng_daily_df['ng_val'].iloc[-35:],'-',color='green')

forecast_data['ds']
ng_daily_df['day_timestamp']
ng_daily_df.index.values[-100:]
#overlay the actual values
plt.show()



#ARIMA models with a drift will be the basis for each MonteCarlo scenario

#added regressor version
ts_prophet.add_regressor('vol_15day')


#model selection with mean absolute error and r-squared
#cumulative sum of prediction errors: actual values and predicted
sum(allo.apply(lambda x: x['yhat_vanilla']-x['actual'],axis=1))/allo.shape[0]