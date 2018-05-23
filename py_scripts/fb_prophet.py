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
from itertools import compress, chain
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
#sort the dataframe
ng_daily_df.sort_values('day_timestamp',axis=0,ascending=True,inplace=True)

#in lieu of mlab
ng_daily_df=quandl.get('EIA/NG_RNGWHHD_D')
ng_daily_df.reset_index(inplace=True)
ng_daily_df.rename(columns={'Value':'ng_val','Date':'day_timestamp'},index=str,inplace=True)
ng_daily_df.index=ng_daily_df.day_timestamp
#finished with mlab formatting



'''Variable creation and analysis - ARMA, regression'''
#calculate the volatility regressors; type: close-to-close volatility
#method 1
temp=ng_daily_df['ng_val'].pct_change().values+1
#log conversion... would get an approximation to this just by calling percent_change() pandas method
ng_daily_df['log_vec']=np.log(temp)
ng_daily_df['roll_10']=ng_daily_df['log_vec'].rolling(9).mean()
test=(ng_daily_df['log_vec']-ng_daily_df['roll_10'])**2
gy=test.rolling(9).sum()
#close-to-close volatility array
yu=np.sqrt((254.5/8)*gy)
#on any given rolling 20-day period, stratify/bin the observations in 1.09488-point ranges

#the distribution by bin for each volatility scenario
high_vol=[8,5,7,10]
low_vol=[30]
mid_vol=[15,10,5,0]

def generateSamps(n,rango):
    #a feeder function to the loop below... selects n number of random samples from the series of numbers contained in rango
    array=np.linspace(rango[0],rango[1],100)
    return(np.random.choice(array,size=n))

rango_list=[(0.0,1.09),(1.09,2.18),(2.18,3.27),(3.27,5.5)]
    
'''the volatility values for each volatility scenario'''
high_vol_list=[]
for i,g in enumerate(high_vol,0):
    high_vol_list.append(generateSamps(g,rango_list[i]))
high_path=list(chain.from_iterable(high_vol_list))

low_vol_list=[]
for i,g in enumerate(low_vol,0):
    low_vol_list.append(generateSamps(g,rango_list[i]))
low_path=list(chain.from_iterable(low_vol_list))

mid_vol_list=[]
for i,g in enumerate(mid_vol,0):
    mid_vol_list.append(generateSamps(g,rango_list[i]))
mid_path=list(chain.from_iterable(mid_vol_list))

#scramble/randomize the list a bit... place into dataframe with the appropriate dates
high_path
mid_path
low_path


def rearrangeDF(lista):
    #randomize/scramble the input list and convert to pandas series
    rango=np.arange(len(lista))
    new_index=np.random.choice(rango,len(lista),replace=False)
    return(pd.Series([lista[i] for i in new_index]))

#can feed this series into a function that concatenates this series with a datetime field to use as input for FBProphet.predict()
rearrangeDF(high_path)
#volatility is only positive an is not an ARMA process, unless it is scaled/standardized... nonetheless a regression can be run on volatility to determine some levels for scenario testing, and perhaps fit a probability distribution
plt.close()
plt.plot(ng_daily_df.index.values[-400:],yu[-400:])
plt.title('rolling 9-day volatility')
plt.show()

''' FBProphet '''
#the dependent variable field is to be labeled 'y'; datetime field labeled 'ds'
#data preparation
fb_version=ng_daily_df.rename(columns={'day_timestamp':'ds','ng_val':'y'})
#ensure that we only select data where there is a volatility measure
fb_version1=fb_version.loc[~np.isnan(yu),:]
#add regressor... optimally this would be standardized
good=yu[~np.isnan(yu)]
fb_version1['volatility']=pd.Series(good)
#shift the volatility by 1 to offset values: only relevant volatility measure is the one the trader knows at time of action and that is the one which measurement period ended prior day
fb_version1['volatility']=fb_version1['volatility'].shift(1)
fb_version1.dropna(how='any',inplace=True)


#instantiate the prophet object
ts_prophet=fbprophet.Prophet(changepoint_prior_scale=0.15, interval_width=0.95)
ts_prophet.add_regressor('volatility')
#will run on the entire dataset, as simulated values will be in the future as of t0
train=fb_version1.iloc[-500:,]
#fit the model
ts_prophet.fit(train)
#will be inserting simulated values
#get path of interest rates
date_df=ts_prophet.make_future_dataframe(periods=30, freq='d')
ir=rearrangeDF(low_path)
date_df['volatility']=train['volatility'].append(ir,ignore_index=True)

#just the date output... this serves as input to the predict() function
#don't know if I'm passing the appropriate arguments here and above
forecast_data=ts_prophet.predict(date_df)
#apply the forecast to the entire dataset... this way I have control over the output

ts_prophet.plot(forecast_data)

#plot the prediction
fig, ax1 = plt.subplots()
ax1.plot(train.ds[-30:,],train.y[-30:,],color='green')
ax1.plot(train.ds[-30:],forecast_data.yhat[-30:], color='red')
ax1.plot(forecast_data.ds,forecast_data.yhat, color='black', linestyle=':')
ax1.fill_between(forecast_data.ds, forecast_data['yhat_upper'], forecast_data['yhat_lower'], alpha=1.5, color='darkgray')
ax1.set_title('Sales (Orange) vs Sales Forecast (Black)')
ax1.set_ylabel('Dollar Sales')
ax1.set_xlabel('Date')
plt.show()



#run the FBProphet model on a parsimoneous dataset w/o a regressor
ts_prophet=fbprophet.Prophet(changepoint_prior_scale=0.15, interval_width=0.95)
train=fb_version1.iloc[-500:-35,]
ts_prophet.fit(train)
future_data = ts_prophet.make_future_dataframe(periods=35, freq = 'd')
simple_set=ts_prophet.predict(future_data)




simple_set['yhat'][-35:-25]
train['y'][-25:]
ts_prophet.plot(simple_set)
#concatenate both dataframes
forecast_all=pd.concat([forecast_data_all,forecast_data])
#append the prediction at the tail of the actual observations

fig, ax1 = plt.subplots()
ax1.plot(train.ds,train.y)
ax1.plot(forecast_data.ds,forecast_data.yhat, color='black', linestyle=':')
ax1.fill_between(forecast_data.ds, forecast_data['yhat_upper'], forecast_data['yhat_lower'], alpha=0.5, color='darkgray')
ax1.set_title('Sales (Orange) vs Sales Forecast (Black)')
ax1.set_ylabel('Dollar Sales')
ax1.set_xlabel('Date')
plt.show()
    
def plot_data(func_df, test_length):
    #end_date = end_date - timedelta(weeks=4) # find the 2nd to last row in the data. We don't take the last row because we want the charted lines to connect
    #mask = (func_df.index > end_date) # set up a mask to pull out the predicted rows of data.
    #predict_df = func_df.loc[mask] # using the mask, we create a new dataframe with just the predicted data.
   
# Now...plot everything
    fig, ax1 = plt.subplots()
    ax1.plot(train.ds,train.y)
    ax1.plot(forecast_data.ds,forecast_data.yhat, color='black', linestyle=':')
    ax1.fill_between(forecast_data.ds, forecast_data['yhat_upper'], forecast_data['yhat_lower'], alpha=0.5, color='darkgray')
    ax1.set_title('Sales (Orange) vs Sales Forecast (Black)')
    ax1.set_ylabel('Dollar Sales')
    ax1.set_xlabel('Date')
  
# change the legend text
    L=ax1.legend() #get the legend
    L.get_texts()[0].set_text('Actual Sales') #change the legend text for 1st plot
    L.get_texts()[1].set_text('Forecasted Sales') #change the legend text for 2nd plot


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

'''some work towards developing the volatility scenarios'''
#develop insight into the distribution of the realized volatility
good=yu[~np.isnan(yu)]
def rolling_apply(ser,window):
    bins=np.array([1.09,2.18,3.27,5.5])
    i=np.arange(ser.shape[0]+1-window)
    #results=np.zeros(ser.shape[0])
    results=[]
    for g in i:
        #results[g+window-1]=np.digitize(ser[g:window+g],bins)
        results.append(np.digitize(ser[g:window+g],bins,right=True))
    return(results)
    
test=rolling_apply(good,30)
good[:30]

def dic_counters(lista):
    master=[]
    for i in lista:
        #counter={1:0,2:0,3:0,4:0,5:0}    
        master.append(np.bincount(i))
    return(master)
        
gh=dic_counters(test)
bools=[len(i) == 4 for i in gh]
final=list(compress(gh,bools))
suma=[sum(i) for i in final]
''' end volatility scenario development '''

#ARIMA models with a drift will be the basis for each MonteCarlo scenario

#added regressor version
ts_prophet.add_regressor('vol_15day')


#model selection with mean absolute error and r-squared
#cumulative sum of prediction errors: actual values and predicted
sum(allo.apply(lambda x: x['yhat_vanilla']-x['actual'],axis=1))/allo.shape[0]


''' Additional regression techniques '''
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






