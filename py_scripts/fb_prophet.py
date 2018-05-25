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
import numpy as np
import quandl
from itertools import chain
#local mongo
#client=MongoClient('localhost',81)
#db=client['commodities']

#def runProphet(commmodity):
def testRunProphet(sym='o'):
#mlab
    client=MongoClient('mongodb://team_nyc:persyy@ds229450.mlab.com:29450/commodities',serverSelectionTimeoutMS=3000)
    db=client.commodities

    #select the asset
    commodities={'ng':'ng_val','o':'oil_val'}
#query ng data from mongo
    selected=commodities[sym]
    values=db['values']
    test=values.find({selected:{'$exists':True}},{'_id':0,'day_timestamp':1,selected:1})
    ng_daily_df=pd.DataFrame(list(test))
    #sort the dataframe
    ng_daily_df.sort_values('day_timestamp',axis=0,ascending=True,inplace=True)
    '''
#in lieu of mlab
    ng_daily_df=quandl.get('EIA/NG_RNGWHHD_D')
    ng_daily_df.reset_index(inplace=True)
    ng_daily_df.rename(columns={'Value':'ng_val','Date':'day_timestamp'},index=str,inplace=True)
    ng_daily_df.index=ng_daily_df.day_timestamp
#finished with mlab formatting
    '''
    '''Variable creation and analysis - ARMA, regression'''
    def createVolArray(df):
#calculate the volatility regressors; type: close-to-close volatility
#method 1
        temp=df.filter(regex=r'val').pct_change().values+1
#log conversion... would get an approximation to this just by calling percent_change() pandas method
        ng_daily_df['log_vec']=np.log(temp)
        ng_daily_df['roll_10']=ng_daily_df['log_vec'].rolling(9).mean()
        test=(ng_daily_df['log_vec']-ng_daily_df['roll_10'])**2
        gy=test.rolling(9).sum()
#close-to-close volatility array
        yu=np.sqrt((254.5/8)*gy)
#on any given rolling 20-day period, stratify/bin the observations in 1.09488-point ranges
        return(yu)

#the distribution by bin for each volatility scenario
    high_vol=[8,5,7,10]
    low_vol=[30,0,0,0]
    mid_vol=[15,10,5,0]

    #calculate volatility array
    yu=createVolArray(ng_daily_df)

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
    all_paths={'high path':high_path,'mid path':mid_path,'low path':low_path}


    def rearrangeDF(lista):
    #randomize/scramble the list of simulated volatility and convert to pandas series
        rango=np.arange(len(lista))
        new_index=np.random.choice(rango,len(lista),replace=False)
        return(pd.Series([lista[i] for i in new_index]))

    #dict to contain all 9-day volatility simulated paths
    dic_ser={}
    for k,v in all_paths.items():
        dic_ser[k]=rearrangeDF(v)

#plot volatility
    '''
    def plotVolatility(yu):
        plt.plot(ng_daily_df.index.values[-400:],yu[-400:])
        plt.xticks(rotation=45)
        plt.title('rolling 9-day volatility')
        return(plt)
    '''
    ''' FBProphet '''

    def prepDataSet(daily_df):
#the dependent variable field is to be labeled 'y'; datetime field labeled 'ds'
#data preparation
        fb_version=daily_df.rename(columns={'day_timestamp':'ds',selected:'y'})
        fb_version['y']=np.log(fb_version['y'].values)
#ensure that we only select data where there is a volatility measure
        #yu = realized historical volatility
        fb_version1=fb_version.loc[~np.isnan(yu),:]
#add regressor... optimally this would be standardized
        good=yu[~np.isnan(yu)]
        fb_version1['volatility']=pd.Series(good)
#shift the volatility by 1 to offset values: only relevant volatility measure is the one the trader knows at time of action and that is the one which measurement period ended prior day
        fb_version1['volatility']=fb_version1['volatility'].shift(1)
        fb_version1.dropna(how='any',inplace=True)
        #train the model on the last 500 trading days
        train=fb_version1.iloc[-500:,]
        return(train)

    def fitProphet(train,ir_path):
        #concatenate the
#instantiate the prophet object
        ts_prophet=fbprophet.Prophet(changepoint_prior_scale=0.15, interval_width=0.95)
        ts_prophet.add_regressor('volatility')
#will run on the entire dataset, as simulated values will be in the future as of t0
#fit the model
        ts_prophet.fit(train)
#will be inserting simulated values
#get path of interest rates
        date_df=ts_prophet.make_future_dataframe(periods=30, freq='d')
        date_df['volatility']=train['volatility'].append(ir_path,ignore_index=True)
#just the date output... this serves as input to the predict() function
#don't know if I'm passing the appropriate arguments here and above
        forecast_data=ts_prophet.predict(date_df)
        forecast_data['yhat']=np.exp(forecast_data['yhat'].values)
        return(forecast_data)
#apply the forecast to the entire dataset... this way I have control over the output

    train=prepDataSet(ng_daily_df)
    result_set={}
    for k, v in dic_ser.items():
        #a tuple of dataframes
        outcome= fitProphet(train,v)
        result_set[k]=outcome


    print(result_set)
        #first two elements are the x,y inputs for the volatility graphs... the other three graphs contain the FB Prophet prediction and fit for each of the volatility scenarios
    return ng_daily_df, yu[-400:], result_set['low path'], result_set['mid path'], result_set['high path']

#all_paths={'high path':high_path,'mid path':mid_path,'low path':low_path}

#1) historical volatility plot: plt.plot(ng_daily_df.index.values[-400:],yu[-400:])

''' legacy stuff
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

#Diagnostics:
#Absolute mean error
#sum or squared residuals
resids=forecast_data['yhat'][-530:-30].values-np.exp(fb_version1['y'][-500:].values)
plt.hist(resids,ins=25)
plt.show()

'''
