#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Building a MonteCarlo model with FBProphet underneath. The variables for the scenarios will be those identified by Silverio. The predicted values will be generated from an ARIMA model with drift. 

"""
import fbprophet
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
import datetime

#local mongo
#client=MongoClient('localhost',81)
#db=client['commodities']

#mlab
client=MongoClient('mongodb://team_nyc:persyy@ds229450.mlab.com:29450/commodities',serverSelectionTimeoutMS=3000)
db=client.commodities

#monthly ng values
values=db['monthlyvalues']
test=values.find({'month_timestamp':{'$gt':datetime.datetime.strptime("2011-01-01","%Y-%m-%d")},
'ng_val':{'$exists':True}},{'_id':0,'month_timestamp':1,'ng_val':1}).limit(5)
test_df=pd.DataFrame(list(test))


#daily oil values
values=db['values']
test=values.find({'oil_val':{'$exists':True}},{'_id':0,'day_timestamp':1,'oil_val':1})
oil_df=pd.DataFrame(list(test))



#the dependent variable field is to labeled 'y'; datetime field labeled 'ds'
#instantiate the prophet object
ts_prophet=fbprophet.Prophet(changepoint_prior_scale=0.15)
oil_df.dtypes
fb_version=oil_df.rename(columns={'day_timestamp':'ds','oil_val':'y'})
fb_version.head()
train=fb_version.iloc[:-75,:]
test=fb_version.iloc[-75:,:]

#fit the model
ts_prophet.fit(train)
ts_forecast=ts_prophet.make_future_dataframe(periods=75,freq='D')
#ts_forecast=ts_prophet.predict(ts_forecast)

#plot the prediction
plt.close()
#plt.title('FB Prophet forecast on only price')
ts_prophet.plot(ts_forecast,xlabel='Date', ylabel='MRO Price')
plt.show()

#a regressor model



#ARIMA models with a drift will be the basis for each MonteCarlo scenario

#added regressor version
ts_prophet.add_regressor('vol_15day')


#model selection with mean absolute error and r-squared
#cumulative sum of prediction errors: actual values and predicted
sum(allo.apply(lambda x: x['yhat_vanilla']-x['actual'],axis=1))/allo.shape[0]