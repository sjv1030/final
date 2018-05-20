# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:33:26 2018

@author: sjv1030_hp
"""


import statsmodels.api as sm
import matplotlib.pyplot as plt

import pandas as pd
import quandl
import json
from urllib.request import urlopen
from pymongo import MongoClient
import datetime

###################################
## This is supposed  to be imported
from py_scripts import get_Data_old
###################################

## Get data
data = get_Data_old.getData(sym='ng',freq='m',eco=1)

## replace the legacy get_data call with a query to the mlab database
#this is a connection to a locally hosted mongo for now and will be updated with mlab credentials after prod code is developed
client=MongoClient('localhost',81)
db=client['commodities']

values=db['monthlyvalues']
test=values.find({'month_timestamp':{'$gt':datetime.datetime.strptime("2011-01-01","%Y-%m-%d")},
'oil_val':{'$exists':True}},{'_id':0,'month_timestamp':1,'oil_val':1}).limit(5)
test_df=pd.DataFrame(list(test))
    


## Convert all the columns to m/m% change
dmm = data.pct_change()[1:]

# Loop through 1:4 lags to see which produces the best rsquared (naive assessment)
rsq_mm = []
for l in range(1,5):
    y = dmm['nat_gas'][l:]
    x = dmm.loc[:,dmm.columns != 'nat_gas'].shift(l).dropna()
        
    # Add a constant with most of your regressions
    x = sm.add_constant(x)
    
    ## Split data into train and test 
    x_train = x.loc[:'20161231']
    x_test = x.loc['20170131':]
    
    y_train = y.loc[:'20161231']
    y_test = y.loc['20170131':]
    
    ols_modelmm = sm.OLS(y_train,x_train) # create model
    ols_fitmm = ols_modelmm.fit() # fit  model
    
    # add lag and rsquared values to list
    rsq_mm.append((l,ols_fitmm.rsquared))

# extract best lag judging by rsquared alone
lstar = max(rsq_mm, key = lambda x: x[1])[0]

y = dmm['nat_gas'][lstar:]
x = dmm.loc[:,dmm.columns != 'nat_gas'].shift(lstar).dropna()
    
# Add a constant with most of your regressions
x = sm.add_constant(x)

## Split data into train and test 
x_train = x.loc[:'20161231']
x_test = x.loc['20170131':]

y_train = y.loc[:'20161231']
y_test = y.loc['20170131':]

ols_modelmm = sm.OLS(y_train,x_train) # create model
ols_fitmm = ols_modelmm.fit() # fit  model

# Print OLS summary
print(ols_fitmm.summary())

# Not sure how to make this qqplot bigger
sm.qqplot(ols_fitmm.resid, fit=True, line='45')
plt.show()


## Convert all the columns to y/y% change
dyy = data.pct_change(4)[4:]


# Loop through 1:4 lags to see which produces the best rsquared (naive assessment)
rsq_yy = []
for l in range(1,5):
    y = dyy['nat_gas'][l:]
    x = dyy.loc[:,dyy.columns != 'nat_gas'].shift(l).dropna()
        
    # Add a constant with most of your regressions
    x = sm.add_constant(x)
    
    ## Split data into train and test 
    x_train = x.loc[:'20161231']
    x_test = x.loc['20170131':]
    
    y_train = y.loc[:'20161231']
    y_test = y.loc['20170131':]
    
    ols_modelyy = sm.OLS(y_train,x_train) # create model
    ols_fityy = ols_modelyy.fit() # fit  model
    
    # add lag and rsquared values to list
    rsq_yy.append((l,ols_fityy.rsquared))

# extract best lag judging by rsquared alone
lstaryy = max(rsq_yy, key = lambda x: x[1])[0]

y = dyy['nat_gas'][lstaryy:]
x = dyy.loc[:,dyy.columns != 'nat_gas'].shift(lstaryy).dropna()
    
# Add a constant with most of your regressions
x = sm.add_constant(x)

## Split data into train and test 
x_train = x.loc[:'20161231']
x_test = x.loc['20170131':]

y_train = y.loc[:'20161231']
y_test = y.loc['20170131':]

ols_modelyy = sm.OLS(y_train,x_train) # create model
ols_fityy = ols_modelyy.fit() # fit  model

# Print OLS summary
print(ols_fityy.summary())

# Not sure how to make this qqplot bigger
sm.qqplot(ols_fityy.resid, fit=True, line='45')
plt.show()




