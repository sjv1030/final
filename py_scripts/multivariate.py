# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:33:26 2018
@author: sjv1030_hp
"""


import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import get_Data_old

## Get data
data = get_Data_old.getData(sym='ng',freq='m',eco=1)

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

# Get model residuals
residmm = ols_fitmm.resid

# Change setting to increase plot font size
matplotlib.rcParams.update({'font.size': 22})

# Make QQ Plot of Model residuals to ensure normal distribution
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,1,1)
sm.qqplot(residmm, fit=True, line='45',ax=ax)
ax.set_title('Q-Q Plot of Residuals')
plt.show()

# Plot Model residuals to check for heteroskedasticity and serial correlation
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,1,1)
ax.scatter(x=residmm.index,y=residmm.values)
ax.set_title('Model Residuals')
plt.axhline(y=0,linestyle='-',color='black')
plt.show()

# Plot of actual data and fitted values
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,1,1)
ax.plot(y_train, label='Actual')
ax.plot(ols_fitmm.fittedvalues, label='Fitted', marker='o', linestyle='dashed')
ax.legend()
ax.set_title('Model Fit on M/M%')
plt.show()

# Get forecasted values by passing testing data set - x_test
predmm = ols_fitmm.predict(x_test)

# Plot Actual and forecasted values of testing set
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,1,1)
ax.plot(y_test, label='Actual', linestyle='dashdot')
ax.plot(predmm, label='Forecast', lw=3)
ax.legend()
ax.set_title('Forecast  of M/M% Values')
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

# Get model residuals
residyy = ols_fityy.resid

# Change setting to increase plot font size
matplotlib.rcParams.update({'font.size': 22})

# Make QQ Plot of Model residuals to ensure normal distribution
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,1,1)
sm.qqplot(residyy, fit=True, line='45',ax=ax)
ax.set_title('Q-Q Plot of Residuals')
plt.show()

# Plot Model residuals to check for heteroskedasticity and serial correlation
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,1,1)
ax.scatter(x=residyy.index,y=residyy.values)
ax.set_title('Model Residuals')
plt.axhline(y=0,linestyle='-',color='black')
plt.show()

# Plot of actual data and fitted values
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,1,1)
ax.plot(y_train, label='Actual')
ax.plot(ols_fityy.fittedvalues, label='Fitted', marker='o', linestyle='dashed')
ax.legend()
ax.set_title('Model Fit on Y/Y%')
plt.show()

# Get forecasted values by passing testing data set - x_test
predyy = ols_fityy.predict(x_test)

# Plot Actual and forecasted values of testing set
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,1,1)
ax.plot(y_test, label='Actual', linestyle='dashdot')
ax.plot(predyy, label='Forecast', lw=3)
ax.legend()
ax.set_title('Forecast  of Y/Y% Values')
plt.show()
