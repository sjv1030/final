# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:33:26 2018

@author: sjv1030_hp
"""

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import pprint
import pandas as pd
import plotly.plotly as py
import plotly.tools as tls
#import get_Data_old
import mongoQueryScripts as mqs

from pymongo import MongoClient
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
init_notebook_mode(connected=True)
import plotly.graph_objs as go

def getOLS(sym='o'):
   
    if sym == 'ng':
        ticker = 'nat_gas'
        data_dict = {}
        data_dict[ticker] = mqs.ng_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['rig'] = mqs.ngrig_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['prod'] = mqs.ngprod_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['cons'] = mqs.ngcons_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['twd'] = mqs.twd_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['ip'] = mqs.ip_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        
        df = data_dict[ticker] # initiate dataframe
        for k,v in data_dict.items():
            if k != ticker:
                # concatenate other series to have one master dataframe
                df = pd.concat([df,v],axis=1,join='inner')
                
        df.dropna(inplace=True)
        cols = [k for k in data_dict]
        df.columns = cols # set dataframe column names  
        # calculate difference between production and consumption
        df['netbal'] = df['prod'] - df['cons']
        
    else:
        ticker = 'oil'
        data_dict = {}
        data_dict[ticker] = mqs.wtc_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['rig'] = mqs.rig_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['prod'] = mqs.prod_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['inv'] = mqs.inv_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['twd'] = mqs.twd_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        data_dict['ip'] = mqs.ip_df.sort_values('month_timestamp').set_index('month_timestamp',drop=True)
        
        df = data_dict[ticker] # initiate dataframe
        for k,v in data_dict.items():
            if k != ticker:
                # concatenate other series to have one master dataframe
                df = pd.concat([df,v],axis=1,join='inner')
                
        df.dropna(inplace=True)
        cols = [k for k in data_dict]
        df.columns = cols # set dataframe column names
    
    # Function to get seasons
    def getSea(row):
        if row.month in [12,1,2]:
            return 'winter'
        elif row.month in [3,4,5]:
            return 'spring'
        elif row.month in [6,7,8]:
            return 'summer'
        else:
            'autumn'
    
    # Add seasons
    df['season'] = ''                
    df['season'] = df.index.map(getSea)
    df = pd.concat([df,pd.get_dummies(df['season'])], axis=1)
    df.drop(['season'], axis=1, inplace=True)

    ###########################################
    ## Convert all the columns to y/y% change except dummies
    dyy = df.iloc[:,:-3].pct_change(12)[12:]

    # Append dummies to y/y% change
    dyy = pd.concat([dyy,df.iloc[12:,-3:]], axis=1)
    
    dyy['springXprod'] = dyy['spring'] * dyy['prod']
    dyy['summerXprod'] = dyy['summer'] * dyy['prod']
    dyy['winterXprod'] = dyy['winter'] * dyy['prod']
    
    if ticker == 'nat_gas':
        dyy['springXcons'] = dyy['spring'] * dyy['cons']
        dyy['summerXcons'] = dyy['summer'] * dyy['cons']
        dyy['winterXcons'] = dyy['winter'] * dyy['cons']
        
        dyy['springXnetbal'] = dyy['spring'] * dyy['netbal']
        dyy['summerXnetbal'] = dyy['summer'] * dyy['netbal']
        dyy['winterXnetbal'] = dyy['winter'] * dyy['netbal']
        
    else:
        dyy['springXcons'] = dyy['spring'] * dyy['inv']
        dyy['summerXcons'] = dyy['summer'] * dyy['inv']
        dyy['winterXcons'] = dyy['winter'] * dyy['inv']
    
    # Loop through 1:4 lags to see which produces the best rsquared (naive assessment)
    rsq_yy = []
    for l in range(1,5):
        y = dyy[ticker][l:]
        x = dyy.loc[:,dyy.columns != ticker].shift(l).dropna()
     
        # Add a constant with most of your regressions
        x = sm.add_constant(x)
    
        ## Split data into train and test
        # Hopefully when all the data is pulled in, this is OK
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
    
    y = dyy[ticker][lstaryy:]
    x = dyy.loc[:,dyy.columns != ticker].shift(lstaryy).dropna()
        
    # Add a constant with most of your regressions
    x = sm.add_constant(x)
    
    ## Split data into train and test 
    # Hopefully when all the data is pulled in, this is OK
    x_train = x.loc[:'20161231']
    x_test = x.loc['20170131':]
    
    y_train = y.loc[:'20161231']
    y_test = y.loc['20170131':]
    
    ols_modelyy = sm.OLS(y_train,x_train) # create model
    ols_fityy = ols_modelyy.fit() # fit  model
    
    # Print OLS summary
    # print(ols_fityy.summary())
    
    # create dataframe of betas and p-values to return
    ols_df = pd.DataFrame({'betas':ols_fityy.params,
                           'p-values':ols_fityy.pvalues})
    
    # Get model residuals
    residyy = ols_fityy.resid
    
    # Change setting to increase plot font size
    matplotlib.rcParams.update({'font.size': 22})
    
    # Make QQ Plot of Model residuals to ensure normal distribution
    qqfig = plt.figure(figsize=(30,15))
    ax = qqfig.add_subplot(1,1,1)
    sm.qqplot(residyy, fit=True, line='45',ax=ax)
    ax.set_title('Q-Q Plot of Residuals')
    plt.show()
    
    # Plot Model residuals to check for heteroskedasticity and serial correlation
    resfig = plt.figure(figsize=(30,15))
    ax = resfig.add_subplot(1,1,1)
    ax.scatter(x=residyy.index,y=residyy.values)
    ax.set_title('Model Residuals')
    plt.axhline(y=0,linestyle='-',color='black')
    plt.show()
    
    # Plot of actual data and fitted values
    fitfig = plt.figure(figsize=(30,15))
    ax = fitfig.add_subplot(1,1,1)
    ax.plot(y_train, label='Actual')
    ax.plot(ols_fityy.fittedvalues, label='Fitted', marker='o', linestyle='dashed')
    ax.legend()
    ax.set_title('Model Fit on Y/Y%')
    plt.show()
    
    # Get forecasted values by passing testing data set - x_test
    predyy = ols_fityy.predict(x_test)
    
    # Plot Actual and forecasted values of testing set
    forefig = plt.figure(figsize=(30,15))
    ax = forefig.add_subplot(1,1,1)
    ax.plot(y_test, label='Actual', linestyle='dashdot')
    ax.plot(predyy, label='Forecast', lw=3)
    ax.legend()
    ax.set_title('Forecast  of Y/Y% Values')
    plt.axhline(y=0,linestyle='-',color='black')
    plt.show()
    
    # Forecasting
    # get features to forecast that werent' included in the test period
    x_fore = dyy.loc[:,dyy.columns != ticker][-lstaryy:]
    
    # add constant to features dataframe
    x_fore = sm.add_constant(x_fore)
    
    # add constant to features dataframe
    if lstaryy == 1:
        x_fore['const'] = 1
    else:
        x_fore = sm.add_constant(x_fore)
    cols = x_fore.columns.tolist()
    x_fore = x_fore[[cols[-1]] + cols[:-1]] 
    
    # shift date forward by number of lags
    x_fore = x_fore.shift(freq=lstaryy)
    # fit features dataframe in model to get forecasted y/y%
    y_fore = ols_fityy.predict(x_fore)
    # Get level prices from one year ago for target variable
    yrago = df[ticker].loc[y_fore.index-pd.DateOffset(years=1)]
    # apply forecasted y/y% change to prices one year ago, to get forecasted levels
    f_df = yrago.values * (1+y_fore)
    
    
    return ols_fityy.nobs, ols_fityy.rsquared_adj, ols_df, f_df

# as an example, run the below code for nat_gas
# n = number of observations
# r = adjusted r-squared
# b = table of betas
# f = table of forecast -- note: this model lags the independent variables by 4 months, so it forecasts out 4 months iteratively
##### run this code for nat gas ---> n,r,b,f = getOLS('ng')

# run below code for oil
# note: this model lags the independent variables by 1 month, so it forecasts out 1 month
# n2,r2,b2,f2 = getOLS()
