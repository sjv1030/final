# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:33:26 2018

@main-author: sjv1030_hp
@co-author: michelebradley
"""

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import json
from urllib.request import urlopen
import probscale

def ng_multivariable():
    data_new = pd.read_csv("py_data/data_silverio.csv")

    headers = ['netbal', 'nat_gas']

    data = data_new[headers]

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

    # Not sure how to make this qqplot bigger
    sm.qqplot(ols_fitmm.resid, fit=True, line='45')

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

    # Not sure how to make this qqplot bigger
    sm.qqplot(ols_fityy.resid, fit=True, line='45')

    plot_position = probscale.plot_pos(ols_fityy.resid)

    return data, ols_fityy.resid, plot_position
