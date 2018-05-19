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

###################################
## This is supposed  to be imported
from py_scripts import get_Data_old
###################################



data = get_Data_old.getData(sym='ng',freq='m',eco=1)

# this is the name of the outter folder and file
# stcuture is [script name].[program]
# t = get_Data_old.getData()


####################################################
## This is supposed  to be imported from pyscripts
####################################################
#
#import pandas as pd
#import quandl
#import json
#from urllib.request import urlopen
#
#def getData(sym='o',freq='d',eco=0):
#   
#    ## EIA API Token
#    api = 'd88caf37dd5c4619bad28016ca4f0379'
#    
#    ## URL to pull data from EIA.gov
#    url = 'http://api.eia.gov/series/?api_key='+api+'&series_id='
#    
#    ########### WTI Spot Prices ###############
##    wti_d = quandl.get('EIA/PET_RWTC_D') # daily
##    wti_w = quandl.get('EIA/PET_RWTC_W') # weekly
##    wti_m = quandl.get('EIA/PET_RWTC_M') # monthly
#    ###########################################
#    
#    # if eco == 0, then return data w/o eco data
#    if eco == 0:
#        # if sym == 'o', then return oil spot prices w/o data
#        if sym == 'o':
#            # if frequency == 'w', then return weekly oil spot prices
#            if freq == 'w':
#                return quandl.get('EIA/PET_RWTC_W')
#            # if frequency == 'm', then return monthly oil spot prices
#            elif freq == 'm':
#                return quandl.get('EIA/PET_RWTC_M')
#            # else return daily oil spot prices
#            else:
#                return quandl.get('EIA/PET_RWTC_D')
#    
#    ########## Nat Gas Spot Prices ############
##    ng_d = quandl.get('EIA/NG_RNGWHHD_D') # daily
##    ng_w = quandl.get('EIA/NG_RNGWHHD_W') # weekly
##    ng_m = quandl.get('EIA/NG_RNGWHHD_M') # monthly
#    ###########################################
#    
#        # if sym == 'ng', then return nat gas spot prices w/o data
#        if sym == 'ng':
#            # if frequency == 'w', then return weekly nat gas spot prices
#            if freq == 'w':
#                return quandl.get('EIA/NG_RNGWHHD_W')
#            # if frequency == 'm', then return monthly nat gas spot prices
#            elif freq == 'm':
#                return quandl.get('EIA/NG_RNGWHHD_M')
#            # else return daily nat gas spot prices
#            else:
#                return quandl.get('EIA/NG_RNGWHHD_D') 
#    
#    # if eco == 1, then return data with eco data
#    if eco == 1:
#        ########## US Economic Data ################
#        twd_m = quandl.get('FRED/TWEXBMTH') # monthly trade-weighted dollar index
#        ip = quandl.get('FRED/IPB50001N') # monthly US industrial production
#        econ_m = pd.concat([twd_m, ip], join='inner', axis=1)
#        econ_m.columns = ['twd','ip']
#        econ_m.index = econ_m.index.to_period('M').to_timestamp('M')
#        ###########################################
#        
#        
#        # if sym == 'o', then return oil spot prices with eco data
#        if sym == 'o':
#            
#            ########### Monthly Rigs  #################
##            rig_id = 'PET.E_ERTRRO_XR0_NUS_C.M'
#            ###########################################
#            
#            ####### Monthly Oil Production ############
##            prod_id = 'PET.MCRFPUS1.M' # Field production
##            import_id = 'PET.MCRIMUS1.M' # Imports
#            ###########################################
#            
#            ######## Monthly Oil Inventory ############
#            ## Consumption (demand) is hard to get from EIA. Additionally, we care more
#            ## about the supply/demand difference. This can be found in the monthly change
#            ## in inventory. If supply > demand, then inventory rises. Converse is true. 
##            inv_id = 'PET.MCESTUS1.M'
#            ###########################################
#            
#            ## Dictionaries to loop through when pulling down data
#            oil_ids = {'rig_id':'PET.E_ERTRRO_XR0_NUS_C.M',
#                       'prod_id':'PET.MCRFPUS1.M',
#                       'import_id':'PET.MCRIMUS1.M',
#                       'inv_id':'PET.MCESTUS1.M',
#                       }
#            
#            # monthly oil spot prices
#            wti_m = quandl.get('EIA/PET_RWTC_M') 
#            
#            ## Dictionary to save data
#            oil_data_dict = {}
#            
#            ## Loop through series dictionary, pull down data,
#            ## make necessary adjustments, then save to data dictionary
#            for k, v in oil_ids.items():
#                dat = urlopen(url+v).read()
#                data = json.loads(dat.decode())
#                
#                df = pd.DataFrame(data['series'][0]['data'],
#                                   columns=['Date',k[:-3]])
#                df.set_index('Date',drop=True,inplace=True)
#                df.sort_index(inplace=True)  
#                oil_data_dict[k[:-3]] = df
#            
#            ## Create dataframe combining all monthly data series
#            oil_data = pd.DataFrame()
#            for v in oil_data_dict.values():
#                oil_data = pd.concat([oil_data, v], axis=1)
#            
#            oil_data['tot_supply'] = oil_data['prod'] + oil_data['import']
#            oil_data.index = pd.to_datetime(oil_data.index+'01')
#            oil_data.index = oil_data.index.to_period('M').to_timestamp('M')
#            
#            oil_data = pd.concat([oil_data,wti_m], join='inner', axis=1)
#            oil_data.rename(columns={'Value':'wti'},inplace=True)
#            oil_data = pd.concat([oil_data, econ_m], join='inner', axis=1)
#            
#            return oil_data
#            
#        # if sym == 'ng', then return nat gas spot prices with eco data
#        if sym == 'ng':
#    
#            ########### Monthly Rigs  #################
##            rig_id = 'PET.E_ERTRRG_XR0_NUS_C.M'
#            ###########################################
#            
#            ###### Monthly Nat Gas Production #########
##            prod_id = 'NG.N9070US2.M'
#            ###########################################
#            
#            ##### Monthly Nat Gas Consumption #########
##            cons_id = 'NG.N9140US1.M'
#            ###########################################
#    
#            ## Dictionaries to loop through when pulling down data
#            ng_ids = {
#                    'rig_id':'PET.E_ERTRRG_XR0_NUS_C.M',
#                    'prod_id':'NG.N9070US2.M',
#                    'cons_id':'NG.N9140US1.M',      
#                    }
#            
#            # monthly nat gas spot prices
#            ng_m = quandl.get('EIA/NG_RNGWHHD_M')
#    
#            ## Dictionary to save data
#            ng_data_dict = {}
#    
#            ## Loop through series dictionary, pull down data,
#            ## make necessary adjustments, then save to data dictionary    
#            for k, v in ng_ids.items():
#                dat = urlopen(url+v).read()
#                data = json.loads(dat.decode())
#                
#                df = pd.DataFrame(data['series'][0]['data'],
#                                   columns=['Date',k[:-3]])
#                
#                df.set_index('Date',drop=True,inplace=True)
#                df.sort_index(inplace=True)
#                
#                # Make nat gas prod same units as nat gas consumption -- billion cubic feet
#                if k[:-3] == 'prod':
#                    df = df/1000
#                ng_data_dict[k[:-3]] = df
#
#    
#            ## Create dataframe combining all monthly data series
#            ng_data = pd.DataFrame()
#            for v in ng_data_dict.values():
#                ng_data = pd.concat([ng_data, v], axis=1)
#                
#            ng_data.dropna(inplace=True)
#            ng_data['netbal'] = ng_data['prod'] - ng_data['cons']
#            ng_data.index = pd.to_datetime(ng_data.index+'01')
#            ng_data.index = ng_data.index.to_period('M').to_timestamp('M')
#            
#            ng_data = pd.concat([ng_data,ng_m], join='inner', axis=1)
#            ng_data.rename(columns={'Value':'nat_gas'},inplace=True)
#            ng_data = pd.concat([ng_data, econ_m], join='inner', axis=1)
#            
#            return ng_data

####################################################
# This is where the function I want to import should stop
####################################################
            
## Get data
#data = getData(sym='ng',freq='m',eco=1)

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




