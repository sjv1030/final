#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 18:41:14 2018

@author: lechuza
"""
import sys
sys.path.append('/home/lechuza/Documents/CUNY/data_607/final_project/data602-finalproject/py_scripts/')
import get_Data_mongo_updates as data
import pandas as pd
import quandl
import json
from urllib.request import urlopen

api = 'd88caf37dd5c4619bad28016ca4f0379'
    ## URL to pull data from EIA.gov
url = 'http://api.eia.gov/series/?api_key='+api+'&series_id='

'''testing'''
#weekly ng
ng_w=quandl.get('EIA/NG_RNGWHHD_W')
data.updateMongoWeekly(ng_w,'ng_val')

#monthly ng
ng_m=quandl.get('EIA/NG_RNGWHHD_M')
data.updateMongoMonthly(ng_m,'ng_val')

#daily ng
ng_d=quandl.get('EIA/NG_RNGWHHD_D') 
data.updateMongoDaily(ng_d,'ng_val')

#daily twd
twd_m = quandl.get('FRED/DTWEXM') 
data.updateMongoDaily(ng_d,'twd_val')
###########
#weekly oil
wtc_weekly=quandl.get('EIA/PET_RWTC_W')
data.updateMongoWeekly(wtc_weekly,'oil_val')

#daily oil
wtc_d=quandl.get('EIA/PET_RWTC_D')
data.updateMongoDaily(wtc_d,'oil_val')

#monthly oil
wtc_m=quandl.get('EIA/PET_RWTC_M')
data.updateMongoMonthly(wtc_m,'oil_val')
wtc_m.tail()

test=df.index.values+'01'
test[0:7]
df.index=df.index.values+'01'

lista=[]
del(v)
oil_ids = {'rig_id':'PET.E_ERTRRO_XR0_NUS_C.M',
                       'prod_id':'PET.MCRFPUS1.M',
                       'import_id':'PET.MCRIMUS1.M',
                       'inv_id':'PET.MCESTUS1.M',
                       }

 ## Loop through series dictionary, pull down data,
## make necessary adjustments, then save to data dictionary
for k, v in oil_ids.items():
    dat = urlopen(url+v).read()
    dats = json.loads(dat.decode())
                
    df = pd.DataFrame(dats['series'][0]['data'],
                                   columns=['Date','Value'])
    df['Value']=df['Value'].astype('float64')
    df.set_index('Date',drop=True,inplace=True)
    
#    df.index=df.index.values+'01'
    #df.index = pd.to_datetime(df.index+'01')
    data.updateMongoMonthly(df,k[:-3],econ=True)
    #print(df.index.values[0:5])
final_df=pd.concat(lista,axis=1)


v='NG.N9070US2.M'
dat = urlopen(url+v).read()
dats = json.loads(dat.decode())
                #converts a list to a dataframe & assigns column names
df = pd.DataFrame(dats['series'][0]['data'],
                                   columns=['Date','Value'])
df['Value']=df['Value'].astype('float64')
df.dtypes
ng_ids = {
                    'ng_rig_id':'PET.E_ERTRRG_XR0_NUS_C.M',
                    'ng_prod_id':'NG.N9070US2.M',
                    'ng_cons_id':'NG.N9140US1.M',      
                    }
    
            ## Loop through series dictionary, pull down data,
            ## make necessary adjustments, then save to data dictionary    
for k, v in ng_ids.items():
    dat = urlopen(url+v).read()
    dats = json.loads(dat.decode())
                #converts a list to a dataframe & assigns column names
    df = pd.DataFrame(dats['series'][0]['data'],
                                   columns=['Date','Value'])
    df['Value']=df['Value'].astype('float64')
                # Make nat gas prod same units as nat gas consumption -- billion cubic feet
    if k[:-3] == 'ng_prod':
        df['Value'] = df['Value']/1000
    df.set_index('Date',drop=True,inplace=True)
    data.updateMongoMonthly(df,k[:-3],econ=True)


twd_m = quandl.get('FRED/TWEXBMTH') # monthly trade-weighted dollar index
data.updateMongoMonthly(twd_m,'twd_val',econ=True)
ip = quandl.get('FRED/IPB50001N') # monthly US industrial production
data.updateMongoMonthly(ip,'ip_val',econ=True)

str(type(twd_m.index)) == "<class 'pandas.core.indexes.datetimes.DatetimeIndex'>"

''' end of test '''

oil_ids = {'rig_id':'PET.E_ERTRRO_XR0_NUS_C.M',
                       'prod_id':'PET.MCRFPUS1.M',
                       'import_id':'PET.MCRIMUS1.M',
                       'inv_id':'PET.MCESTUS1.M',
                       }
            
            # monthly oil spot prices
wti_m = quandl.get('EIA/PET_RWTC_M') 
            
            ## Dictionary to save data
oil_data_dict = {}
            
            ## Loop through series dictionary, pull down data,
            ## make necessary adjustments, then save to data dictionary
            
dat = urlopen(url+oil_ids['rig_id']).read()
data = json.loads(dat.decode())
ma=data['series'][0]['data']
df = pd.DataFrame(data['series'][0]['data'],
                                   columns=['Date','Value'])
df['Value']=df['Value'].astype('float64')
df.set_index('Date',drop=True,inplace=True)
df.index = pd.to_datetime(df.index+'01')
#df.index = DF.index.to_period('M').to_timestamp('M')
df.index.values[0:2]
df.head()
df.dtypes

df_dic=df.to_dict(orient='index')
    #iterate through the dictionary making updates to the mongodb
for key, val in df_dic.items():
        # "Value" below is compatible for quandl, and may need to be amended for EIA data
    dic={'month_timestamp':key,'rig_val':val['Value']}
    values.update_one({'month_timestamp':key},{"$set":dic},upsert=True)
        
        
        
        


for k, v in oil_ids.items():
    dat = urlopen(url+v).read()
    data = json.loads(dat.decode())
                
    
    df = pd.DataFrame(data['series'][0]['data'],
                                   columns=['Date',k[:-3]])
    df.set_index('Date',drop=True,inplace=True)
    df.sort_index(inplace=True)  
    oil_data_dict[k[:-3]] = df
            
            ## Create dataframe combining all monthly data series
oil_data = pd.DataFrame()
for v in oil_data_dict.values():
    oil_data = pd.concat([oil_data, v], axis=1)
            
    oil_data['tot_supply'] = oil_data['prod'] + oil_data['import']
    oil_data.index = pd.to_datetime(oil_data.index+'01')
    oil_data.index = oil_data.index.to_period('M').to_timestamp('M')
            
    oil_data = pd.concat([oil_data,wti_m], join='inner', axis=1)
    oil_data.rename(columns={'Value':'wti'},inplace=True)
    oil_data = pd.concat([oil_data, econ_m], join='inner', axis=1)
            
    
    
            
    # if sym == 'ng', then return nat gas spot prices with eco data
    if sym == 'ng':

        ########### Monthly Rigs  #################
#            rig_id = 'PET.E_ERTRRG_XR0_NUS_C.M'
        ###########################################

        ###### Monthly Nat Gas Production #########
#            prod_id = 'NG.N9070US2.M'
        ###########################################

        ##### Monthly Nat Gas Consumption #########
#            cons_id = 'NG.N9140US1.M'
        ###########################################

        ## Dictionaries to loop through when pulling down data
        ng_ids = {
                'rig_id':'PET.E_ERTRRG_XR0_NUS_C.M',
                'prod_id':'NG.N9070US2.M',
                'cons_id':'NG.N9140US1.M',      
                }

        # monthly nat gas spot prices
        ng_m = quandl.get('EIA/NG_RNGWHHD_M')

        ## Dictionary to save data
        ng_data_dict = {}

        ## Loop through series dictionary, pull down data,
        ## make necessary adjustments, then save to data dictionary    
        for k, v in ng_ids.items():
            dat = urlopen(url+v).read()
            data = json.loads(dat.decode())

            df = pd.DataFrame(data['series'][0]['data'],
                               columns=['Date',k[:-3]])

            df.set_index('Date',drop=True,inplace=True)
            df.sort_index(inplace=True)

            # Make nat gas prod same units as nat gas consumption -- billion cubic feet
            if k[:-3] == 'prod':
                df = df/1000
            ng_data_dict[k[:-3]] = df


        ## Create dataframe combining all monthly data series
        ng_data = pd.DataFrame()
        for v in ng_data_dict.values():
            ng_data = pd.concat([ng_data, v], axis=1)

        ng_data.dropna(inplace=True)
        ng_data['netbal'] = ng_data['prod'] - ng_data['cons']
        ng_data.index = pd.to_datetime(ng_data.index+'01')
        ng_data.index = ng_data.index.to_period('M').to_timestamp('M')

        ng_data = pd.concat([ng_data,ng_m], join='inner', axis=1)
        ng_data.rename(columns={'Value':'nat_gas'},inplace=True)
        ng_data = pd.concat([ng_data, econ_m], join='inner', axis=1)

    return ng_data

