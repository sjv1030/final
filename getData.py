# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:42:49 2018

@author: sjv1030_hp
"""


import pandas as pd
import quandl
import json
from urllib.request import urlopen

## EIA API Token
api = 'd88caf37dd5c4619bad28016ca4f0379'

########### WTI Spot Prices ###############
wti_d = quandl.get('EIA/PET_RWTC_D') # daily
wti_w = quandl.get('EIA/PET_RWTC_W') # weekly
wti_m = quandl.get('EIA/PET_RWTC_M') # monthly
###########################################

########## Nat Gas Spot Prices ############
ng_d = quandl.get('EIA/NG_RNGWHHD_D') # daily
ng_w = quandl.get('EIA/NG_RNGWHHD_W') # weekly
ng_m = quandl.get('EIA/NG_RNGWHHD_M') # monthly
###########################################

########### Monthly Rigs  #################
oil_rig_id = 'PET.E_ERTRRO_XR0_NUS_C.M'
ng_rig_id = 'PET.E_ERTRRG_XR0_NUS_C.M'
###########################################

###### Monthly Nat Gas Production #########
ng_prod_id = 'NG.N9070US2.M'
###########################################

####### Monthly Oil Production ############
oil_prod_id = 'PET.MCRFPUS1.M' # Field production
oil_import_id = 'PET.MCRIMUS1.M' # Imports
###########################################

######## Monthly Oil Inventory ############
## Consumption (demand) is hard to get from EIA. Additionally, we care more
## about the supply/demand difference. This can be found in the monthly change
## in inventory. If supply > demand, then inventory rises. Converse is true. 
oil_inv_id = 'PET.MCESTUS1.M'
###########################################

##### Monthly Nat Gas Consumption #########
ng_cons_id = 'NG.N9140US1.M'
###########################################

########## US Economic Data ################
gdpr = quandl.get('FRED/GDPMC1') # quarterly US real GDP index
twd_m = quandl.get('FRED/TWEXBMTH') # monthly trade-weighted dollar index
twd_d = quandl.get('FRED/DTWEXB') # daily trade-weighted dollar index
ip = quandl.get('FRED/IPB50001N') # monthly US industrial production
econ_m = pd.concat([twd_m, ip], join='inner', axis=1)
econ_m.columns = ['twd','ip']
econ_m.index = econ_m.index.to_period('M').to_timestamp('M')
###########################################


## Dictionaries to loop through when pulling down data
oil_ids = {'rig_id':'PET.E_ERTRRO_XR0_NUS_C.M',
        'prod_id':'PET.MCRFPUS1.M',
        'import_id':'PET.MCRIMUS1.M',
        'inv_id':'PET.MCESTUS1.M',
        }


ng_ids = {
        'rig_id':'PET.E_ERTRRG_XR0_NUS_C.M',
        'prod_id':'NG.N9070US2.M',
        'cons_id':'NG.N9140US1.M',      
        }

## Dictionaries to save data
oil_data_dict = {}
ng_data_dict = {}

url = 'http://api.eia.gov/series/?api_key='+api+'&series_id='

## Loop through series dictionary, pull down data,
## many necessary adjustments, then save to data dictionary
for k, v in oil_ids.items():
    dat = urlopen(url+v).read()
    data = json.loads(dat.decode())
    
    df = pd.DataFrame(data['series'][0]['data'],
                       columns=['Date',k[:-3]])
    df.set_index('Date',drop=True,inplace=True)
    df.sort_index(inplace=True)  
    oil_data_dict[k[:-3]] = df


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
oil_data = pd.DataFrame()
for v in oil_data_dict.values():
    oil_data = pd.concat([oil_data, v], axis=1)

oil_data['tot_supply'] = oil_data['prod'] + oil_data['import']
oil_data['netbal'] = oil_data['inv'].diff() ## change in inventory is what we're after
oil_data.dropna(inplace=True)
oil_data.index = pd.to_datetime(oil_data.index+'01')
oil_data.index = oil_data.index.to_period('M').to_timestamp('M')

oil_data = pd.concat([oil_data,wti_m], join='inner', axis=1)
oil_data.rename(columns={'Value':'wti'},inplace=True)
oil_data = pd.concat([oil_data, econ_m], join='inner', axis=1)

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
