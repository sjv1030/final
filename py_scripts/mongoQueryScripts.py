# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:08:19 2018

@author: CallejaL
"""

from pymongo import MongoClient
import pandas as pd

#mlab
client=MongoClient('mongodb://team_nyc:persyy@ds229450.mlab.com:29450/commodities',serverSelectionTimeoutMS=3000)
db=client.commodities

'''
monthly values: ng_val, oil_val, rig (oil rigs), prod (oil prod), inv (oil inv), ng_rig, ng_prod, ng_cons, twd_val, ip_val
weekly values: oil_val, ng_val
daily values: ng_val, oil_val, twd_val
'''

#monthly ng values
values=db['monthlyvalues']
test=values.find({'ng_val':{'$exists':True}},{'_id':0,'month_timestamp':1,'ng_val':1})
ng_df=pd.DataFrame(list(test))

#monthly wtc prices
values=db['monthlyvalues']
test_mo=values.find({'oil_val':{'$exists':True}},{'_id':0,'month_timestamp':1,'oil_val':1})
wtc_df=pd.DataFrame(list(test_mo))

#daily ng values
values=db['values']
test=values.find({'ng_val':{'$exists':True}},{'_id':0,'day_timestamp':1,'ng_val':1})
ng_daily_df=pd.DataFrame(list(test))

#daily wtc prices
values=db['values']
test_mo=values.find({'oil_val':{'$exists':True}},{'_id':0,'day_timestamp':1,'oil_val':1})
wtc_daily_df=pd.DataFrame(list(test_mo))

db['values']
test_mo=values.find({'twd_val':{'$exists':True}},{'_id':0,'day_timestamp':1,'twd_val':1})
twd_val=pd.DataFrame(list(test_mo))

#weekly ng values
values=db['weeklyvalues']
test=values.find({'ng_val':{'$exists':True}},{'_id':0,'week_timestamp':1,'ng_val':1})
ng_weekly_df=pd.DataFrame(list(test))

#weekly wtc prices
values=db['weeklyvalues']
test_mo=values.find({'oil_val':{'$exists':True}},{'_id':0,'week_timestamp':1,'oil_val':1})
wtc_weekly_df=pd.DataFrame(list(test_mo))

#oil rigs monthly
values=db['monthlyvalues']
test_mo=values.find({'rig':{'$exists':True}},{'_id':0,'month_timestamp':1,'rig':1})
rig_df=pd.DataFrame(list(test_mo))

# oil inventory
values=db['monthlyvalues']
test_mo=values.find({'prod':{'$exists':True}},{'_id':0,'month_timestamp':1,'prod':1})
prod_df=pd.DataFrame(list(test_mo))

#oil inventory monthly
values=db['monthlyvalues']
test_mo=values.find({'inv':{'$exists':True}},{'_id':0,'month_timestamp':1,'inv':1})
inv_df=pd.DataFrame(list(test_mo))

#ng rigs monthly
values=db['monthlyvalues']
test_mo=values.find({'ng_rig':{'$exists':True}},{'_id':0,'month_timestamp':1,'ng_rig':1})
ngrig_df=pd.DataFrame(list(test_mo))

#ng production monthly
values=db['monthlyvalues']
test_mo=values.find({'ng_prod':{'$exists':True}},{'_id':0,'month_timestamp':1,'ng_prod':1})
ngprod_df=pd.DataFrame(list(test_mo))

#ng cons
values=db['monthlyvalues']
test_mo=values.find({'ng_cons':{'$exists':True}},{'_id':0,'month_timestamp':1,'ng_cons':1})
ngcons_df=pd.DataFrame(list(test_mo))

#twd monthly
values=db['monthlyvalues']
test_mo=values.find({'twd_val':{'$exists':True}},{'_id':0,'month_timestamp':1,'twd_val':1})
twd_df=pd.DataFrame(list(test_mo))

#industrial production monthly
values=db['monthlyvalues']
test_mo=values.find({'ip_val':{'$exists':True}},{'_id':0,'month_timestamp':1,'ip_val':1})
ip_df=pd.DataFrame(list(test_mo))


