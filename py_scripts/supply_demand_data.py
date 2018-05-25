import pandas as pd
from py_scripts import mongoQueryScripts as mqs

def get_supply_demand_data(sym):
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

    return df, ticker
