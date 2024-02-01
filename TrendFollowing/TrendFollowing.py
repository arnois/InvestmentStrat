# -*- coding: utf-8 -*-
"""
Trend following investemnt strategy tool kit

@author: arnulf.q@gmail.com
"""

#%% MODULES
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import os
import datetime as dt
#from matplotlib.ticker import StrMethodFormatter
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

#%% Futures Data
# for demo purposes we are using ZN (10Y T-Note) futures contract

# Dates range
str_idt, str_fdt = '2000-12-31', '2023-12-31'

# Futures symbols
yh_tkrlst = ['ZN=F']

# Data pull from yahoo finance
path = r'C:\Users\arnoi\Documents\Python Scripts\db\futures_10Y_yf.parquet'
path = r'H:\db\futures_ZN_yf.parquet'
if os.path.isfile(path):
    # From saved db
    data = pd.read_parquet(path)
else:
    # Download db
    data = pdr.get_data_yahoo(yh_tkrlst, 
                              start=str_idt, 
                              end=str_fdt)
    # Save db
    data.to_parquet(path)

#%% ZN Price Viz
## Plotting T3Y Price
fig, ax = plt.subplots()
ax.plot(data.loc['2021':,'Adj Close'], '-', color='darkcyan')
myFmt = mdates.DateFormatter('%b%y')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('')
ax.set_ylabel('Price')
ax.set_title('10Y T-Note Future')
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()

#%% TSM
# Log-prices and returns
data_logP = data[['Adj Close']].apply(np.log)
data_ret = data_logP.diff()

# Dates cutoff
tstmp_first_available_dt = data_ret.index[0] + dt.timedelta(days = 91)
tstmp_last_available_dt = data_ret.index[-1] - dt.timedelta(days = 91)
dt_cutoff = data_ret[tstmp_first_available_dt:].\
    apply(lambda x: x.index - dt.timedelta(days=91)).\
        rename(columns={'Adj Close':'Start'})
dt_cutoff['End'] = dt_cutoff['Start'] + dt.timedelta(days=90)

# 3M Returns
df_3M_Returns = pd.\
    DataFrame(columns=['Start','End','T3M_Ret', 'Ret', 'ERet'],
              index=dt_cutoff.index)
for i,r in dt_cutoff.iterrows():
    logP1, logP2 = data_logP.loc[r['Start']:r['End']].iloc[[0,-1]].to_numpy()
    logCurrP = data_logP.loc[i].to_numpy()
    lookback_ret = logP2 - logP1
    curr_ret = logCurrP - logP2
    lookback_exc_ret = curr_ret - lookback_ret
    tmp_arr = np.concatenate([lookback_ret, curr_ret, lookback_exc_ret])
    df_3M_Returns.loc[i,['Start', 'End']] = r[['Start','End']]
    df_3M_Returns.loc[i,['T3M_Ret', 'Ret', 'ERet']] = tmp_arr
    










