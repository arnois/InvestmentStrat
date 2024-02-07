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

# Data pull
path = r'C:\Users\arnoi\Documents\Python Scripts\db\futures_10Y_yf.parquet'
path = r'H:\db\futures_ZN_yf.parquet'
if os.path.isfile(path):
    # From saved db
    data = pd.read_parquet(path)
else:
    # Download data from yahoo finance
    data = pdr.get_data_yahoo(yh_tkrlst, 
                              start=str_idt, 
                              end=str_fdt)
    # Save db
    data.to_parquet(path)

# Data update
last_date_data = data.index[-1]
todays_date = pd.Timestamp(dt.datetime.today().date()) - pd.tseries.offsets.BDay(1)
isDataOOD = last_date_data < todays_date
if isDataOOD:
    print("Updating data... ")
    str_idt = last_date_data.strftime("%Y-%m-%d")
    str_fdt = todays_date.strftime("%Y-%m-%d")
    new_data = pdr.get_data_yahoo(yh_tkrlst, start=str_idt, end=str_fdt)
    updated_data = pd.concat([data,new_data]).\
        reset_index().drop_duplicates('Date').set_index('Date')
    print("Saving data...")
    updated_data.to_parquet(path)
    data = pd.read_parquet(path)

#%% ZN Price Viz
## Plotting T3Y Price
fig, ax = plt.subplots()
ax.plot(data.loc['2023':,'Adj Close'], '-', color='darkcyan')
myFmt = mdates.DateFormatter('%b%y')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('')
ax.set_ylabel('Price')
ax.set_title('10Y T-Note Future')
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()

#%% TSM
# Daily log-prices
data_logP = data[['Adj Close']].apply(np.log)

# Daily returns
data_ret = data_logP.diff().fillna(0)
ema_alpha = 1-60/61
data_ret['EMA'] = data_ret.ewm(span=121, adjust=False).mean()

# Daily ex ante volatility
def exante_var_estimate(t=2, alpha=1-60/61):
    if t<=0:
        t=1
    sumvar = 0
    var_rt = data_ret['EMA'].iloc[t]
    for n in range(t-1):
        rt_1_i = data_ret['Adj Close'].iloc[t-1-n]
        sumvar += (1-alpha)*alpha**n*(rt_1_i - var_rt)**2
    return 261*sumvar

#
#lst_eave = []
#for t in range(data_ret.shape[0]):
#    lst_eave.append(exante_var_estimate(t=t))
#data_ret['eaSigma'] = np.sqrt(lst_eave)

# Current ex ante volatility estimate
volat_estimate = np.sqrt(exante_var_estimate(t=data_ret.shape[0]-1))

# Position size - constant vol
pos_size = 0.40/volat_estimate

# Last 12 months smoothed return
df_idx_dates = pd.DataFrame(
    [todays_date - pd.tseries.offsets.DateOffset(months=n) for n in [12,9,6,3]],
    columns=['Idx_Date'],
    index=[f"T{n}M" for n in [12,9,6,3]])
df_kM_ret = df_idx_dates.apply(lambda d: np.exp(data_ret['EMA'].loc[d[0]:].sum())-1, axis=1)

# Cum return index
data_idx = data_ret.cumsum().apply(np.exp)
t1y_date = todays_date - pd.tseries.offsets.DateOffset(years=1)
data_idx.loc[t1y_date:].plot(color=['darkcyan','orange'])

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
    










