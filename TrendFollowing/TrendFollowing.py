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
    if new_data.index[-1] != todays_date:
        str_fdt = (todays_date + pd.tseries.offsets.DateOffset(days=1)).\
            strftime("%Y-%m-%d")
        new_data = pdr.get_data_yahoo(yh_tkrlst, start=str_idt, end=str_fdt)
    updated_data = pd.concat([data,new_data]).\
        reset_index().drop_duplicates('Date').set_index('Date')
    print("Saving data...")
    updated_data.to_parquet(path)
    data = pd.read_parquet(path)

#%% ZN Price Viz
## Plotting T3Y Price
fig, ax = plt.subplots()
ax.plot(data.loc['2024':,'Adj Close'], '-', color='darkcyan')
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

# Cum return index
data_idx = data_ret.cumsum().apply(np.exp)
t1y_date = todays_date - pd.tseries.offsets.DateOffset(years=1)
data_idx.loc[t1y_date:].plot(color=['darkcyan','orange'])

# Function to compute daily ex ante variance estimate
def exante_var_estimate(t: pd.Timestamp = pd.Timestamp("2024-02-07"), 
                        alpha: float = 1-60/61) -> float:
    """
    Parameters
    ----------
    t : pd.Timestamp
        Estimate date. The default is pd.Timestamp("2024-02-07").
    alpha : float, optional
        EMA parameter. The default is 1-60/61.

    Returns
    -------
    float
        Exponentially weighted average of squared returns.
    """
    # Index for t as date
    idx_t = np.where(data_ret.index == t)[0][0]
    if idx_t<=0:
        idx_t = 1
    # Exp. weighted average return
    var_rt = data_ret['EMA'].iloc[idx_t]
    # EWASR
    sumvar = 0
    for n in range(idx_t-1):
        rt_1_i = data_ret['Adj Close'].iloc[idx_t-1-n]
        sumvar += (1-alpha)*(alpha**n)*(rt_1_i - var_rt)**2
    return 261*sumvar
#
#lst_eave = []
#for t in range(data_ret.shape[0]):
#    lst_eave.append(exante_var_estimate(t=t))
#data_ret['eaSigma'] = np.sqrt(lst_eave)

# Current ex ante volatility estimate
volat_estimate = np.sqrt(exante_var_estimate(t=todays_date))

# Position size - constant vol
pos_size = np.floor(0.40/volat_estimate)

# Last k months smoothed return
kmonths = [12,9,6,3,1]
idxdt = todays_date - pd.tseries.offsets.DateOffset(days=1)
df_idx_dates = pd.DataFrame(
    [idxdt - pd.tseries.offsets.DateOffset(months=n) for n in kmonths],
    columns=['Idx_Date'],
    index=[f"T{n}M" for n in kmonths])

df_kM_ret = df_idx_dates.\
    apply(lambda d: np.exp(data_ret['EMA'].loc[d[0]:idxdt].sum())-1, axis=1)

# Last k-month returns for each previous dates
data_ret_sgnl = pd.DataFrame(index=data_ret.index)
for i,r in data_ret.iterrows():
    # End date
    j = i - pd.tseries.offsets.DateOffset(days=1)
    # Starting dates
    curr_idx_dates = pd.DataFrame(
        [j - pd.tseries.offsets.DateOffset(months=n) for n in kmonths],
        columns=['Idx_Date'],
        index=[f"T{n}M" for n in kmonths])
    # k-month smoother returns
    curr_kM_ret = curr_idx_dates.\
        apply(lambda d: np.exp(data_ret['EMA'].loc[d[0]:j].sum())-1, axis=1)
    # Fill df
    data_ret_sgnl.loc[i,curr_kM_ret.index.tolist()] = curr_kM_ret.values

# Coherent signals
df_TkM_sgnls = data_ret_sgnl[data_ret_sgnl.index[0] + \
                    pd.tseries.offsets.DateOffset(months=np.max(kmonths)+1):]

# Rebalancing dates
idx_reb_date = 3 # Thursday
df_TkM_sgnls['isRebDate'] = [d.weekday() for d in df_TkM_sgnls.index]
df_TkM_sgnls['isRebDate'] = df_TkM_sgnls['isRebDate'] == idx_reb_date
df_reb_dates = df_TkM_sgnls[df_TkM_sgnls['isRebDate']].reset_index()['Date']

# Example
ct_pt_value = 32*31.25
reb_date = pd.Timestamp("2024-02-01")
sgnl_date = reb_date - pd.tseries.offsets.BDay(1)
str_strat_name = 'T1M'
reb_sgnl = np.sign(df_TkM_sgnls.loc[sgnl_date, str_strat_name])
reb_sigma = np.sqrt(exante_var_estimate(t=sgnl_date))
reb_size = np.floor(0.40/volat_estimate)
reb_pos_size = reb_size*reb_sgnl
reb_px_enter = data.loc[reb_date, 'Adj Close']
reb_date_end = reb_date + pd.tseries.offsets.DateOffset(days=6)
reb_px_exit = data.loc[reb_date_end, 'Adj Close']
reb_pnl = (reb_px_exit - reb_px_enter)*ct_pt_value*reb_pos_size
reb_ret = np.log(reb_px_exit) - np.log(reb_px_enter)







