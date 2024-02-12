"""
Trend following investemnt strategy tool kit.

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
import holidays
import calendar
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

#%% RETURNS
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

#%% TREND
# Annualized k-month returns
df_TkM_sgnls = data_ret_sgnl[data_ret_sgnl.index[0] + \
                    pd.tseries.offsets.DateOffset(months=np.max(kmonths)+1):]
df_TkM_sgnls = df_TkM_sgnls.apply(lambda x: x*np.array([1,12/9,12/6,12/3,12]), axis=1)

# Trend rules
idx_down = df_TkM_sgnls.apply(lambda x: x['T12M']>x['T9M'] and x['T9M']>x['T6M'],axis=1)
idx_up = df_TkM_sgnls.apply(lambda x: x['T12M']<x['T9M'] and x['T9M']<x['T6M'],axis=1)
df_TkM_sgnls['Trend'] = 0
df_TkM_sgnls.loc[idx_down,'Trend'] = -1
df_TkM_sgnls.loc[idx_up,'Trend'] = 1

# Rebalancing months
tmp = set([(d.year,d.month) for d in df_TkM_sgnls.index])
first_thursday = []
for y,m in tmp:
    weekly_thursday=[]
    for week in calendar.monthcalendar(y, m):
        if week[3] != 0:
            weekly_thursday.append(week[3])
    first_thursday.append(
        pd.Timestamp(dt.date(y,m,weekly_thursday[0]).strftime("%Y-%m-%d")))
df_reb_dates = pd.DataFrame(first_thursday, columns=['RebDate']).\
    sort_values(by='RebDate').reset_index(drop=True)['RebDate']

# Rebalancing dates weekly
idx_reb_date = 3 # Thursday
df_TkM_sgnls['isRebDate'] = [d.weekday() for d in df_TkM_sgnls.index]
df_TkM_sgnls['isRebDate'] = df_TkM_sgnls['isRebDate'] == idx_reb_date
df_reb_dates = df_TkM_sgnls[df_TkM_sgnls['isRebDate']].reset_index()['Date']

# Example: investment signal
ct_pt_value = 32*31.25
reb_date = pd.Timestamp("2024-01-04")
sgnl_date = reb_date - pd.tseries.offsets.BDay(1)
reb_sgnl = df_TkM_sgnls.loc[sgnl_date, 'Trend']
reb_sigma = np.sqrt(exante_var_estimate(t=sgnl_date))
reb_size = np.floor(0.40/volat_estimate)
reb_pos_size = reb_size*reb_sgnl
reb_px_enter = data.loc[reb_date, 'Adj Close']
# reb_date_end = reb_date + pd.tseries.offsets.DateOffset(days=6) # a week later
reb_date_end = sgnl_date + pd.tseries.offsets.BDay(20) # a month later
reb_px_exit = data.loc[reb_date_end, 'Adj Close']
reb_pnl = (reb_px_exit - reb_px_enter)*ct_pt_value*reb_pos_size
reb_ret = (np.log(reb_px_exit) - np.log(reb_px_enter))*reb_sgnl

#%% STRAT
df_TkM_sgnls['xover'] = 0
# Lagged returns xovers
idx_xover_up = pd.concat([df_TkM_sgnls.shift().\
                                   apply(lambda x: x['T1M']<x['T3M'], axis=1),
                                   df_TkM_sgnls.\
               apply(lambda x: x['T1M']>x['T3M'], axis=1)], axis=1).all(axis=1)
df_TkM_sgnls.loc[idx_xover_up, 'xover'] = 1
idx_xover_down = pd.concat([df_TkM_sgnls.shift().\
                                   apply(lambda x: x['T1M']>x['T3M'], axis=1),
                                   df_TkM_sgnls.\
               apply(lambda x: x['T1M']<x['T3M'], axis=1)], axis=1).all(axis=1)
df_TkM_sgnls.loc[idx_xover_down, 'xover'] = -1

# Backtest
ct_pt_value = 32*31.25
tmpcal = holidays.country_holidays('US')
lst_holidays = [date.date() for date in pd.date_range('1999-01-01', '2034-01-01') if date in tmpcal]
bdc = np.busdaycalendar(holidays=lst_holidays)
df_strat = pd.DataFrame()
for i,r in df_reb_dates.items():
    try:
        # r is rebalance date
        sgnl_date = r - pd.tseries.offsets.CustomBusinessDay(1, calendar=bdc)
        ## Trend assessment
        reb_trend = df_TkM_sgnls.loc[sgnl_date, 'Trend']
        reb_sgnl=reb_trend
        ## Trade signal assessment
        reb_sigma = np.sqrt(exante_var_estimate(t=sgnl_date))
        reb_size = np.floor(0.40/reb_sigma)
        reb_pos_size = reb_size*reb_sgnl
        reb_px_enter = data.loc[r, 'Adj Close']
        # reb_date_end = r + pd.tseries.offsets.CustomBusinessDay(4, calendar=bdc)
        reb_date_end = sgnl_date + pd.tseries.offsets.CustomBusinessDay(20, calendar=bdc)
        reb_px_exit = data.loc[reb_date_end, 'Adj Close']
        reb_pnl = (reb_px_exit - reb_px_enter)*ct_pt_value*reb_pos_size
        reb_ret = (np.log(reb_px_exit) - np.log(reb_px_enter))*reb_sgnl
        df_strat.loc[r,['Signal','Pos_Size','Px_Enter', 'Px_Exit','PnL','R']] = \
            [reb_sgnl,reb_pos_size,reb_px_enter,reb_px_exit,reb_pnl,reb_ret]
    except:
        continue

np.exp(df_strat['R'].cumsum()).plot()
df_strat['PnL'].cumsum().plot()

