"""
Trend following investemnt strategy tool kit.

@author: arnulf.q@gmail.com
"""

#%% MODULES
import numpy as np
import pandas as pd
import pandas_ta as ta
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
import sys
#%% CUSTOM MODULES
str_cwd = r'H:\Python\TrendFollowing' # str_cwd=r'C:\Users\arnoi\Documents\Python Scripts\TrendFollowing'
sys.path.append(str_cwd)
from DataMan import DataMan
from TrendMan import TrendMan

#%% UDF

#%% DATA
# Path
path = r'C:\Users\arnoi\Documents\db\etf_bbg.parquet'
path = r'H:\db\etf_bbg.parquet'

# Data Manager
dfMgr = DataMan(datapath=path)

# Data
dfMgr.set_data_from_path()
data = dfMgr.get_data()

# Data subset
data_slice = dfMgr.sliceDataFrame(dt_start='2016-12-31', 
                     dt_end='2020-12-31', 
                     lst_securities=['TLT','GLD','IWM'])
# Data stats
dfMgr.statistics(data_slice.apply(np.log).diff().dropna()['Close']).T

# Correlation matrix between selected assets
dfMgr.plot_corrmatrix(data_slice['Close'].diff(), txtCorr=True)

# Data update
xlpath_update = r'C:\Users\jquintero\db\datapull_etf_1D.xlsx'
dfMgr.update_data(xlpath = xlpath_update)
data = dfMgr.get_data()

#%% TREND
nes, nel = 21, 64
TrendMgr = TrendMan(nes, nel)
TrendMgr.set_trend_data(data,'Close')
data_trend = TrendMgr.get_trend_data()

# Check any assets TA
name = 'IWM'
namecol = [f'{name}_trend',f'{name}_trend_strength',
           f'{name}_trend_status',f'{name}_ema_d']

# Trend + dynamics
data_trend[namecol].iloc[-21:].merge(
    data.loc[data_trend[namecol].iloc[-21:].index]['Close'][name]
    , left_index=True, right_index=True)
    
#%% TREND ANALYSIS

# TrendStrength-based weigths
data_strength_Z = TrendMgr.get_trend_strength_Z()
data_w = data_strength_Z.apply(lambda x: x/x.sum(), axis=1)
data_w.columns = [s.replace('_ema_d','') for s in data_w.columns]
print(f"Last W's:\n{round(100*data_w.iloc[-5:].T,0)}")

# Long-only subcase
data_w_lOnly = (((data_strength_Z.apply(np.sign)+1)/2).\
    fillna(0)*data_strength_Z).apply(lambda x: x/x.sum(), axis=1)
data_w_lOnly.columns = [s.replace('_ema_d','') for s in data_w_lOnly.columns]
print(f"Last Long-Only W's:\n{round(100*data_w_lOnly.iloc[-5:].T,0)}")

# Filter out non-weak trends
nonwTrends = []
for name in data.columns.levels[1]:
    # Trend strength
    tmpcol = f'{name}_trend_strength'
    if data_trend.iloc[-1][f'{name}_trend_strength'] == 'weak':
        continue
    else:
        nonwTrends.append(name)

# Non-weak Trending Assets
data_trend.iloc[-1][[f'{c}_trend' for c in nonwTrends]]

#%% PRICE VIZ
# Todays date
todays_date = pd.Timestamp.today().date() 

## Plotting Price Levels
fig, ax = plt.subplots()
tmpetf = ['GLD','XLY','XLI','XLV']
tmptkrs = [('Close', s) for s in tmpetf]
ax.plot(data.loc['2024':,tmptkrs].dropna(), '-')
myFmt = mdates.DateFormatter('%d%b')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('')
ax.set_ylabel('Price')
ax.set_title('ETFs')
ax.legend(tmpetf, loc='best', bbox_to_anchor=(1.01,1.01))
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()
                                    
# Single Plot
tmpetf = 'GLD'
tmptkr = ('Close', tmpetf)
t1y_date = todays_date - pd.tseries.offsets.DateOffset(years=1)
fig, ax = plt.subplots()
ax.plot(data.loc[t1y_date:,tmptkr].dropna(),c='darkcyan')
ax.plot(data_trend.loc[t1y_date:,tmpetf+'_EMA_ST'],'--', c='C0')
ax.plot(data_trend.loc[t1y_date:,tmpetf+'_EMA_LT'],'--',c='orange')
myFmt = mdates.DateFormatter('%d%b%Y')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('')
ax.set_ylabel('Price')
ax.set_title(f'{tmpetf} ETF')
ax.legend(['Price','Short EMA', 'Long EMA'], 
          loc='best', bbox_to_anchor=(1.01,1.01))
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()

#%% RETURNS
# Daily log-prices
data_logP = data[['Close']].apply(np.log)

# Daily returns + smoothed returns
data_ret = data_logP.diff().fillna(0).droplevel(0, axis=1)
data_ret = data_ret.merge(
    data_ret.apply(lambda s: ta.ema(s, nes)).rename(
        columns=dict(zip(data_ret.columns,
                         [f'{s}_EMA_ST' for s in data_ret.columns]))).\
        merge(data_ret.apply(lambda s: ta.ema(s, nel)).\
            rename(
        columns=dict(zip(data_ret.columns,
                         [f'{s}_EMA_LT' for s in data_ret.columns]))),                      
        left_index=True, right_index=True),
    left_index=True, right_index=True)

# Cum return index
data_idx = data_ret.cumsum().apply(np.exp)
clrlst = ['darkcyan','blue','orange','red']
tmpNames = ['XLV','IWM','GLD','EEM']
data_idx.loc[t1y_date:, tmpNames].plot(color=clrlst, title='Returns Idx')

# Volatility
data_ret_sigma = (data_ret[tmpNames].\
                  rolling(21).std()*np.sqrt(252)).dropna()
data_ret_sigma = (data_ret[tmpNames].\
                  ewm(span=252, adjust=False).std()*np.sqrt(252)).\
    iloc[252:].rename(columns={'Close':'EMA'}).merge(data_ret_sigma, 
                                                         left_index=True,
                                                         right_index=True)
dfMgr.statistics(data_ret_sigma).T
data_ret_sigma.loc[t1y_date:].plot(title='Volatility', alpha=0.65)

#%% VOLATILITY
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
    var_rt = data_ret.iloc[:,1].iloc[idx_t]
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

#%% TRAILING RETURNS
# Last k months smoothed return
kmonths = [12,9,6,3,1]

df_idx_dates = pd.DataFrame(
    [todays_date - pd.tseries.offsets.DateOffset(months=n) for n in kmonths],
    columns=['Idx_Date'],
    index=[f"T{n}M" for n in kmonths])

df_kM_ret = df_idx_dates.\
    apply(lambda d: np.exp(data_ret['EMA'].loc[d[0]:todays_date].sum())-1, 
          axis=1)*np.array([1,12/9,12/6,12/3,12])
    
# Last k-month returns for each previous dates
data_ret_kM = pd.DataFrame(index=data_ret.index)
for i,r in data_ret.iterrows():
    # End date
    j = i - pd.tseries.offsets.DateOffset(days=1)
    # Starting dates
    curr_idx_dates = pd.DataFrame(
        [j - pd.tseries.offsets.DateOffset(months=n) for n in kmonths],
        columns=['Idx_Date'],
        index=[f"T{n}M" for n in kmonths])
    # k-month returns
    curr_kM_ret = curr_idx_dates.\
        apply(lambda d: 
              np.exp(data_ret['Adj Close'].loc[d[0]:j].sum())-1, axis=1)
    # Fill df
    data_ret_kM.loc[i,curr_kM_ret.index.tolist()] = curr_kM_ret.values

#data_ret_kM = data_ret_kM*np.array([1,12/9,12/6,12/3,12])

# Last k-month smoothed returns for each previous dates
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

# data_ret_sgnl = data_ret_sgnl*np.array([1,12/9,12/6,12/3,12])

#%% FWD RETURNS
# Fwd k-weeks returns
kweeks = [12,4,2,1]

df_fwd_dates = pd.DataFrame(
    [todays_date + pd.tseries.offsets.DateOffset(weeks=n) for n in kweeks],
    columns=['Idx_Date'],
    index=[f"F{n}W" for n in kweeks])

df_fwd_dates_ret = df_fwd_dates.\
    apply(lambda d: np.exp(data_ret['EMA'].loc[todays_date:d[0]].sum())-1, 
          axis=1)

# Fwd k-weeks returns for each previous dates
tmpdt_st = data_ret.index[1] + pd.tseries.offsets.DateOffset(weeks=12)
tmpdt_ed = data_ret.index[-2] - pd.tseries.offsets.DateOffset(weeks=12)
data_ret_fwd = pd.DataFrame(index=data_ret.\
                            loc[data_ret.index[1]:tmpdt_ed].index)
for i,r in data_ret.loc[data_ret.index[1]:tmpdt_ed].iterrows():
    # Ending fwd dates
    curr_idx_fwd_dates = pd.DataFrame(
        [i + pd.tseries.offsets.DateOffset(weeks=n) for n in kweeks],
        columns=['Idx_Date'],
        index=[f"F{n}W" for n in kweeks])
    # Fwd k-week returns
    curr_fwd_ret = curr_idx_fwd_dates.\
        apply(lambda d: 
              np.exp(data_ret['Adj Close'].loc[i:d[0]].sum())-1, 
              axis=1)
    # Fill df
    data_ret_fwd.loc[i, curr_fwd_ret.index.tolist()] = curr_fwd_ret.values
    
# data_ret_fwd = data_ret_fwd*np.array([52/12,52/4,52/2,52/1])

#%% TRAILING VS FWD RETURNS
df_ret_TvF = data_ret_fwd.merge(data_ret_kM,left_index=True,right_index=True)
# scatter
from pandas.plotting import scatter_matrix
scatter_matrix(df_ret_TvF[['F12W','F4W','F2W','F1W','T12M']], alpha=0.5, 
               figsize=(10, 6), diagonal='kde')
# corr
dfMgr.plot_corrmatrix(df_ret_TvF, dt_start = '2021-02-20', dt_end = '2024-02-20', 
                lst_securities = ['F12W','F4W','F2W','F1W',
                                  'T12M','T9M','T6M','T3M','T1M'], 
                plt_size = (10,8), txtCorr = False, corrM = 'spearman')
# Corr between fwd and trailing returns
df_ret_TvF.corr(method='spearman').loc[['F12W','F4W','F2W','F1W'],['T1M','T3M','T6M','T9M','T12M']]

## REGAN
from scipy import stats
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()

# Row slice
row_ed_dt = pd.Timestamp('2023-11-28')
row_st_dt = row_ed_dt - pd.tseries.offsets.DateOffset(years=8)
# Linear features
col_feats = ['T3M','T6M','T9M','T12M']
col_resp = ['F12W']
X = df_ret_TvF.loc[row_st_dt:row_ed_dt, col_feats]
y = df_ret_TvF.loc[row_st_dt:row_ed_dt, col_resp]
mlr.fit(X, y);print(f'\nScore TrainSet: {mlr.score(X,y):,.2%}\n')
pd.DataFrame([np.insert(mlr.coef_[0],0,mlr.intercept_[0])], columns = ['b']+col_feats)
y_hat = mlr.predict(X)
sse = np.sum((mlr.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])
mlr_tstat = mlr.coef_ / se
mlr_tstat_p = 2 * (1 - stats.t.cdf(np.abs(mlr_tstat), y.shape[0] - X.shape[1]))
mlr_insample = y.merge(pd.DataFrame(y_hat, columns=['Model'], index=y.index), 
        left_index=True,right_index=True)
mlr_insample.plot(title='In-Sample Data Returns')
(mlr_insample.diff(axis=1).dropna(axis=1)*-1).plot(title='In-Sample Model Errors')

#%% TREND
# Annualized k-month returns
df_TkM_sgnls = data_ret_sgnl[data_ret_sgnl.index[0] + \
                    pd.tseries.offsets.DateOffset(months=np.max(kmonths)+1):]
df_TkM_sgnls = df_TkM_sgnls.apply(lambda x: x*np.array([1,12/9,12/6,12/3,12]), axis=1)

# Trend rules
idx_down = df_TkM_sgnls.apply(lambda x: 0 > x['T9M'], axis=1) # df_TkM_sgnls.apply(lambda x: x['T12M']>x['T9M'], axis=1)
idx_up = df_TkM_sgnls.apply(lambda x: 0 < x['T9M'], axis=1) # df_TkM_sgnls.apply(lambda x: x['T12M']<x['T9M'], axis=1)
## fill
df_TkM_sgnls['Trend'] = 0
df_TkM_sgnls.loc[idx_down,'Trend'] = -1
df_TkM_sgnls.loc[idx_up,'Trend'] = 1

#%% Example: investment signal
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

#%% STRAT: Trend signal + rebalance period by holding time /RETURNS

# MONTHLY rebalance
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

# WEEKLY rebalance
idx_reb_date = 3 # Thursday
df_TkM_sgnls['isRebDate'] = [d.weekday() for d in df_TkM_sgnls.index]
df_TkM_sgnls['isRebDate'] = df_TkM_sgnls['isRebDate'] == idx_reb_date
df_reb_dates = df_TkM_sgnls[df_TkM_sgnls['isRebDate']].reset_index()['Date']

# calendar mgmt
ct_pt_value = 32*31.25
tmpcal = holidays.country_holidays('US')
lst_holidays = [date.date() for date in pd.date_range('1999-01-01', '2034-01-01') if date in tmpcal]
bdc = np.busdaycalendar(holidays=lst_holidays)

# Backtest
df_strat = pd.DataFrame()
## Reb by period
for i,r in df_reb_dates.items():
    try:
        # r is rebalance date
        sgnl_date = r - pd.tseries.offsets.CustomBusinessDay(1, calendar=bdc)
        ## Trend assessment
        reb_trend = df_TkM_sgnls.loc[sgnl_date, 'Trend']
        reb_sgnl = reb_trend*(np.sign(df_TkM_sgnls.loc[sgnl_date, 'T1M']) == reb_trend)
        ## Trade signal assessment
        reb_sigma = np.sqrt(exante_var_estimate(t=sgnl_date))
        reb_size = np.floor(0.40/reb_sigma)
        reb_pos_size = reb_size*reb_sgnl
        reb_px_enter = data.loc[r, 'Adj Close']
        reb_date_end = r + pd.tseries.offsets.CustomBusinessDay(4, calendar=bdc) # Weekly
        # reb_date_end = sgnl_date + pd.tseries.offsets.CustomBusinessDay(20, calendar=bdc) # Monthly
        if reb_date_end > data.index[-1]:
            reb_px_exit = data.iloc[-1]['Adj Close']
        else:
            reb_px_exit = data.loc[reb_date_end, 'Adj Close']
        reb_pnl = (reb_px_exit - reb_px_enter)*ct_pt_value*reb_pos_size
        reb_ret = (np.log(reb_px_exit) - np.log(reb_px_enter))*reb_sgnl
        df_strat.loc[r,['Signal','Pos_Size','Px_Enter', 'Px_Exit','PnL','R']] = \
            [reb_sgnl,reb_pos_size,reb_px_enter,reb_px_exit,reb_pnl,reb_ret]
    except:
        continue

# Plot backtest res
np.exp(df_strat['R'].cumsum()).plot(title='Trend: T9M Sign\nReb: Weekly+T1M')
dfMgr.statistics(df_strat[['R','PnL']]).T
df_strat['PnL'].cumsum().plot()

#%% STRAT: Trend filter + Xover signal /RETURNS

# Lagged returns xovers: 1M vs 3M returns
df_TkM_sgnls['xover'] = 0
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

# 1M returns sign xover
df_TkM_sgnls['xover'] = 0
idx_xover_up = pd.concat([df_TkM_sgnls.shift().\
                                   apply(lambda x: x['T1M'] < 0, axis=1),
                                   df_TkM_sgnls.\
               apply(lambda x: x['T1M'] > 0, axis=1)], axis=1).all(axis=1)
df_TkM_sgnls.loc[idx_xover_up, 'xover'] = 1
idx_xover_down = pd.concat([df_TkM_sgnls.shift().\
                                   apply(lambda x: x['T1M'] > 0, axis=1),
                                   df_TkM_sgnls.\
               apply(lambda x: x['T1M'] < 0, axis=1)], axis=1).all(axis=1)
df_TkM_sgnls.loc[idx_xover_down, 'xover'] = -1

## Reb by signal
df_reb_events = df_TkM_sgnls[['Trend','xover']][df_TkM_sgnls['xover'] != 0]
df_strat_byXover = pd.DataFrame()
for i,r in df_reb_events.iterrows():
    if r['Trend']*r['xover'] == 1: # Trade
        try:
            # i is rebalance date provided a xover
            sgnl_date = i
        
            ## Signal
            reb_sgnl = r['xover']
           
            ## Trade signal assessment
            reb_sigma = np.sqrt(exante_var_estimate(t=sgnl_date))
            reb_size = np.floor(0.40/reb_sigma)
            reb_pos_size = reb_size*reb_sgnl
            reb_px_enter = data.loc[i, 'Adj Close']
            df_strat_byXover.loc[i,['Signal','Pos_Size','Px_Enter']] = \
                [reb_sgnl,reb_pos_size,reb_px_enter]
        except:
            continue
    else: # Cannot trade
        continue
### Trades evaluation
tmpdf = pd.DataFrame()
for j,q in df_strat_byXover.iterrows():
    # Trade signal position
    curr_signal = q['Signal']
    # Closing date
    if not np.any(df_reb_events.index > j): # If still not closing date
        idx_close_date = data.index[-1] # last date available
    else:
        idx_close_date = df_reb_events.\
            index[pd.concat([df_reb_events['xover'] == curr_signal*-1, 
                             pd.Series(df_reb_events.index > j, 
                                       index=df_reb_events.index)], 
                            axis=1).all(axis=1)][0]
    # Closing price
    px_exit = data.loc[idx_close_date, 'Adj Close']
    closing_ret = (px_exit/q['Px_Enter']-1)*curr_signal
    
    # df fill
    tmpdf.loc[j,['Close_Date', 'Px_Exit', 'R']] = [idx_close_date, px_exit, closing_ret]
### update
df_strat_byXover = df_strat_byXover.merge(tmpdf,left_index=True, right_index=True) 
df_strat_byXover['PnL'] = df_strat_byXover.\
    apply(lambda x: 
          (x['Px_Exit'] - x['Px_Enter'])*x['Pos_Size']*x['Signal']*ct_pt_value, 
          axis=1)

# Results plot
np.exp(df_strat_byXover['R'].cumsum()).plot(title='Trend: T9M Sign\nReb: 1Mxo0')
dfMgr.statistics(df_strat_byXover[['R','PnL']]).T
df_strat_byXover['PnL'].cumsum().plot()

#%% STRAT: Trend filter + price xover signal /PRICE
# Trend is determined by the relative position of the short over the long EMA
# Signal is defined by price xover over the short EMA
from StratMan import StratMan

# Strategy
StratMgr = StratMan(data, data_trend)
StratMgr.set_strat()

# Strat backtest results
StratMgr.set_strat_results()
dic_strat_res = StratMgr.get_strat_results()

# Strat signals
dicSgnls = StratMgr.get_signals_data()

# Base data for strat
name = 'EFA'
df_signals = dicSgnls[name]
df_pxi = StratMgr.get_PricePlusIndicators_data(name)
# Valid entries
df_valid_entry = StratMgr.get_valid_entry(name)
# Backtest results
df_strat_bt = dic_strat_res[name](name)

# BT res for each asset strat
tmpdic = {}
for name in data.columns.levels[1]:
    tmpdic[name] = dic_strat_res[name](name)
    
# PnL of each asset
res = pd.DataFrame(index = data.columns.levels[1],
                   columns = ['TRet','avgRet','stdRet','avgPnL','avgDur',
                              'avgMFE','avgMAE','avgM2M_Mean','avgM2M_Std', 
                              'avgM2M_Skew'])
for name in data.columns.levels[1]:
    res.loc[name, 'TRet'] = (tmpdic[name]['PnL']/tmpdic[name]['Entry_price']+1).prod()-1
    res.loc[name, 'avgRet'] = (tmpdic[name]['PnL']/tmpdic[name]['Entry_price']).mean()
    res.loc[name, 'stdRet'] = (tmpdic[name]['PnL']/tmpdic[name]['Entry_price']).std()
    res.loc[name, 'avgPnL'] = tmpdic[name]['PnL'].mean()
    res.loc[name, 'avgDur'] = tmpdic[name]['Dur_days'].mean()
    res.loc[name, 'avgMFE'] = (tmpdic[name]['MaxGain(MFE)']/tmpdic[name]['Entry_price']).mean()
    res.loc[name, 'avgMAE'] = (tmpdic[name]['MaxLoss(MAE)']/tmpdic[name]['Entry_price']).mean()
    res.loc[name, 'avgM2M_Mean'] =(tmpdic[name]['M2M_Mean']/tmpdic[name]['Entry_price']).mean()
    res.loc[name, 'avgM2M_Std'] = (tmpdic[name]['M2M_Std']/tmpdic[name]['Entry_price']).std()
    res.loc[name, 'avgM2M_Skew'] = (tmpdic[name]['M2M_Skew']/tmpdic[name]['Entry_price']).mean()

###############################################################################
# Backtest run
bt_cols = ['Start','End','Signal','Entry_price','Exit_price', 'PnL',
           'Dur_days', 'MaxGain(MFE)', 'MaxLoss(MAE)', 'M2M_Mean', 'M2M_TMean',
           'M2M_IQR', 'M2M_Std', 'M2M_Skew', 'M2M_Med', 'M2M_Kurt']
df_strat_bt = pd.DataFrame(columns = bt_cols)
last_exit_day = df_valid_entry.index[0]
col_px = [(s, name) for s in ['Open', 'High', 'Low']]

for i,r in df_valid_entry.iterrows(): # i,r = list(df_valid_entry.iterrows())[0]
    # Valid entry day should be after last trade exit date
    if i < last_exit_day:
        continue
    # Check if signal is from today
    if df_pxi[i:].shape[0] < 1 or df_pxi[i:].index[-1] == i:
        # We actually are seeing this signal live. Trade at tomorrows open.
        break
    
    # Trade signal triggers a trade at next day
    entry_day = df_pxi[i:].iloc[1].name
    
    # Entry price at OHL mean price to account for price action
    entry_price = data.loc[entry_day, col_px].mean()
    
    # Trade is closed at next day when exit signal is triggered
    if r['Entry'] == 1: # Open position is long
        # Filter data by possible exits from a long position
        dates_valid_exit = df_pxi.index[df_signals['Exit_Long'] != 0]
        col_maxProfit = 'High'
        col_maxLoss = 'Low'
    else: # Open position is short
        # Filter out by short-only exit signals
        dates_valid_exit = df_pxi.index[df_signals['Exit_Short'] != 0]
        col_maxProfit = 'Low'
        col_maxLoss = 'High'
    
    # Valid exits 
    tmpdf_valid_exit = data.loc[dates_valid_exit, col_px].loc[i:]

    # Check if trade still open
    if tmpdf_valid_exit.shape[0] < 2:
        if tmpdf_valid_exit.shape[0] < 1:
            # Trade is still on, so possible exit price is on last observed day
            exit_day = data.iloc[-1].name
        else:
            exit_day = tmpdf_valid_exit.iloc[0].name
            
    else: # Exit signal triggered
        # Exit signal date
        date_exit_signal = tmpdf_valid_exit.iloc[0].name
        
        # Check if closing trade is tomorrow
        if data.loc[date_exit_signal:].shape[0] < 2:
            exit_day = data.loc[date_exit_signal:].iloc[0].name + \
                pd.tseries.offsets.BDay(1)
        else: 
            # Exit day following exit signal date
            exit_day = data.loc[date_exit_signal:, col_px].iloc[1].name
        
    # Exit price at OHL mean price to account for price action
    exit_price = data.loc[exit_day, col_px].mean()
    
    # Trade PnL
    pnl = (exit_price - entry_price)*r['Entry']
    
    # Trade duration
    dur_days = (exit_day - entry_day).days
    
    # Mark2Market PnL
    m2m_ = (data.loc[entry_day:exit_day, ('Close', name)] - entry_price)*r['Entry']
    
    # Mark-2-market stats
    m2m_stats = dfMgr.statistics(m2m_.to_frame())
    mfe = (max(data.loc[entry_day:exit_day, (col_maxProfit, name)]) - \
        entry_price)*r['Entry']
    mae = (min(data.loc[entry_day:exit_day, (col_maxLoss, name)]) - \
        entry_price)*r['Entry']

    # BT row data
    tmp_bt_row = [entry_day, exit_day, r['Entry'], entry_price, exit_price, 
                  pnl, dur_days, mfe, mae]+m2m_stats.to_numpy().tolist()[0]
    # Add obs to bt df
    df_strat_bt.loc[i] = tmp_bt_row
    
    # Update last exit day
    last_exit_day = exit_day

#%% DISTRIBUTION ANALYSIS
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

# Returns distribution
rhist = df_strat_byXover['R'].plot.hist(title='Xover Returns', density=True)
rhist = df_strat['R'].plot.hist(title='1M-Holding Returns', density=True)
recdf = ECDF(df_strat_byXover['R'])
recdf = ECDF(df_strat['R'])

#%% EMPIRICAL DISTRIBUTION SAMPLING

# function for random samples generator from empirical distribution
def empirical_sample(ecdf, size):
    u = np.random.uniform(0,1,size)
    ecdf_x = np.delete(ecdf.x,0)
    sample = []
    for u_y in u:
        idx = np.argmax(ecdf.y >= u_y)-1
        e_x = ecdf_x[idx]
        sample.append(e_x)
    return pd.Series(sample,name='emp_sample')

# function for paths generator from empirical distribution
def sim_path_R(ecdf, sample_size=1000, paths_size=1000):
    runs = []
    for n in range(paths_size):
        run = empirical_sample(ecdf,sample_size)
        run.name = run.name + f'{n+1}'
        runs.append(run.cumsum())
    df_runs = pd.concat(runs, axis = 1)
    df_runs.index = df_runs.index + 1
    df_runs = pd.concat([pd.DataFrame(np.zeros((1,paths_size)),
                                      columns=df_runs.columns),
               df_runs])
    return df_runs

#%% SAMPLING RETURNS PATHS

# Sim
N_sim = 10000
N_sample = df_strat_byXover.shape[0]
runs_myR = sim_path_R(recdf, sample_size=N_sample, paths_size=N_sim)
runs_myR.plot(title=f'Cumulative Returns Simulations\n'\
              f'N(paths)={N_sim}, N(sample) ={N_sample}',
              legend=False)
plt.grid(alpha=0.5, linestyle='--')
# Simulation results
simmean = runs_myR.mean(axis=1)
simstd = runs_myR.std(axis=1)
simqtl = runs_myR.quantile(q=(0.025,1-0.025),axis=1).T
simres = [simmean, simmean-2*simstd, simmean+2*simstd, 
          simqtl]
avgsim = pd.concat(simres,axis=1)
avgsim.columns = ['Avg Path','LB-2std','UB-2std',
                  f'{2.5}%-q',f'{97.5}%-q']
avgsim.plot(title='Cumulative Returns Simulations\nMean path\n'\
            f'N(paths)={N_sim}, N(sample) ={N_sample}', 
            style=['-','--','--','--','--'])
plt.grid(alpha=0.5, linestyle='--')

# Simulation stats
(runs_myR.mean()/runs_myR.std()).mean()
(runs_myR.mean()/runs_myR.std()).std()
(runs_myR.mean()/runs_myR.std()).plot.hist(title =\
                                'Sharpe Ratios\nCum. Returns Sims', 
                                density = True)

#%% POSITION SIZING

N_sim = 1000
N_sample = df_strat_byXover.shape[0]
dic_bal = {}
for n in range(N_sim):
    sim_r = empirical_sample(recdf,N_sample)
    init_capital = 100 
    pos_size_pct = 0.002
    pnl = np.array([])
    balance = np.array([init_capital])
    for r in sim_r:
        trade_pos_size = pos_size_pct*balance[-1]
        trade_pnl = r*trade_pos_size
        pnl = np.append(pnl,trade_pnl)
        balance = np.append(balance,balance[-1]+trade_pnl)
    dic_bal[f'run{n+1}'] = balance

# Equity curve sim results    
df_bal = pd.DataFrame(dic_bal)
df_bal.plot(title=f'Equity Curve Sim\n'\
              f'N(paths)={N_sim}, N(sample) ={N_sample}',
              legend=False)
plt.grid(alpha=0.5, linestyle='--')

# Sharpe ratios dist
sharpes = np.array((df_bal.mean()-100)/df_bal.std())
plt.hist(sharpes, density = True)
plt.title('Sharpe Ratios\nEquity Curve Sims')
plt.show()

# Sharpes stats
sharpes_mu = np.mean(sharpes)
sharpes_me = np.median(sharpes)
sharpes_mo = 3*sharpes_me - 2*sharpes_mu
(sharpes_mu, sharpes_me, sharpes_mo)
np.std(sharpes)
stats.mode(sharpes)

# Original method for mode
sharpes_histogram = np.histogram(sharpes)
L = sharpes_histogram[1][np.argmax(sharpes_histogram[0])]
f1 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])]
f0 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])-1]
f2 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])+1]
cls_i = np.diff(sharpes_histogram[1])[0]
mo = L + (f1-f0)/(2*f1-f0-f2)*cls_i