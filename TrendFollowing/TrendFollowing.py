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

#%% UDF
# function to get statistics for different variables in a dataframe
def statistics(data):
    """
    Returns: Summary stats for each serie in the input DataFrame.
    """
    from scipy import stats
    tmpmu = np.round(data.mean().values,4)
    tmptmu_5pct = np.round(stats.trim_mean(data, 0.05),4)
    tmpiqr = np.round(data.apply(stats.iqr).values,4)
    tmpstd = np.round(data.std().values,4)
    tmpskew = np.round(data.apply(stats.skew).values,4)
    tmpme = np.round(data.median().values,4)
    tmpkurt = np.round(data.apply(stats.kurtosis).values,4)
    tmpcol = ['Mean', '5% Trimmed Mean', 'Interquartile Range', 
              'Standard Deviation', 'Skewness', 'Median', 'Kurtosis']
    tmpdf = pd.DataFrame([tmpmu, tmptmu_5pct, tmpiqr,
                        tmpstd, tmpskew, tmpme, tmpkurt]).T
    tmpdf.columns = tmpcol
    tmpdf.index = data.columns
    return tmpdf

# function to manage row and column ranges
def sliceDataFrame(df, dt_start = None, dt_end = None, lst_securities = None):
    # columns verification
    lstIsEmpty = (lst_securities is None) or (lst_securities == [])
    if lstIsEmpty:
        tmpsecs = df.columns.tolist()
    else:
        tmpsecs = lst_securities
    
    # column-filtered df
    tmpdfret = df[tmpsecs]

    # date range verification
    if (dt_start is None) or not (np.any(tmpdfret.index >= pd.to_datetime(dt_start))):
        tmpstr1 = df.index.min()
    else:
        tmpstr1 = dt_start
    if (dt_end is None) or not (np.any(tmpdfret.index >= pd.to_datetime(dt_end))):
        tmpstr2 = df.index.max()
    else:
        tmpstr2 = dt_end
        
    return tmpdfret.loc[tmpstr1:tmpstr2,tmpsecs].dropna()

# funciton to plot correlation matrix among features
def plot_corrmatrix(df, dt_start = None, dt_end = None, lst_securities = None, 
                    plt_size = (10,8), txtCorr = False, corrM = 'spearman'):
    """
    Returns: Correlation matrix.
    """
    from seaborn import heatmap
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    tmpdfret = df_
    
    # corrmatrix
    plt.figure(figsize=plt_size)
    heatmap(
        tmpdfret.corr(method=corrM.lower()),        
        cmap='RdBu', 
        annot=txtCorr, 
        vmin=-1, vmax=1,
        fmt='.2f')
    plt.title(corrM.title(), size=20)
    plt.tight_layout();plt.show()
    return None

#%% DATA
# for demo purposes we are using ZN (10Y T-Note) futures contract

# Dates range
str_idt, str_fdt = '2000-12-31', '2023-12-31'

# Futures symbols
yh_tkrlst = ['ZT=F'] # ZT=F, ZF=F

# Data pull
path = r'C:\Users\arnoi\Documents\Python Scripts\db\futures_'+yh_tkrlst[0][:2]+'_yf.parquet'
path = r'H:\db\futures_'+yh_tkrlst[0][:2]+'_yf.parquet'
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
todays_date = pd.Timestamp(dt.datetime.today().date()) - pd.tseries.offsets.\
    BDay(1)
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

#%% PRICE VIZ
## Plotting T3Y Price
fig, ax = plt.subplots()
ax.plot(data.loc['2024':,'Adj Close'], '-', color='darkcyan')
myFmt = mdates.DateFormatter('%d%b')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('')
ax.set_ylabel('Price')
ax.set_title(f'{yh_tkrlst[0][:2]} Future')
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()

#%% RETURNS
# Daily log-prices
data_logP = data[['Adj Close']].apply(np.log)

# Daily returns
data_ret = data_logP.diff().fillna(0)
ema_alpha = 1-60/61
data_ret['EMA'] = data_ret['Adj Close'].ewm(span=16, adjust=False).mean()

# Cum return index
data_idx = data_ret.cumsum().apply(np.exp)
t1y_date = todays_date - pd.tseries.offsets.DateOffset(years=1)
data_idx.loc['2024':].plot(color=['darkcyan','orange'], 
                           title=f'{yh_tkrlst[0][:2]} Returns Idx')
data_idx.loc[t1y_date:].plot(color=['darkcyan','orange'], 
                             title=f'{yh_tkrlst[0][:2]} Returns Idx')

# Volatility
data_ret_sigma = (data_ret[['Adj Close']].\
                  rolling(21).std()*np.sqrt(252)).dropna()
data_ret_sigma = (data_ret[['Adj Close']].\
                  ewm(span=252, adjust=False).std()*np.sqrt(252)).\
    iloc[252:].rename(columns={'Adj Close':'EMA'}).merge(data_ret_sigma, 
                                                         left_index=True,
                                                         right_index=True)
statistics(data_ret_sigma).T
data_ret_sigma.loc['2023-02-21':].plot(title=f'{yh_tkrlst[0][:2]} Volatility', alpha=0.65)

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
plot_corrmatrix(df_ret_TvF, dt_start = '2021-02-20', dt_end = '2024-02-20', 
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

#%% STRAT: Trend signal + rebalance period by holding time

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
statistics(df_strat[['R','PnL']]).T
df_strat['PnL'].cumsum().plot()

#%% STRAT: Trend filter + Xover signal

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
statistics(df_strat_byXover[['R','PnL']]).T
df_strat_byXover['PnL'].cumsum().plot()

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