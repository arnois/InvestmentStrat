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
#%% DATA FROM XL FILE (BBG)
# Savepath
path = r'C:\Users\arnoi\Documents\db\eqty_bbg.parquet'
path = r'H:\db\eqty_bbg.parquet'

# Data pull
if os.path.isfile(path):
    # From saved db
    data = pd.read_parquet(path)
else:
    # Inception from xl file
    xlpath = r'C:\Users\jquintero\db\datapull_eqty_1D.xlsx'
    data = pd.read_excel(xlpath)
    tmpcols = data.iloc[4]; tmpcols[0] = 'Date'; tmpcols = tmpcols.tolist()
    tmpdf = data.iloc[7:]; tmpdf.columns = tmpcols
    tmpdf = tmpdf.set_index('Date').astype(float)
    # Save db
    tmpdf.to_parquet(path)
    data = tmpdf
    
# Data update
last_date_data = data.index[-1]
todays_date = pd.Timestamp(dt.datetime.today().date()) - pd.tseries.offsets.\
    BDay(1)
isDataOOD = last_date_data < todays_date
if isDataOOD:
    print("Updating data... ")
    # date range for new data
    str_idt = last_date_data.strftime("%Y-%m-%d")
    str_fdt = todays_date.strftime("%Y-%m-%d")
    # new data
    xlpath_update = r'C:\Users\jquintero\db\datapull_eqty_1D.xlsx'
    new_data = pd.read_excel(xlpath_update)
    tmpcols = new_data.iloc[4]; tmpcols[0] = 'Date'; tmpcols = tmpcols.tolist()
    tmpdf = new_data.iloc[7:]; tmpdf.columns = tmpcols
    tmpdf = tmpdf.set_index('Date').astype(float)
    # updated data
    updated_data = pd.concat([data, tmpdf.loc[str_idt:str_fdt]]).\
        reset_index().drop_duplicates('Date').set_index('Date')
    print("Saving data...")
    updated_data.to_parquet(path)
    data = pd.read_parquet(path)
    
#%% TREND ANALYSIS
nes = 21
nel = 64

# Smoothed data
data_ema = data.\
    apply(lambda s: ta.ema(s, nes)).\
    rename(columns=dict(zip(data.columns,
                            [f'{s}_EMA_ST' for s in data.columns]))).merge(
    data.\
        apply(lambda s: ta.ema(s, nel)).\
        rename(columns=dict(zip(data.columns,
                                [f'{s}_EMA_LT' for s in data.columns]))),                      
    left_index=True, right_index=True)

# Trend + dynamics
data_ta = pd.DataFrame()
for name in data.columns:
    # Short EMA
    tmp_sema = data_ema[f'{name}_EMA_ST']
    
    # Long EMA
    tmp_lema = data_ema[f'{name}_EMA_LT']
    
    # Trend strength
    tmp_emadiff = (tmp_sema - tmp_lema)
    tmp_strength = pd.Series('normal',index=tmp_emadiff.index)
    tmp_strength[abs(tmp_emadiff) >= tmp_emadiff.std()] = 'strong'
    tmp_strength[tmp_emadiff <= tmp_emadiff.std()/2] = 'weak'
    
    # Trend status; 5-EMA of the RoC of the EMA difference
    tmp_status = tmp_emadiff.diff().rename('Close').to_frame().ta.ema(5)
    
    # Trend specs
    data_ta = pd.\
        concat([data_ta, 
                tmp_emadiff.apply(np.sign).rename(f'{name}_trend').to_frame(),
                tmp_strength.rename(f'{name}_trend_strength').to_frame(),
                tmp_status.rename(f'{name}_trend_status').to_frame(),
                tmp_emadiff.rename(f'{name}_ema_d').to_frame()], 
               axis=1)    
    
# Check any assets TA
name = 'IWM'
namecol = [f'{name}_trend',f'{name}_trend_strength',
           f'{name}_trend_status',f'{name}_ema_d']
data_ta[namecol].iloc[-21:].merge(
    data.loc[data_ta[namecol].iloc[-21:].index][name]
    , left_index=True, right_index=True)

# Assets trend stengthness by ST-LT EMA Differences
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Trend strength - as a smoothed short-long EMA difference
X_strength = data_ta.loc['2004-03-29':,[f'{s}_ema_d' for s in data.columns]]

# Z-scaled strength
data_strength_Z = pd.DataFrame(sc.fit_transform(X_strength), 
             columns = data.columns, index=X_strength.index).fillna(0).abs()

# Strength-based weights
data_w = data_strength_Z.apply(lambda x: x/x.sum(), axis=1)

# Filter out non-weak trends
nonwTrends = []
for name in data.columns:
    # Trend strength
    tmpcol = f'{name}_trend_strength'
    if data_ta.iloc[-1][f'{name}_trend_strength'] == 'weak':
        continue
    else:
        nonwTrends.append(name)

# Non-weak Trending Assets
data_ta.iloc[-1][[f'{c}_trend' for c in nonwTrends]]

#%% DATA FROM YFINANCE
# Ticker list
path_tkr_list = r'C:\Users\arnoi\Documents\Python Scripts\TrendFollowing'
fname_tkr_list = r'\tickers_etf_yf.txt'
yh_tkrlst = pd.read_csv(path_tkr_list+fname_tkr_list, 
                        sep=",", header=None).to_numpy()[0].tolist()

# for demo purposes we are using ZN (10Y T-Note) futures contract

# Dates range
str_idt, str_fdt = '2000-12-31', '2024-04-06'

# Futures symbols
#yh_tkrlst = ['ZT=F'] # ZT=F, ZF=F, ZN=F, TN=F

# Data pull
path = r'C:\Users\arnoi\Documents\db\etf_yf.parquet'
path = r'H:\db\etf_yf.parquet'
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

# Data columns mgmt
data.rename(columns={'BTC-USD':'BTC'}, level=1, inplace=True)

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
## Plotting Price Levels
fig, ax = plt.subplots()
tmpetf = ['GLD','XLY','XLI','XLV']
tmptkrs = [('Adj Close', s) for s in tmpetf]
ax.plot(data.loc['2024':,tmptkrs].dropna(), '-')
myFmt = mdates.DateFormatter('%d%b')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('')
ax.set_ylabel('Price')
ax.set_title('ETFs')
ax.legend(tmpetf, loc='best', bbox_to_anchor=(1.01,1.01))
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()

#%% PRICE TREND INDICATOR
nes = 21
nel = 64

# Smoothed data over closing prices
data_ema = data['Adj Close'].dropna().\
    apply(lambda s: ta.ema(s, nes)).\
    rename(columns=dict(zip(data['Adj Close'].columns,
                            [f'{s}_EMA_ST' for s in data['Adj Close'].columns]))).merge(
    data['Adj Close'].dropna().\
        apply(lambda s: ta.ema(s, nel)).\
        rename(columns=dict(zip(data['Adj Close'].columns,
                                [f'{s}_EMA_LT' for s in data['Adj Close'].columns]))),                      
    left_index=True, right_index=True)
                                    
# Plot
tmpetf = 'GLD'
tmptkr = ('Adj Close', tmpetf)
t1y_date = todays_date - pd.tseries.offsets.DateOffset(years=1)
fig, ax = plt.subplots()
ax.plot(data.loc[t1y_date:,tmptkr].dropna(),c='darkcyan')
ax.plot(data_ema.loc[t1y_date:,tmpetf+'_EMA_ST'],'--', c='C0')
ax.plot(data_ema.loc[t1y_date:,tmpetf+'_EMA_LT'],'--',c='orange')
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
data_logP = data[['Adj Close']].apply(np.log)

# Daily returns
data_ret = data_logP.diff().fillna(0)
data_ret['EMA_ST'] = data_ret[['Adj Close']].\
    rename(columns={'Adj Close':'Close'}).ta.ema(length=21)
data_ret['EMA_LT'] = data_ret[['Adj Close']].\
    rename(columns={'Adj Close':'Close'}).ta.ema(length=50)

# Cum return index
data_idx = data_ret.cumsum().apply(np.exp)
clrlst = ['darkcyan','blue','orange','red']
data_idx.loc[t1y_date:].plot(color=clrlst, 
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
data_ret_sigma.loc[t1y_date:].\
    plot(title=f'{yh_tkrlst[0][:2]} Volatility', alpha=0.65)

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
statistics(df_strat[['R','PnL']]).T
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
statistics(df_strat_byXover[['R','PnL']]).T
df_strat_byXover['PnL'].cumsum().plot()

#%% STRAT: Trend filter + price xover signal /PRICE
# Trend is determined by the relative position of the short over the long EMA
# Signal is determined by the price xover the short EMA

# Base data for strat
name = 'IWM'
tmpcols = [f'{name}_EMA_{s}' for s in ['ST','LT']]
df_strat = data_ema[tmpcols]

# Temporal price+indicator info
df_tmp = data[[name]].merge(
    data[[name]].shift().rename(columns={name:f'{name}_1'}), 
    left_index=True, right_index=True).\
    merge(data_ema[tmpcols], left_index=True, right_index=True)

# Entry signals
df_signals = pd.DataFrame(index = df_tmp.index)
df_signals['Trend'] = 0
df_signals['Entry'] = 0
df_signals['Exit_Long'] = 0
df_signals['Exit_Short'] = 0

# Trend signals
df_signals['Trend'] += data_ta[f'{name}_trend'].fillna(0).astype(int)
    
# Buy signals
df_signals['Entry'] += df_tmp.apply(lambda s: 
    (s[name+'_1'] < s[name+'_EMA_ST'])*(s[name] > s[name+'_EMA_ST'])*1, axis=1)

# Sell signals
df_signals['Entry'] += df_tmp.apply(lambda s: 
    (s[name+'_1'] > s[name+'_EMA_ST'])*(s[name] < s[name+'_EMA_ST'])*-1, axis=1)
    
# Exit longs
df_signals['Exit_Long'] += df_tmp.apply(lambda x: 
    (x[name+'_1'] > x[name+'_EMA_LT'])*(x[name] < x[name+'_EMA_LT'])*1, axis=1)

# Exit shorts
df_signals['Exit_Short'] += df_tmp.apply(lambda x: 
    (x[name+'_1'] < x[name+'_EMA_LT'])*(x[name] > x[name+'_EMA_LT']), axis=1)

# Valid entries
df_valid_entry = ((df_signals['Trend']*df_signals['Entry'] > 0)*df_signals['Entry']).\
    rename('Entry').to_frame()
df_valid_entry = df_valid_entry[df_valid_entry['Entry'] != 0]

df_tmp.loc[df_valid_entry.index[0]-pd.Timedelta(days=3):df_valid_entry.index[0]+pd.Timedelta(days=1)]

###############################################################################
# Backtest run
bt_cols = ['Start','End','Signal','Entry_price','Exit_price', 'PnL',
           'Dur_days', 'MaxGain(MFE)', 'MaxLoss(MAE)', 'M2M_Mean', 'M2M_TMean',
           'M2M_IQR', 'M2M_Std', 'M2M_Skew', 'M2M_Med', 'M2M_Kurt']
df_strat_bt = pd.DataFrame(columns = bt_cols)
last_exit_day = df_valid_entry.index[0]
for i,r in df_valid_entry.iterrows():
    # Valid entry day should be after last trade exit date
    if i < last_exit_day:
        continue
    # Check if signal is from today
    if df_strat[i:].shape[0] < 1:
        # We actually are seeing this singal live. Trade at tomorrows open.
        break
    
    # Trade signal triggers a trade at next day
    entry_day = df_strat[i:].iloc[1].name
    
    # Entry price at OHL mean price to account for price action
    entry_price = data.loc[entry_day, ['Open', 'High', 'Low']].mean()
    
    # Trade is closed at next day when exit signal is triggered
    if r['Signal'] == 1: # Open position is long
        # Filter data by possible exits from a long position
        dates_valid_exit = df_strat.index[df_strat['Exit_Long'] != 0]
        col_maxProfit = 'High'
        col_maxLoss = 'Low'
    else: # Open position is short
        # Filter out by short-only exit signals
        dates_valid_exit = df_strat.index[df_strat['Exit_Short'] != 0]
        col_maxProfit = 'Low'
        col_maxLoss = 'High'
    
    # Valid exits 
    tmpdf_valid_exit = data.loc[dates_valid_exit].loc[i:]

    # Check if trade still open
    if tmpdf_valid_exit.shape[0] < 2:
        # Trade is still on, so possible exit price is on last observed day
        exit_day = data.iloc[-1].name
    else: # Exit signal triggered
        # Exit signal date
        date_exit_signal = tmpdf_valid_exit.iloc[0].name
        
        # Check if closing trade is tomorrow
        if data.loc[date_exit_signal:].shape[0] < 2:
            exit_day = data.loc[date_exit_signal:].iloc[0].name + \
                pd.tseries.offsets.BDay(1)
        else: 
            # Exit day following exit signal date
            exit_day = data.loc[date_exit_signal:].iloc[1].name
        
    # Exit price at OHL mean price to account for price action
    exit_price = data.loc[exit_day][['Open', 'High', 'Low']].mean()
    
    # Trade PnL
    pnl = (exit_price - entry_price)*r['Signal']
    
    # Trade duration
    dur_days = (exit_day - entry_day).days
    
    # Mark2Market PnL
    m2m_ = (data.loc[entry_day:exit_day, 'Close'] - entry_price)*r['Signal']
    
    # Mark-2-market stats
    m2m_stats = statistics(m2m_.to_frame())
    mfe = (max(data.loc[entry_day:exit_day, col_maxProfit]) - \
        entry_price)*r['Signal']
    mae = (min(data.loc[entry_day:exit_day, col_maxLoss]) - \
        entry_price)*r['Signal']

    # BT row data
    tmp_bt_row = [entry_day, exit_day, r['Signal'], entry_price, exit_price, 
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