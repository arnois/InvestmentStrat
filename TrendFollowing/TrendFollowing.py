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
import sys
#%% WORKING DIRECTORY
str_cwd = r'H:\Python\TrendFollowing' 
if not os.path.exists(str_cwd):
    # Current working path
    cwdp = os.getcwd()
    cwdp_parent = os.path.dirname(cwdp)
    sys.path.append(cwdp)
else:
    sys.path.append(str_cwd)
#%% CUSTOM MODULES
from DataMan import DataMan
from TrendMan import TrendMan
#%% DATA
aclss = 'fut' # etf, eqty, fut
# Path
path = rf'H:\db\{aclss}_bbg.parquet'
if not os.path.exists(path):    path = cwdp_parent+rf'\{aclss}_bbg.parquet' 

# Data Manager
dfMgr = DataMan(datapath=path)

# Data
dfMgr.set_data_from_path()
data = dfMgr.get_data()

# Data subset
## ['TLT','GLD','IWM','URA','USO']
## ['CORN','COPPER','R10Y','MXN','SP500']
## ['SBUX','PLUG','WMT','MSFT']
selSec = ['CORN','COPPER','R10Y','MXN','SP500']
data_slice = dfMgr.sliceDataFrame(dt_start='2022-12-31', 
                                  dt_end='2024-03-31', 
                                  lst_securities=selSec)
# Data stats
print(dfMgr.statistics(data_slice.apply(np.log).diff().dropna()['Close']).T)

# Correlation matrix between selected assets
dfMgr.plot_corrmatrix(data_slice['Close'].diff(), txtCorr=True)

# Data update
try:
    xlpath_update = rf'C:\Users\jquintero\db\datapull_{aclss}_1D.xlsx'
    dfMgr.update_data(xlpath = xlpath_update)
    data = dfMgr.get_data()
except FileNotFoundError:
    print('\nData for updates not found!')

# Asset names
names = np.array([s for s in data.columns.levels[1] if s != ''])
exit()
#%% TREND
nes, nel = 21, 64
TrendMgr = TrendMan(nes, nel)
TrendMgr.set_trend_data(data,'Close')
data_trend = TrendMgr.get_trend_data()

# Check any assets TA
name = 'RUSSELL' # SBUX, R10Y, IWM, RUSSELL
namecol = [f'{name}_trend',f'{name}_trend_strength',
           f'{name}_trend_status',f'{name}_ema_d']

# Trend + dynamics
tmpTD = data_trend[namecol].iloc[-21:].merge(
    data.loc[data_trend[namecol].iloc[-21:].index]['Close'][name]
    , left_index=True, right_index=True)
print(f'\n{name} Trend Dynamics:\n{tmpTD}')
    
#%% TREND ANALYSIS
last4Mo = np.arange(-1,-28*4,-27).tolist()
# TrendStrength-based weigths
data_strength_Z = TrendMgr.get_trend_strength_Z()
data_w = data_strength_Z.apply(lambda x: x/x.sum(), axis=1)
data_w.columns = [s.replace('_ema_d','') for s in data_w.columns]
print(f"Last W's:\n{round(100*data_w.iloc[last4Mo].T,0)}\n")

# Long-only subcase
data_w_lOnly = (((data_strength_Z.apply(np.sign)+1)/2).\
    fillna(0)*data_strength_Z).apply(lambda x: x/x.sum(), axis=1)
data_w_lOnly.columns = [s.replace('_ema_d','') for s in data_w_lOnly.columns]
tmpLOP = round(100*data_w_lOnly.iloc[last4Mo].T,0)
print(f"Last Long-Only W's:\n{tmpLOP.loc[~(tmpLOP == 0).all(axis=1),]}\n")

# Filter out non-weak trends
nonwTrends = []
for name in names:
    # Trend strength
    tmpcol = f'{name}_trend_strength'
    if data_trend.iloc[-1][f'{name}_trend_strength'] == 'weak':
        continue
    else:
        nonwTrends.append(name)

# Non-weak Trending Assets
print(f"\nNon-Weak Trends\n{data_trend.iloc[-1][[f'{c}_trend' for c in nonwTrends]]}")

#%% PRICE & TREND VIZ
es_tkrs = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

# Todays date
todays_date = pd.Timestamp.today().date() 

## Plotting Price Levels
tmpetf = ['CORN','COPPER','R10Y','MXN','RUSSELL'] # ['WMT','XOM','DIS','SBUX'], ['GLD','COPX','URA','IWM'], ['CORN','COPPER','R10Y','MXN','SP500']
fig, ax = plt.subplots(figsize=(9,7))
tmptkrs = [('Close', s) for s in tmpetf]
tmpdf = data.loc['2024':,tmptkrs].dropna()/data.loc['2024':,tmptkrs].dropna().iloc[0]*100
ax.plot(tmpdf, '-')
myFmt = mdates.DateFormatter('%d%b')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('')
ax.set_ylabel('Index')
ax.set_title('Asset Cumulative Returns')
ax.legend(tmpetf, loc='best', bbox_to_anchor=(1.01,1.01))
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()
                                    
# Single Plot
tmpetf = tmpetf[-1]
tmptkr = ('Close', tmpetf)
t1y_date = todays_date - pd.tseries.offsets.DateOffset(years=1)
fig, ax = plt.subplots(figsize=(13,7))
ax.plot(data.loc[t1y_date:,tmptkr].dropna(),c='darkcyan')
ax.plot(data_trend.loc[t1y_date:,tmpetf+'_EMA_ST'],'--', c='C0')
ax.plot(data_trend.loc[t1y_date:,tmpetf+'_EMA_LT'],'--',c='orange')
myFmt = mdates.DateFormatter('%d%b%Y')
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('')
ax.set_ylabel('Price')
ax.set_title(f'{tmpetf}')
ax.legend(['Price','Fast EMA', 'Slow EMA'], 
          loc='best', bbox_to_anchor=(1.01,1.01))
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()
##
print(pd.concat([data[tmptkr],data_trend[[tmpetf+'_EMA_ST',tmpetf+'_EMA_LT']]],axis=1).iloc[-21:])
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

# BT res for each asset strat
tmpdic = {}
for name in names:
    print(f'\n BT Results for {name}...')
    tmpdic[name] = dic_strat_res[name](name)
    
# PnL of each asset
res = pd.DataFrame(index = data.columns.levels[1],
                   columns = ['TRet','avgRet','Acc','stdRet','avgDur',
                              'avgMFE','avgMAE','avgM2M_Mean','avgM2M_Std', 
                              'avgM2M_Skew'])
for name in names:
    res.loc[name, 'TRet'] = (tmpdic[name]['PnL']/tmpdic[name]['Entry_price']+1).prod()-1
    res.loc[name, 'avgRet'] = (tmpdic[name]['PnL']/tmpdic[name]['Entry_price']).mean()
    res.loc[name, 'Acc'] = tmpdic[name][tmpdic[name]['PnL']>0].shape[0]/tmpdic[name].shape[0]
    res.loc[name, 'stdRet'] = (tmpdic[name]['PnL']/tmpdic[name]['Entry_price']).std()
    res.loc[name, 'avgDur'] = tmpdic[name]['Dur_days'].mean()
    res.loc[name, 'avgMFE'] = (tmpdic[name]['MaxGain(MFE)']/tmpdic[name]['Entry_price']).mean()
    res.loc[name, 'avgMAE'] = (tmpdic[name]['MaxLoss(MAE)']/tmpdic[name]['Entry_price']).mean()
    res.loc[name, 'avgM2M_Mean'] =(tmpdic[name]['M2M_Mean']/tmpdic[name]['Entry_price']).mean()
    res.loc[name, 'avgM2M_Std'] = (tmpdic[name]['M2M_Std']/tmpdic[name]['Entry_price']).std()
    res.loc[name, 'avgM2M_Skew'] = (tmpdic[name]['M2M_Skew']/tmpdic[name]['Entry_price']).mean()

# Cum returns
cumRet = pd.DataFrame()
for name in names:
    cumRet = pd.concat([cumRet, tmpdic[name].\
                          apply(lambda x: 1+x['PnL']/x['Entry_price'], 
                                axis=1).cumprod().rename(name).to_frame()],
                 axis=1)

df_RetIdx = cumRet.ffill().fillna(1)*100
if 'BTC' in df_RetIdx:
    df_RetIdx[df_RetIdx.columns[df_RetIdx.iloc[-1] >= 100]].drop('BTC', axis=1).\
        plot(title='BT Cumulative Returns', figsize=(12,10)).\
            legend(loc='center left', bbox_to_anchor=(1.0, 0.5)); plt.show()
else:
    df_RetIdx[df_RetIdx.columns[df_RetIdx.iloc[-1] >= 100]].\
        plot(title='BT Cumulative Returns', figsize=(12,10)).\
            legend(loc='center left', bbox_to_anchor=(1.0, 0.5)); plt.show()


# Top performers
names_top = res['TRet'].sort_values(ascending=False)[:5].index.\
    insert(-1,res['avgRet'].sort_values(ascending=False)[:5].index).unique()

print(f'\nTop Performers\n {res.loc[names_top].T}')

# Strat signals
dicSgnls = StratMgr.get_signals_data()

# Base data for strat
name = 'R10Y' # 'SBUX', 'EEM', 'R10Y'
df_signals = dicSgnls[name]
df_pxi = StratMgr.get_PricePlusIndicators_data(name)
# Valid entries
df_valid_entry = StratMgr.get_valid_entry(name)
# Backtest results
df_strat_bt = dic_strat_res[name](name)

#%% DISTRIBUTION ANALYSIS
from statsmodels.distributions.empirical_distribution import ECDF

# Returns distribution
bt_ret = df_strat_bt['PnL']/df_strat_bt['Entry_price']
rhist = bt_ret.plot.hist(title=f'Xover Returns\n{name}', 
                         density=True); plt.show()
recdf = ECDF(bt_ret)
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
N_sample = bt_ret.shape[0]
runs_myR = sim_path_R(recdf, sample_size=N_sample, paths_size=N_sim)
runs_myR.plot(title=f'Cumulative Returns Simulations\n'\
              f'N(paths)={N_sim}, N(sample) ={N_sample}',
              legend=False)
plt.grid(alpha=0.5, linestyle='--');plt.show()
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
plt.grid(alpha=0.5, linestyle='--');plt.show()

# Simulation stats
(runs_myR.mean()/runs_myR.std()).mean()
(runs_myR.mean()/runs_myR.std()).std()
(runs_myR.mean()/runs_myR.std()).plot.hist(title =\
                                'Sharpe Ratios\nCum. Returns Sims', 
                                density = True);plt.show()

#%% POSITION SIZING

N_sim = 1000
N_sample = bt_ret.shape[0]
dic_bal = {}
init_capital = 100 
pos_size_pct = 0.005
for n in range(N_sim):
    sim_r = empirical_sample(recdf,N_sample)
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
              f'Pos Size={pos_size_pct:,.2%}',
              legend=False)
plt.grid(alpha=0.5, linestyle='--'); plt.show()

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
# stats.mode(sharpes)

# Original method for mode
sharpes_histogram = np.histogram(sharpes)
L = sharpes_histogram[1][np.argmax(sharpes_histogram[0])]
f1 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])]
f0 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])-1]
f2 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])+1]
cls_i = np.diff(sharpes_histogram[1])[0]
mo = L + (f1-f0)/(2*f1-f0-f2)*cls_i