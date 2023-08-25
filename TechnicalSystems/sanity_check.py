#%% DESCRIPTION
# The purpose of this module is to test all classes and functions. 
# Also, develop the logic behind the code.

#%%
# Repo Mgmt
import sys
import os
str_cwd = r'H:\Python\KaxaNuk\Open_Investment-Strategies-Module-main\TechnicalSystems'
sys.path.append(str_cwd)
# Libraries 
import pandas as pd
from Formulas_ToolKit import FinancialIndicators, sep, mpf
import Backtest_ToolKit as bk
from Formulas_ToolKit import macd_mfi_plot, macd_rsi_plot, entry_plot, inter_graph
import fmpmodule as fmp
# %matplotlib inline
#%%
'First, let´s read our parameters'
os.chdir(str_cwd)
params = pd.read_excel('parameters.xlsx', index_col=0)
tickers = pd.read_excel('parameters.xlsx', sheet_name='Stocks')
tickers = tickers['Tickers'].to_list()

initial_capital = params['Values'].loc['Initial_capital']
commission_percentage = params['Values'].loc['Commission']
variable_commission = params['Values'].loc['Variable_Commission']
dollar_amount = params['Values'].iloc[-1]
start_date = pd.to_datetime(params['Values'].loc['Start_date'])
end_date = pd.to_datetime(params['Values'].loc['End_date'])
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')
#%%
'Read our data'

nvda = fmp.daily_prices('NVDA', start_date, end_date)
nvda.columns = [col.capitalize() for col in nvda.columns]
nvda = nvda.loc[:, 'Open':'Volume']
nvda = nvda.sort_index()
#%%
'Let´s test some strategies from our module'
analysis2 = FinancialIndicators(nvda)  # Save the object
vpvmas = pd.concat([analysis2.VPVMA(12, 26), analysis2.VPVMAS(12, 26, 9)], axis=1)
vpvmas.loc['2018-12-20':'2019-03-22',:].plot(legend=False, style=['-','--'])

#%% VPVMA SIGNAL

vpvmas_sgnl = analysis2.vpvma_signal()
vpvmas_sgnl2 = analysis2.vpvma_signal2()
analysis2.plot_signals(start_date='2018-12-21', end_date='2019-02-11')

# plotting
start_date, end_date, stock_name = '2018-12-21', '2019-06-31', 'NVDA'
ti_vpv = vpvmas.rename({0:'VPV',1:'VPVMA'},axis=1).merge(vpvmas_sgnl2,left_index=True,right_index=True)
ti_vpv = sep(ti_vpv,start_date,end_date)
close = sep(nvda,start_date,end_date)
ti_vpv.index = pd.to_datetime(ti_vpv.index)
close.index = pd.to_datetime(close.index)

buy_dates = ti_vpv[ti_vpv['Signal'] == 1].index
sell_dates = ti_vpv[ti_vpv['Signal'] == -1].index

buy_dates_str = [date.strftime("%Y-%m-%d") for date in buy_dates]
sell_dates_str = [date.strftime("%Y-%m-%d") for date in sell_dates]
all_dates = buy_dates_str + sell_dates_str
all_colors = ['g']*len(buy_dates_str) + ['r']*len(sell_dates_str)

indicators  = [    
    mpf.make_addplot(ti_vpv['VPV'], color='#606060', panel=1, ylabel='VPV', secondary_y=False),
    mpf.make_addplot(ti_vpv['VPVMA'], color='#1f77b4', panel=1, secondary_y=False, linestyle = '--')     
    ]

fig, axlist = mpf.plot(close,
                     type='candle',
                     volume=False,
                     addplot=indicators,
                     style =  'classic',
                     xrotation = 0,
                     datetime_format = '%Y-%m-%d',
                     figsize = (40,20),
                     returnfig = True,
                     #mav = (20,55),
                     vlines=dict(vlines=all_dates, colors=all_colors, 
                                 linewidths=1, alpha=0.25)      
                     )

fig.suptitle(str(stock_name) +' from ' + str(start_date) + ' to ' +str(end_date),y=.90,fontsize=20,  x=0.59)
for i, ax in enumerate(axlist):
    ylabel = ax.get_ylabel()
    if i % 2 == 0:  
        ax.yaxis.tick_left()
        ax.set_ylabel(ylabel, fontsize=17, labelpad=20)
    else:  
        # ax.yaxis.tick_right()
        ax.set_ylabel(ylabel, fontsize=17, labelpad=20)
        

fig.show()
#%% RSI MACD
rsi_nvda = analysis2.rsi(14, False)
rsi_nvda = rsi_nvda.to_frame(name='RSI')
macd_nvda = analysis2.MACD(12, 26, 9)
macd_rsi_nvda = analysis2.macd_rsi_signal(12, 26, 9, 35, 70)
macd_mfi_nvda = analysis2.macd_mfi_signal(12, 26, 9, 25, 70)
mfi_nvda = analysis2.mfi(14)
mfi_nvda = mfi_nvda.to_frame(name='MFI')
#%%
'Let´s check the signals (NVDA)'

values_lower_zero_rsi_nvda = macd_rsi_nvda[macd_rsi_nvda['Signal'] < 0]
values_bigger_zero_rsi_nvda = macd_rsi_nvda[macd_rsi_nvda['Signal'] > 0]

values_lower_zero_mfi_nvda = macd_mfi_nvda[macd_mfi_nvda['Signal'] < 0]
values_bigger_zero_mfi_nvda = macd_mfi_nvda[macd_mfi_nvda['Signal'] > 0]

subset_nvda = macd_nvda[macd_nvda['MACD'] > macd_nvda['Signal_Line']]
#%%
'Plot'

start_date_nvda_rsi = '2004-03-01'
end_date_nvda_rsi = '2004-12-30'

start_date_nvda_mfi = '2002-05-10'
end_date_nvda_mfi = '2003-07-01'

macd_rsi_plot(rsi_nvda, macd_nvda, macd_rsi_nvda, nvda, start_date_nvda_rsi, end_date_nvda_rsi, 'NVDA')
macd_mfi_plot(mfi_nvda, macd_nvda, macd_mfi_nvda, nvda, start_date_nvda_mfi, end_date_nvda_mfi, 'NVDA')

entry_plot('NVDA', macd_rsi_nvda, nvda)
inter_graph(nvda, macd_rsi_nvda, 'NVDA')

#%%

'Build our dataframe for Backtest'

dic_prices = {}

for i in range(len(tickers)):
    dic_prices[tickers[i]] = fmp.daily_prices(tickers[i], start_date, end_date)

latest_df = max(dic_prices, key=lambda x: dic_prices[x].index.min())
latest_start_date = dic_prices[latest_df].index.min()

for key in dic_prices:
    df = dic_prices[key]
    df = df.loc[df.index >= latest_start_date]
    df.columns = [col.capitalize() for col in df.columns]
    df = df.sort_index()
    df = df.loc[:, 'Open':'Close']
    dic_prices[key] = df

result_df = pd.DataFrame()

for key, df in dic_prices.items():
    indicators = FinancialIndicators(df)
    signals = indicators.macd_rsi_signal(12, 26, 9, 35, 70)
    result_df[key] = signals['Signal']


def column_report(df_x):
    """
    Parameters
    ----------
    df_x: DataFrame
        Give a list of tickers.

    Returns
    -------
    Dataframe with columns dtype', 'nulls', '%nulls', 'count', 'unique', 'top', 'freq', 'mean',
           'std', 'min', '25%', '50%', '75%', 'max'.

    """

    col1 = {'dtype'}
    col2 = {'nulls'}

    a = pd.DataFrame(df_x.dtypes, columns=list(col1))

    b = pd.DataFrame(df_x.isna().sum(), columns=list(col2))
    b['%nulls'] = round(100 * b['nulls'] / df_x.shape[0], 2)

    c = df_x.describe(include='all').transpose()

    return a.join(b).join(c).sort_values('dtype')


df_report = column_report(result_df)

'Signals ready, now prices'

df_close = pd.DataFrame({ticker: df['Close'] for ticker, df in dic_prices.items()})

backtest = bk.backtest(df_close, result_df, initial_capital, commission_percentage, variable_commission,
                       dollar_amount)
