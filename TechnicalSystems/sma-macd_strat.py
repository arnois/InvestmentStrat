# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:19:29 2023

@author: suprl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf


archivo = 'AMZN_Mkt_Data.csv'
amazon = pd.read_csv(archivo, index_col=0, parse_dates=True)
data_available = amazon.columns

amazon_ohlc = amazon[['Open',
                      'High',
                      'Low',
                      'Close',
                      'Adjclose',
                      'Volume',
                      'Returns',
                      'RSI',
                      '20_EMA',
                      '55_EMA',
                      '200_EMA']]


def MACD(df, window_st, window_lt, window_signal):
    
    macd = pd.DataFrame()
    macd['ema_st'] = df['Close'].ewm(span=window_st).mean()
    macd['ema_lt'] = df['Close'].ewm(span=window_lt).mean()
    macd['macd'] = macd['ema_st'] - macd['ema_lt']
    macd['signal'] = macd['macd'].ewm(span=window_signal).mean()
    macd['diff'] = macd['macd'] - macd['signal']
    macd['bar_positive'] = macd['diff'].map(lambda x: x if x > 0 else 0) #De la diferencia, mapea con 0 los que son menores a 0
    macd['bar_negative'] = macd['diff'].map(lambda x: x if x < 0 else 0) # mapea la diferencia que sea mayor a 0

    return macd

macd = MACD(amazon_ohlc, 12, 26, 9)
# rsi = ma_up / ma_down
# rsi = 100 - (100/(1 + rsi))
# df_x['RSI'] = rsi
    

price_info = pd.concat([amazon_ohlc, macd], axis = 1)

price_info['rolling_vol'] = price_info['Volume'].rolling(10).mean()

#%%

start = '2020'

price_info = price_info[start:]


# macd = macd.loc[start:]

indicators  = [
    mpf.make_addplot(price_info['20_EMA'], color='#606060', panel = 0),
    mpf.make_addplot(price_info['55_EMA'], color = '#1f77b4', panel = 0),
    mpf.make_addplot(price_info['200_EMA'], color ='#F0F020', panel = 0),
    
    mpf.make_addplot(price_info['rolling_vol'], color = 'dodgerblue', panel = 1),
    
    mpf.make_addplot((price_info['macd']), color='#606060', panel=2, ylabel='MACD', secondary_y=False),
    mpf.make_addplot((price_info['signal']), color='#1f77b4', panel=2, secondary_y=False),
    mpf.make_addplot((price_info['bar_positive']), type='bar', color='#4dc790', panel=2),
    mpf.make_addplot((price_info['bar_negative']), type='bar', color='#fd6b6c', panel=2),
    
    mpf.make_addplot(price_info['RSI'], color = 'orange', panel = 3, ylabel = 'RSI'),
    mpf.make_addplot([70]* len(price_info), color = 'gray', panel = 3, secondary_y = False, linestyle = '-'),
    mpf.make_addplot([35]* len(price_info), color = 'gray', panel = 3, secondary_y = False, linestyle = '-'),
    mpf.make_addplot([50]* len(price_info), color = 'gray', panel = 3, secondary_y = False, alpha = 0.4)
    
]


#%%

mpf.plot(price_info,
                     type='candle',
                     volume=True,
                     addplot=indicators,
                     style =  'yahoo',
                     axtitle = 'Amazon price since '+start,
                     xrotation = 0,
                     datetime_format = '%Y-%m-%d',
                     figsize = (30,15),
                     savefig = (f'amazon_plot_{start}.png')
         )

mpf.show()
#%% ----------------

# fig_size = (30,15)
# fig = mpf.figure(figsize= fig_size)
# ax1 = fig.add_subplot(2, 1, 1)
# mpf.plot(price_info,
#                      type='candle',
#                      volume=True,
#                      addplot=indicators,
#                      style =  'yahoo',
#                      figscale=2,
#                      axtitle = 'Amazon price since '+start,
#                      figsize = (30,15),
#                      ax = ax1
#                      # savefig = (f'amazon_plot_{start}.png')
#          )
# # Add the subplot for the RSI indicator
# ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

# # Plot the RSI indicator on the lower subplot
# mpf.plot(data['RSI'], ax=ax2, ylabel='RSI')


#%%
# Start with the signal strat

# -------------------

"""
La señal va a estar dada por un cruce de MACD y cruce de EMAS, el pex es que hay
que hacer un rolling window porque nunca va a haber un periodo en el que el mismo
dia se den los cruces, entonces tendrá que ser en una ventana de X dias!!

Quiero que haya una señal de compra si en una vetana de 10 dias tuve un cruce de 
MACD y de EMAs

para esto construyo un Df que tiene las señales por individual de emas y macd
para despues tener un Df que sea el rolling window de las dos para la ventana

"""
signals_df = pd.DataFrame(index = price_info.index)
draft_df =  pd.DataFrame(index = price_info.index)

index = price_info.index

draft_df['EMA_buy'] = np.where(price_info['20_EMA'] > price_info['55_EMA'], 1, 0)
draft_df['ema_buy_signal'] = draft_df['EMA_buy'].diff() #Este nomre se puede cambiar

draft_df['EMA_sell'] = np.where(price_info['55_EMA'] > price_info['20_EMA'], 1, 0)
draft_df['ema_sell_signal'] = draft_df['EMA_sell'].diff()

draft_df['RSI_buy'] = np.where(price_info['RSI'] < 35, 1, 0)
draft_df['rsi_buy_signal'] = -(draft_df['RSI_buy'].diff()) # +1 es señal de compra
draft_df['rsi_buy_signal'] = draft_df['rsi_buy_signal'].map(lambda x: x if x > 0 else 0)

draft_df['RSI_sell'] = np.where(price_info['RSI'] > 70, 1, 0)
draft_df['rsi_sell_signal'] = draft_df['RSI_sell'].diff() # -1 señal de venta 
draft_df['rsi_sell_signal'] = draft_df['rsi_sell_signal'].map(lambda x: x if x < 0 else 0)

draft_df['RSI_signal'] = draft_df['rsi_buy_signal'] + draft_df['rsi_sell_signal'] 

draft_df['MACD_buy'] =  np.where(price_info['macd'] > price_info['signal'], 1, 0)
draft_df['macd_buy_signal'] = draft_df['MACD_buy'].diff()

draft_df['MACD_sell'] =  np.where(price_info['signal'] > price_info['macd'], 1, 0)
draft_df['macd_sell_signal'] = (draft_df['MACD_sell'].diff())



# %%

signals_df['EMA_signal'] = draft_df['ema_buy_signal']
signals_df['MACD_signal'] = draft_df['macd_buy_signal']
signals_df['RSI_signal'] = draft_df['rsi_buy_signal'] + draft_df['rsi_sell_signal'] 


#%%

