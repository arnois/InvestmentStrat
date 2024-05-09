# -*- coding: utf-8 -*-
"""
KaxaNukTrendAnalyzer (KANTA)

@author: arnulf.q@gmail.com
"""
#%% MODULES
import numpy as np
import os
import sys
from random import choice
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
# Path
path = r'H:\db\etf_bbg.parquet'
if not os.path.exists(path):    path = cwdp_parent+r'\etf_bbg.parquet'

# Data Manager
dfMgr = DataMan(datapath=path)

# Data
dfMgr.set_data_from_path()
data = dfMgr.get_data()

# Data update
try:
    xlpath_update = r'C:\Users\jquintero\db\datapull_etf_1D.xlsx'
    dfMgr.update_data(xlpath = xlpath_update)
    data = dfMgr.get_data()
except FileNotFoundError:
    print('\nData for updates not found!')

# Asset names
names = np.array([s for s in data.columns.levels[1] if s != ''])

#%% TREND SETTINGS
nes, nel = 21, 64
TrendMgr = TrendMan(nes, nel)
TrendMgr.set_trend_data(data,'Close')
data_trend = TrendMgr.get_trend_data()

#%% TREND ANALYZER
if __name__ == '__main__':
    print('---------------------------------------')
    print('Welcome to the KaxaNuk Trend Analyzer (KANTA)')
    print('---------------------------------------')
    procFinished = False
    # Check any assets TA
    while not procFinished:
        name = input('\nETF ticker: ').upper()
        if name in names:
            namecol = [f'{name}_trend',f'{name}_trend_strength',
                       f'{name}_trend_status',f'{name}_ema_d']
            
            # Trend + dynamics
            tau = 5
            ecolx = ['Trend','Status','Strength','EMAD','Closing Price']
            tmpTD = data_trend[namecol].iloc[-tau:].merge(
                data.loc[data_trend[namecol].iloc[-tau:].index]['Close'][name]
                , left_index=True, right_index=True).\
                rename(columns=dict(zip(namecol+[name], ecolx)))
            print(f"Last {tau} days trend dynamics:\n")
            print(tmpTD)
            procFinished = True
        else:
            print(f'\n{name} ticker not found in data!')
            print(f'\nTry another one like {choice(names)} perhaps...')
        
