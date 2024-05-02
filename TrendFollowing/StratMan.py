# -*- coding: utf-8 -*-
"""
Class to define, set and analyize trend in OHLCV data.

@author: arnulf.q@gmail.com
"""
#%% MODULES
import numpy as np
import pandas as pd
from scipy import stats
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
#%% CLASS STRATEGY MANAGER
class StratMan:
    """
    A class set, handle, compute and analyize trading signals from trend data.
    """
    def __init__(self, price_data: pd.DataFrame, 
                 trend_data: pd.DataFrame) -> None:
        # Attributes
        self.__price_data = price_data
        self.__trend_data = trend_data
        self.__names = [s.replace('_EMA_ST','') 
                        for s in self.__trend_data.columns if '_EMA_ST' in s]
        self.__dict_sgnls = None
        self.__strat_results = None
        
    def get_trend_data(self):
        return self.__trend_data
    
    def get_signals_data(self):
        return self.__dict_sgnls
    
    def get_strat_results(self):
        return self.__strat_results
    
    def get_PricePlusIndicators_data(self, name: str) -> pd.DataFrame:
        # Verify for asset name in data
        if name not in self.__names:
            print('Asset name not found in data!')
            return pd.DataFrame()
        return self.__get_PricePlusIndicators__(name)
    
    # Function to merge price data with its indicators
    def __get_PricePlusIndicators__(self, name: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        name : str
            Asset name.

        Returns
        -------
        df_pxi : pd.DataFrame
            Price and indicator data merged.
        """
        # Price column names
        px_col = ('Close', name)
        
        # MA column names
        macol = [f'{name}_EMA_{s}' for s in ['ST','LT']]
        
        # Temporal price+indicator info
        df_pxi = self.__price_data[[px_col]].merge(
            self.__price_data[[px_col]].shift().\
                rename(columns={name:f'{name}_1'}, level=1), 
            left_index=True, right_index=True).droplevel(0, axis=1).\
            merge(self.__trend_data[macol], 
                  left_index=True, right_index=True)
        
        return df_pxi
    
    
    # Function to compute signals for PxMA strat
    def __get_signals_PxMA(self, df_pxi: pd.DataFrame) -> pd.DataFrame():
        """
        Parameters
        ----------
        df_pxi : pd.DataFrame
            Price and indicators data from which signals are created.

        Returns
        -------
        df_signals : pd.DataFrame
            Dataframe with signals data.
        """
        
        # Asset name
        name = df_pxi.columns[0]
        
        # Signals dataframe
        df_signals = pd.DataFrame(index = df_pxi.index)
        df_signals['Trend'] = 0
        df_signals['Entry'] = 0
        df_signals['Exit_Long'] = 0
        df_signals['Exit_Short'] = 0
        
        # Trend signals
        df_signals['Trend'] += self.__trend_data[f'{name}_trend'].\
            fillna(0).astype(int)
            
        # Buy signals
        df_signals['Entry'] += df_pxi.apply(lambda s: 
            (s[name+'_1'] < s[name+'_EMA_ST'])*(s[name] > s[name+'_EMA_ST'])*1, 
            axis=1)

        # Sell signals
        df_signals['Entry'] += df_pxi.apply(lambda s: 
            (s[name+'_1'] > s[name+'_EMA_ST'])*(s[name] < s[name+'_EMA_ST'])*-1, 
            axis=1)
            
        # Exit longs
        df_signals['Exit_Long'] += df_pxi.apply(lambda x: 
            (x[name+'_1'] > x[name+'_EMA_LT'])*(x[name] < x[name+'_EMA_LT'])*1, 
            axis=1)

        # Exit shorts
        df_signals['Exit_Short'] += df_pxi.apply(lambda x: 
            (x[name+'_1'] < x[name+'_EMA_LT'])*(x[name] > x[name+'_EMA_LT']), 
            axis=1)
            
        return df_signals

    
    # Function to define entry+exit signals given strat
    def set_strat(self, strat_type: str = 'PxMA') -> None:
        
        # Dictionary to save signals for each asset
        dict_sgnls = {}
        
        # Loop through asset
        for name in self.__names:
            # Price+Indicators merged data
            df_pxi = self.__get_PricePlusIndicators__(name)
            
            # Signals data
            df_signals = self.__get_signals_PxMA(df_pxi)
            
            # Save
            dict_sgnls[name] = df_signals
        
        self.__dict_sgnls = dict_sgnls
        return None
    
    
    # Function to calc valid entry signals for given asset
    def get_valid_entry(self, name: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        name : str
            Asset name.

        Returns
        -------
        df_valid_entry : pd.DataFrame
            Entry signals that are chronologically valid.
        """
        # Verify if signals data exist and asset name
        if  self.__dict_sgnls is None:
            print('Signals data still not computed!')
            return pd.DataFrame()
        elif name not in self.__names:
            print('Asset name not found in data!')
            return pd.DataFrame()
        
        # Signals data
        df_signals = self.__dict_sgnls[name]
        
        # Valid entries
        df_valid_entry = ((df_signals['Trend']*df_signals['Entry'] > 0)*\
                          df_signals['Entry']).rename('Entry').to_frame()
        df_valid_entry = df_valid_entry[df_valid_entry['Entry'] != 0]
        
        return df_valid_entry
    
    
    # Function to run backtest results given asset name
    def __get_backtestrun__(self, name: str) -> pd.DataFrame:
        
        # Results fields
        bt_cols = ['Start','End','Signal','Entry_price','Exit_price', 'PnL',
                   'Dur_days', 'MaxGain(MFE)', 'MaxLoss(MAE)', 'M2M_Mean', 
                   'M2M_TMean', 'M2M_IQR', 'M2M_Std', 'M2M_Skew', 'M2M_Med', 
                   'M2M_Kurt']
        
        # Price+indicators data
        df_pxi = self.__get_PricePlusIndicators__(name)
        
        # Strat signals
        df_signals = self.__dict_sgnls[name]
        
        # Valid entries
        df_valid_entry = self.get_valid_entry(name)
        
        # Control vars
        df_strat_bt = pd.DataFrame(columns = bt_cols)
        last_exit_day = df_valid_entry.index[0]
        col_px = [(s, name) for s in ['Open', 'High', 'Low']]
        
        # loop through entries
        for i,r in df_valid_entry.iterrows():
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
            entry_price = self.__price_data.loc[entry_day, col_px].mean()
            
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
            tmpdf_valid_exit = self.__price_data.\
                loc[dates_valid_exit, col_px].loc[i:]

            # Check if trade still open
            if tmpdf_valid_exit.shape[0] < 2:
                if tmpdf_valid_exit.shape[0] < 1:
                    # Trade is still on, so possible exit price is on last observed day
                    exit_day = self.__price_data.iloc[-1].name
                else:
                    exit_day = tmpdf_valid_exit.iloc[0].name
                    
            else: # Exit signal triggered
                # Exit signal date
                date_exit_signal = tmpdf_valid_exit.iloc[0].name
                
                # Check if closing trade is tomorrow
                if self.__price_data.loc[date_exit_signal:].shape[0] < 2:
                    exit_day = self.__price_data.\
                        loc[date_exit_signal:].iloc[0].name + \
                        pd.tseries.offsets.BDay(1)
                else: 
                    # Exit day following exit signal date
                    exit_day = self.__price_data.\
                        loc[date_exit_signal:, col_px].iloc[1].name
                
            # Exit price at OHL mean price to account for price action
            exit_price = self.__price_data.loc[exit_day, col_px].mean()
            
            # Trade PnL
            pnl = (exit_price - entry_price)*r['Entry']
            
            # Trade duration
            dur_days = (exit_day - entry_day).days
            
            # Mark2Market PnL
            m2m_ = (self.__price_data.\
                    loc[entry_day:exit_day, ('Close', name)] - entry_price)*\
                r['Entry']
            
            # Mark-2-market stats
            m2m_stats = self.__statistics__(m2m_.to_frame())
            mfe = (max(self.__price_data.\
                       loc[entry_day:exit_day, (col_maxProfit, name)]) - \
                entry_price)*r['Entry']
            mae = (min(self.__price_data.\
                       loc[entry_day:exit_day, (col_maxLoss, name)]) - \
                entry_price)*r['Entry']

            # BT row data
            tmp_bt_row = [entry_day, exit_day, r['Entry'], entry_price, 
                          exit_price, pnl, dur_days, mfe, mae] + \
                m2m_stats.to_numpy().tolist()[0]
                
            # Add obs to bt df
            df_strat_bt.loc[i] = tmp_bt_row
            
            # Update last exit day
            last_exit_day = exit_day
        
        return df_strat_bt
    
    # Function to calc strategy results
    def set_strat_results(self) -> None:

        # Verify if signals data exist and asset name
        if  self.__dict_sgnls is None:
            print('Signals data still not computed!')
            return pd.DataFrame()
        
        # Backtest results for each asset
        tmpdict = {}
        for name in self.__dict_sgnls.keys():
            tmpdict[name] = self.__get_backtestrun__
        # Attribute update
        self.__strat_results = tmpdict
        return None


    ###########################################################################
    # FUNCTIONS OVER DATAFRAME
    ###########################################################################
    def __statistics__(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns: Summary stats for each serie in the input DataFrame.
        """
        if data is None:
            data = self.get_returns_OpenVsClose().dropna()
        # Mean
        tmpmu = np.round(data.mean().values,4)
        # Trimmed mean
        tmptmu_5pct = np.round(stats.trim_mean(data, 0.05),4)
        # Interquartile Range
        tmpiqr = np.round(data.apply(stats.iqr).values,4)
        # Standard Deviation
        tmpstd = np.round(data.std().values,4)
        # Skewness
        tmpskew = np.round(data.apply(stats.skew).values,4)
        # Median
        tmpme = np.round(data.median().values,4)
        # Kurtosis
        tmpkurt = np.round(data.apply(stats.kurtosis).values,4)
        # Dataframe column names
        tmpcol = ['Mean', '5% Trimmed Mean', 'Interquartile Range', 
                  'Standard Deviation', 'Skewness', 'Median', 'Kurtosis']
        # Dataframe format statistics
        tmpdf = pd.DataFrame([tmpmu, tmptmu_5pct, tmpiqr,
                            tmpstd, tmpskew, tmpme, tmpkurt]).T
        tmpdf.columns = tmpcol
        tmpdf.index = data.columns
        return tmpdf
            
            