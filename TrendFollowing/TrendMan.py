# -*- coding: utf-8 -*-
"""
Class to define, set and analyize trend in OHLCV data.

@author: arnulf.q@gmail.com
"""
#%% MODULES
import numpy as np
import pandas as pd
import pandas_ta as ta
#%% CLASS TREND MANAGER
class TrendMan:
    """
    A class to make trend analysis from timeseries data based on two 
    exponential moving averages.
    """
    def __init__(self, n_short_period: int = 50, n_long_period: int = 100) -> None:
        self.__nes = n_short_period
        self.__nel = n_long_period
        
    
    def __smooth_data__(self, data: pd.DataFrame, 
                        price_field: str = 'Close') -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with at least closing price data. Dataframe with column-
            levels as (OHLCV, Asset) is expected.
        price_field : str, optional
            Price field to which the exponential moving average(EMA) is applied. 
            The default is 'Close'.

        Returns
        -------
        data_ema : pd.DataFrame
            Dataframe with smoothed prices for each provided asset as column.
        """
        # Smoothed data over price field
        data_ema = data[price_field].apply(lambda s: ta.ema(s, self.__nes)).\
            rename(columns=dict(zip(data[price_field].columns,
                [f'{s}_EMA_ST' for s in data[price_field].columns]))).merge(
            data['Close'].apply(lambda s: ta.ema(s, self.__nel)).\
                rename(columns=dict(zip(data['Close'].columns,
                [f'{s}_EMA_LT' for s in data['Close'].columns]))),                      
            left_index=True, right_index=True)
        return data_ema
    
    
    def __set_trend__(self, data_ema: pd.DataFrame, 
                      price_field: str = 'Close') -> pd.DataFrame:
        """
        Parameters
        ----------
        data_ema : pd.DataFrame
            Smoothed prices.
        price_field : str, optional
            Price field to which trend analysis is applied. 
            The default is 'Close'.

        Returns
        -------
        data_ta : pd.DataFrame
            Dataframe with trend direction, strength, and status (RoC).
        """
        # Assets names
        names = list(set(s.replace('_EMA_ST','').replace('_EMA_LT','') 
                         for s in data_ema.columns))
        # Trend + dynamics
        data_ta = pd.DataFrame()
        for name in names:
            # Short EMA
            tmp_sema = data_ema[f'{name}_EMA_ST']
            
            # Long EMA
            tmp_lema = data_ema[f'{name}_EMA_LT']
            
            # Trend strength
            tmp_emadiff = (tmp_sema - tmp_lema)
            tmp_strength = pd.Series(tmp_emadiff ,index=tmp_emadiff.index)
            
            # Trend status as 5-EMA of the RoC of the EMA difference
            tmp_status = tmp_emadiff.diff().rename(price_field).to_frame().ta.ema(5)
            
            # Trend specs
            data_ta = pd.concat([data_ta, 
                tmp_emadiff.apply(np.sign).rename(f'{name}_trend').to_frame(),
                tmp_strength.rename(f'{name}_trend_strength').to_frame(),
                tmp_status.rename(f'{name}_trend_status').to_frame(),
                tmp_emadiff.rename(f'{name}_ema_d').to_frame()], axis=1)
        return data_ta
        
        
    def get_trend_data(self, data: pd.DataFrame, 
                       price_field: str = 'Close') -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with at least closing price data. Dataframe with column-
            levels as (OHLCV, Asset) is expected.
        price_field : str, optional
            Price field to which the exponential moving average(EMA) 
            is applied. The default is 'Close'.

        Returns
        -------
        pd.DataFrame
            Smoothed prices with corresponding trend characteristics: 
                direction, strength, and status (RoC).
        """
        # Default field to Close
        if price_field is None:
            price_field = 'Close'
            
        # Smoothed prices
        data_ema = self.__smooth_data__(data=data, price_field=price_field)
        
        # Trend dynamics
        ta = self.__set_trend__(data_ema=data_ema, price_field=price_field)
        
        return pd.concat([data_ema, ta], axis=1)
            
            