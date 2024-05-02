# -*- coding: utf-8 -*-
"""
Class to define, set and analyize trend in OHLCV data.

@author: arnulf.q@gmail.com
"""
#%% MODULES
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
#%% CLASS STRATEGY MANAGER
class StratMan:
    """
    A class set, handle, compute and analyize trading signals from trend data.
    """
    def __init__(self, trend_data: pd.DataFrame) -> None:
        self.__trend_data = trend_data
        self.__data = None
        self.__names = None
        self.__trend_strength_Z = None
        
    def get_trend_data(self):
        return self.__data
    
    def get_trend_strength_Z(self):
        return self.__trend_strength_Z
    
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
    
    
    def __set_trend_strength_Z__(self) -> None:
        """
        Returns
        -------
        None
            Sets scaled data for trend strength.
        """
        # Columns for EMA differences
        selcols = [f'{s}_ema_d' for s in self.__names if s != '']
        # Total years to eval
        years = self.__data.index.year.unique().sort_values(ascending=False)
        # Scaler for each year
        sc = StandardScaler()
        tmpdf = pd.DataFrame()
        for yr in years[:-8]:
            # Years range
            y1, y2 = yr-8, yr-1
            # EMA diff sliced data
            X = self.__data.loc[str(y1):str(y2), selcols]
            # Verify data availability
            keepCol = ~X.isna().all(axis=0)
            X = X[X.columns[keepCol]]
            # Scaler
            tmpSC = sc.fit(X)
            # Target
            y = self.__data.loc[str(yr), X.columns]
            # Scaled data
            yZ = pd.DataFrame(tmpSC.transform(y), columns = y.columns, index = y.index)
            # Save
            tmpdf = pd.concat([tmpdf, yZ])
        # Set attribute   
        self.__trend_strength_Z = tmpdf.sort_index()
        return None
    
    
    def __set_trend_strength_class__(self) -> None:
        """
        Returns
        -------
        None
            Modifies trend strength number for category.
        """
        # Scaled trend strength
        tsZ = self.__trend_strength_Z
        # Trend strength category
        for name in self.__names:
            # Columns
            selcol = f'{name}_ema_d'
            mcol = f'{name}_trend_strength'
            # Strong dates
            idx_strong = tsZ[abs(tsZ[selcol]) >= 1].index
            self.__data.loc[idx_strong, mcol] = 'strong'
            # Weak dates
            idx_weak = tsZ[abs(tsZ[selcol]) < 0.25].index
            self.__data.loc[idx_weak, mcol] = 'weak'
            # Normal dates
            idx_nrm = self.__data[mcol][(self.__data[mcol] != 'weak')*\
                                        (self.__data[mcol] != 'strong')].index
            self.__data.loc[idx_nrm, mcol] = 'normal'
            
        return None
        
        
    def set_trend_data(self, data: pd.DataFrame, 
                       price_field: str = 'Close') -> None:
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
        
        # Set attributes
        self.__data = pd.concat([data_ema, ta], axis=1)
        self.__names = data.columns.levels[1]
        
        # Set trend strength class
        self.__set_trend_strength_Z__()
        self.__set_trend_strength_class__()
        
        return None
            
            