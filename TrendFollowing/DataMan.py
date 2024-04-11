# -*- coding: utf-8 -*-
"""
Class to help manage the data for the trend following system.

@author: arnulf.q@gmail.com
"""
#%% MODULES
import os
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
from scipy import stats

#%% CLASS DATA MANAGER
class DataMan:
    """
    A class to help work easier with timeseries and cross section price data 
    for trading systems or systematics strategies.
    """
    def __init__(self, data: pd.DataFrame = None, datapath: str = None) -> None:
        self._data = data
        self._path_input_data = datapath
    
    
    def _pull_data_yf(self, yf_tickers_path: str = 'tickers_etf_yf.txt') -> pd.DataFrame():
        """ 
        Create dataframe with historic OHLCV data downloaded from YahooFinance.
        """
        # Yahoo finance tickers from txt file
        yh_tkrlst = pd.read_csv(yf_tickers_path, sep=",", header=None).\
            to_numpy()[0].tolist()
        # Date delimiters
        todays_date = pd.Timestamp(pd.Timestamp.today().date()) - \
            pd.tseries.offsets.BDay(1)
        T25Y_date = todays_date - pd.tseries.offsets.BDay(252*25)
        # Download data from yahoo finance
        tmpdf = pdr.get_data_yahoo(yh_tkrlst, 
                                  start=todays_date.strftime('%Y-%m-%d'), 
                                  end=T25Y_date.strftime('%Y-%m-%d'))
        # Column-levels standard labels mgmt
        tmpdf.rename(columns={'BTC-USD':'BTC'}, level=1, inplace=True)
        return tmpdf
    
    
    def _pull_data_bbg_(self, xlpath: str = 'datapull_etf_1D.xlsx') -> pd.DataFrame:
        """ 
        Create dataframe with historic OHLCV data downloaded from Bloomberg 
        into an excel file.
        """
        # Data downloaded from Bloomberg into an xl file as historic data table
        data = pd.read_excel(xlpath)
        # Column-levels extraction
        tmpcols1 = data.iloc[4]; tmpcols2 = data.iloc[2]
        tmpcols1[0] = 'Date'; tmpcols1 = tmpcols1.unique().tolist()
        # Column-levels mgmt
        if np.nan in tmpcols1: tmpcols1.remove(np.nan)
        tmpcols2[0] = 'Date'; tmpcols2 = tmpcols2.unique().tolist()
        if np.nan in tmpcols2: tmpcols2.remove(np.nan)
        if 'Date' in tmpcols1: tmpcols1.remove('Date')
        if 'Date' in tmpcols2: tmpcols2.remove('Date')
        # Data separation from column-levels
        tmpdf = data.iloc[5:]
        tmpdf = tmpdf.rename(columns={tmpdf.columns[0]:'Date'})
        tmpdf = tmpdf.set_index('Date').astype(float)
        # Column-levels format as (OHLC, Asset)
        tmplist = []
        for a in tmpcols2:
            tmplist+=[(s,a) for s in tmpcols1]
        # MultiIndex
        mulidx = pd.MultiIndex.from_tuples(tmplist)
        # Column-levels assignment to dataframe
        tmpdf.columns = mulidx
        # Column-levels standard labels mgmt
        cols2chg_l1 = dict([(s,
                             s.replace('PX_','').lower().capitalize().\
                                 replace('Last','Close')) 
                            for s in tmpdf.columns.levels[0]])
        cols2chg_l2 = dict([(s,
                             s.replace(' US Equity','').replace(' Curncy','').\
                                 replace('XBTUSD','BTC')) 
                            for s in tmpdf.columns.levels[1]])
        tmpdf.rename(columns=cols2chg_l1, level=0, inplace=True)
        tmpdf.rename(columns=cols2chg_l2, level=1, inplace=True)
        return tmpdf
        
    
    def set_data_from_path(self, xlpath: str = None, 
                 yf_tickers_path: str = None) -> None:
        """
        Dataframe with historic OHLCV data.
        """
        # Check for database existance
        if os.path.isfile(self.path_input_data):
            # Import saved db
            data = pd.read_parquet(self.path_input_data)
        else:
            # Create db
            if xlpath is not None and yf_tickers_path is None:
                # Inception from xl file
                #xlpath = r'C:\Users\jquintero\db\datapull_etf_1D.xlsx'
                data = self._pull_data_bbg_(xlpath)
            elif xlpath is None and yf_tickers_path is not None:
                # Inception from yahoo finance
                #r'C:\Users\arnoi\Documents\Python Scripts\TrendFollowing\tickers_etf_yf.txt'
                data = self._pull_data_yf(yf_tickers_path)
                
            # Save db
            if self._path_input_data is not None: 
                data.to_parquet(self._path_input_data)
            
        # Data as standardized (OHLCV,Asset) Dataframe
        self._data = data
        
        
        def get_data(self) -> pd.DataFrame:
            """
            Returns the Dataframe with standardized (OHLCV, Asset) price data.
            """
            return self._data
        
        
        def get_returns_OpenVsClose(self) -> pd.DataFrame:
            """
            Returns: Log-returns of Open vs Close price data.
            """
            tmpdf = pd.DataFrame(index = self._data.index)
            for asset in self._data.columns.levels[1]:
                # Open, Close Prices
                tmpcol = [(s, asset) for s in ['Open', 'Close']]
                # Log-returns
                tmpr = self._data[tmpcol].apply(np.log).diff(axis=1).Close
                # Dataconcat
                tmpdf = pd.concat([tmpdf, tmpr], axis=1)
            return tmpdf
        
        # FUNCTIONS OVER DATAFRAME
        def statistics(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        
        
        def sliceDataFrame(self, df: pd.DataFrame = None, 
                           dt_start: str = None, dt_end: str = None, 
                           lst_securities: list = None) -> pd.DataFrame:
            """
            Returns: Dataframe sliced between given dates and filtered
            by the list of securities provided.
            """
            # Columns verification
            lstIsEmpty = (lst_securities is None) or (lst_securities == [])
            if lstIsEmpty:
                tmpsecs = df.columns.levels[1].tolist()
            else:
                tmpsecs = [[('Open', s),('High', s),('Low', s),('Close', s)] 
                           for s in lst_securities]
                tmpsecs = [item for sublist in tmpsecs for item in sublist]
            
            # Column-filtered dataframe
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
                
            return tmpdfret.loc[tmpstr1:tmpstr2, tmpsecs]

