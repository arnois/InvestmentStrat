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
from seaborn import heatmap
from matplotlib import pyplot as plt
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
    
    
    def _update_data_yf_(self, data: pd.DataFrame = None) -> None:
        """
        Updates dataframe with new historic OHLCV data downloaded from yFinance.
        """
        # Data verification
        if data is None:
            data = self._data
        # Date delimiters
        last_date_data = data.index[-1]
        todays_date = pd.Timestamp(pd.Timestamp.today().date()) - \
            pd.tseries.offsets.BDay(1)
        str_idt = last_date_data.strftime("%Y-%m-%d")
        str_fdt = todays_date.strftime("%Y-%m-%d")
        # Data fields
        yh_tkrlst = [s.replace('BTC','BTC-USD') for s in data.columns.levels[1]]
        # Data download from yahoo finance
        new_data = pdr.get_data_yahoo(yh_tkrlst, start=str_idt, end=str_fdt)
        if new_data.index[-1] != todays_date:
            str_fdt = (todays_date + pd.tseries.offsets.DateOffset(days=1)).\
                strftime("%Y-%m-%d")
            new_data = pdr.get_data_yahoo(yh_tkrlst, start=str_idt, end=str_fdt)
        # New available data
        if new_data.index[-1] >= todays_date:
            # New data format
            new_data = new_data[data.columns.levels[0].tolist()].\
                rename(columns={'BTC-USD':'BTC'}, level=1)
            # Update
            print("Updating data...")
            updated_data = pd.concat([data,new_data]).reset_index().\
                drop_duplicates(('Date','')).set_index('Date')
            print("Saving data...")
            updated_data.to_parquet(self._path_input_data)
            self._data = updated_data
        else:
            print("No new data.")
        
    
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
    
    
    def _update_data_bbg_(self, data: pd.DataFrame = None, 
                          xlpath_update: str = 'datapull_etf_1D.xlsx') -> None:
        """
        Updates dataframe with new historic OHLCV data from an excel file with
        data downloaded from Bloomberg.
        """
        # Data verification
        if data is None:
            data = self._data
        # Import new data
        new_data = pd.read_excel(xlpath_update)
        # Column-levels mgmt
        tmpcols1 = new_data.iloc[4]; tmpcols2 = new_data.iloc[2]
        tmpcols1[0] = 'Date'; tmpcols1 = tmpcols1.unique().tolist()
        if np.nan in tmpcols1: tmpcols1.remove(np.nan)
        tmpcols2[0] = 'Date'; tmpcols2 = tmpcols2.unique().tolist()
        if np.nan in tmpcols2: tmpcols2.remove(np.nan)
        if 'Date' in tmpcols1: tmpcols1.remove('Date')
        if 'Date' in tmpcols2: tmpcols2.remove('Date')
        # Data separation
        tmpdf = new_data.iloc[5:]; tmpdf = tmpdf.rename(columns={tmpdf.columns[0]:'Date'})
        tmpdf = tmpdf.set_index('Date').astype(float)
        # Column-levels format as (OHLC, Asset)
        tmplist = []
        for a in tmpcols2:
            tmplist+=[(s,a) for s in tmpcols1]
        mulidx = pd.MultiIndex.from_tuples(tmplist)
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
        # Date delimiters
        last_date_data = data.index[-1]
        todays_date = pd.Timestamp(pd.Timestamp.today().date()) - \
            pd.tseries.offsets.BDay(1)
        # New available data
        new_data_ldt = tmpdf.index[-1]
        if new_data_ldt >= last_date_data:
            # Date range for new data
            str_idt = last_date_data.strftime("%Y-%m-%d")
            str_fdt = todays_date.strftime("%Y-%m-%d")
            print("Updating data... ")
            # Updated data
            updated_data = pd.concat([data, tmpdf.loc[str_idt:str_fdt]]).\
                reset_index().drop_duplicates(('Date','')).set_index('Date')
            print("Saving data...")
            updated_data.to_parquet(self._path_input_data)
            self._data = updated_data
        else:
            print("No new data.")
    
    
    def set_data_from_path(self, xlpath: str = None, 
                           yf_tickers_path: str = None) -> None:
        """
        Dataframe with historic OHLCV data.
        """
        # Check for database existance
        if os.path.isfile(self._path_input_data):
            # Import saved db
            data = pd.read_parquet(self._path_input_data)
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
            else:
                print("No xl file path nor YahooFinance tickers provided!")
                return None
                
            # Save db
            if self._path_input_data is not None: 
                data.to_parquet(self._path_input_data)
            
        # Data as standardized (OHLCV,Asset) Dataframe
        self._data = data
        
    
    def update_data(self, data: pd.DataFrame = None, xlpath: str = None, 
                    yf_tickers_path: str = None) -> None:
        """
        Updates current historic OHLCV data.
        """
        # Data verification
        if data is None:
            data = self._data
        # Out of date verification
        last_date_data = data.index[-1]
        todays_date = pd.Timestamp(pd.Timestamp.today().date()) - \
            pd.tseries.offsets.BDay(1)
        isDataOOD = last_date_data < todays_date
        if isDataOOD:
            print("Database out of date...")
            if xlpath is not None and yf_tickers_path is None:
                # Update from xl file from Bloomberg
                self._update_data_bbg_(data=data, xlpath_update=xlpath)
            else:
                # Update from yahoo finance
                self._update_data_yf_(data=data)
        else:
            print("Database up to date.")
        
        
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
    
    ###########################################################################
    # FUNCTIONS OVER DATAFRAME
    ###########################################################################
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
        # Data verification
        if df is None:
            df = self._data
        # Columns filter verification
        lstIsEmpty = (lst_securities is None) or (lst_securities == [])
        
        # Columns level verification
        if df.columns.nlevels > 1:
            # For multiple index
            if lstIsEmpty:
                tmplst = df.columns.levels[1].tolist()
                tmpsecs = [[('Open', s),('High', s),('Low', s),('Close', s)] 
                           for s in tmplst]
            else:
                tmpsecs = [[('Open', s),('High', s),('Low', s),('Close', s)] 
                           for s in lst_securities]
            # Unlisting sublists
            tmpsecs = [item for sublist in tmpsecs for item in sublist]
        else:
            # For single index
            if lstIsEmpty:
                tmpsecs = df.columns.tolist()
            else:
                tmpsecs = lst_securities        
        
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

   
    def plot_corrmatrix(self, df: pd.DataFrame = None, dt_start: str = None, 
                        dt_end: str = None, lst_securities: list = None, 
                        plt_size: tuple = (10,8), txtCorr: bool = False, 
                        corrM: str = 'spearman') -> None:
        """
        Returns: Correlation matrix plot.
        """
        # Data verification
        if df is None:
            df = self._data
            # Securities and date ranges filter
            df_ = self.sliceDataFrame(df=df, dt_start=dt_start, dt_end=dt_end, 
                                      lst_securities=lst_securities)
        else:
            df_ = df
        
        tmpdfret = df_
        
        # Corrmatrix
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
