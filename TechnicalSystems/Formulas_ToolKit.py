# Libraries

import pandas as pd
import numpy as np
import fmpmodule as fmp
import time
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
from scipy.optimize import minimize
import calendar
import math as math
from math import sqrt
import openpyxl
from datetime import date
from datetime import timedelta
import datetime
global p
import time
import warnings
import plotly.graph_objects as go 
from IPython.display import display
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import plotly.offline as offline

#%% 'Extra Functions'

def sep(df_x,start_date,end_date):
  df1 = df_x.loc[(df_x.index >= start_date)
                     & (df_x.index < end_date)]
  return df1

#%% 'Main Class'
# Class to buildup setups and generate technical trading signals
class FinancialIndicators:
    
    def __init__(self, df):
        # UPDATE: attributes should be computed at init class process.
        #   like typycal prices; which are a derivation of H-L-C prices.
        # First check for proper input given
        colnames = ['Open','High','Low','Close','Volume']
        if all(x in df.columns for x in colnames):
            names_typrice = ['High', 'Low', 'Close']
            self.df = df
            self.df['TP'] = self.df[names_typrice].mean(axis=1)
        else:
            raise ValueError('Inappropiate column names or configuration.'+\
                f'\nProvide DataFrame with {colnames}'+\
                ' column names at least.')
        
    # Function to compute the volume weighted average price
    def VWAP(self, period: int = 21) -> pd.Series:
        # NEW: this function is added as a basis for further indicators.
        """
        Computes the Volume Weighted Average Price (VWAP).
        
        Parameters:
        - period (int): The number of time steps to calculate the VWMA.
        
        Returns:
        pd.Series: VWAP for given period.
        """
        volumes = self.df['Volume']
        typrices = self.df['TP']
        cum_PV = (typrices*volumes).rolling(period).sum()
        cum_V = volumes.rolling(period).sum()
        return cum_PV/cum_V
    
    # Function to compute a moving average over the VWAP
    def VWMA(self, vwap_p: int = 21, ma_p: int = 5) -> pd.Series:
        # UPDATE: there is no need for arguments other than self and periods.
        """
        Computes the Volume Weighted Moving Average (VWMA).

        Parameters:
        - vwap_p (int): The number of time steps to calculate the VWAP.
        - ma_p (int): The length of the MA over the VWAP.

        Returns:
        pd.Series: VWMA for the given period.
        """
        vwap = self.VWAP(vwap_p)
        return vwap.rolling(ma_p).mean()

    def DV(self, vol_window: int = 10) -> pd.Series:
        # UPADTE: added vol-period as argument to provide generalization.
        
        """
        Calculate daily volatility.
        
        Parameters:
        - vol_window (int): The number of periods to use for volatility calc.

        Returns:
        pd.Series: Daily volatility.
        
        """
        daily_returns = self.df['Close'].pct_change()
        daily_volatility = daily_returns.rolling(window=vol_window).std()  # Adjust the window size to capture daily volatility
        return daily_volatility

    def ESVMap(self, short_term_period: int) -> pd.Series:
        # UPDATE: use VWMA for SVWMA.
        """
        Calculate Exponential Smoothed Volume Moving Average Product (ESVMap).

        Parameters:
        - short_term_period (int): The number of periods over which to calculate the ESVMap.

        Returns:
        pd.Series: ESVMap.
        """
        svwma = self.VWMA(vwap_p=short_term_period) # svwma = self.SVWMA(short_term_period)
        dv = self.DV()
        ema_dvSVWMA = pd.Series.ewm(svwma * dv, span=short_term_period).mean()
        return ema_dvSVWMA

    def VPVMA(self, short_term_period: int, long_term_period: int) -> pd.Series:
        # UPDATE: use ESVMap for ElVMap
        """
        Calculate Volume Price Volume Moving Average (VPVMA).

        Parameters:
        - short_term_period (int): The number of periods over which to calculate the short-term VPVMA.
        - long_term_period (int): The number of periods over which to calculate the long-term VPVMA.

        Returns:
        pd.Series: VPVMA.
        """
        
        esvmap = self.ESVMap(short_term_period)
        elvmap = self.ESVMap(long_term_period) #elvmap = self.ElVMap(long_term_period)
        vpvma = esvmap - elvmap
        return vpvma

    def VPVMAS(self, short_term_period: int, long_term_period: int, signal_period: int) -> pd.Series:
        
        """
        Calculate Volume Price Volume Moving Average Smoothed (VPVMAS).

        Parameters:
        - short_term_period (int): The number of periods over which to calculate the short-term VPVMAS.
        - long_term_period (int): The number of periods over which to calculate the long-term VPVMAS.
        - signal_period (int): The number of periods over which to calculate the signal VPVMAS.

        Returns:
        pd.Series: VPVMAS.
        """
        vpvma = self.VPVMA(short_term_period, long_term_period)
        vpvmas = vpvma.rolling(window=signal_period).mean()
        return vpvmas

    def vpvma_signal(
            self, 
            short_term_period: int = 12, 
            long_term_period: int = 26, 
            signal_period: int = 9, 
            bandwidth: float = 0.10) -> pd.DataFrame:
        
        """
        Generate VPVMA signals based on the given parameters.

        Parameters:
        - short_term_period (int): Periods to use for fast VPVMA.
        - long_term_period (int): Periods to use for slow VPVMA.
        - signal_period (int): Periods for smooth put fast(VPVMA)-slow(VPVMA).
        - bandwidth (float): Threshold xover for buy and sell signals.

        Returns:
        pd.DataFrame: DataFrame with VPVMA signals.
        """
        # VPVMA fast vs slow
        vpvma = self.VPVMA(short_term_period, long_term_period)
        # VPVMAs distances smoothed out
        vpvmas = self.VPVMAS(short_term_period,long_term_period,signal_period)
        # Buy/Sell signals calc
        signals = np.zeros(len(vpvma))
        for i in range(1, len(vpvma)):
            if (vpvma.iloc[i] > (1 + bandwidth) * vpvmas.iloc[i]) and \
                (vpvma.iloc[i-1] <= vpvmas.iloc[i-1]):
                signals[i] = 1  # Buy signal
            elif (vpvma.iloc[i] < (1 - bandwidth * 2) * vpvmas.iloc[i]) and \
                (vpvma.iloc[i-1] >= vpvmas.iloc[i-1]):
                signals[i] = -1  # Sell signal

        self.df['Signal'] = signals

        return pd.DataFrame({'Signal': signals}, index=self.df.index)

    def vpvma_signal2(
            self, 
            short_term_period: int = 12, 
            long_term_period: int = 26, 
            signal_period: int = 9, 
            bandwidth: float = 0.01) -> pd.DataFrame:
        
        """
        Generate VPVMA signals (vectorized implementation) based on the given parameters.

        Parameters:
        - short_term_period (int): Periods to use for fast VPVMA.
        - long_term_period (int): Periods to use for slow VPVMA.
        - signal_period (int): Periods for smooth put fast(VPVMA)-slow(VPVMA).
        - bandwidth (float): Threshold xover for buy and sell signals.

        Returns:
        pd.DataFrame: DataFrame with VPVMA signals.
        """
        # VPVMA fast vs slow
        vpvma = self.VPVMA(short_term_period, long_term_period)
        # VPVMAs distances smoothed out
        vpvmas = self.VPVMAS(short_term_period,long_term_period,signal_period)
        # Buy signals
        ## NOTE: Error on bandwidth consideration for negative values. 
        ## UPDATE: Mod bandwidth to account for negative level values
        adj_bwth = bandwidth*vpvmas.apply(np.sign)
        buy_signals = np.where((vpvma > (1 + adj_bwth) * vpvmas) & \
                               (vpvma.shift(1) <= vpvmas.shift(1)), 1, 0)
        # Sell signals
        sell_signals = np.where((vpvma < (1 - adj_bwth*2) * vpvmas) & \
                                (vpvma.shift(1) >= vpvmas.shift(1)), -1, 0)

        signals = buy_signals + sell_signals

        self.df['Signal'] = signals

        return pd.DataFrame({'Signal': signals}, index=self.df.index)

    def plot_signals(self, 
                     start_date: str = '2018-01-01', 
                     end_date: str = '2023') -> None:
        
        """
        Plot the buy and sell signals.

        Returns:
        None
        """
        
        fig, ax = plt.subplots(figsize=(15, 7))

        # plot prices
        ax.plot(self.df[start_date:end_date].index, 
                self.df[start_date:end_date]['Close'], label='Close Price')

        # plot signals
        buy_signals = self.df[start_date:end_date][self.df['Signal'] == 1]
        sell_signals = self.df[start_date:end_date][self.df['Signal'] == -1]

        ax.scatter(buy_signals.index, buy_signals['Close'], 
                   color='g', label='Buy Signal')
        ax.scatter(sell_signals.index, sell_signals['Close'], 
                   color='r', label='Sell Signal')

        ax.set(title='Buy and Sell signals', xlabel='Date', ylabel='Price')
        ax.legend()
        plt.show()    
    
    def rsi(self, periods=14, ema=True) -> pd.Series:
        
        """
        
        Calculate the Relative Strength Index (RSI).

        Parameters:
        - periods (int): The number of periods over which to calculate the RSI. Default is 14.
        - ema (bool): Whether to use exponential moving average for calculations. Default is True.

        Returns:
        pd.Series: RSI values.
        
        """
        
        close_delta = self.df['Close'].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if ema:
            ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        else:
            ma_up = up.rolling(window=periods).mean()
            ma_down = down.rolling(window=periods).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))

        return rsi

    def MACD(self, window_st: int, window_lt: int, window_signal: int) -> pd.DataFrame:
        
        """
        
        Calculate Moving Average Convergence Divergence (MACD).

        Parameters:
        - window_st (int): The short-term window size for EMA calculations.
        - window_lt (int): The long-term window size for EMA calculations.
        - window_signal (int): The window size for the signal line.

        Returns:
        pd.DataFrame: DataFrame with MACD values, signal line, histogram, and bar values.
        
        """
        macd = pd.DataFrame()
        macd['ema_st'] = self.df['Close'].ewm(span=window_st).mean()
        macd['ema_lt'] = self.df['Close'].ewm(span=window_lt).mean()
        macd['MACD'] = macd['ema_st'] - macd['ema_lt']
        macd['Signal_Line'] = macd['MACD'].ewm(span=window_signal).mean()
        macd['Hist'] = macd['MACD'] - macd['Signal_Line']
        macd['bar_positive'] = macd['Hist'].map(lambda x: x if x > 0 else 0)
        macd['bar_negative'] = macd['Hist'].map(lambda x: x if x < 0 else 0)

        return macd

    def signal_line_crossover(self, window_st: int, window_lt: int, window_signal: int) -> pd.DataFrame:
        
        """
        
        Generate MACD signal based on signal line crossover.

        Parameters:
        - window_st (int): The short-term window size for EMA calculations.
        - window_lt (int): The long-term window size for EMA calculations.
        - window_signal (int): The window size for the signal line.

        Returns:
        pd.DataFrame: DataFrame with MACD signal.
        
        """
        
        macd = self.MACD(window_st, window_lt, window_signal)
        signals = np.zeros(len(macd))

        for i in range(1, len(macd)):
            if macd['MACD'].iloc[i-1] < macd['Signal_Line'].iloc[i-1] and macd['MACD'].iloc[i] > macd['Signal_Line'].iloc[i]:
                signals[i] = 1  # Buy signal
            elif macd['MACD'].iloc[i-1] > macd['Signal_Line'].iloc[i-1] and macd['MACD'].iloc[i] < macd['Signal_Line'].iloc[i]:
                signals[i] = -1  # Sell signal

        return pd.DataFrame({'Signal': signals}, index=self.df.index)

    def zero_crossover(self, window_st: int, window_lt: int, window_signal: int) -> pd.DataFrame:
        
        """
        
        Generate MACD signal based on zero crossover.

        Parameters:
        - window_st (int): The short-term window size for EMA calculations.
        - window_lt (int): The long-term window size for EMA calculations.
        - window_signal (int): The window size for the signal line.

        Returns:
        pd.DataFrame: DataFrame with MACD signal.
        
        """
        
        macd = self.MACD(window_st, window_lt, window_signal)
        signals = np.zeros(len(macd))

        for i in range(1, len(macd)):
            if macd['MACD'].iloc[i-1] < 0 and macd['MACD'].iloc[i] > 0:
                signals[i] = 1  # Buy signal
            elif macd['MACD'].iloc[i-1] > 0 and macd['MACD'].iloc[i] < 0:
                signals[i] = -1  # Sell signal

        return pd.DataFrame({'Signal': signals}, index=self.df.index)

    def histogram(self, window_st: int, window_lt: int, window_signal: int) -> pd.DataFrame:
        
        """
        
        Generate MACD signal based on histogram pattern.

        Parameters:
        - window_st (int): The short-term window size for EMA calculations.
        - window_lt (int): The long-term window size for EMA calculations.
        - window_signal (int): The window size for the signal line.

        Returns:
        pd.DataFrame: DataFrame with MACD signal.
        
        """
        
        macd = self.MACD(window_st, window_lt, window_signal)
        signals = np.zeros(len(macd))

        for i in range(2, len(macd)):
            if all([macd['Hist'].iloc[i] < 0, macd['Hist'].iloc[i-1] == macd['Hist'].iloc[i-2:i].min()]):
                signals[i] = 1  # Buy signal
            elif all([macd['Hist'].iloc[i] > 0, macd['Hist'].iloc[i-1] == macd['Hist'].iloc[i-2:i].max()]):
                signals[i] = -1  # Sell signal

        return pd.DataFrame({'Signal': signals}, index=self.df.index)

    def signal_line_crossover_above_zero(
            self, window_st: int, window_lt: int, window_signal: int) -> pd.DataFrame:
        
        """
        
        Generate MACD signal based on signal line crossover above zero.

        Parameters:
        - window_st (int): The short-term window size for EMA calculations.
        - window_lt (int): The long-term window size for EMA calculations.
        - window_signal (int): The window size for the signal line.

        Returns:
        pd.DataFrame: DataFrame with MACD signal.
        
        """
        
        macd = self.MACD(window_st, window_lt, window_signal)
        signals = np.zeros(len(macd))

        for i in range(1, len(macd)):
            if macd['MACD'].iloc[i-1] < macd['Signal_Line'].iloc[i-1] and macd['MACD'].iloc[i] > macd['Signal_Line'].iloc[i] and macd['MACD'].iloc[i] > 0:
                signals[i] = 1  # Buy signal
            elif macd['MACD'].iloc[i-1] > macd['Signal_Line'].iloc[i-1] and macd['MACD'].iloc[i] < macd['Signal_Line'].iloc[i] and macd['MACD'].iloc[i] < 0:
                signals[i] = -1  # Sell signal

        return pd.DataFrame({'Signal': signals}, index=self.df.index)

    def bollinger_bands(self, window=14, std_dev=2) -> pd.DataFrame:
        
        """
        
        Calculate Bollinger Bands.

        Parameters:
        - window (int): The window size for calculating the moving average.
        - std_dev (int): The number of standard deviations to use for the upper and lower bands. Default is 2.

        Returns:
        pd.DataFrame: DataFrame with Bollinger Bands.
        
        """
        
        middle_band = self.df['Close'].rolling(window=window).mean()
        std = self.df['Close'].rolling(window=window).std()
        lower_band = middle_band - std * std_dev
        upper_band = middle_band + std * std_dev
        return pd.DataFrame({'Lower Band': lower_band, 'Middle Band': middle_band, 'Upper Band': upper_band}, index=self.df.index)

    def bollinger_band_signal(self, window_short=10, window_long=50) -> pd.DataFrame:
        
        """
        
        Generate Bollinger Bands signal.

        Parameters:
        - window_short (int): The window size for the short-term moving average.
        - window_long (int): The window size for the long-term moving average.

        Returns:
        pd.DataFrame: DataFrame with Bollinger Bands signal.
        
        """
        
        bbw = ((self.bollinger_bands()['Upper Band'] - self.bollinger_bands()['Lower Band']) / self.bollinger_bands()['Middle Band']).rolling(window_short).mean()
        signal = bbw.rolling(window_long).mean()
        signals = pd.DataFrame({'Signal': np.where(bbw > signal, 1, -1)}, index=self.df.index)
        return signals

    def rsi_signal(self, buy_threshold=30, sell_threshold=70) -> pd.DataFrame:
        
        """
        
        Generate RSI signal.

        Parameters:
        - buy_threshold (int): The threshold value for a buy signal. Default is 30.
        - sell_threshold (int): The threshold value for a sell signal. Default is 70.

        Returns:
        pd.DataFrame: DataFrame with RSI signal.
        
        """
        
        rsi = self.rsi()
        signals = pd.DataFrame({'Signal': np.zeros(len(rsi))}, index=self.df.index)

        for i in range(1, len(rsi)):
            if rsi.iloc[i] <= buy_threshold:
                signals.iloc[i] = 1  # Buy signal
            elif rsi.iloc[i] >= sell_threshold:
                signals.iloc[i] = -1  # Sell signal

        return signals

    def mfi(self, periods=14) -> pd.Series:
        
        """
        
        Calculate the Money Flow Index (MFI).

        Parameters:
        - periods (int): The number of periods over which to calculate the MFI. Default is 14.

        Returns:
        pd.Series: MFI values.
        
        """
        
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow_raw = typical_price * self.df['Volume']
        positive_money_flow = money_flow_raw.where(typical_price > typical_price.shift(1), 0)
        negative_money_flow = money_flow_raw.where(typical_price < typical_price.shift(1), 0)
        positive_money_flow_sum = positive_money_flow.rolling(window=periods).sum()
        negative_money_flow_sum = negative_money_flow.rolling(window=periods).sum()
        mfi = 100 - (100 / (1 + positive_money_flow_sum / negative_money_flow_sum))

        return mfi

    def mfi_signal(self, buy_threshold=25, sell_threshold=75) -> pd.DataFrame:
        
        """
        Generate MFI signal.

        Parameters:
        - buy_threshold (int): The threshold value for a buy signal. Default is 25.
        - sell_threshold (int): The threshold value for a sell signal. Default is 75.

        Returns:
        pd.DataFrame: DataFrame with MFI signal.
        """
        
        mfi = self.mfi()
        signals = pd.DataFrame({'Signal': np.zeros(len(mfi))}, index=self.df.index)

        for i in range(len(mfi)):
            if mfi.iloc[i] <= buy_threshold:
                signals.iloc[i] = 1  # Buy signal
            elif mfi.iloc[i] >= sell_threshold:
                signals.iloc[i] = -1  # Sell signal

        return signals

    def macd_rsi_signal(
            self, 
            window_st: int, 
            window_lt: int, 
            window_signal: int, 
            lower_threshold: int, 
            upper_threshold: int, 
            rsi_periods=14) -> pd.DataFrame:
        
        """
        Generate MACD-RSI signal.

        Parameters:
        - window_st (int): The short-term window size for EMA calculations.
        - window_lt (int): The long-term window size for EMA calculations.
        - window_signal (int): The window size for the signal line.
        - lower_threshold (int): The lower threshold value for RSI.
        - upper_threshold (int): The upper threshold value for RSI.
        - rsi_periods (int): The number of periods over which to calculate RSI. Default is 14.

        Returns:
        pd.DataFrame: DataFrame with MACD-RSI signal.
        """
        
        macd = self.MACD(window_st, window_lt, window_signal)
        rsi = self.rsi(rsi_periods)

        signals = np.zeros(len(macd))

        for i in range(5, len(macd)):
            if macd['MACD'].iloc[i] > macd['Signal_Line'].iloc[i] and all(rsi.iloc[i-5:i] <= lower_threshold):
                signals[i] = 1  # Buy signal
            elif macd['MACD'].iloc[i] < macd['Signal_Line'].iloc[i] and all(rsi.iloc[i-5:i] >= upper_threshold):
                signals[i] = -1  # Sell signal

        return pd.DataFrame({'Signal': signals}, index=self.df.index)

    def macd_rsi_signal_vectorized(
            self, 
            window_st: int, 
            window_lt: int, 
            window_signal: int, 
            lower_threshold: int, 
            upper_threshold: int, 
            rsi_periods=14) -> pd.DataFrame:
        
        """
        Generate MACD-RSI signal (vectorized implementation).

        Parameters:
        - window_st (int): The short-term window size for EMA calculations.
        - window_lt (int): The long-term window size for EMA calculations.
        - window_signal (int): The window size for the signal line.
        - lower_threshold (int): The lower threshold value for RSI.
        - upper_threshold (int): The upper threshold value for RSI.
        - rsi_periods (int): The number of periods over which to calculate RSI. Default is 14.

        Returns:
        pd.DataFrame: DataFrame with MACD-RSI signal.
        """
        
        macd = self.MACD(window_st, window_lt, window_signal)
        rsi = self.rsi(rsi_periods)

        # Create a boolean series for each condition
        condition_buy_macd = macd['MACD'] > macd['Signal_Line']
        condition_buy_rsi = rsi.rolling(5).min() <= lower_threshold
        condition_sell_macd = macd['MACD'] < macd['Signal_Line']
        condition_sell_rsi = rsi.rolling(5).max() >= upper_threshold

        # Combine the conditions to obtain the signals
        signals = np.zeros(len(macd))
        signals[(condition_buy_macd & condition_buy_rsi)] = 1
        signals[(condition_sell_macd & condition_sell_rsi)] = -1

        return pd.DataFrame({'Signal': signals}, index=self.df.index)
    
    
    
    def macd_mfi_signal(
            self, 
            window_st: int, 
            window_lt: int, 
            window_signal: int, 
            lower_threshold: int, 
            upper_threshold: int,
            mfi_periods=14):
        
        macd = self.MACD(window_st, window_lt, window_signal)
        mfi = self.mfi(mfi_periods)
        
        signals = np.zeros(len(macd))
    
        for i in range(5, len(macd)):
            if macd['MACD'].iloc[i] > macd['Signal_Line'].iloc[i] and all(mfi.iloc[i-5:i+1] <= lower_threshold):
                signals[i] = 1  # Buy signal
            elif macd['MACD'].iloc[i] < macd['Signal_Line'].iloc[i] and all(mfi.iloc[i-5:i+1] >= upper_threshold):
                signals[i] = -1  # Sell signal

        return pd.DataFrame({'Signal': signals}, index=self.df.index)

#%% Class to evaluate technical signals performance
class PortfolioStats:
    
    def drawdown(self, return_series: pd.Series):
        """
        Takes a time series of asset returns
        Compute and returns a DataFrame that contains
        the wealth index
        the previous peaks
        percent drawdowns
        """
        wealth_index = 1000 * (1 + return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return pd.DataFrame({"Wealth": wealth_index,
                             "Peaks": previous_peaks,
                             "Drawdown": drawdowns})

    def semideviation(self, r):
        """
        Returns the semideviation aka negative semideviation of r
        r must be a Series or a DataFrame
        """
        is_negative = r < 0
        return r[is_negative].std(ddof=0)

    def skewness(self, r):
        """
        Alternative to scipy.stats.skew()
        Computes Skewness of a series or DataFrame
        Returns either a float or a Series
        """
        demeaned_r = r - r.mean()
        # use the pop sd, so set dof = 0
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()
        return exp / sigma_r**3

    def kurtosis(self, r):
        """
        Alternative to scipy.stats.kurtosis()
        Computes Kurtosis of a series or DataFrame
        Returns either a float or a Series
        """
        demeaned_r = r - r.mean()
        # use the pop sd, so set dof = 0
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()
        return exp / sigma_r**4

    def is_normal(self, r, level=0.01):
        """
        Applies the Jarque Bera test to determine if a Series is normal or not
        Test is applied at the 1% level by default
        Returns True if the hypothesis of normality is accepted, False otherwise
        """
        # The function returns a tuple
        # Unpack the tuple
        statistic, p_value = scipy.stats.jarque_bera(r)
        # result = scipy.stats.jarque_bera(r)
        return p_value > level

    def var_historic(self, r, level=5):
        """
        VaR Historic
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(self.var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError('Expected r to be series or a DataFrame')

    def var_gaussian(self, r, level=5, modified=False):
        """
        Returns the Parametric Gaussian VaR of a Series or DataFrame
        """
        # Compute the Z score assuming it was Gaussian
        z = scipy.stats.norm.ppf(level / 100)
        if modified:
            # Compute the Z score based on observed skewness and kurtosis
            s = self.skewness(r)
            k = self.kurtosis(r)
            z = (z +
                 (z**2 - 1) * s / 6 +
                 (z**3 - 3 * z) * (k - 3) / 24 -
                 (2 * z**3 - 5 * z) * (s**2) / 36
                 )
        return -(r.mean() + z * r.std(ddof=0))

    def cvar_historic(self, r, level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(r, pd.Series):
            is_beyond = r <= -self.var_historic(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(self.cvar_historic, level=level)
        else:
            raise TypeError('Expected r to be a Series or DataFrame')

    def compound(self, r):
        """
        returns the result of compounding the set of returns in r
        """
        return np.expm1(np.log1p(r).sum())

    def annualize_rets(self, r, periods_per_year):
        """
        Annualizes a set of returns
        We should infer the periods per year
        """
        compounded_growth = (1 + r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year / n_periods) - 1

    def annualize_vol(self, r, periods_per_year):
        """
        Annualizes the vol of a set of returns
        We should infer the periods per year
        """
        return r.std() * (periods_per_year**0.5)

    def sharpe_ratio(self, r, riskfree_rate, periods_per_year):
        """
        Computes the annualized sharpe ratio of a set of returns
        """
        # Computes the annual risk free rate per period
        rf_per_period = (1 + riskfree_rate)**(1 / periods_per_year) - 1
        excess_ret = r - rf_per_period
        ann_ex_ret = self.annualize_rets(excess_ret, periods_per_year)
        ann_vol = self.annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol

    def portfolio_return(self, weights, returns):
        """
        Weights --> Returns
        """
        return weights.T @ returns

    def portfolio_vol(self, weights, covmat):
        """
        Weights --> Vol
        """
        return (weights.T @ covmat @ weights)**0.5

    def summary_stats(self, r, riskfree_rate=0.03, periods_per_year=252):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
        """
        ann_r = r.aggregate(self.annualize_rets, periods_per_year=periods_per_year)
        ann_vol = r.aggregate(self.annualize_vol, periods_per_year=periods_per_year)
        ann_sr = r.aggregate(self.sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
        dd = r.aggregate(lambda r: self.drawdown(r).Drawdown.min())
        skew = r.aggregate(self.skewness)
        kurt = r.aggregate(self.kurtosis)
        cf_var5 = r.aggregate(self.var_gaussian, modified=True)
        hist_cvar5 = r.aggregate(self.cvar_historic)
        return pd.DataFrame({'Annualized Return': ann_r,
                             'Annualized Vol': ann_vol,
                             'Skewness': skew,
                             'Kurtosis': kurt,
                             'Cornish-Fisher VaR(5%)': cf_var5,
                             'Historic CVaR': hist_cvar5,
                             'Sharpe Ratio': ann_sr,
                             'Max Drawdown': dd
                             })            


#%% Plotting functions

# 'MACD_RSI'
def macd_rsi_plot(rsi_df, macd_df, macd_rsi_df, close_df,start_date, end_date,stock_name):
    
    rsi2 = sep(rsi_df,start_date,end_date)
    macd2 = sep(macd_df,start_date,end_date)
    macd_rsi2 = sep(macd_rsi_df,start_date,end_date)
    close = sep(close_df,start_date,end_date)

    close.index = pd.to_datetime(close.index)
    rsi2.index = pd.to_datetime(rsi2.index)
    macd2.index = pd.to_datetime(macd2.index)
    macd_rsi2.index = pd.to_datetime(macd_rsi2.index)
    
    buy_dates = macd_rsi2[macd_rsi2['Signal'] == 1].index
    sell_dates = macd_rsi2[macd_rsi2['Signal'] == -1].index

    buy_dates_str = [date.strftime("%Y-%m-%d") for date in buy_dates]
    sell_dates_str = [date.strftime("%Y-%m-%d") for date in sell_dates]
    all_dates = buy_dates_str + sell_dates_str
    all_colors = ['g']*len(buy_dates_str) + ['r']*len(sell_dates_str)
    
    prices = close.loc[:,'Close']
    
    indicators  = [    
        mpf.make_addplot((macd2['MACD']), color='#606060', panel=2, ylabel='MACD', secondary_y=False),
        mpf.make_addplot((macd2['Signal_Line']), color='#1f77b4', panel=2, secondary_y=False),
        mpf.make_addplot((macd2['bar_positive']), type='bar', color='#4dc790', panel=2),
        mpf.make_addplot((macd2['bar_negative']), type='bar', color='#fd6b6c', panel=2),
        
        mpf.make_addplot(rsi2['RSI'], color = 'orange', panel = 3, ylabel = 'RSI'),
        mpf.make_addplot([70]* len(close), color = 'gray', panel = 3, secondary_y = False, linestyle = '-'),
        mpf.make_addplot([35]* len(close), color = 'gray', panel = 3, secondary_y = False, linestyle = '-'),
        
        mpf.make_addplot(close['Close'], color = 'red', panel = 4, ylabel = 'Close')
        
    ]

    
    fig, axlist = mpf.plot(close,
                         type='candle',
                         volume=True,
                         addplot=indicators,
                         style =  'yahoo',
                         xrotation = 0,
                         datetime_format = '%Y-%m-%d',
                         figsize = (40,20),
                         returnfig = True,
                         mav = (20,55),
                         vlines=dict(vlines=all_dates, colors=all_colors, linewidths=1, alpha=0.5)
                    
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

    fig = plt.figure(figsize=(35,20))
    # fig.suptitle(stock_name)

    ax1 = fig.add_subplot(221, ylabel="Price")
    ax2 = fig.add_subplot(223, ylabel="RSI")

    ax1.set_title("MACD/RSI Strategy: " + str(stock_name)+' from ' +str(start_date) +' to '+ str(end_date),fontsize=18)
    ax1.get_xaxis().set_visible(True)

    prices.plot(ax=ax1, color='b', lw=1.1)
    ax1.plot(prices[macd_rsi2['Signal'] == 1], '^', markersize=7, color='g')
    ax1.plot(prices[macd_rsi2['Signal'] == -1], 'v', markersize=7, color='r')

    rsi2.RSI.plot(ax=ax2, color='b')
    ax2.set_ylim(0,100)
    ax2.set_title("RSI for: " + str(stock_name),fontsize=15)
    ax2.axhline(70, color='r', linestyle='--')
    ax2.axhline(30, color='r', linestyle='--')
    
    return mpf.show(), fig.show()
    
# 'MACD_MFI'
def macd_mfi_plot(mfi_df, macd_df, macd_mfi_df, close_df,start_date, end_date,stock_name):
    
    mfi2 = sep(mfi_df,start_date,end_date)
    macd2 = sep(macd_df,start_date,end_date)
    macd_mfi2 = sep(macd_mfi_df,start_date,end_date)
    close = sep(close_df,start_date,end_date)

    close.index = pd.to_datetime(close.index)
    mfi2.index = pd.to_datetime(mfi2.index)
    macd2.index = pd.to_datetime(macd2.index)
    macd_mfi2.index = pd.to_datetime(macd_mfi2.index)
    
    prices = close.loc[:,'Close']
    
    buy_dates = macd_mfi2[macd_mfi2['Signal'] == 1].index
    sell_dates = macd_mfi2[macd_mfi2['Signal'] == -1].index

    buy_dates_str = [date.strftime("%Y-%m-%d") for date in buy_dates]
    sell_dates_str = [date.strftime("%Y-%m-%d") for date in sell_dates]
    all_dates = buy_dates_str + sell_dates_str
    all_colors = ['g']*len(buy_dates_str) + ['r']*len(sell_dates_str)

    indicators  = [    
        mpf.make_addplot((macd2['MACD']), color='#606060', panel=2, ylabel='MACD', secondary_y=False),
        mpf.make_addplot((macd2['Signal_Line']), color='#1f77b4', panel=2, secondary_y=False),
        mpf.make_addplot((macd2['bar_positive']), type='bar', color='#4dc790', panel=2),
        mpf.make_addplot((macd2['bar_negative']), type='bar', color='#fd6b6c', panel=2),
        
        mpf.make_addplot(mfi2['MFI'], color = 'orange', panel = 3, ylabel = 'MFI'),
        mpf.make_addplot([70]* len(close), color = 'gray', panel = 3, secondary_y = False, linestyle = '-'),
        mpf.make_addplot([25]* len(close), color = 'gray', panel = 3, secondary_y = False, linestyle = '-'),
        
        mpf.make_addplot(close['Close'], color = 'red', panel = 4, ylabel = 'Close')
        
    ]

    
    fig,axlist = mpf.plot(close,
                         type='candle',
                         volume=True,
                         addplot=indicators,
                         style =  'yahoo',
                         xrotation = 0,
                         returnfig = True,
                         datetime_format = '%Y-%m-%d',
                         figsize = (40,20),
                         mav = (20,55),
                         vlines=dict(vlines=all_dates, colors=all_colors, linewidths=1, alpha=0.5)
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

    fig = plt.figure(figsize=(35,20))
    # fig.suptitle(stock_name)

    ax1 = fig.add_subplot(221, ylabel="Price")
    ax2 = fig.add_subplot(223, ylabel="RSI")

    ax1.set_title("MACD/MFI Strategy: " + str(stock_name) +' from ' +str(start_date) +' to '+ str(end_date), fontsize = 18)
    ax1.get_xaxis().set_visible(True)

    prices.plot(ax=ax1, color='b', lw=1.1)
    ax1.plot(prices[macd_mfi2['Signal'] == 1], '^', markersize=7, color='g')
    ax1.plot(prices[macd_mfi2['Signal'] == -1], 'v', markersize=7, color='r')

    mfi2.MFI.plot(ax=ax2, color='b')
    ax2.set_ylim(0,100)
    ax2.set_title("MFI for: " + str(stock_name),fontsize=15)
    ax2.axhline(70, color='r', linestyle='--')
    ax2.axhline(25, color='r', linestyle='--')
    
    return mpf.show(), fig.show()


def entry_plot(stock_name,signal_df,prices_df):    
    
    prices_df.index = pd.to_datetime(prices_df.index)
    signal_df.index = pd.to_datetime(signal_df.index)
    prices = prices_df.loc[:,'Close'].to_frame()
    returns = prices.pct_change().dropna(axis=0)
    returns.rename(columns = {'Close':'Returns'},inplace=True)    
    
    fig = plt.figure(figsize=(35,20))
    # fig.suptitle(stock_name)
    
    ax1 = fig.add_subplot(221, ylabel="Price")
    ax2 = fig.add_subplot(223, ylabel="Returns")
    
    ax1.set_title("Signals for: " + str(stock_name),fontsize=18)
    ax1.get_xaxis().set_visible(True)
    
    prices.plot(ax=ax1, color='b', lw=1)
    ax1.plot(prices[signal_df['Signal'] == 1], '^', markersize=7, color='g')
    ax1.plot(prices[signal_df['Signal'] == -1], 'v', markersize=7, color='r')
    
    sns.histplot(returns['Returns'], kde=True, ax=ax2)
    ax2.set_title("Returns Histogram", fontsize=15)

    return plt.show()

    
def inter_graph(df_x, macd_rsi_df, titulo=None):
    colors = ['green' if close >= open else 'red' for open, close in zip(df_x['Open'], df_x['Close'])]

    candlesticks = go.Candlestick(
        x=df_x.index,
        open=df_x['Open'],
        high=df_x['High'],
        low=df_x['Low'],
        close=df_x['Close'],
        showlegend=False,
        increasing_line_color='green',  
        decreasing_line_color='red',  
        name='Price'
    )
    volume_bars = go.Bar(
        x=df_x.index,
        y=df_x['Volume'],
        showlegend=False,
        marker={
            "color": colors,  
            "line": {
                "color": colors,
                "width": 1
            }
        },
        name='Volume'
    )

    
    macd_rsi_df.index = pd.to_datetime(macd_rsi_df.index)
    buy_dates = macd_rsi_df[macd_rsi_df['Signal'] == 1].index
    sell_dates = macd_rsi_df[macd_rsi_df['Signal'] == -1].index

    
    buy_points = go.Scatter(
        x=buy_dates[buy_dates.isin(df_x.index)],
        y=df_x.loc[buy_dates[buy_dates.isin(df_x.index)]]['Close'],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            symbol='triangle-up'
        ),
        showlegend=True,
        name='Buy'
    )

    
    sell_points = go.Scatter(
        x=sell_dates[sell_dates.isin(df_x.index)],
        y=df_x.loc[sell_dates[sell_dates.isin(df_x.index)]]['Close'],
        mode='markers',
        marker=dict(
            size=10,
            color='yellow',
            symbol='triangle-down'
        ),
        showlegend=True,
        name='Sell'
    )

   
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(candlesticks, secondary_y=True)
    fig.add_trace(volume_bars, secondary_y=False)
    fig.add_trace(buy_points, secondary_y=True)
    fig.add_trace(sell_points, secondary_y=True)
    
    fig.update_layout(
        title=titulo,
        height=800,
        template='plotly_dark',  
        title_font=dict(size=20),  
        autosize=True,
        hovermode="x",  
    )
    fig.update_yaxes(title_text="Price $", secondary_y=True, showgrid=True)
    fig.update_yaxes(title_text="Volume $", secondary_y=False, showgrid=False)

    offline.plot(fig, filename='plot2.html')
    fig.show()   
