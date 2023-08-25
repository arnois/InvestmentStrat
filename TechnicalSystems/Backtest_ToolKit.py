# Description

"""
-Our Python code for our trading strategy with backtesting:

This code allows us to backtest a trading strategy for multiple 
assets using predefined signals. Price and position data are loaded 
from CSV files, while initial parameters are loaded from an Excel file.

The backtesting is performed by simulating buy and sell transactions based 
on the provided signals. The backtesting results, including portfolio value, 
executed transactions, commissions, and other details, are stored and can be 
visualized through charts.

The code is structured in a single class, 'BacktestStrategy', which includes 
methods for data loading, backtesting, return calculation, and result visualization.

Dependencies:

-pandas: for data manipulation and analysis.
-numpy: for mathematical operations.
-matplotlib and mplfinance: for data visualization.
-seaborn: for data visualization.
-talib: for technical analysis tools.
-datetime and time: for operations related to date and time.
-xlsxwriter: for writing Excel files.
-typing: for data type support.

"""

# Libraries 

import pandas as pd
import numpy as np
import datetime
import time 
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from typing import List


# ------------------------ #

# Backtesting

# ------------------------ #

# 'We import our dataframes and parameters'

# prices = pd.read_csv('Prices.csv', index_col=0)
# positions = pd.read_csv('Positions.csv', index_col=0)
# parameters = pd.read_excel('Parameters.xlsx', index_col=0) 
# initial_capital = parameters['Values'].loc['Initial_capital']
# commission_percentage = parameters['Values'].loc['Commission']
# variable_commission = parameters['Values'].loc['Variable_Commission']
# dollar_amount = parameters['Values'].iloc[-1]
# start_date = pd.to_datetime(parameters['Values'].loc['Start_date'])
# end_date = pd.to_datetime(parameters['Values'].loc['End_date'])
# prices.index = pd.to_datetime(prices.index)
# positions.index = pd.to_datetime(positions.index)

def find_closest_date(df, target_date):
    
    """Find the closest date in a DataFrame to the target date.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dates.
    target_date : str
        Target date in string format.

    Returns
    -------
    str
        Closest date to the target date in string format.
    """
    
    closest_date = df.index.get_loc(target_date, method='nearest')
    return df.index[closest_date]

def backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float,
    commission_percentage: float,
    variable_commission: bool,
    dollar_amount: float,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    
    """
    Backtest a trading strategy

    Parameters
    ----------
    prices : pd.DataFrame
        Dataframe containing prices for different assets
    signals : pd.DataFrame
        Dataframe containing buy/sell signals for different assets
    initial_capital : float
        Initial capital to start trading
    commission_percentage : float
        Percentage of commission to be applied on trades
    variable_commission : bool
        Indicator whether the commission is variable or fixed
    dollar_amount : float
        Dollar amount to be used for trades
    start_date : str
        Start date for backtesting
    end_date : str
        End date for backtesting

    Returns
    -------
    pd.DataFrame
        Dataframe with results of the backtest
    """

    accumulated_shares = pd.Series(0, index=prices.columns)
    results_df = pd.DataFrame(index=prices.index, columns=['Bought_Amount', 'Num_Shares_Bought', 'Sold_Amount', 
                                                           'Num_Shares_Sold', 'Accumulated_Shares', 'Portfolio_Value', 
                                                           'Cash', 'Profit_Loss', 'Total_Portfolio_Value', 'Shares_Bought', 
                                                           'Shares_Sold', 'Commissions'])
    cash = initial_capital
    portfolio_value = initial_capital
    previous_portfolio_value = 0
    prices = prices.fillna(0)

    if start_date is None:
        start_date = max(prices.index.min(), signals.index.min())
    elif start_date not in prices.index or start_date not in signals.index:
        closest_start_date = find_closest_date(prices, start_date)
        start_date = closest_start_date

    if end_date is None:
        end_date = min(prices.index.max(), signals.index.max())
    elif end_date not in prices.index or end_date not in signals.index:
        closest_end_date = find_closest_date(prices, end_date)
        end_date = closest_end_date

    prices = prices.loc[start_date:end_date]
    signals = signals.loc[start_date:end_date]
    
    for date, prices_row in prices.iterrows():
        signals_row = signals.loc[date]
        bought_amount = sold_amount = num_shares_bought = num_shares_sold = 0
        commissions = 0  
        shares_bought: List[str] = []
        shares_sold: List[str] = []

        for ticker, signal in signals_row.iteritems():
            price = prices_row[ticker]

            if signal == 1:
                dollars_per_share = dollar_amount / price
                num_shares_to_buy = int(dollars_per_share)
                buy_amount = price * num_shares_to_buy
                commission = commission_percentage * buy_amount
                buy_amount += commission

                if cash >= buy_amount:
                    cash -= buy_amount
                    accumulated_shares[ticker] += num_shares_to_buy
                    bought_amount += buy_amount
                    num_shares_bought += num_shares_to_buy
                    shares_bought.append(ticker)
                    commissions += commission

            elif signal == -1:
                quantity = accumulated_shares[ticker]
                if quantity > 0:
                    sell_amount = price * quantity
                    commission = commission_percentage * sell_amount
                    sell_amount -= commission
                    cash += sell_amount
                    accumulated_shares[ticker] -= quantity
                    sold_amount += sell_amount
                    num_shares_sold += quantity
                    shares_sold.append(ticker)
                    commissions += commission

        portfolio_value = sum(accumulated_shares * prices_row)
        profit_loss = portfolio_value - previous_portfolio_value
        previous_portfolio_value = portfolio_value

        total_portfolio_value = portfolio_value + cash
        results_df.loc[date] = [bought_amount, num_shares_bought, sold_amount, num_shares_sold, accumulated_shares.sum(),
                                portfolio_value, cash, profit_loss, total_portfolio_value, shares_bought, shares_sold,
                                commissions]

    return results_df.dropna(axis=0)

# results = backtest(prices, positions, initial_capital, commission_percentage, variable_commission, dollar_amount)
# returns = results['Total_Portfolio_Value'].pct_change().to_frame()
# returns.columns = ['Returns']

# 'Example 2'

# results2 = backtest(prices, positions, initial_capital, commission_percentage, variable_commission, dollar_amount,
#                     start_date,end_date)

#%%


"Let's plot"

def plot_portfolio_analysis(results: pd.DataFrame, returns: pd.DataFrame, initial_capital: float) -> None:
    
    """Plot portfolio analysis charts.

    Parameters:
        results (pd.DataFrame): Dataframe with backtest results.
        returns (pd.DataFrame): Dataframe with portfolio returns.
        initial_capital (float): Initial capital used for the backtest.

    Returns:
        None
    """

    # Plot portfolio value over time
    ax = results.loc[results['Portfolio_Value'] > 0, 'Portfolio_Value'].plot(grid=True, figsize=(15, 10))
    ax.set_title('Portfolio Value Over Time', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()

    # Plot total portfolio value over time
    ax2 = results.loc[results['Total_Portfolio_Value'] > initial_capital, 'Total_Portfolio_Value'].plot(grid=True, figsize=(15, 10))
    ax2.set_title('Total Portfolio Value Over Time')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    plt.show()

    # Plot returns over time
    ax3 = returns.loc[returns['Returns'] != 0, 'Returns'].plot(grid=True, figsize=(15, 10))
    ax3.set_title('Returns (Total Portfolio Value)', fontsize=14)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    plt.show()

    # Plot distribution of returns
    plt.figure(figsize=(15, 10))
    sns.set_style('whitegrid')
    sns.distplot(returns.dropna(), kde=True, hist_kws={'edgecolor': 'black', 'linewidth': 1})
    plt.title('Distribution of Returns (Total Portfolio Value)', fontsize=14)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()

# plot_portfolio_analysis(results, returns, initial_capital)

# results.index = pd.to_datetime(results.index)
# returns.index = pd.to_datetime(returns.index)

def subplot_portfolio_analysis(results: pd.DataFrame, returns: pd.DataFrame) -> None:
    
    """Plot portfolio analysis charts in subplots.

    Parameters:
        results (pd.DataFrame): Dataframe with backtest results.
        returns (pd.DataFrame): Dataframe with portfolio returns.

    Returns:
        None
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 15))

    # Plot portfolio value over time
    axs[0, 0].plot(results.loc[results['Portfolio_Value'] > 0, 'Portfolio_Value'])
    axs[0, 0].set_title('Portfolio Value Over Time', fontsize=18)
    axs[0, 0].set_facecolor('white')
    axs[0, 0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[0, 0].xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))

    start_index = (results['Total_Portfolio_Value'] != results.iloc[0]['Total_Portfolio_Value']).idxmax()

    # Plot total portfolio value over time
    axs[0, 1].plot(results.loc[start_index:, 'Total_Portfolio_Value'])
    axs[0, 1].set_title('Total Portfolio Value Over Time', fontsize=18)
    axs[0, 1].set_facecolor('white')
    axs[0, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[0, 1].xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))

    # Plot returns over time
    axs[1, 0].plot(returns.loc[returns['Returns'] != 0, 'Returns'])
    axs[1, 0].set_title('Returns (Total Portfolio Value)', fontsize=18)
    axs[1, 0].set_xlabel('Date', fontsize=16)
    axs[1, 0].set_ylabel('Returns', fontsize=16)
    axs[1, 0].set_facecolor('white')
    axs[1, 0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[1, 0].xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))

    # Plot distribution of returns
    sns.set_style('whitegrid')
    sns.distplot(returns.dropna(), kde=True, hist_kws={'edgecolor': 'black', 'linewidth': 1}, ax=axs[1, 1])
    axs[1, 1].set_title('Distribution of Returns (Total Portfolio Value)', fontsize=18)
    axs[1, 1].set_xlabel('Returns', fontsize=16)
    axs[1, 1].set_ylabel('Frequency', fontsize=16)

    axs[0, 0].tick_params(axis='x', labelsize=14)
    axs[0, 1].tick_params(axis='x', labelsize=14)
    axs[1, 0].tick_params(axis='x', labelsize=14)
    axs[1, 1].tick_params(axis='x', labelsize=14)

    plt.subplots_adjust(hspace=1)
    plt.tight_layout()
    plt.show()

# subplot_portfolio_analysis(results, returns)
