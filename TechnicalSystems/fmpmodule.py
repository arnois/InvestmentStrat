
# Libraries 

import pandas as pd
import numpy as np
import datetime
import time
import requests
import urllib.request as urllib2, json
import xlsxwriter
from io import StringIO

# -------------------------------------------- #

# Financial Modeling Prep

# -------------------------------------------- #

"""
Financial Modeling Prep Api:
It gives us access to various historical financial data, such as financial statements, market data, metrics, ratios, etc.    
"""

api_key = '70f5d7b2d3b80682f0f157c8d097955d'
# Api Key
# api_key = 'your api key'

'Functions'

def mkt_data(stock):
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.

    Returns
    -------
    data : DataFrame
        Give us a DataFrame with the
        prices of the stock.
    '''
    url = 'https://financialmodelingprep.com/api/v3/historical-price-full/'+stock+'?apikey='+api_key
    response = urllib2.urlopen(url)
    data = json.loads(response.read())
    data = data['historical']
    data = pd.DataFrame(data).set_index('date')
    return data 


# apple = mkt_data('AAPL')
# stocks = ['AAPL','NVDA']

def mkt_data_dict(stocks):
    '''
    Parameters
    ----------
    stocks : List of strings
        Receives a list with
        tickers.

    Returns
    -------
    dict_stocks : Dictionary
        Returns a dictionary of dataframes
        of the prices of the given stocks.
    '''
    dict_stocks = {}
    for i in range(len(stocks)):
        url = 'https://financialmodelingprep.com/api/v3/historical-price-full/'+stocks[i]+'?apikey='+api_key
        response = urllib2.urlopen(url)
        data = json.loads(response.read())
        data = data['historical']
        data = pd.DataFrame(data).set_index('date')
        dict_stocks[str(stocks[i])] = data
    return dict_stocks

        
# dict1 = mkt_data_dict(stocks)

def inc_state(stock):
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.

    Returns
    -------
    data : DataFrame
        Returns the company´s income statement.
    '''
    
    url_fin = 'https://financialmodelingprep.com/api/v3/income-statement/'+stock+'?limit=120&apikey='+api_key
    response = urllib2.urlopen(url_fin)
    data = json.loads(response.read())
    data = pd.DataFrame(data).set_index('date')
    return data

# femsa_inc = inc_state('KOF')

def bal_sheet(stock):
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.

    Returns
    -------
    data : DataFrame
        Returns the company´s balance sheet.
    '''
    url_bal = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/'+stock+'?limit=120&apikey='+api_key
    response = urllib2.urlopen(url_bal)
    data = json.loads(response.read())
    data = pd.DataFrame(data).set_index('date')
    return data

# apple_bal = bal_sheet('AAPL')

# dict_incs = {}
# for i in range(len(stocks)):
#   dict_incs[str(stocks[i])] = inc_state(stocks[i])
    
    
# dict_bals = {}
# for i in range(len(stocks)):
#   dict_bals[str(stocks[i])] = bal_sheet(stocks[i])

# start_date1 = '2018-01-01'
# end_date1 = '2021-09-30'
# stock = 'AAPL'
# interval = '1min'


def daily_prices(stock,start_date,end_date):
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.
    
    starte_date : str initial date
    end_date : str final date

    Returns
    -------
    data : DataFrame
        Returns the company´s daily prices.
    '''
    url = 'https://financialmodelingprep.com/api/v3/historical-price-full/'+stock+'?from='+start_date+'&to='+end_date+'&apikey='+api_key
    response = urllib2.urlopen(url)
    data = json.loads(response.read())
    data = data['historical']
    data = pd.DataFrame(data).set_index('date')
    return data

# apple_prices = daily_prices(stock, start_date1, end_date1)  

  
def historical_prices(stock,interval):
    
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.
    
    interval : str
        Receives 1min, 5min, 15min, 30min, 1,hour
        https://site.financialmodelingprep.com/developer/docs/#Stock-Historical-Price

    Returns
    -------
    data : DataFrame
        Returns the company´s historical prices.
    '''
    url = 'https://financialmodelingprep.com/api/v3/historical-chart/'+interval+'/'+stock+'?&apikey='+api_key
    response = urllib2.urlopen(url)
    data = json.loads(response.read())
    data = pd.DataFrame(data).set_index('date')
    return data

# nvda_prices_m = historical_prices(stock,'1min')
    
def inc_state_per(stock, period = 'annual'):
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.
    
    period : str
        Receives annual or quarter

    Returns
    -------
    data : DataFrame
        Returns the company´s Income Statement.
    '''
    
    if period == 'quarter':
        url_fin = 'https://financialmodelingprep.com/api/v3/income-statement/'+stock+'?period=quarter&limit=120&apikey='+api_key
        response = urllib2.urlopen(url_fin)
        data = json.loads(response.read())
        data = pd.DataFrame(data).set_index('date')
    
    elif period == 'annual':
        url_fin = 'https://financialmodelingprep.com/api/v3/income-statement/'+stock+'?limit=120&apikey='+api_key
        response = urllib2.urlopen(url_fin)
        data = json.loads(response.read())
        data = pd.DataFrame(data).set_index('date')
        
    else:
        print('invalid period')
    return data


# apple_inc = inc_state_per('AAPL','annual')
# arca = inc_state_per('AC.MX','quarter')
# alfaa = inc_state_per('ALFAA.MX','quarter')
# volara = inc_state_per('VOLARA.MX','quarter')

    
def balance_sheet_per(stock, period = 'annual'):
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.
    
    period : str
        Receives annual or quarter

    Returns
    -------
    data : DataFrame
        Returns the company´s balance sheet.
    '''
    
    
    if period == 'quarter':
        url_fin = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/'+stock+'?period=quarter&limit=120&apikey='+api_key
        response = urllib2.urlopen(url_fin)
        data = json.loads(response.read())
        data = pd.DataFrame(data).set_index('date')
    
    elif period == 'annual':
        url_fin = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/'+stock+'?limit=120&apikey='+api_key
        response = urllib2.urlopen(url_fin)
        data = json.loads(response.read())
        data = pd.DataFrame(data).set_index('date')
        
    else:
        print('invalid period')
    return data
    
# volarabs = balance_sheet_per('VOLARA.MX','quarter')



def inc_statement(stock,start_date,end_date, period = 'annual'):
    
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.
    
    period : str
        Receives annual or quarter

    Returns
    -------
    data : DataFrame
        Returns the company´s Income Statement.
    '''
    
    if period == 'annual':
        
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        diff = end_date.year - start_date.year
        diff = diff + 1
        periods = str(diff)
    
        url_fin = 'https://financialmodelingprep.com/api/v3/income-statement/'+stock+'?period=annual&limit='+periods+'&apikey='+api_key
        response = urllib2.urlopen(url_fin)
        data = json.loads(response.read())
        data = pd.DataFrame(data).set_index('date')
    
    elif period == 'quarter':
        
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        diff = int((diff/3)+1)
        periods = str(diff)
    
        url_fin = 'https://financialmodelingprep.com/api/v3/income-statement/'+stock+'?period=quarter&limit='+periods+'&apikey='+api_key
        response = urllib2.urlopen(url_fin)
        data = json.loads(response.read())
        data = pd.DataFrame(data).set_index('date')
    
    else:
        print('No available')
    
    return data

# url = 'https://financialmodelingprep.com/api/v4/financial-reports-dates?symbol=AAPL&apikey='+api_key
# response = urllib2.urlopen(url)
# data = json.loads(response.read())
# data = pd.DataFrame(data).set_index('date')

'BIEN '

# url = 'https://financialmodelingprep.com/api/v4/financial-reports-json?symbol=AAPL&year=2020&period=FY&apikey='+api_key
# response = urllib2.urlopen(url)
# data = json.loads(response.read())
# data = data['Cover Page'][4]
# data = data['Document Period End Date'][0]

# stock = 'AAPL'
def company_sector(stock):
    
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.
    
    Returns
    -------
    data : DataFrame
        Returns the company´s sector.
    '''
    url = 'https://financialmodelingprep.com/api/v3/profile/'+stock+'?apikey='+api_key
    response = urllib2.urlopen(url)
    data = json.loads(response.read())
    data = pd.DataFrame(data[0],index = [0]).set_index('symbol')
    return data



def company_ratios(stock):
    
    '''
    Parameters
    ----------
    stock : str
        Receives the ticker.
    
    Returns
    -------
    data : DataFrame
        Returns the company´s fundamental ratios.
    '''
    url2 = 'https://financialmodelingprep.com/api/v3/profile/'+stock+'?apikey='+api_key
    response2 = urllib2.urlopen(url2)
    data = json.loads(response2.read())
    data = pd.DataFrame(data[0],index = [0]).set_index('symbol')
    url = 'https://financialmodelingprep.com/api/v3/ratios-ttm/'+stock+'?apikey='+api_key
    response = urllib2.urlopen(url)
    data2 = json.loads(response.read())
    data2 = pd.DataFrame(data2[0],index = [stock])
    data3 = pd.concat([data,data2.loc[:,['returnOnAssetsTTM', 'returnOnEquityTTM','peRatioTTM',
                                         'ebtPerEbitTTM','netProfitMarginTTM']]],
                                          axis=1)
    data4 = data3[['sector','industry','returnOnAssetsTTM', 'returnOnEquityTTM','peRatioTTM',
                   'ebtPerEbitTTM','netProfitMarginTTM']]
    return data4



# apple_test = company_ratios('AAPL')

# lista_stocks = ['AAPL','NVDA','MSFT','TSLA','AMZN','META']

# try:
#     lista_dfs = [company_ratios(stocks) for stocks in lista_stocks]
# except:
#     pass

# final_df = pd.concat(lista_dfs)



