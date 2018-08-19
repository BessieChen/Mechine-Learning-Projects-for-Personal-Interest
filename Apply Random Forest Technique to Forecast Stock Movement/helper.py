# Set of helper functions for main algorithm

import numpy as np
import quandl
import pandas as pd
import pandas_talib as talib
from Bessie_feature import EWMA,BBANDS,STOK,STO,MACD,CCI,RSI,Chaikin
import datetime
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf



# Adopted from: https://stackoverflow.com/questions/29721228/given-a-date-range-how-can-we-break-it-up-into-n-contiguous-sub-intervals
def date_range_new(start, end, prop, set_time = "None"):

    if set_time == "None":
        from datetime import datetime
        # start = datetime.strptime(start,"%b %d %Y")
        # end = datetime.strptime(end,"%b %d %Y")
        format = "%b %d %Y"
        start = start.strftime(format)
        end = end.strftime(format)
        diff = (end  - start ) * prop
        yield (start + diff).strftime("%b %d %Y")

    # If specific split time is included, return the specific time
    else:
        yield set_time

def date_range(start, end, prop, set_time = "None"):

    if set_time == "None":
        from datetime import datetime
        start = datetime.strptime(start,"%b %d %Y")
        end = datetime.strptime(end,"%b %d %Y")
        diff = (end  - start ) * prop
        yield (start + diff).strftime("%b %d %Y")

    # If specific split time is included, return the specific time
    else:
        yield set_time

def get_curr_date():

    # Gets current date as month-day-year
    now = datetime.datetime.now()
    now = now.strftime('%m %d %Y')

    # Reformats date with month as a string
    time_arr = now.split(" ")
    month = datetime.date(1900, int(time_arr[0]), 1).strftime('%b')
    print month
    time_arr[0] = month
    curr_time = " ".join(time_arr)

    return curr_time

def read_txt(fname):

    # Opens the file
    with open(fname) as f:
        content = f.readlines()

    # Reads the lines of the file and removes empty lines
    content = [line.strip() for line in content] #Reads the lines of the file
    # print content #['Algorithm Parameters', "Note: Not all stocks are supported in the Quandl 'Wiki' database.", '=======================', 'Stocks to Forecast (Comma Separated Valid Tickers):', 'AAPL, ADBE, ABBV, LUK, MAC, MSFT, PNR, PPL, PSA, SLB', '', 'Forecast Period (Integer):', '7']
    # content = [s for s in content if s != ""] #removes empty lines
    # print content #['Algorithm Parameters', "Note: Not all stocks are supported in the Quandl 'Wiki' database.", '=======================', 'Stocks to Forecast (Comma Separated Valid Tickers):', 'AAPL, ADBE, ABBV, LUK, MAC, MSFT, PNR, PPL, PSA, SLB', 'Forecast Period (Integer):', '7']

    # Finds index of the "Ticker" input and gets the stock array
    matching = [s for s in content if "Ticker" in s]
    # print matching #print: ['Stocks to Forecast (Comma Separated Valid Tickers):']
    tickers_index = content.index(matching[0])
    # print matching[0] #Stocks to Forecast (Comma Separated Valid Tickers):
    # print tickers_index #3
    stocks = [ticker.strip() for ticker in content[tickers_index+1].split(',')]

    # Finds the index of the "Forecast Period" input and gets the forecast period
    matching = [s for s in content if "Forecast Period" in s]
    pred_period = content.index(matching[0])
    period = content[pred_period+1]

    return stocks, int(period)

def exponential_smoothing(alpha, input_data):

    # Exponentially smooths input prices beginning from the most recent
    for i in reversed(range(0, len(input_data)-1)):
        input_data.iloc[i] = input_data.iloc[i+1]*(1-alpha) + alpha * input_data.iloc[i]

def create_shifted_orderbook(ticker, start_date, end_date, lag_period = 5, pred_period = 7):

    # Retrieve the Nifty data from Yahoo finance:
    format = '%Y-%m-%d'  # Formatting directives
    start = start_date.strftime(format)
    end = end_date.strftime(format)

    yf.pdr_override()  # <== that's all it takes :-)
    stock_data = pdr.get_data_yahoo(ticker, start=start, end=end)

    # Creates stock lag
    stock_data.dropna()
    stock_lag = pd.DataFrame(data = stock_data, index=stock_data.index)

    stock_returns = pd.DataFrame()

    # Initializes dataframe values and smooths the closing price data
    stock_data_smooth = stock_data['Adj Close']
    exponential_smoothing(0.7, stock_data_smooth) #so the stock_data_smooth is smoothing


    stock_lag["Close"] = stock_data_smooth #so, now the stock_lag["Close"] is derive from Adj Close + smoothing.

    # Sets lagging price data (previous days' price data as feature inputs)
    for i in range(0, lag_period):
        column_label = 'Lag{:d}'.format(i)
        stock_lag[column_label] = stock_lag['Close'].shift(1+i)

    # EMA- Momentum
    ndays = 30
    name_EWMA = 'EWMA_' + str(ndays)
    stock_lag['EWMA_'] = EWMA(stock_lag,ndays )[name_EWMA]

    # Bollinger Bands
    aa = BBANDS(stock_lag, ndays=30)
    stock_lag['upperband'] = aa['Upper BollingerBand']
    stock_lag['lowerband'] = aa['Lower BollingerBand']

    # StochK
    n = 30
    name_slowk = 'SO%k'
    name_slowd = 'SO%d_' + str(n)
    stock_lag['slowk'] = STOK(stock_lag)[name_slowk]
    stock_lag['slowd'] = STO(stock_lag, n)[name_slowd]

    # MACD- Momentum
    n_fast = 12
    n_slow = 26
    name_macd = 'MACD_' + str(n_fast) + '_' + str(n_slow)
    name_macdsignal = 'MACDsign_' + str(n_fast) + '_' + str(n_slow)
    name_macdhist = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow)
    macd = MACD(stock_lag, n_fast, n_slow)[name_macd]
    macdsignal = MACD(stock_lag, n_fast, n_slow)[name_macdsignal]
    stock_lag['macdhist'] = MACD(stock_lag, n_fast, n_slow)[name_macdhist]

    # CCI- Momentum
    stock_lag['CCI'] = CCI(stock_lag, ndays = 30)["CCI"]

    # Chaikin- Volume
    stock_lag['Chaikin'] = Chaikin(stock_lag)['Chaikin']

    stock_returns['Day Returns'] = stock_data['Adj Close'].pct_change() * 100
    # Sets lagging percent change data
    for i in range(0, lag_period):
        column_label = 'Lag{:d}'.format(i)
        stock_returns[column_label] = stock_lag[column_label].pct_change() * 100


    # Remove NaN's from stock lag
    print "shape of stock_lag before dropna: ",stock_lag.shape[0]
    stock_lag = stock_lag.dropna(axis=0, how='any')
    print "shape of stock_lag before dropna: ",stock_lag.shape[0]

    print "shape of stock_returns before dropna: ",stock_returns.shape[0]
    # Adjusts stock_return data to same length as stock_lag
    stock_returns = stock_returns.tail(stock_lag.shape[0])
    print "shape of stock_returns after dropna: ",stock_returns.shape[0]


    # Determine stock movement direction and lagging movement
    stock_movement = pd.DataFrame(index=stock_returns.index)
    stock_movement['Movement_0'] = np.sign(stock_returns['Day Returns'])
    stock_movement['Movement_0'][0] = 1
    for i in range(0, pred_period):
        column_label = 'Movement_{:d}'.format(i + 1)
        stock_movement[column_label] = stock_movement['Movement_0'].shift(i + 1)

    # Removes NaNs from 'stock_movement' and resizes 'stocks_returns' and 'stock_lag' accordingly
    print "shape of stock_movement before dropna: ",stock_movement.shape[0]
    stock_movement = stock_movement.dropna(axis=0, how='any')
    print "shape of stock_movement after dropna: ",stock_movement.shape[0]

    stock_returns = stock_returns[stock_returns.index <= stock_movement.index[stock_movement.index.__len__() - 1]]
    stock_returns = stock_returns.tail(stock_movement.shape[0])
    stock_lag = stock_lag[stock_lag.index <= stock_movement.index[stock_movement.index.__len__() - 1]]
    stock_lag = stock_lag.tail(stock_movement.shape[0])

    return stock_data, stock_returns, stock_lag, stock_movement
