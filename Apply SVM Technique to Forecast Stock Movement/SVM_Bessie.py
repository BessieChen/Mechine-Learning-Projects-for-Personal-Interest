import csv
import numpy as np
from sklearn.svm import SVR# for regression
from sklearn.svm import SVC# for classification
from sklearn.svm import SVC# for classification

import matplotlib.pyplot as plt
from fit_model import fit_model, fit_model_cross_validation
from helper import date_range, create_shifted_orderbook, read_txt, get_curr_date,exponential_smoothing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import datetime
import pandas as pd
import datetime
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import quandl
import pandas_talib as talib
from Bessie_feature import EWMA,BBANDS,STOK,STO,MACD,CCI,RSI,Chaikin
import datetime
from pandas_datareader import data as pdr
from sklearn.model_selection import TimeSeriesSplit


def fit_feature_importance_cross_validation(ticker, feature_label_list, forest, X_data, y_data, splits=3):
    from sklearn.model_selection import TimeSeriesSplit
    #an example of TimeSeriesSplit
    # >> > for train_index, test_index in tscv.split(X):
    #     ...
    #     print("TRAIN:", train_index, "TEST:", test_index)
    # ...
    # X_train, X_test = X[train_index], X[test_index]
    # ...
    # y_train, y_test = y[train_index], y[test_index]
    # TRAIN: [0]
    # TEST: [1]
    # TRAIN: [0 1]
    # TEST: [2]
    # TRAIN: [0 1 2]
    # TEST: [3]

    # Initializes time series split object
    time_series_cv = TimeSeriesSplit(n_splits=splits)
    split_cnt = 1

    # Create time series split indices. Trains and tests
    # model on split data
    for train_index, test_index in time_series_cv.split(X_data):

        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print accuracy
        print " Cross Valid "+ str(split_cnt) + " for %s Finished"  %ticker
        split_cnt = split_cnt + 1


        # Print the feature ranking
        print "Feature ranking for %s:" %ticker

        for f in range(X_train.shape[1]):
            print "No.%d feature %d %s (%f)" % (f + 1, indices[f],feature_label_list[indices[f]], importances[indices[f]])

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importance for %s:" %ticker)
        plt.bar(range(X_train.shape[1]), importances[indices],
                color="g", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.grid()
        plt.show()


def create_shifted_orderbook_more_feature(ticker, start_date, end_date, lag_period = 5, pred_period = 7):
    """
    Shifts order book data. Determines stock movement direction. Incorporates technical indicators as training features.
    :param ticker: stock ticker
    :param start_date: start date for stock data collection period
    :param end_date: end date for stock data collection period
    :param lag_period: number of previous prices trained on
    :param pred_period: number of days forecast
    :return:
            stock_data: the price/volume info for a stock
            stock_returns: percent change between days and lagging percent changes (i.e. previous days' changes)
            stock_lag: stock price and lagging stock price
            stock_movement: stock movement direction (+1: increase, -1: decrease)
    """
    # Retrieve the Nifty data from Yahoo finance:
    format = '%Y-%m-%d'  # Formatting directives
    start = start_date.strftime(format)
    end = end_date.strftime(format)

    yf.pdr_override()  # <== that's all it takes :-)
    stock_data = pdr.get_data_yahoo(ticker, start=start, end=end)

    # Creates stock lag
    stock_data.dropna()
    stock_lag = pd.DataFrame(data = stock_data, index=stock_data.index)

    stock_returns = pd.DataFrame(data = stock_data, index=stock_data.index)

    # Initializes dataframe values and smooths the closing price data
    stock_data_smooth = stock_data['Adj Close']
    exponential_smoothing(0.7, stock_data_smooth) #so the stock_data_smooth is smoothing


    #stock_lag['Volume'] = stock_returns['Volume'] = stock_data['Volume']
    stock_lag["Close"] = stock_data_smooth #so, now the stock_lag["Close"] is derive from Adj Close + smoothing.
    #print stock_lag["Close"]

    # Sets lagging price data (previous days' price data as feature inputs)
    for i in range(0, lag_period):
        column_label = 'Lag{:d}'.format(i)
        stock_lag[column_label] = stock_lag['Close'].shift(1+i)

    # EMA- Momentum
    #stock_lag['EMA'] = talib.EMA(close, timeperiod = 30)
    ndays = 30
    name_EWMA = 'EWMA_' + str(ndays)
    stock_lag['EWMA_'] = EWMA(stock_lag,ndays )[name_EWMA]

    # Bollinger Bands
    #stock_lag['upperband'], stock_lag['middleband'], stock_lag['lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    aa = BBANDS(stock_lag, ndays=30)
    stock_lag['upperband'] = aa['Upper BollingerBand']
    stock_lag['lowerband'] = aa['Lower BollingerBand']

    # StochK
    #stock_lag['slowk'], stock_lag['slowd'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3,
    #                     slowd_matype=0)
    n = 30
    name_slowk = 'SO%k'
    name_slowd = 'SO%d_' + str(n)
    stock_lag['slowk'] = STOK(stock_lag)[name_slowk]
    stock_lag['slowd'] = STO(stock_lag, n)[name_slowd]

    # MACD- Momentum
    #macd, macdsignal, stock_lag['macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    n_fast = 12
    n_slow = 26
    name_macd = 'MACD_' + str(n_fast) + '_' + str(n_slow)
    name_macdsignal = 'MACDsign_' + str(n_fast) + '_' + str(n_slow)
    name_macdhist = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow)
    macd = MACD(stock_lag, n_fast, n_slow)[name_macd]
    macdsignal = MACD(stock_lag, n_fast, n_slow)[name_macdsignal]
    stock_lag['macdhist'] = MACD(stock_lag, n_fast, n_slow)[name_macdhist]

    # CCI- Momentum
    #stock_lag['CCI'] = talib.CCI(high, low, close)
    stock_lag['CCI'] = CCI(stock_lag, ndays = 30)["CCI"]
    #print stock_lag['CCI']

    # # RSI- Momentum
    # #stock_lag['RSI'] = talib.RSI(close, timeperiod=14)
    # ndays = 14
    # name_RSI = 'RSI_' + str(ndays)
    # stock_lag['RSI'] = RSI(stock_lag, n = ndays)[name_RSI]
    # #print stock_lag['RSI']


    # Chaikin- Volume
    #stock_lag['Chaikin'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    stock_lag['Chaikin'] = Chaikin(stock_lag)['Chaikin']
    #print stock_lag['Chaikin']

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



def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def predict_price(ticker, X, Y, x = 20 , splits = 3):


    time_series_cv = TimeSeriesSplit(n_splits=splits)
    split_cnt = 1

    for train_index, test_index in time_series_cv.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models
        svr_lin = SVR(kernel='linear', C=1e3)
        svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        print "doing the fitting"
        svr_rbf.fit(X_train, y_train)  # fitting the data points in the models
        print "complete the fitting"
        #svr_lin.fit(X_train, y_train)
        svr_poly.fit(X_train, y_train)
        print "complete the fitting2"

        # Print accuracy
        print " Cross Valid " + str(split_cnt) + " for %s Finished" % ticker
        split_cnt = split_cnt + 1

        plt.scatter(X_train, y_train, color='black', label='Data')  # plotting the initial datapoints
        plt.plot(X_train, svr_rbf.predict(X_train), color='red', label='RBF model')  # plotting the line made by the RBF kernel
        #plt.plot(X_train, svr_lin.predict(X_train), color='green', label='Linear model')  # plotting the line made by linear kernel
        plt.plot(X_train, svr_poly.predict(X_train), color='blue',
                 label='Polynomial model')  # plotting the line made by polynomial kernel
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Support Vector Regression')
        plt.legend()
        plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]



if __name__ == "__main__":

    # Reads user parameters from text file
    stocks, period = read_txt('parameters.txt')

    # List of stocks to input
    ticker_arr = stocks

    # Number of days' price data used for training
    lag = 20
    # Forward prediction period
    pred_period = period

    # Start and end dates of stock data set queried
    # start_date = "Jun 1 2000"
    # end_date = get_curr_date()
    #print get_curr_date()
    start = datetime.datetime(2017, 6, 1)
    end = datetime.datetime(2018, 4, 1)

    # Trains and tests on each ticker
    for ticker in ticker_arr[0]:
        # Retrieves stock info and returns training/testing data
        stock_data, stock_returns, stock_lag, stock_movement = create_shifted_orderbook(ticker, start_date=start,
                                                                                        end_date=end,
                                                                                        lag_period=lag,
                                                                                        pred_period=pred_period)

        # Technical indicator list
        # feature_label_list = ['Volume', 'EMA', 'upperband', 'middleband', 'lowerband', 'macdhist', 'CCI', 'RSI', 'WILLR',
        #                     'Chaikin', 'slowk', 'slowd']
        feature_label_list = ['EWMA_', 'upperband', 'lowerband', 'slowk', 'slowd', "macdhist", 'CCI', 'Chaikin']

        # Adds price features to feature list
        for i in range(0, lag):
            feature_label = 'Lag{:d}'.format(i)
            feature_label_list.append(feature_label)

        # Gets data with the features from feature list
        stock_feature_df = stock_lag[feature_label_list]


        # feature_importance
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)

        # print stock_lag.axes
        # print stock_returns.axes  #get the column name of stock_returns
        # print stock_feature_df.axes #get the column name of stock_feature_df
        print stock_returns["Adj Close"]
        aa = pd.DataFrame()
        aa["Lag0"] = stock_returns["Lag0"]
        print aa.values
        print aa.shape
        aa = np.ravel(aa)
        print len(aa)
        print aa

        predicted_price = predict_price(ticker, stock_feature_df.values, stock_returns.values)

        print "\nThe stock open price for 29th Feb is:"
        print "RBF kernel: $", str(predicted_price[0])
        print "Linear kernel: $", str(predicted_price[1])
        print "Polynomial kernel: $", str(predicted_price[2])

#predicted_price = predict_price(dates, prices, 29)

