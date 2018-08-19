import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlencode
from sklearn import cluster, covariance, manifold
import quandl
import pandas as pd
import datetime
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

print(__doc__)

# def retry(f, n_attempts=3):
#     "Wrapper function to retry function calls in case of exceptions"
#     def wrapper(*args, **kwargs):
#         for i in range(n_attempts):
#             try:
#                 return f(*args, **kwargs)
#             except Exception:
#                 if i == n_attempts - 1:
#                     raise
#     return wrapper
#
# def quotes_historical_google(symbol, start_date, end_date):
#
#     format = '%Y-%m-%d'  # Formatting directives
#     start = start_date.strftime(format)
#     end = end_date.strftime(format)
#
#     yf.pdr_override()  # <== that's all it takes :-)
#     data = pdr.get_data_yahoo(symbol, start=start, end=end)
#
#     return data
#
#
#
# symbol_dict = {
#     'NYSE:TOT': 'Total',
#     'NYSE:XOM': 'Exxon',
#
#     }
#
# # #############################################################################
# symbols, names = np.array(sorted(symbol_dict.items())).T
#
# symbols_new = [] #symbols without "NYSE"
# for n in range(len(symbols)):
#     symbols_new.append(symbols[n].split(":")[1])
# print symbols_new
#
#
# start = datetime.datetime(2018, 3, 10)
# end = datetime.datetime(2018, 4, 1)
#
# quotes = []
#
# for symbol in symbols_new:
#     print('Fetching quote history for %r' % symbol)
#     quotes.append(quotes_historical_google(
#         symbol, start, end))
# print type(quotes[0])
#
# empty_dataframe = []
# for q in quotes:
#     empty_dataframe.append(q.empty)
#
# symbols_valid = []
# name_valid = []
#
# for i,j in enumerate(empty_dataframe):
#     if j == False:
#         symbols_valid.append(symbols_new[i])
#         name_valid.append(names[i])
#
# print "drop the empty dataframe and then download the stock price data...."
#
# # #############################################################################
# start = datetime.datetime(2017, 1, 1)
# end = datetime.datetime(2018, 4, 1)
# quotes = []
#
# for symbol in symbols_valid:
#     print('Fetching quote history for %r' % symbol)
#     quotes.append(quotes_historical_google(
#         symbol, start, end))
#
# print "complete fetching the stock price data ...."
#
#
# close_prices = np.vstack([q.iloc[:,4] for q in quotes]) #4 stands for Adj Close
# open_prices = np.vstack([q.iloc[:,0] for q in quotes]) #0 stands for Open
#
# # The daily variations of the quotes are what carry most information
# variation = close_prices - open_prices
#
#
# # #############################################################################
#
#
# # standardize the time series: using correlations rather than covariance
# # is more efficient for structure recovery
# X = variation.copy().T
# print variation.shape
# print X.shape
# print X.std(axis=0)
# X /= X.std(axis=0)
# print X.shape

df = pd.DataFrame({"X": [1,1,2,1,1], "Y": ["A","D","B","C", "D"]})
print df[(df.X == 1) & df.Y.isin(["A","B","C"])]
print df[df.X == 1]
df.query("X == 1 and Y in ['A','B','C']")
