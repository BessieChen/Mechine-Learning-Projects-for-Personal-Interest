import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from markov_Bessie001 import MarkowitzOpt
from helper import read_txt, create_shifted_orderbook, get_curr_date,date_range



interest_rate = 0.03/12								# Fixed interest rate
min_return = 0.02									# Minimum desired return


stocks, period = read_txt('parameters.txt')
print stocks,period

#stocks = ['AAPL','IBM','MSFT','GOOG','QCOM']		# Stock ticker names
ticker_arr = stocks

start_date = datetime.datetime(2005, 1, 1)
end_date = datetime.datetime(2018, 4, 1)
format = '%Y-%m-%d'  # Formatting directives
start = start_date.strftime(format)
end = end_date.strftime(format)

# Specify number of days to shift
shift = 20
shift_returns = pd.DataFrame()
stock_price = pd.DataFrame()
print shift_returns

for ticker in ticker_arr:
    yf.pdr_override()  # <== that's all it takes :-)
    stock_data = pdr.get_data_yahoo(ticker, start=start, end=end)
    stock_data.dropna()
    print type(stock_data["Adj Close"])
    # Compute returns over the time period specified by shift
    name_price = "Price of " +ticker
    name = "Return of "+ticker
    stock_price[name_price] = stock_data["Adj Close"]
    shift_returns[name] = stock_data["Adj Close"] / stock_data["Adj Close"].shift(shift) - 1

shift_returns = shift_returns.dropna(axis=0, how='any')
stock_price = stock_price.dropna(axis=0, how='any')

print shift_returns






# Creates stock lag


#
# print "ticker_adj_close",ticker_adj_close
# print "shift_returns",shift_returns
# print type(shift_returns[0])


# Specify filter "length"
filter_len = shift
#
# shift_returns = shift_returns[0]  #previously, shift_returns is list, now convert it to 'pandas.core.series.Series'
# print shift_returns[1]
# Compute mean and variance
shift_returns_mean = shift_returns.ewm(span=filter_len).mean()
shift_returns_mean = shift_returns_mean[shift_returns_mean.index != shift_returns_mean.index[0]]
print "shift_returns_mean",shift_returns_mean
#print "shift_returns_mean",shift_returns_mean
shift_returns_var = shift_returns.ewm(span=filter_len).var()
shift_returns_var = shift_returns_var.dropna(axis=0, how='any')
print "shift_returns_var",shift_returns_var
#print "shift_returns_var",shift_returns_var
# Compute covariances
NumStocks = len(ticker_arr)
covariance = pd.DataFrame()
for FirstStock in np.arange(NumStocks-1):
    for SecondStock in np.arange(FirstStock+1,NumStocks):
        print ticker_arr[FirstStock]
        ColumnTitle = ticker_arr[FirstStock] + '-' + ticker_arr[SecondStock]
        print ColumnTitle
        covariance[ColumnTitle] = shift_returns["Return of "+ ticker_arr[FirstStock]].ewm(span=filter_len).cov(shift_returns["Return of "+ ticker_arr[SecondStock]])
print covariance

# Variable Initialization
start_date = '2013-02-04'
index = shift_returns.index
print 'shift_returns',shift_returns
print index
start_index = index.get_loc(start_date)
start_index2 = index[2]
end_date = index[-1]
end_index = index.get_loc(end_date)
date_index_iter = start_index
ticker_arr.append('Interest_Rate')
distribution = DataFrame(index=ticker_arr)
print distribution
returns = Series(index=index)
# Start Value
total_value = 1.0
returns[index[date_index_iter]] = total_value

while date_index_iter + 20 < end_index:
    date = index[date_index_iter]
    portfolio_alloc = MarkowitzOpt(shift_returns_mean.ix[date], shift_returns_var.ix[date], covariance.ix[date],
                                   interest_rate, min_return)
    distribution[date.strftime('%Y-%m-%d')] = portfolio_alloc

    # Calculating portfolio return
    date2 = index[date_index_iter + shift]
    temp1 = stock_price.ix[date2] / stock_price.ix[date]
    print temp1
    temp1.ix[ticker_arr[-1]] = interest_rate + 1
    temp2 = Series(np.array(portfolio_alloc.ravel()).reshape(len(portfolio_alloc)), index=ticker_arr)
    total_value = np.sum(total_value * temp2 * temp1)
    # Increment Date
    date_index_iter += shift
    returns[index[date_index_iter]] = total_value
    print date_index_iter

print returns

# Remove dates that there are no trades from returns
returns = returns[np.isfinite(returns)]

print returns
# Plot portfolio allocation of last 10 periods
ax = distribution.T.ix[-10:].plot(kind='bar',stacked=True)
plt.ylim([0,1])
plt.xlabel('Date')
plt.ylabel('distribution')
plt.title('distribution vs. Time')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('allocation.png')

# Plot stock prices and shifted returns
fig, axes = plt.subplots(nrows=2,ncols=1)
stock_price.plot(ax=axes[0])
shift_returns.plot(ax=axes[1])
axes[0].set_title('Stock Prices')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')
axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes[1].set_title(str(shift)+ ' Day Shift returns')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('returns ' + str(shift) + ' Days Apart')
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('stocks.png', pad_inches=1)
fig.tight_layout()

# Plot portfolio returns vs. time
plt.figure()
returns.plot()
plt.xlabel('Date')
plt.ylabel('Portolio returns')
plt.title('Portfolio returns vs. Time')
# plt.savefig('returns.png')

plt.show()