import matplotlib.pyplot as plt
import numpy as np
from fit_model import fit_model, fit_model_cross_validation
from helper import read_txt, create_shifted_orderbook, get_curr_date,date_range
from sklearn.ensemble import RandomForestClassifier
import datetime
import pandas as pd



if __name__ == "__main__":
    # Reads user parameters from text file
    stocks, period = read_txt('parameters_RF.txt')
    print stocks,period

    # Technical indicator list
    feature_label_list = ['EWMA_', 'upperband', 'lowerband', 'slowk', 'slowd', "macdhist", 'CCI','Chaikin']

    # List of stocks to input
    ticker_arr = stocks

    # Number of days' price data used for training
    lag = 20
    # Forward prediction period
    pred_period = period

    # Start and end dates of stock data set queried
    start = datetime.datetime(2018, 1, 1)
    end = datetime.datetime(2018, 4, 1)

    # Train-test splits for cross-validation
    splits = 3

    # Proportion of data between the start date and end date used for training
    train_prop = 0.9
    test_prop = 1 - train_prop

    # Hit-Rate score for one train-test split
    hit_rate_one_split = []
    # Hit-Rate score for cross-validation results
    hit_rate_cv = []

    # Trains and tests on each ticker
    for ticker in ticker_arr:
        # Retrieves stock info and returns training/testing data
        stock_data, stock_returns, stock_lag, stock_movement = create_shifted_orderbook(ticker, start_date=start,
                                                                                        end_date=end,
                                                                                        lag_period=lag,
                                                                                     pred_period=pred_period)
        ##download the data into csv.
        #name_stock_data = ticker + "_stock_data.csv"
        # stock_data.to_csv(name_stock_data, sep=',')
        # name_stock_returns = ticker + "_stock_returns.csv"
        # stock_returns.to_csv(name_stock_returns, sep=',')
        # name_stock_lag = ticker + "_stock_lag.csv"
        # stock_lag.to_csv(name_stock_lag, sep=',')
        # name_stock_movement = ticker + "_stock_movement.csv"
        # stock_movement.to_csv(name_stock_movement, sep=',')

        ##download the data into xlsx.
        name_stock_data = ticker + "_stock_data_from_" + str(start) +".xlsx"
        # stock_data.to_excel(name_stock_data)
        # name_stock_returns = ticker + "_stock_returns_from_" + str(start) +".xlsx"
        # stock_returns.to_excel(name_stock_returns)
        # name_stock_lag = ticker + "_stock_lag_from_" + str(start) +".xlsx"
        # stock_lag.to_excel(name_stock_lag)
        # name_stock_movement = ticker + "_stock_movement_from_" + str(start) +".xlsx"
        # stock_movement.to_excel(name_stock_movement)


        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('2'+ name_stock_data, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        stock_data.to_excel(writer, sheet_name='Sheet1')

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()