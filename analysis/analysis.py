# Import modules
import numpy as np
import ta
import pandas as pd
from matplotlib import pyplot as plt
from pandas_datareader import DataReader as reader
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from pickle import load, dump
from random import random

# Method : Options for technical analysis
def get_options():
    return ['HDFCBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS']

# Method : Indicators for technical analysis
def get_indicators():
    return ['Close', 'MACD', 'BB', 'RSI', 'ADX']

# Method : Get trends for a particular option
def get_trends(option):
    data = [get_test_data(option)]
    model = load(open('analysis/data/'+option+'_model.pickle', 'rb'))
    buy = model.predict(data)[0]
    return buy, 1-buy

# Method : Get graphs for a particular option and indicator
def get_graphs(option, indicator):
    return option+'_'+indicator+'.png'

# Method : Save and update datasets
def save_datasets():
    end_date = (datetime.today()+timedelta(days=3)).strftime('%Y-%m-%d')
    for option in get_options():
        with open('analysis/data/'+option+'_dataset.csv', 'w') as f:
            f.write(create_dataset(option, end_date).to_csv())
        create_graphs(option)
        get_model(option)

# Method : Create dataset
def create_dataset(option, end_date):
    data = reader(option, 'yahoo', '2000-01-01', end_date)

    data['MACD'] = ta.trend.macd(close=data['Close'], fillna=True)
    data['MACD_Signal'] = ta.trend.macd_signal(close=data['Close'], fillna=True)

    data['ADX'] = ta.trend.adx(high=data['High'], low=data['Low'], close=data['Close'], fillna=True)
    data['ADX_POS'] = ta.trend.adx_pos(high=data['High'], low=data['Low'], close=data['Close'], fillna=True)
    data['ADX_NEG'] = ta.trend.adx_neg(high=data['High'], low=data['Low'], close=data['Close'], fillna=True)

    data['BBH'] = ta.volatility.bollinger_hband(close=data['Close'], fillna=True)
    data['BBL'] = ta.volatility.bollinger_lband(close=data['Close'], fillna=True)

    data['RSI'] = ta.momentum.rsi(close=data['Close'], fillna=True)

    data.reset_index(level=0)
    return data.dropna()

# Method : Create graphs
def create_graphs(option):
    create_graphs_MACD(option)
    create_graphs_BB(option)
    create_graphs_ADX(option)
    create_graphs_RSI(option)
    create_graphs_Close(option)

# Methods : MACD Graph
def create_graphs_MACD(option):
    data = pd.read_csv('analysis/data/'+option+'_dataset.csv')[-64:]
    plt.figure()
    line1, = plt.plot(list(data['MACD']), color='red', label='MACD')
    line2, = plt.plot(list(data['MACD_Signal']), color='g', label='MACD Signal')
    plt.legend(handles=[line1, line2])
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('MACD for '+option)
    plt.savefig('static/'+option+'_'+'MACD.png')
    plt.close()

# Methods : BB Graph
def create_graphs_BB(option):
    data = pd.read_csv('analysis/data/'+option+'_dataset.csv')[-64:]
    plt.figure()
    line1, = plt.plot(list(data['BBH']), color='red', label='Bollinger High Band')
    line2, = plt.plot(list(data['BBL']), color='g', label='Bollinger Low Band')
    line3, = plt.plot(list(data['Close']), label='Close Value')
    plt.legend(handles=[line1, line2, line3])
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Bollinger Bands for '+option)
    plt.savefig('static/'+option+'_'+'BB.png')
    plt.close()

# Methods : Close
def create_graphs_Close(option):
    data = pd.read_csv('analysis/data/'+option+'_dataset.csv', index_col=0)[-64:]
    plt.figure()
    line1, = plt.plot(list(data['Close']), label='Close Value')
    plt.legend(handles=[line1])
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Closing prices for '+option)
    plt.savefig('static/'+option+'_'+'Close.png')
    plt.close()

# Methods : ADX Graph
def create_graphs_ADX(option):
    data = pd.read_csv('analysis/data/'+option+'_dataset.csv')[-64:]
    plt.figure()
    line1, = plt.plot(list(data['ADX_NEG']), color='red', label='-DI')
    line2, = plt.plot(list(data['ADX_POS']), color='g', label='+DI')
    line3, = plt.plot(list(data['ADX']), label='ADX')
    plt.legend(handles=[line1, line2, line3])
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Advanced Directional Index for '+option)
    plt.savefig('static/'+option+'_'+'ADX.png')
    plt.close()

# Methods : RSI Graph
def create_graphs_RSI(option):
    data = pd.read_csv('analysis/data/'+option+'_dataset.csv')[-64:]
    plt.figure()
    line1, = plt.plot(list(data['RSI']), label='RSI')
    plt.legend(handles=[line1])
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Relative Strength Index for '+option)
    plt.savefig('static/'+option+'_'+'RSI.png')
    plt.close()

# Methods : Get test data
def get_test_data(option):
    df = pd.read_csv('analysis/data/'+option+'_dataset.csv')
    df = df[['ADX', 'ADX_POS', 'ADX_NEG', 'MACD', 'MACD_Signal', 'RSI']].iloc[-1]
    df = np.array(df)
    return df

# Methods : Get model
def get_model(option):
    print('Generating model for ', option)
    df = pd.read_csv('analysis/data/'+option+'_dataset.csv')
    df['Target'] = df['Close'].shift(-1)
    df['Trend'] = df.apply(lambda x : 1 if x['Target'] > x['Close'] else 0, axis=1)
    df = df.dropna()
    X = np.array(df[['ADX', 'ADX_POS', 'ADX_NEG', 'MACD', 'MACD_Signal', 'RSI']])
    Y = np.array(df['Trend'])
    model = RandomForestRegressor()
    model.fit(X, Y)
    dump(model, open('analysis/data/'+option+'_model.pickle', 'wb'))