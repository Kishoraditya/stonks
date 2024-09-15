import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from prophet import Prophet
import xgboost as xgb
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from tensorflow import keras
#from keras import layers
#from keras import models

#from keras.models import Sequential
#from keras.layers import LSTM, Dense


from sklearn.preprocessing import MinMaxScaler

def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_relative_strength_index(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window):
    ma = calculate_moving_average(data, window)
    std = data['Close'].rolling(window=window).std()
    upper_band = ma + (std * 2)
    lower_band = ma - (std * 2)
    return upper_band, ma, lower_band

# ... (existing functions)

def perform_analysis(data, analysis_type):
    if analysis_type == 'technical':
        return technical_analysis(data)
    elif analysis_type == 'prediction':
        return price_prediction(data)
    else:
        return {'error': 'Invalid analysis type'}

def technical_analysis(data):
    analysis = {}
    analysis['moving_average_50'] = calculate_moving_average(data, 50)
    analysis['moving_average_200'] = calculate_moving_average(data, 200)
    analysis['rsi_14'] = calculate_relative_strength_index(data, 14)
    analysis['upper_band'], analysis['middle_band'], analysis['lower_band'] = calculate_bollinger_bands(data, 20)
    return analysis

def price_prediction(data):
    model = ARIMA(data['Close'], order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=30)
    return {'forecast': forecast.tolist()}

# ... (existing functions)

def visualize_analysis(data, analysis):
    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.plot(data.index, analysis['moving_average_50'], label='50-day MA')
    plt.plot(data.index, analysis['upper_band'], label='Upper Bollinger Band')
    plt.plot(data.index, analysis['lower_band'], label='Lower Bollinger Band')
    plt.title('Stock Price Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('stock_analysis.png')
    plt.close()

def predict_price_linear_regression(data):
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_days = 30
    future_X = np.array(range(len(X), len(X) + future_days)).reshape(-1, 1)
    future_prices = model.predict(future_X)
    return future_prices.tolist()


def predict_price_arima(data):
    # Ensure the index is a DatetimeIndex with frequency information
    data = data.asfreq('D')  # 'D' for daily frequency
    model = ARIMA(data['Close'], order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=30)
    return forecast.tolist()


def predict_prophet(data):
    df = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)  # Remove timezone information
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)


def predict_xgboost(data):
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(X_train, y_train)
    last_30_days = X.tail(30)
    predictions = model.predict(last_30_days)
    return predictions

def predict_lstm(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)
    
    last_60_days = scaled_data[-60:].reshape(1, -1, 1)
    predictions = []
    
    for _ in range(30):
        next_pred = model.predict(last_60_days)
        predictions.append(next_pred[0, 0])
        last_60_days = np.roll(last_60_days, -1, axis=1)
        last_60_days[0, -1, 0] = next_pred
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def predict_random_forest(data):
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    future_data = X.tail(30)
    predictions = model.predict(future_data)
    return predictions



def analyze_stock(data):
    # Ensure the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)
    
    analysis = {}
    analysis['moving_average_50'] = calculate_moving_average(data, 50)
    analysis['rsi_14'] = calculate_relative_strength_index(data, 14)
    analysis['upper_band'], analysis['middle_band'], analysis['lower_band'] = calculate_bollinger_bands(data, 20)
    analysis['linear_regression_prediction'] = predict_price_linear_regression(data)
    analysis['arima_prediction'] = predict_price_arima(data)
    analysis['prophet_prediction'] = predict_prophet(data)
    analysis['xgboost_prediction'] = predict_xgboost(data)
    analysis['lstm_prediction'] = predict_lstm(data)
    analysis['random_forest_prediction'] = predict_random_forest(data)
    return analysis
