# -*- coding: utf-8 -*-
"""
Stock Price Forecasting
Author: Kapil Dev Bhandari
---------------------------------------
This notebook forecasts Apple (AAPL) stock prices using:
1. ARIMA
2. Prophet
3. LSTM

Data Range: 2015–2023
"""

# =================================================
# 1. Import Libraries
# =================================================
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =================================================
# 2. Data Collection
# =================================================
# Download Apple stock data
data = yf.download("AAPL", start="2015-01-01", end="2023-12-31", auto_adjust=True)

# Keep only 'Close' price
df = data[['Close']]
print(df.head())

# Plot closing prices
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'])
plt.title("Apple Stock Closing Price (2015-2023)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# =================================================
# 3. Trend Analysis
# =================================================
# Rolling mean (30-day smoothing)
df['rolling_mean_30'] = df['Close'].rolling(window=30).mean()

plt.figure(figsize=(12,6))
plt.plot(df['Close'], label="Original")
plt.plot(df['rolling_mean_30'], label="30-Day Rolling Mean")
plt.legend()
plt.show()

# Autocorrelation
autocorrelation_plot(df['Close'])

# Stationarity check (ADF Test)
result = adfuller(df['Close'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Differencing
df_diff = df['Close'].diff().dropna()
result = adfuller(df_diff)
print("ADF Statistic after differencing:", result[0])
print("p-value after differencing:", result[1])

# ACF and PACF plots
plot_acf(df_diff, lags=50)
plot_pacf(df_diff, lags=50)
plt.show()

# =================================================
# 4. Train-Test Split
# =================================================
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

plt.figure(figsize=(12,6))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.title("Train-Test Split (Apple Stock)")
plt.legend()
plt.show()

# =================================================
# 5. ARIMA Model
# =================================================
model = ARIMA(train['Close'], order=(5,1,0))  # Example order
model_fit = model.fit()

# Forecast
forecast_arima = model_fit.forecast(steps=len(test))

plt.figure(figsize=(12,6))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index, test['Close'], label='Test')
plt.plot(test.index, forecast_arima, label='ARIMA Forecast')
plt.legend()
plt.show()

# =================================================
# 6. Prophet Model
# =================================================
# Prepare data
df_prophet = df.reset_index()[['Date','Close']]
df_prophet.columns = ['ds','y']

train_p = df_prophet.iloc[:train_size]
test_p = df_prophet.iloc[train_size:]

# Train Prophet
m = Prophet(daily_seasonality=True)
m.fit(train_p)

future = m.make_future_dataframe(periods=len(test_p))
forecast_prophet = m.predict(future)

# Plot
m.plot(forecast_prophet)

# Align predictions
forecast_p = forecast_prophet.set_index('ds')['yhat'].reindex(test.index, method='nearest')

# =================================================
# 7. LSTM Model
# =================================================
# Normalize
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[['Close']])

# Train-test split
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, callbacks=[early_stop])

# Predict
y_pred = model.predict(X_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# Align indexes
test_index = df.index[train_size+time_step+1:train_size+time_step+1+len(y_test)]

# Plot
plt.figure(figsize=(12,6))
plt.plot(test_index, y_test, label='Actual Price')
plt.plot(test_index, y_pred, label='LSTM Predicted Price')
plt.legend()
plt.show()

# =================================================
# 8. Evaluation
# =================================================
rmse_arima = np.sqrt(mean_squared_error(test['Close'], forecast_arima))
rmse_prophet = np.sqrt(mean_squared_error(test['Close'], forecast_p))
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE - ARIMA:", rmse_arima)
print("RMSE - Prophet:", rmse_prophet)
print("RMSE - LSTM:", rmse_lstm)
