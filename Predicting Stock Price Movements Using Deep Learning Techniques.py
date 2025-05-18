#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import subprocess

# List of required packages
packages = [
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "statsmodels",
    "arch", "tensorflow", "ta"
]

# Function to check and install missing packages
def install_missing_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install missing packages
install_missing_packages(packages)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[4]:


# Load stock price data
mag7_closing_prices = pd.read_csv("MAG7_closing_prices_2019_2024.csv", parse_dates=["Date"])
mag7_closing_prices.head()


# In[5]:


macro_data = pd.read_csv("macroeconomic_data_2019_2024.csv", parse_dates=["Date"])
macro_data.head()


# In[6]:


# Ensure data is sorted by date
mag7_closing_prices.sort_values("Date", inplace=True)
macro_data.sort_values("Date", inplace=True)


# In[7]:


# Fill missing values in macroeconomic data
# Inflation_CPI is filled using forward-fill within the same month
macro_data["Inflation_CPI"] = macro_data.groupby(macro_data["Date"].dt.to_period("M"))["Inflation_CPI"].ffill()

# Fill missing values in VIX and 10Y Treasury Yield using linear interpolation
macro_data["VIX"] = macro_data["VIX"].interpolate(method="linear")
macro_data["10Y_Treasury_Yield"] = macro_data["10Y_Treasury_Yield"].interpolate(method="linear")


# In[8]:


# Merge datasets
data = pd.merge(mag7_closing_prices, macro_data, on="Date", how="left")


# In[9]:


data.head()


# In[10]:


# Exploratory Data Analysis
# Plot all stock prices in a single chart
def plot_all_stocks(df, stock_columns=["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"], title="Stock Prices Over Time"):
    plt.figure(figsize=(15, 6))
    
    for stock in stock_columns:
        plt.plot(df["Date"], df[stock], label=stock)

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(title)
    plt.legend()
    plt.show()

# Call the function
plot_all_stocks(data, title="MAG7 Stock Prices from 2019 to 2024")


# In[11]:


def plot_time_series(df, column, title):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df[column], label=column)
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.title(title)
    plt.legend()
    plt.show()

# Plot the macroeconomic indicators
plot_time_series(data, "VIX", "VIX Volatility Index")
plot_time_series(data, "10Y_Treasury_Yield", "10Y Treasury Yield")
plot_time_series(data, "Inflation_CPI", "Inflation CPI")


# In[12]:


from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Attention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import ta  # Technical Analysis Library


# In[13]:


# Stationarity Tests
def test_stationarity(df, column):
    adf_test = adfuller(df[column])
    kpss_test = kpss(df[column], regression="c")

    print(f"ADF Test for {column}: p-value = {adf_test[1]}")
    print(f"KPSS Test for {column}: p-value = {kpss_test[1]}")


# In[14]:


# Compute Technical Indicators
def compute_technical_indicators(df, stock_columns):
    for stock in stock_columns:
        df[f"{stock}_SMA_50"] = ta.trend.sma_indicator(df[stock], window=50)
        df[f"{stock}_SMA_200"] = ta.trend.sma_indicator(df[stock], window=200)
        df[f"{stock}_RSI"] = ta.momentum.rsi(df[stock], window=14)
        df[f"{stock}_MACD"] = ta.trend.macd(df[stock])
        df[f"{stock}_Bollinger_High"] = ta.volatility.bollinger_hband(df[stock])
        df[f"{stock}_Bollinger_Low"] = ta.volatility.bollinger_lband(df[stock])
    return df


# In[15]:


# Normalize Data
def normalize_data(df, stock_columns):
    scaler = MinMaxScaler()
    df[stock_columns] = scaler.fit_transform(df[stock_columns])
    return df, scaler


# In[16]:


# Prepare Data for LSTM/GRU
def prepare_lstm_data(df, target_column, time_steps=60):
    X, y = [], []
    for i in range(len(df) - time_steps):
        X.append(df.iloc[i : i + time_steps].values)
        y.append(df.iloc[i + time_steps][target_column])
    return np.array(X), np.array(y)


# In[17]:


# Attention Mechanism
def attention_layer(inputs):
    query = Dense(32, activation="relu")(inputs)
    key = Dense(32, activation="relu")(inputs)
    value = Dense(32, activation="relu")(inputs)
    attention_output = Attention()([query, key, value])
    return Concatenate()([inputs, attention_output])

# Build Models
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def build_gru_model(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        GRU(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def build_attention_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    x = LSTM(50, return_sequences=True)(x)
    x = attention_layer(x)
    x = LSTM(50, return_sequences=False)(x)
    x = Dense(25, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def build_attention_gru_model(input_shape):
    inputs = Input(shape=input_shape)
    x = GRU(50, return_sequences=True)(inputs)
    x = GRU(50, return_sequences=True)(x)
    x = attention_layer(x)
    x = GRU(50, return_sequences=False)(x)
    x = Dense(25, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Train Model
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    early_stop = EarlyStopping(monitor="val_loss", patience=5)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_val, y_val), callbacks=[early_stop])
    return history, model


# In[18]:


# ARIMA Model
def train_arima_model(df, target_column):
    model = ARIMA(df[target_column], order=(5,1,0))
    arima_fit = model.fit()
    return arima_fit

# GARCH Model
def train_garch_model(df, target_column):
    model = arch_model(df[target_column], vol="Garch", p=1, q=1)
    garch_fit = model.fit(disp="off")
    return garch_fit


# In[19]:


# Plot Predictions
def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Actual Prices")
    plt.plot(y_pred, label="Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title(f"{model_name}: Actual vs Predicted Stock Prices")
    plt.legend()
    plt.show()


# In[20]:


# Main Function
def main():
    stock_columns = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
    
    df = compute_technical_indicators(data, stock_columns)
    df, scaler = normalize_data(data, stock_columns)
    
    # Time series forecasting with Deep Learning
    X, y = prepare_lstm_data(data, "AAPL")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        "LSTM": build_lstm_model((X_train.shape[1], X_train.shape[2])),
        "GRU": build_gru_model((X_train.shape[1], X_train.shape[2])),
        "Attention LSTM": build_attention_lstm_model((X_train.shape[1], X_train.shape[2])),
        "Attention GRU": build_attention_gru_model((X_train.shape[1], X_train.shape[2]))
    }

    for name, model in models.items():
        print(f"Training {name}...")
        _, trained_model = train_model(model, X_train, y_train, X_test, y_test)
        y_pred = trained_model.predict(X_test)
        plot_predictions(y_test, y_pred, name)

    # Traditional Models
    arima_model = train_arima_model(df, "AAPL")
    print(arima_model.summary())

    garch_model = train_garch_model(df, "AAPL")
    print(garch_model.summary())

if __name__ == "__main__":
    main()

