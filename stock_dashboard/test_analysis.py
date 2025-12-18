import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import numpy as np
import pandas as pd
import ta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Test data loading and analysis
try:
    print("📥 Loading data...")
    df = yf.download("AAPL", start="2018-01-01", end="2024-12-18", progress=False)
    df.dropna(inplace=True)
    print(f"✅ Data loaded: {len(df)} rows")
    
    print("\n📊 Computing technical indicators...")
    close_data = df['Close'].values.flatten()
    df['RSI'] = ta.momentum.RSIIndicator(pd.Series(close_data)).rsi().values
    print("✅ RSI computed")
    
    macd = ta.trend.MACD(pd.Series(close_data))
    df['MACD'] = macd.macd().values
    df['MACD_Signal'] = macd.macd_signal().values
    print("✅ MACD computed")
    
    bb = ta.volatility.BollingerBands(pd.Series(close_data))
    df['BB_High'] = bb.bollinger_hband().values
    df['BB_Low'] = bb.bollinger_lband().values
    print("✅ Bollinger Bands computed")
    
    print("\n🤖 Training LSTM model...")
    close_prices = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)
    
    X, y = [], []
    lookback = 60
    
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    print("✅ LSTM model trained")
    
    print("\n🔮 Making 10-day predictions...")
    last_60 = scaled_data[-60:].flatten()
    future_prices = []
    
    for _ in range(10):
        pred = model.predict(last_60.reshape(1, 60, 1), verbose=0)
        future_prices.append(pred[0][0])
        last_60 = np.append(last_60[1:], pred[0][0])
    
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    print("✅ Predictions completed")
    print(f"\n📈 Next 10 days predictions:\n{future_prices}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
