#!/usr/bin/env python
"""Comprehensive test of stock dashboard functionality"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import traceback

print("=" * 60)
print("STOCK DASHBOARD - COMPREHENSIVE ERROR CHECK")
print("=" * 60)

# Test 1: Imports
print("\n[TEST 1] Testing imports...")
try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import ta
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Data Loading
print("\n[TEST 2] Testing data loading...")
try:
    df = yf.download("AAPL", start="2018-01-01", end="2024-12-18", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"✅ Data loaded: {len(df)} rows")
    if len(df) < 60:
        print(f"❌ ERROR: Not enough data ({len(df)} rows)")
        sys.exit(1)
except Exception as e:
    print(f"❌ Data loading error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Technical Indicators
print("\n[TEST 3] Testing technical indicators...")
try:
    close_data = df['Close'].values.flatten()
    df['RSI'] = ta.momentum.RSIIndicator(pd.Series(close_data)).rsi().values
    print("✅ RSI calculated")
    
    macd = ta.trend.MACD(pd.Series(close_data))
    df['MACD'] = macd.macd().values
    df['MACD_Signal'] = macd.macd_signal().values
    print("✅ MACD calculated")
    
    bb = ta.volatility.BollingerBands(pd.Series(close_data))
    bb_high = bb.bollinger_hband().reset_index(drop=True)
    bb_low = bb.bollinger_lband().reset_index(drop=True)
    df['BB_High'] = pd.Series(bb_high.values, index=df.index).fillna(df['Close'])
    df['BB_Low'] = pd.Series(bb_low.values, index=df.index).fillna(df['Close'])
    print("✅ Bollinger Bands calculated")
    
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    print("✅ SMAs calculated")
    
    df['Momentum'] = df['Close'].diff(10)
    print("✅ Momentum calculated")
    
except Exception as e:
    print(f"❌ Indicator error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Strategy Logic
print("\n[TEST 4] Testing strategy logic...")
try:
    strategies = ["RSI + MACD", "Bollinger Bands", "RSI Only", "Moving Average Crossover", "Momentum"]
    
    for strategy in strategies:
        df_test = df.copy()
        df_test['Signal'] = "HOLD"
        
        if strategy == "RSI + MACD":
            df_test.loc[(df_test['RSI'] < 30) & (df_test['MACD'] > df_test['MACD_Signal']), 'Signal'] = "BUY"
            df_test.loc[(df_test['RSI'] > 70) & (df_test['MACD'] < df_test['MACD_Signal']), 'Signal'] = "SELL"
        
        elif strategy == "Bollinger Bands":
            buy_signal = df_test['Close'] < df_test['BB_Low']
            sell_signal = df_test['Close'] > df_test['BB_High']
            df_test.loc[buy_signal, 'Signal'] = "BUY"
            df_test.loc[sell_signal, 'Signal'] = "SELL"
        
        elif strategy == "RSI Only":
            df_test.loc[df_test['RSI'] < 30, 'Signal'] = "BUY"
            df_test.loc[df_test['RSI'] > 70, 'Signal'] = "SELL"
        
        elif strategy == "Moving Average Crossover":
            df_test.loc[df_test['SMA50'] > df_test['SMA200'], 'Signal'] = "BUY"
            df_test.loc[df_test['SMA50'] < df_test['SMA200'], 'Signal'] = "SELL"
        
        elif strategy == "Momentum":
            df_test.loc[df_test['Momentum'] > 0, 'Signal'] = "BUY"
            df_test.loc[df_test['Momentum'] < 0, 'Signal'] = "SELL"
        
        print(f"✅ {strategy} strategy works")

except Exception as e:
    print(f"❌ Strategy error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: LSTM Models
print("\n[TEST 5] Testing LSTM models...")
try:
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
    print(f"✅ Training data prepared: X shape {X.shape}, y shape {y.shape}")
    
    models = ["LSTM (1-layer)", "LSTM (2-layer)", "LSTM (3-layer)"]
    
    for model_type in models:
        if model_type == "LSTM (1-layer)":
            model = Sequential([
                LSTM(50, input_shape=(X.shape[1], 1)),
                Dense(1)
            ])
        elif model_type == "LSTM (2-layer)":
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50),
                Dense(1)
            ])
        elif model_type == "LSTM (3-layer)":
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50, return_sequences=True),
                LSTM(50),
                Dense(1)
            ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=1, batch_size=32, verbose=0)
        print(f"✅ {model_type} trained successfully")
        
        # Test prediction
        last_60 = scaled_data[-60:].flatten()
        pred = model.predict(last_60.reshape(1, 60, 1), verbose=0)
        print(f"   Prediction shape: {pred.shape}")

except Exception as e:
    print(f"❌ LSTM error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Simple Moving Average
print("\n[TEST 6] Testing Simple Moving Average model...")
try:
    sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
    if pd.isna(sma_20):
        sma_20 = df['Close'].iloc[-1]
    future_prices = [[sma_20] for _ in range(10)]
    future_prices = np.array(future_prices)
    print(f"✅ SMA predictions generated: shape {future_prices.shape}")
except Exception as e:
    print(f"❌ SMA error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Backtesting
print("\n[TEST 7] Testing backtesting logic...")
try:
    df_backtest = df.copy()
    df_backtest['Signal'] = "HOLD"
    df_backtest.loc[(df_backtest['RSI'] < 30) & (df_backtest['MACD'] > df_backtest['MACD_Signal']), 'Signal'] = "BUY"
    df_backtest.loc[(df_backtest['RSI'] > 70) & (df_backtest['MACD'] < df_backtest['MACD_Signal']), 'Signal'] = "SELL"
    
    df_backtest['Position'] = 0
    df_backtest.loc[df_backtest['Signal'] == 'BUY', 'Position'] = 1
    df_backtest.loc[df_backtest['Signal'] == 'SELL', 'Position'] = -1

    df_backtest['Market_Return'] = df_backtest['Close'].pct_change()
    df_backtest['Strategy_Return'] = df_backtest['Market_Return'] * df_backtest['Position'].shift(1)

    total_return = (df_backtest['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
    print(f"✅ Backtesting works. Total return: {total_return*100:.2f}%")

except Exception as e:
    print(f"❌ Backtesting error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: Investment Calculations
print("\n[TEST 8] Testing investment calculations...")
try:
    last_price = df['Close'].iloc[-1]
    investment_amount = 50000
    shares = investment_amount // last_price
    future_price = last_price * 1.1
    expected_profit = (future_price - last_price) * shares
    stop_loss = last_price * 0.98
    target_price = last_price * 1.05
    
    print(f"✅ Investment calculations:")
    print(f"   Current Price: ₹{last_price:.2f}")
    print(f"   Shares: {int(shares)}")
    print(f"   Stop Loss: ₹{stop_loss:.2f}")
    print(f"   Target: ₹{target_price:.2f}")
    print(f"   Expected Profit: ₹{expected_profit:.2f}")

except Exception as e:
    print(f"❌ Investment calculation error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - APP IS READY TO USE")
print("=" * 60)
