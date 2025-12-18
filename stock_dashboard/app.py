# ==========================================================
# ADVANCED STOCK MARKET DASHBOARD
# LSTM (10-Day Prediction) + RSI + MACD + Bollinger Bands
# Backtesting + Risk Management + Investment Calculator
# ==========================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import GRU

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📈 AI Stock Market Analysis Dashboard")
st.markdown("### LSTM (10-Day Prediction) | RSI | MACD | Bollinger Bands")

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Strategy Selection
    strategy = st.selectbox(
        "🎯 Select Trading Strategy",
        [
            "RSI + MACD",
            "Bollinger Bands",
            "RSI Only",
            "Moving Average Crossover",
            "Momentum"
        ]
    )
    
    st.markdown("---")
    
    # Model Selection
    model_type = st.selectbox(
        "🤖 Select Prediction Model",
        [
            "LSTM (1-layer)",
            "LSTM (2-layer)",
            "LSTM (3-layer)",
            "GRU (1-layer)",
            "Prophet",
            "Random Forest Classifier",
            "Simple Moving Average"
        ]
    )
    # Ensemble option
    use_ensemble = st.checkbox("🧩 Use Ensemble (average available models)", value=False)
    auto_weight = st.checkbox("⚖️ Auto-weight ensemble by recent accuracy (slow)", value=False)
    if auto_weight:
        weight_window = st.slider("Backtest window (days)", min_value=3, max_value=20, value=7)
    else:
        weight_window = 0
    
    st.markdown("---")
    st.subheader("📋 Strategy Details")
    
    if strategy == "RSI + MACD":
        st.write("**Buy Signal:** RSI < 30 AND MACD > Signal")
        st.write("**Sell Signal:** RSI > 70 AND MACD < Signal")
    elif strategy == "Bollinger Bands":
        st.write("**Buy Signal:** Price < Lower Band")
        st.write("**Sell Signal:** Price > Upper Band")
    elif strategy == "RSI Only":
        st.write("**Buy Signal:** RSI < 30")
        st.write("**Sell Signal:** RSI > 70")
    elif strategy == "Moving Average Crossover":
        st.write("**Buy Signal:** SMA50 > SMA200")
        st.write("**Sell Signal:** SMA50 < SMA200")
    elif strategy == "Momentum":
        st.write("**Buy Signal:** Momentum > 0")
        st.write("**Sell Signal:** Momentum < 0")

# ---------------- USER INPUT ----------------
col1, col2 = st.columns([2, 1])

with col1:
    symbol_input = st.text_input("Or Enter Stock Symbol", "AAPL", label_visibility="collapsed")

with col2:
    # Popular ticker dropdown
    popular_tickers = st.selectbox(
        "Popular Tickers",
        [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "TCS.NS",
            "RELIANCE.NS",
            "INFY.NS",
            "WIPRO.NS",
            "HDFC.NS"
        ]
    )
    # Use selected ticker if different from input
    symbol = popular_tickers if popular_tickers else symbol_input
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
investment_amount = st.number_input("💰 Amount to Invest (₹)", min_value=1000, step=1000)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    # flatten columns when yfinance returns MultiIndex (ticker level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# ---------------- RUN ANALYSIS ----------------
if st.button("🚀 Run Full Analysis"):
    try:
        with st.spinner("📊 Loading data..."):
            df = load_data(symbol, start_date, end_date)
        
        if len(df) < 10:
            st.error("❌ Not enough data. Please select a longer date range (minimum 10 rows).")
            st.stop()

        # Warn if less than recommended lookback
        if len(df) < 60:
            st.warning("⚠ Limited history: using a smaller lookback for modeling (results may be less accurate).")

        # ========== TECHNICAL INDICATORS ==========
        close_data = df['Close'].values.flatten()
        df['RSI'] = ta.momentum.RSIIndicator(pd.Series(close_data)).rsi().values

        macd = ta.trend.MACD(pd.Series(close_data))
        df['MACD'] = macd.macd().values
        df['MACD_Signal'] = macd.macd_signal().values

        bb = ta.volatility.BollingerBands(pd.Series(close_data))
        bb_high = bb.bollinger_hband().reset_index(drop=True)
        bb_low = bb.bollinger_lband().reset_index(drop=True)
        df['BB_High'] = pd.Series(bb_high.values, index=df.index).fillna(df['Close'])
        df['BB_Low'] = pd.Series(bb_low.values, index=df.index).fillna(df['Close'])

        # ========== STRATEGY ==========
        df['Signal'] = "HOLD"
        
        if strategy == "RSI + MACD":
            df.loc[(df['RSI'] < 30) & (df['MACD'] > df['MACD_Signal']), 'Signal'] = "BUY"
            df.loc[(df['RSI'] > 70) & (df['MACD'] < df['MACD_Signal']), 'Signal'] = "SELL"
        
        elif strategy == "Bollinger Bands":
            buy_signal = df['Close'] < df['BB_Low']
            sell_signal = df['Close'] > df['BB_High']
            df.loc[buy_signal, 'Signal'] = "BUY"
            df.loc[sell_signal, 'Signal'] = "SELL"
        
        elif strategy == "RSI Only":
            df.loc[df['RSI'] < 30, 'Signal'] = "BUY"
            df.loc[df['RSI'] > 70, 'Signal'] = "SELL"
        
        elif strategy == "Moving Average Crossover":
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()
            df.loc[df['SMA50'] > df['SMA200'], 'Signal'] = "BUY"
            df.loc[df['SMA50'] < df['SMA200'], 'Signal'] = "SELL"
        
        elif strategy == "Momentum":
            df['Momentum'] = df['Close'].diff(10)
            df.loc[df['Momentum'] > 0, 'Signal'] = "BUY"
            df.loc[df['Momentum'] < 0, 'Signal'] = "SELL"

        # ========== LSTM MODEL ==========
        close_prices = df[['Close']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        # adaptive lookback: use up to 60 but no more than available history-1
        lookback = min(60, len(scaled_data) - 1)
        X, y = [], []

        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])

        # Initialize variables
        model = None
        future_prices = None

        # Helper to compute prediction for a given model name
        def compute_prediction_for(model_name):
            result = None
            try:
                if model_name.startswith('LSTM') or model_name.startswith('GRU'):
                    # build sequence arrays if not already
                    if X is None:
                        return None
                    Xloc = X.copy()
                    yloc = y.copy()
                    # choose model
                    if model_name.startswith('LSTM'):
                        if model_name == 'LSTM (1-layer)':
                            mdl = Sequential([LSTM(50, input_shape=(Xloc.shape[1], 1)), Dense(1)])
                        elif model_name == 'LSTM (2-layer)':
                            mdl = Sequential([LSTM(50, return_sequences=True, input_shape=(Xloc.shape[1], 1)), LSTM(50), Dense(1)])
                        else:
                            mdl = Sequential([LSTM(50, return_sequences=True, input_shape=(Xloc.shape[1], 1)), LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
                    else:
                        mdl = Sequential([GRU(50, input_shape=(Xloc.shape[1], 1)), Dense(1)])

                    mdl.compile(optimizer='adam', loss='mean_squared_error')
                    mdl.fit(Xloc, yloc, epochs=3, batch_size=32, verbose=0)

                    last_n = scaled_data[-min(60, scaled_data.shape[0]):].flatten()
                    # pad if needed
                    if len(last_n) < Xloc.shape[1]:
                        pad = np.repeat(last_n[0], Xloc.shape[1]-len(last_n))
                        last_n = np.concatenate([pad, last_n])
                    preds = []
                    for _ in range(10):
                        pred = mdl.predict(last_n.reshape(1, Xloc.shape[1], 1), verbose=0)
                        preds.append(pred[0][0])
                        last_n = np.append(last_n[1:], pred[0][0])
                    result = scaler.inverse_transform(np.array(preds).reshape(-1,1))

                elif model_name == 'Prophet':
                    try:
                        from prophet import Prophet
                        df_prop = df.reset_index().rename(columns={df.index.name or 'Date':'ds', 'Close':'y'})
                        if 'ds' not in df_prop.columns:
                            df_prop = df_prop.rename(columns={df_prop.columns[0]:'ds'})
                        m = Prophet(daily_seasonality=True)
                        m.fit(df_prop[['ds','y']])
                        future = m.make_future_dataframe(periods=20)
                        forecast = m.predict(future)
                        last_ds = pd.to_datetime(df.index[-1])
                        fut = forecast[forecast['ds'] > last_ds].head(10)
                        result = fut['yhat'].values.reshape(-1,1)
                    except Exception:
                        result = None

                elif model_name == 'Simple Moving Average':
                    sma_20 = df['Close'].rolling(window=min(20, len(df))).mean().iloc[-1]
                    if pd.isna(sma_20):
                        sma_20 = df['Close'].iloc[-1]
                    result = np.array([[sma_20] for _ in range(10)])
            except Exception:
                result = None
            return result

        if len(X) == 0:
            # Not enough data to build sequences for LSTM; fall back to SMA/last-price predictions
            sma_20 = df['Close'].rolling(window=min(20, len(df))).mean().iloc[-1]
            if pd.isna(sma_20):
                sma_20 = df['Close'].iloc[-1]
            future_prices = np.array([[sma_20] for _ in range(10)])
            X = None
        else:
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)

        with st.spinner("🤖 Training & Predicting..."):
            # If ensemble enabled, compute multiple model predictions and average them
            if use_ensemble:
                # Helper: quick walk-forward backtest to estimate 1-day RMSE for recent period
                def compute_backtest_error(model_name, eval_window=7):
                    try:
                        errors = []
                        n = len(df)
                        # limit window to available history minus lookback
                        eval_window = min(eval_window, max(1, n - 5))
                        for k in range(eval_window, 0, -1):
                            # train on data up to index -k
                            cutoff = n - k - 1
                            if cutoff < 10:
                                continue
                            df_train = df.iloc[:cutoff+1]
                            # simple 1-day prediction using same helper but on df_train
                            try:
                                # Temporarily build scaled arrays for training
                                cp = df_train[['Close']].values
                                sc = MinMaxScaler().fit(cp)
                                sd = sc.transform(cp)
                                lb = min(60, len(sd)-1)
                                Xb, yb = [], []
                                for i in range(lb, len(sd)):
                                    Xb.append(sd[i-lb:i,0])
                                    yb.append(sd[i,0])
                                if len(Xb) == 0:
                                    # fallback: naive persistence
                                    pred = df_train['Close'].iloc[-1]
                                else:
                                    Xb = np.array(Xb).reshape(len(Xb), lb, 1)
                                    if model_name.startswith('LSTM') or model_name.startswith('GRU'):
                                        # small model and 1 epoch for speed
                                        if model_name.startswith('LSTM'):
                                            mloc = Sequential([LSTM(16, input_shape=(Xb.shape[1],1)), Dense(1)])
                                        else:
                                            mloc = Sequential([GRU(16, input_shape=(Xb.shape[1],1)), Dense(1)])
                                        mloc.compile(optimizer='adam', loss='mean_squared_error')
                                        mloc.fit(Xb, np.array(yb), epochs=1, batch_size=16, verbose=0)
                                        last_seq = sd[-lb:].flatten()
                                        if len(last_seq) < lb:
                                            pad = np.repeat(last_seq[0], lb - len(last_seq))
                                            last_seq = np.concatenate([pad, last_seq])
                                        p = mloc.predict(last_seq.reshape(1, lb, 1), verbose=0)[0][0]
                                        pred = float(sc.inverse_transform(np.array([[p]]))[0][0])
                                    elif model_name == 'Prophet':
                                        try:
                                            from prophet import Prophet
                                            dfp = df_train.reset_index().rename(columns={df_train.index.name or 'Date':'ds','Close':'y'})
                                            if 'ds' not in dfp.columns:
                                                dfp = dfp.rename(columns={dfp.columns[0]:'ds'})
                                            mm = Prophet(daily_seasonality=True)
                                            mm.fit(dfp[['ds','y']])
                                            fut = mm.make_future_dataframe(periods=1)
                                            fc = mm.predict(fut)
                                            pred = float(fc[fc['ds'] > dfp['ds'].iloc[-1]]['yhat'].iloc[0])
                                        except Exception:
                                            pred = df_train['Close'].iloc[-1]
                                    else:
                                        # SMA fallback
                                        pred = df_train['Close'].rolling(window=min(20,len(df_train))).mean().iloc[-1]
                            except Exception:
                                pred = df_train['Close'].iloc[-1]
                            actual = df.iloc[cutoff+1]['Close']
                            errors.append((pred - actual)**2)
                        if len(errors) == 0:
                            return float('inf')
                        mse = np.mean(errors)
                        rmse = np.sqrt(mse)
                        return rmse
                    except Exception:
                        return float('inf')

                preds_list = []
                candidate_models = ['LSTM (1-layer)', 'GRU (1-layer)', 'Prophet', 'Simple Moving Average']
                model_errors = {}
                # If auto-weight requested, compute errors (may be slow)
                if auto_weight and weight_window > 0:
                    with st.spinner("🧪 Running short backtest to estimate model accuracy..."):
                        for mname in candidate_models:
                            model_errors[mname] = compute_backtest_error(mname, eval_window=weight_window)

                for mname in candidate_models:
                    p = compute_prediction_for(mname)
                    if p is not None:
                        preds_list.append((mname, p))

                if len(preds_list) > 0:
                    # If auto_weight: compute normalized inverse-error weights
                    if auto_weight and len(model_errors) > 0:
                        weights = []
                        for mname, p in preds_list:
                            err = model_errors.get(mname, float('inf'))
                            w = 0.0 if err == float('inf') else 1.0 / (err + 1e-6)
                            weights.append(w)
                        weights = np.array(weights)
                        if np.sum(weights) == 0:
                            weights = np.ones_like(weights) / len(weights)
                        else:
                            weights = weights / np.sum(weights)
                        stacked = np.stack([p.flatten() for (_, p) in preds_list], axis=1)
                        avg = (stacked * weights.reshape(1, -1)).sum(axis=1).reshape(-1,1)
                    else:
                        stacked = np.stack([p.flatten() for (_, p) in preds_list], axis=1)
                        avg = np.mean(stacked, axis=1).reshape(-1,1)
                    future_prices = avg
                else:
                    future_prices = None
            else:
                # If Random Forest classifier selected, run it to possibly override Signal
                if model_type == "Random Forest Classifier":
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        df_feat = df.copy()
                        df_feat['SMA50'] = df_feat['Close'].rolling(window=50).mean()
                        df_feat['SMA200'] = df_feat['Close'].rolling(window=200).mean()
                        df_feat['Return1'] = df_feat['Close'].pct_change().fillna(0)
                        df_feat['Momentum'] = df_feat['Close'].diff(10).fillna(0)
                        df_feat['FutureReturn3'] = df_feat['Close'].pct_change(periods=3).shift(-3)
                        df_feat['Label'] = 0
                        df_feat.loc[df_feat['FutureReturn3'] > 0.01, 'Label'] = 1
                        df_feat.loc[df_feat['FutureReturn3'] < -0.01, 'Label'] = -1
                        feat_cols = ['RSI','MACD','MACD_Signal','SMA50','SMA200','Return1','Momentum']
                        df_feat = df_feat.dropna()
                        if len(df_feat) > 30:
                            Xclf = df_feat[feat_cols].values
                            yclf = df_feat['Label'].values
                            clf = RandomForestClassifier(n_estimators=100, random_state=42)
                            clf.fit(Xclf, yclf)
                            last_feat = df_feat[feat_cols].iloc[-1:].values
                            pred_label = clf.predict(last_feat)[0]
                            pred_map = {1:'BUY', -1:'SELL', 0:'HOLD'}
                            df.loc[df.index[-1], 'Signal'] = pred_map.get(pred_label, 'HOLD')
                    except Exception:
                        pass

                # For single-model selection, compute prediction via helper when possible
                if model_type in ['LSTM (1-layer)', 'LSTM (2-layer)', 'LSTM (3-layer)', 'GRU (1-layer)', 'Prophet', 'Simple Moving Average']:
                    future_prices = compute_prediction_for(model_type)
                else:
                    future_prices = None

            # Ensure fallback if still None
            if future_prices is None:
                sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
                if pd.isna(sma_20):
                    sma_20 = df['Close'].iloc[-1]
                future_prices = np.array([[sma_20] for _ in range(10)])

        # ========== INVESTMENT ==========
        last_price = df['Close'].iloc[-1]
        shares = investment_amount // last_price
        expected_profit = (future_prices[-1][0] - last_price) * shares

        # ========== RISK MANAGEMENT ==========
        stop_loss = last_price * 0.98
        target_price = last_price * 1.05

        # ========== BACKTESTING ==========
        df['Position'] = 0
        df.loc[df['Signal'] == 'BUY', 'Position'] = 1
        df.loc[df['Signal'] == 'SELL', 'Position'] = -1

        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Market_Return'] * df['Position'].shift(1)

        total_return = (df['Strategy_Return'] + 1).cumprod().iloc[-1] - 1

        # ================= DISPLAY =================

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Latest Stock Data")
            st.dataframe(df.tail())

        with col2:
            st.subheader("🤖 LSTM 10-Day Prediction")
            st.table(pd.DataFrame(future_prices, columns=["Predicted Price"]))

        # ---------- STRATEGY INFO ----------
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"📌 **Strategy:** {strategy}")
        with col_info2:
            st.info(f"🤖 **Model:** {model_type}")

        st.subheader("📈 Price with Bollinger Bands")
        # Prepare prediction series and confidence bands
        try:
            last_date = pd.to_datetime(df.index[-1])
            future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices))
            predicted_series = pd.Series(future_prices.flatten(), index=future_index)

            # Estimate recent volatility (daily returns std) using lookback if available
            vol_window = min(60, len(df))
            rel_vol = df['Close'].pct_change().rolling(vol_window).std().iloc[-1]
            if pd.isna(rel_vol) or rel_vol == 0:
                rel_vol = df['Close'].pct_change().std()
            # Use a multiplier to widen/narrow bands; 1.5 is a reasonable visual
            mult = 1.5
            upper_band = predicted_series * (1 + rel_vol * mult)
            lower_band = predicted_series * (1 - rel_vol * mult)
        except Exception:
            predicted_series = None
            upper_band = None
            lower_band = None

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['Close'], label="Close Price")
        ax.plot(df['BB_High'], linestyle='--', label="BB High")
        ax.plot(df['BB_Low'], linestyle='--', label="BB Low")

        # Plot predicted prices and confidence band if available
        if predicted_series is not None:
            ax.plot(predicted_series.index, predicted_series.values, marker='o', color='red', label='Predicted Price (10d)')
            ax.fill_between(predicted_series.index, lower_band.values, upper_band.values, color='red', alpha=0.15, label='Confidence Band')

        ax.legend()
        st.pyplot(fig)

        # ---------- EQUITY CURVE ----------
        st.subheader("📉 Strategy Equity Curve")
        fig2, ax2 = plt.subplots(figsize=(12,4))
        (df['Strategy_Return'] + 1).cumprod().plot(ax=ax2)
        ax2.set_title("Equity Curve")
        st.pyplot(fig2)

        # ---------- INVESTMENT SUMMARY ----------
        st.subheader("💰 Investment Summary")
        st.write(f"📌 Current Price: ₹{round(last_price,2)}")
        st.write(f"📦 Shares You Can Buy: {int(shares)}")
        st.write(f"🎯 Target Price: ₹{round(target_price,2)}")
        st.write(f"🔻 Stop Loss: ₹{round(stop_loss,2)}")
        st.write(f"📈 Expected Price (10 Days): ₹{round(future_prices[-1][0],2)}")
        st.write(f"💵 Expected Profit/Loss: ₹{round(expected_profit,2)}")

        # ---------- PREDICTION EXPLANATION ----------
        with st.expander("🔎 What these prediction values mean", expanded=False):
            st.markdown("- **Predicted Price:** The model's estimated closing price for each of the next 10 trading days. If an LSTM model was selected, this is produced by a trained neural network using recent historical Close prices. If there wasn't enough history, a Simple Moving Average (SMA) fallback is used.")
            st.markdown("- **Expected Price (10 Days):** The final value in the 10-day prediction list (used above for profit estimates).")
            st.markdown("- **Expected Profit/Loss:** Estimated profit (or loss) if you buy the computed number of shares now and sell at the predicted 10-day price. Calculated as `(predicted_price - current_price) * shares`.")
            st.markdown("- **Target Price:** A simple projected upside level (current price × 1.05). Use as an example target, not a guarantee.")
            st.markdown("- **Stop Loss:** A risk-control level set to protect capital (current price × 0.98). Adjust based on your risk tolerance.")
            st.markdown("- **Model notes:** LSTM models learn temporal patterns from recent data (lookback up to 60 days). Short histories use smaller lookbacks or fall back to SMA, so predictions may be less reliable.")
            st.info("Interpret predictions cautiously — they are model estimates, not guaranteed outcomes.")

        # ---------- BEGINNER FRIENDLY GUIDE ----------
        with st.expander("🧭 Beginner's Guide: How to interpret these results", expanded=False):
            st.markdown("**Quick summary for non-experts:**")
            st.markdown("- **BUY / SELL / HOLD:** These are signals derived from technical indicators. `BUY` suggests conditions historically associated with price increases. `SELL` suggests conditions historically associated with declines. `HOLD` means no clear signal. These are not guarantees — they are suggestions based on patterns.")
            st.markdown("- **When to invest:** If the dashboard shows `BUY` and the predicted 10-day price is higher than the current price, that indicates the model and indicators are aligned. Consider your own risk tolerance, time horizon, and only invest money you can afford to lose.")
            st.markdown("- **When not to invest:** If the signal is `SELL` or the predicted price is below the current price, it's a warning that price may fall in the short-term. You may choose to wait, reduce position size, or set a strict stop-loss.")
            st.markdown("- **How profit/loss is calculated:** `Expected Profit/Loss = (Predicted Price - Current Price) × Shares`. Example: Current = ₹100, Predicted = ₹110, Shares = 10 → Profit = (110-100)*10 = ₹100.")
            st.markdown("- **Time horizon:** Predictions are for the next 10 trading days. This is a short-term horizon — for long-term investing, use fundamental research and consider a longer timeframe.")
            st.markdown("- **Risk management:** Use `Stop Loss` to limit downside (example uses current × 0.98). Use `Target Price` to lock partial profits. Never allocate your full portfolio to a single trade.")
            st.markdown("- **Confidence & reliability:** Model confidence depends on the amount and quality of historical data. Short histories reduce reliability. Always combine model output with your own research.")
            st.info("This guide is educational — it does not constitute financial advice. Consider consulting a licensed financial advisor for personal recommendations.")

        # ---------- BACKTEST RESULT ----------
        st.subheader("📊 Strategy Performance")
        st.write(f"📈 Total Strategy Return: {round(total_return*100,2)} %")

        # ---------- FINAL AI DECISION ----------
        st.subheader("🧠 Final AI Decision")

        if (df['Signal'].iloc[-1] == "BUY") and (future_prices[-1][0] > last_price):
            st.success("✅ STRONG BUY – Indicators + AI aligned")
        elif (df['Signal'].iloc[-1] == "SELL") and (future_prices[-1][0] < last_price):
            st.error("❌ STRONG SELL – High Risk")
        else:
            st.warning("⚠ HOLD – No clear confirmation")

        # ---------- ACTIONABLE RECOMMENDATIONS ----------
        try:
            predicted_price = float(future_prices[-1][0])
            pct_change = (predicted_price - last_price) / last_price * 100
            current_signal = df['Signal'].iloc[-1]

            with st.expander("📝 Actionable Recommendations (based on current output)", expanded=True):
                # Quick one-line actionable summary
                quick_action = "HOLD — No clear action"
                abs_pct = abs(pct_change)
                if current_signal == 'BUY' and predicted_price > last_price:
                    if pct_change >= 1.0:
                        quick_action = f"STRONG BUY — Predicted ↑ {round(pct_change,2)}% — consider Moderate position"
                    elif pct_change >= 0.2:
                        quick_action = f"BUY — Predicted ↑ {round(pct_change,2)}% — consider Small position or scale-in"
                    else:
                        quick_action = f"HOLD — Predicted change small ({round(pct_change,2)}%)"
                elif current_signal == 'SELL' and predicted_price < last_price:
                    if pct_change <= -1.0:
                        quick_action = f"STRONG SELL — Predicted ↓ {round(pct_change,2)}% — consider Reducing exposure"
                    elif pct_change <= -0.2:
                        quick_action = f"SELL — Predicted ↓ {round(pct_change,2)}% — consider Small reduction"
                    else:
                        quick_action = f"HOLD — Predicted change small ({round(pct_change,2)}%)"

                st.markdown(f"**Quick action:** {quick_action}")
                st.markdown(f"**Current signal:** **{current_signal}**  ")
                st.markdown(f"**Current price:** ₹{round(last_price,2)} — **Predicted (10d):** ₹{round(predicted_price,2)} ({round(pct_change,2)}%)")

                if current_signal == 'BUY' and predicted_price > last_price:
                    st.subheader("Recommended action: Consider Buying")
                    st.markdown("- **Why:** Indicators and model expect the price to rise in the next 10 trading days.")
                    st.markdown("- **Suggested entry:** Buy now or scale in across 2–3 purchases to average price.")
                    cons_amt = int(investment_amount * 0.10)
                    mod_amt = int(investment_amount * 0.25)
                    aggr_amt = int(investment_amount * 0.50)
                    st.markdown(f"- **Position sizing examples:** Conservative: invest ₹{cons_amt}, Moderate: ₹{mod_amt}, Aggressive: ₹{aggr_amt}.")
                    st.markdown(f"- **Suggested stop-loss:** approx ₹{round(stop_loss,2)} (current × 0.98).")
                    st.markdown(f"- **Suggested take-profit:** target the model price ₹{round(predicted_price,2)} or use the dashboard target ₹{round(target_price,2)}.")
                    st.markdown(f"- **Estimated short-term ROI:** {round(pct_change,2)}% (before fees/taxes).")
                    st.info("Scale position size to your risk tolerance; consider using limit orders and a stop-loss.")

                elif current_signal == 'SELL' and predicted_price < last_price:
                    st.subheader("Recommended action: Consider Selling / Reducing Exposure")
                    st.markdown("- **Why:** Indicators and model expect short-term downside.")
                    st.markdown("- **Suggested actions:** Close or reduce positions, move proceeds to cash or hedged positions. Consider trailing stop if you remain invested.")
                    st.markdown(f"- **If holding shares:** consider setting a tighter stop-loss below current price (e.g., {round(stop_loss,2)}).")
                    st.markdown(f"- **If you want to re-enter later:** set a price alert around the dashboard's predicted price ₹{round(predicted_price,2)} or wait for a clear BUY signal.")
                    st.info("Avoid panic-selling; consider gradual reductions and re-evaluate as new data arrives.")

                else:
                    st.subheader("Recommended action: Hold / Monitor")
                    st.markdown("- **Why:** No clear agreement between indicators and model, or the direction is uncertain.")
                    st.markdown("- **Suggested actions:** Do not open large new positions. Consider setting alerts for: price crossing SMA50/SMA200, RSI < 30 or > 70, or the model prediction moving > ±3%.")
                    st.markdown("- **Risk controls:** Keep existing stop-losses in place and reassess in 1–3 trading days.")
                    st.info("Waiting avoids taking positions during uncertainty — consider paper-trading strategies first.")

                st.warning("This is educational guidance, not financial advice. Always perform your own research and consider consulting a licensed advisor.")
                # ---------- CHARTS & TABLES EXPLAINED ----------
                st.markdown("**Charts & Tables Explained:**")
                st.markdown("- **Latest Stock Data (table):** Shows recent Open/High/Low/Close/Volume rows. Use the `Close` price for indicators and prediction inputs.")
                st.markdown("- **LSTM 10-Day Prediction (table):** The model's estimated closing price for each of the next 10 trading days. Treat these as short-term estimates, not guarantees.")
                st.markdown("- **Price with Bollinger Bands (chart):** Displays the historical `Close` price with upper and lower Bollinger Bands. Price near the lower band may indicate oversold conditions; near the upper band may indicate overbought conditions. Predicted prices and the shaded confidence band indicate the model's forecast and uncertainty.")
                st.markdown("- **Strategy Equity Curve (chart):** Plots cumulative returns from applying the chosen strategy historically. A rising curve means the strategy produced gains over the tested period; a falling curve means losses.")
                st.markdown("- **Investment Summary (section):** Quick financials: current price, suggested shares for your input amount, suggested stop-loss and target. These are examples — adjust for fees and taxes.")
                st.markdown("- **Strategy Performance (metric):** The total percentage return from the backtest period using generated signals. This is historical performance and does not guarantee future returns.")
                st.info("Tip: Combine signals — model prediction, indicator signal, and equity-curve trend — before taking action. If they disagree, prefer conservative sizing or hold.")
        except Exception:
            st.info("Actionable recommendations unavailable (missing prediction data).")

        # ---------- EXPORT RECOMMENDATION (CSV / PDF) ----------
        try:
            # Build a predictions DataFrame
            if 'predicted_series' in locals() and predicted_series is not None:
                preds_df = pd.DataFrame({
                    'Date': predicted_series.index.strftime('%Y-%m-%d'),
                    'Predicted_Price': predicted_series.values.flatten()
                })
            else:
                # fallback to numeric index if dates unavailable
                preds_df = pd.DataFrame({
                    'Day': list(range(1, len(future_prices) + 1)),
                    'Predicted_Price': np.array(future_prices).flatten()
                })

            # Build details for export
            details = {
                'Symbol': symbol,
                'Current Price': round(last_price, 2),
                'Predicted (10d)': round(float(future_prices[-1][0]), 2),
                'Signal': df['Signal'].iloc[-1],
                'Predicted Change (%)': round(((float(future_prices[-1][0]) - last_price) / last_price) * 100, 2),
                'Shares (suggested)': int(shares),
                'Investment Amount': int(investment_amount),
                'Stop Loss': round(stop_loss, 2),
                'Target Price': round(target_price, 2)
            }

            details_df = pd.DataFrame(list(details.items()), columns=['Metric', 'Value'])

            # CSV download: combine predictions and a header with details
            csv_buf = preds_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="⬇️ Download Predictions (CSV)",
                data=csv_buf,
                file_name=f"{symbol}_predictions.csv",
                mime='text/csv'
            )

            # PDF download: render a simple report using matplotlib text
            from io import BytesIO
            pdf_buf = BytesIO()
            fig_pdf = plt.figure(figsize=(8.27, 11.69))
            fig_pdf.patch.set_facecolor('white')
            txt = fig_pdf.text(0.02, 0.98, f"Stock Prediction Report - {symbol}", fontsize=14, weight='bold')
            y = 0.94
            for k, v in details.items():
                fig_pdf.text(0.02, y, f"{k}: {v}", fontsize=10)
                y -= 0.03
                if y < 0.1:
                    break

            y -= 0.02
            fig_pdf.text(0.02, y, "\nPredicted Prices (next 10 trading days):", fontsize=11, weight='bold')
            y -= 0.03
            # print up to 10 predictions
            for idx, val in enumerate(preds_df.iloc[:, -1].values[:10], start=1):
                fig_pdf.text(0.02, y, f"Day {idx}: {round(float(val),2)}", fontsize=10)
                y -= 0.025
                if y < 0.05:
                    break

            plt.axis('off')
            fig_pdf.savefig(pdf_buf, format='pdf', bbox_inches='tight')
            plt.close(fig_pdf)
            pdf_buf.seek(0)

            st.download_button(
                label="⬇️ Download Report (PDF)",
                data=pdf_buf,
                file_name=f"{symbol}_prediction_report.pdf",
                mime='application/pdf'
            )
        except Exception as e:
            st.info("Export unavailable: " + str(e))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please check your inputs and try again.")
