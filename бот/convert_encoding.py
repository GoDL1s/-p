import ccxt
import pandas as pd
import telegram
import asyncio
import numpy as np
import os
import logging
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from dotenv import load_dotenv

# Load environment variables
dotenv_path = r'C:\Users\lisyu\бот\API.env'
load_dotenv(dotenv_path)

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

# Check if all tokens are loaded
if None in (API_KEY, API_SECRET, TELEGRAM_TOKEN, CHAT_ID):
    raise ValueError("All tokens must be specified in the .env file.")

# Configuration
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'BNB/USDT']
TIMEFRAME = '1m'
INITIAL_CAPITAL = 10.0
CURRENT_CAPITAL = {symbol: INITIAL_CAPITAL for symbol in SYMBOLS}
TRADE_PERCENTAGE = 0.5
TRADING_ENABLED = True

# Initialize exchange
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})

# Configure logging
logging.basicConfig(level=logging.INFO)

# Fetch data function
async def fetch_data(symbol, timeframe):
    logging.info(f"Fetching data for {symbol}...")
    try:
        data = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, timeframe)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

# Calculate indicators
def calculate_indicators(df):
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = compute_rsi(df['close'])
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df['atr'] = compute_atr(df['high'], df['low'], df['close'])
    df.dropna(inplace=True)
    return df

# Compute RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute MACD
def compute_macd(series):
    short_window = 12
    long_window = 26
    signal_window = 9
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

# Compute ATR
def compute_atr(high, low, close):
    return np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift()))).rolling(window=14).mean()

# Train XGBoost model
def train_xgboost(X_train, y_train):
    model = XGBRegressor()
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'xgboost_model.pkl')
    return best_model

# Predict price using XGBoost
def predict_price_xgboost(df):
    df = calculate_indicators(df)

    # Updated required features, removed sentiment
    required_features = ['open', 'high', 'low', 'volume', 'sma_5', 'sma_20', 'rsi', 'macd', 'atr']
    
    # Check if all required features are in the DataFrame
    for feature in required_features:
        if feature not in df.columns:
            logging.error(f"Missing feature: {feature}")
            return None

    X = df[required_features]
    df['returns'] = df['close'].pct_change()
    y = df['returns'].dropna()

    # Synchronize indices to prevent feature mismatch
    X = X.loc[y.index]

    if X.empty or y.empty:
        raise ValueError("No data available for training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model_path = 'xgboost_model.pkl'
    model = joblib.load(model_path) if os.path.exists(model_path) else train_xgboost(X_train, y_train)

    predicted_return = model.predict(X_test[-1:])
    predicted_price = df['close'].iloc[-1] * (1 + predicted_return[0])
    return predicted_price

# Predict price using LSTM
def predict_price_lstm(df):
    df = calculate_indicators(df)

    required_features = ['close', 'sma_5', 'sma_20', 'rsi', 'macd', 'atr']
    
    # Check if all required features are in the DataFrame
    for feature in required_features:
        if feature not in df.columns:
            logging.error(f"Missing feature: {feature}")
            return None

    data = df[required_features].values
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))

    if len(data) < 1:
        raise ValueError("Not enough data to train LSTM.")

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(data, data[:, 0], epochs=50, batch_size=32)

    predicted_price = model.predict(data[-1:])
    return predicted_price[0, 0]

# Open order function
async def open_order(symbol, order_type, amount):
    try:
        if order_type == 'buy':
            order = await asyncio.to_thread(exchange.create_market_buy_order, symbol, amount)
            logging.info(f"Buying {amount} {symbol}: {order}")
        elif order_type == 'sell':
            order = await asyncio.to_thread(exchange.create_market_sell_order, symbol, amount)
            logging.info(f"Selling {amount} {symbol}: {order}")
    except Exception as e:
        logging.error(f"Error executing {order_type} order for {symbol}: {e}")

# Trade function
async def trade(symbol):
    global CURRENT_CAPITAL

    while True:
        df = await fetch_data(symbol, TIMEFRAME)
        if df is not None:
            predicted_price_xgboost = predict_price_xgboost(df)
            predicted_price_lstm = predict_price_lstm(df)

            if predicted_price_xgboost is not None and predicted_price_lstm is not None:
                current_price = df['close'].iloc[-1]
                logging.info(f"Current price {symbol}: {current_price}, XGBoost prediction: {predicted_price_xgboost}, LSTM prediction: {predicted_price_lstm}")

                if TRADING_ENABLED:
                    if predicted_price_xgboost > current_price and predicted_price_lstm > current_price:
                        amount_to_buy = CURRENT_CAPITAL[symbol] * TRADE_PERCENTAGE / current_price
                        await open_order(symbol, 'buy', amount_to_buy)
                        CURRENT_CAPITAL[symbol] -= CURRENT_CAPITAL[symbol] * TRADE_PERCENTAGE

                    elif predicted_price_xgboost < current_price and predicted_price_lstm < current_price:
                        amount_to_sell = CURRENT_CAPITAL[symbol] * TRADE_PERCENTAGE
                        await open_order(symbol, 'sell', amount_to_sell)
                        CURRENT_CAPITAL[symbol] += CURRENT_CAPITAL[symbol] * TRADE_PERCENTAGE

        await asyncio.sleep(60)  # Wait before next cycle

# Main function
async def main():
    await asyncio.gather(*(trade(symbol) for symbol in SYMBOLS))

if __name__ == "__main__":
    asyncio.run(main())
