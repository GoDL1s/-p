import os
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
from aiogram.filters import Command
import ccxt.async_support as ccxt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, IsolationForest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from deep_translator import GoogleTranslator
import xgboost as xgb
import httpx
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import io
import ta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import minimize
import aiohttp
import functools
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import joblib
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import logging
from logging.handlers import RotatingFileHandler
import gc
import time
import psutil
from collections import OrderedDict
import asyncio
from typing import Dict, Any
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import os
import asyncio
import aiohttp
import gc
from datetime import datetime
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Настройка для работы с Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Настройка для TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Оптимизация для CPU
num_cores = 24  # AMD EPYC 7402 имеет 24 ядра
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.txt", encoding='utf-8'),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = "bot_log.txt"
log_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Добавляем новый логгер для анализа криптовалют и машинного обучения
crypto_ml_logger = logging.getLogger('crypto_ml')
crypto_ml_logger.setLevel(logging.INFO)
crypto_ml_handler = RotatingFileHandler("crypto_ml_log.txt", maxBytes=5*1024*1024, backupCount=2)
crypto_ml_handler.setFormatter(log_formatter)
crypto_ml_logger.addHandler(crypto_ml_handler)

# Загрузка переменных окружения
load_dotenv(r'C:\Users\lisyu\бот\API.env')

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

# Проверка наличия всех токенов
required_tokens = ['API_KEY', 'API_SECRET', 'TELEGRAM_TOKEN', 'CHAT_ID']
missing_tokens = [token for token in required_tokens if os.getenv(token) is None]
if missing_tokens:
    raise ValueError(f"Следующие токены отсутствуют в файле .env: {', '.join(missing_tokens)}")

# Конфигурации
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'XRPUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'LINKUSDT', 'UNIUSDT']
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
TRADING_ENABLED = False
UPDATE_INTERVAL = 60  # Обновление каждые 60 секунд
MODEL_UPDATE_INTERVAL = timedelta(minutes=30)  # Обновляем модели каждые 30 минут
HISTORICAL_DATA_DAYS = 90
PORTFOLIO_PERCENTAGE = 0.1
MAX_PORTFOLIO_RISK = 5

# Определение признаков
features = ['open', 'high', 'low', 'close', 'volume', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_Signal', 
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'EMA20', 'ATR', 'OBV', 'close_lag1', 'close_lag2',
            'Stochastic_K', 'Stochastic_D', 'ROC', 'Williams_R', 'CCI', 'ADX']

# Инициализация клиентов
bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher()
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
})

# Создание кнопок
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="СТАРТ"), KeyboardButton(text="СТОП")],
        [KeyboardButton(text="БАЛАНС 💰"), KeyboardButton(text="АНАЛИЗ 📊")],
        [KeyboardButton(text="ГРАФИК 📈"), KeyboardButton(text="СТАТИСТИКА 📉")],
        [KeyboardButton(text="НОВОСТИ 📰"), KeyboardButton(text="НАСТРОЙКИ ⚙️")],
        [KeyboardButton(text="ПЕРЕВОДЫ 💸"), KeyboardButton(text="ПОРТФЕЛЬ 💼")],
        [KeyboardButton(text="РИСК-МЕНЕДЖМЕНТ 🛡️"), KeyboardButton(text="ОПТИМИЗАЦИЯ 🔧")]
    ],
    resize_keyboard=True
)

balance_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="СПОТОВЫЙ БАЛАНС", callback_data="spot_balance")],
        [InlineKeyboardButton(text="БАЛАНС P2P 💳", callback_data="p2p_balance")]
    ]
)

settings_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="10%", callback_data="set_percentage_10"),
         InlineKeyboardButton(text="20%", callback_data="set_percentage_20"),
         InlineKeyboardButton(text="30%", callback_data="set_percentage_30")],
        [InlineKeyboardButton(text="50%", callback_data="set_percentage_50"),
         InlineKeyboardButton(text="100%", callback_data="set_percentage_100")],
        [InlineKeyboardButton(text="Свой процент", callback_data="set_custom_percentage")]
    ]
)

transfer_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Перевод крипты", callback_data="transfer_crypto")],
        [InlineKeyboardButton(text="Перевод P2P", callback_data="transfer_p2p")]
    ]
)

# Глобальные переменные
latest_analysis_results = {}
model_performance = {}
user_portfolios = {}
is_trading = False
daily_balance_start = None
daily_pnl = 0
trade_history = []

# Новые глобальные переменные
DATA_CACHE = {}
DATA_CACHE_TIME = 300  # 5 минут
NEWS_CACHE = {}
NEWS_CACHE_TIME = 3600  # 1 час
ANALYSIS_RESULTS_CACHE = OrderedDict()
MAX_CACHE_SIZE = 1000

# ThreadPoolExecutor для параллельных вычислений
executor = ThreadPoolExecutor(max_workers=num_cores)

# Функция для кэширования результатов
def async_cache(ttl=300):
    def wrapper(func):
        cache = {}
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            result = await func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result
        return wrapped
    return wrapper

# Глобальные переменные для хранения исторических данных и результатов анализа
HISTORICAL_DATA_CACHE = OrderedDict()
ANALYSIS_RESULTS_CACHE = OrderedDict()

# Настройка Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Crypto Trading Bot'

# Функция для получения сервиса Google Drive
def get_google_drive_service():
    try:
        # Попробуем использовать service account
        creds = None
        if os.path.exists('service_account.json'):
            creds = service_account.Credentials.from_service_account_file(
                'service_account.json', scopes=['https://www.googleapis.com/auth/drive.file'])
        elif all(key in os.environ for key in ['GOOGLE_ACCOUNT_TYPE', 'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_EMAIL', 'GOOGLE_PRIVATE_KEY']):
            creds_info = {
                "type": os.environ['GOOGLE_ACCOUNT_TYPE'],
                "client_id": os.environ['GOOGLE_CLIENT_ID'],
                "client_email": os.environ['GOOGLE_CLIENT_EMAIL'],
                "private_key": os.environ['GOOGLE_PRIVATE_KEY'].replace('\\n', '\n'),
            }
            creds = service_account.Credentials.from_service_account_info(
                creds_info, scopes=['https://www.googleapis.com/auth/drive.file'])
        
        if creds:
            return build('drive', 'v3', credentials=creds)
        else:
            logger.warning("Google Drive аутентификация не удалась. Функциональность Google Drive будет отключена.")
            return None
    except Exception as e:
        logger.error(f"Ошибка при инициализации Google Drive сервиса: {e}")
        return None

# Инициализация Google Drive сервиса
drive_service = get_google_drive_service()

async def upload_to_drive(file_path, file_name):
    try:
        file_metadata = {'name': file_name}
        media = MediaFileUpload(file_path, resumable=True)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        logger.info(f"File ID: {file.get('id')} uploaded to Google Drive")
    except Exception as e:
        logger.error(f"Error uploading file to Google Drive: {e}")

@async_cache(ttl=DATA_CACHE_TIME)
async def get_historical_data(symbol, timeframe):
    """
    Получает исторические данные для заданной пары и таймфрейма
    """
    try:
        end_time = exchange.milliseconds()
        start_time = end_time - (HISTORICAL_DATA_DAYS * 24 * 60 * 60 * 1000)
        
        # Проверка, чтобы не запрашивать данные из будущего
        current_time = exchange.milliseconds()
        if start_time > end_time:
            start_time = end_time - (24 * 60 * 60 * 1000)  # Начнем с 24 часов назад
        
        all_ohlcv = []
        while start_time < end_time:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, start_time, limit=1000)
            if len(ohlcv) == 0:
                break
            all_ohlcv.extend(ohlcv)
            start_time = ohlcv[-1][0] + 1
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df.index = pd.DatetimeIndex(df.index).to_period(timeframe)
        
        # Обновляем кэш исторических данных
        cache_key = f"{symbol}_{timeframe}"
        HISTORICAL_DATA_CACHE[cache_key] = df.tail(MAX_CACHE_SIZE)
        
        # Если кэш превысил максимальный размер, удаляем старые записи
        if len(HISTORICAL_DATA_CACHE) > MAX_CACHE_SIZE:
            HISTORICAL_DATA_CACHE.popitem(last=False)
        
        logger.info(f"Получены исторические данные для {symbol} на {timeframe}")
        return df
    except Exception as e:
        logger.error(f"Ошибка получения данных для {symbol} на {timeframe}: {e}")
        return pd.DataFrame()
    
def add_features(df):
    """
    Добавляет технические индикаторы к DataFrame
    """
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    
    # Новые индикаторы
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stochastic_K'] = stoch.stoch()
    df['Stochastic_D'] = stoch.stoch_signal()
    df['ROC'] = ta.momentum.roc(df['close'], window=12)
    df['Williams_R'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    return df

def prepare_data_for_ml(df):
    """
    Подготавливает данные для машинного обучения
    """
    df = add_features(df)
    
    df_clean = df.dropna()
    
    if df_clean.empty:
        logger.warning("После удаления NaN значений данные стали пустыми")
        return None, None, None

    X = df_clean[features]
    y = df_clean['close'].shift(-1)  # Целевая переменная - цена закрытия следующего периода

    X = X[:-1]
    y = y[:-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def optimize_rf(X, y):
    """
    Оптимизирует гиперпараметры для Random Forest
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=num_cores)
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=num_cores, verbose=1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

class LSTMRegressor(BaseEstimator, RegressorMixin):
    """
    Класс для LSTM регрессора
    """
    def __init__(self, input_shape, epochs=50, batch_size=32):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = build_lstm_model(self.input_shape)
        self.model.fit(X.reshape(-1, self.input_shape, 1), y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X.reshape(-1, self.input_shape, 1)).flatten()

def build_lstm_model(input_shape):
    """
    Создает модель LSTM
    """
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(input_shape, 1)),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def validate_model(X, y, model):
    """
    Валидирует модель с использованием кросс-валидации временных рядов
    """
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    return np.mean(scores), np.std(scores)

def backtest_strategy(df, model, initial_balance=1000):
    """
    Проводит бэктестинг стратегии
    """
    balance = initial_balance
    position = None
    for i in range(len(df) - 1):
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i + 1]
        prediction = model.predict(df.iloc[i:i+1][features])
        
        if prediction > current_price and position is None:
            position = balance / current_price
            balance = 0
        elif prediction < current_price and position is not None:
            balance = position * current_price
            position = None
    
    if position is not None:
        balance = position * df['close'].iloc[-1]
    
    return balance - initial_balance

def analyze_residuals(y_true, y_pred):
    """
    Анализирует остатки модели
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title('Анализ остатков')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig('residuals_analysis.png')
    plt.close()
    
    # Загрузка графика на Google Drive
    asyncio.create_task(upload_to_drive('residuals_analysis.png', 'residuals_analysis.png'))

def monitor_model_performance(model, X, y, window_size=100):
    """
    Отслеживает производительность модели во времени
    """
    performance = []
    for i in range(window_size, len(X), window_size // 10):
        X_window = X[i-window_size:i]
        y_window = y[i-window_size:i]
        y_pred = model.predict(X_window)
        mse = mean_squared_error(y_window, y_pred)
        performance.append(mse)
    return performance

def remove_anomalies(X, y, contamination=0.01):
    """
    Удаляет аномалии из данных
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=num_cores)
    yhat = iso_forest.fit_predict(X)
    mask = yhat != -1
    return X[mask], y[mask]

def naive_forecast(y):
    """
    Создает наивный прогноз
    """
    return np.roll(y, 1)

async def plot_feature_importance(model, feature_names):
    """
    Визуализирует важность признаков
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pos, importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(feature_names)[sorted_idx])
    ax.set_title('Важность признаков')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Загрузка графика на Google Drive
    asyncio.create_task(upload_to_drive('feature_importance.png', 'feature_importance.png'))

async def analyze_autocorrelation(y):
    """
    Анализирует автокорреляцию временного ряда
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plot_acf(y, ax=ax1, lags=50)
    plot_pacf(y, ax=ax2, lags=50)
    plt.savefig('autocorrelation.png')
    plt.close()
    
    # Загрузка графика на Google Drive
    asyncio.create_task(upload_to_drive('autocorrelation.png', 'autocorrelation.png'))

async def train_and_evaluate_models(X, y):
    """
    Обучает и оценивает модели машинного обучения
    """
    if X is None or y is None:
        logger.error("Невозможно обучить модели: отсутствуют данные")
        return {}

    X, y = remove_anomalies(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf_model = load_model('rf', 'all')
    if rf_model is None:
        rf_model = optimize_rf(X_train, y_train)
    else:
        rf_model.fit(X_train, y_train)
    
    xgb_model = load_model('xgb', 'all')
    if xgb_model is None:
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=num_cores)
    
    lstm_model = load_model('lstm', 'all')
    if lstm_model is None:
        lstm_model = LSTMRegressor(input_shape=X_train.shape[1], epochs=50, batch_size=32)
    
    elastic_net = load_model('elastic_net', 'all')
    if elastic_net is None:
        elastic_net = ElasticNet(random_state=42)
    
    svr = load_model('svr', 'all')
    if svr is None:
        svr = SVR()
    
    # Используем ThreadPoolExecutor для параллельного обучения моделей
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        future_xgb = executor.submit(xgb_model.fit, X_train, y_train)
        future_lstm = executor.submit(lstm_model.fit, X_train, y_train)
        future_elastic = executor.submit(elastic_net.fit, X_train, y_train)
        future_svr = executor.submit(svr.fit, X_train, y_train)
        
        xgb_model = future_xgb.result()
        lstm_model = future_lstm.result()
        elastic_net = future_elastic.result()
        svr = future_svr.result()
    
    ensemble = VotingRegressor([
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lstm', lstm_model),
        ('elastic', elastic_net),
        ('svr', svr)
    ])
    
    ensemble.fit(X_train, y_train)
    
    # Сохраняем обученные модели
    save_model(rf_model, 'rf', 'all')
    save_model(xgb_model, 'xgb', 'all')
    save_model(lstm_model, 'lstm', 'all')
    save_model(elastic_net, 'elastic_net', 'all')
    save_model(svr, 'svr', 'all')
    save_model(ensemble, 'ensemble', 'all')
    
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    val_score, val_std = validate_model(X, y, ensemble)
    backtest_profit = backtest_strategy(pd.DataFrame(X, columns=features), ensemble)
    analyze_residuals(y_test, y_pred)
    performance = monitor_model_performance(ensemble, X, y)
    naive_mse = mean_squared_error(y[1:], naive_forecast(y)[1:])
    
    await plot_feature_importance(rf_model, features)
    await analyze_autocorrelation(y)
    
    # Логирование результатов анализа и машинного обучения
    crypto_ml_logger.info(f"Результаты обучения моделей:")
    crypto_ml_logger.info(f"MSE: {mse:.4f}")
    crypto_ml_logger.info(f"R2: {r2:.4f}")
    crypto_ml_logger.info(f"MAE: {mae:.4f}")
    crypto_ml_logger.info(f"MAPE: {mape:.2f}%")
    crypto_ml_logger.info(f"Validation Score: {val_score:.4f} +/- {val_std:.4f}")
    crypto_ml_logger.info(f"Backtest Profit: {backtest_profit:.2f}")
    crypto_ml_logger.info(f"Naive MSE: {naive_mse:.4f}")
    
    return {
        'ensemble': {'model': ensemble, 'mse': mse, 'r2': r2, 'mae': mae, 'mape': mape},
        'rf': {'model': rf_model},
        'xgb': {'model': xgb_model},
        'lstm': {'model': lstm_model},
        'elastic_net': {'model': elastic_net},
        'svr': {'model': svr},
        'validation_score': val_score,
        'validation_std': val_std,
        'backtest_profit': backtest_profit,
        'performance': performance,
        'naive_mse': naive_mse
    }
def save_model(model, symbol, timeframe):
    """
    Сохраняет модель машинного обучения
    """
    filename = f"model_{symbol}_{timeframe}.joblib"
    joblib.dump(model, filename)
    logger.info(f"Модель сохранена: {filename}")
    
    # Загрузка модели на Google Drive
    asyncio.create_task(upload_to_drive(filename, filename))

def load_model(symbol, timeframe):
    """
    Загружает модель машинного обучения
    """
    filename = f"model_{symbol}_{timeframe}.joblib"
    if os.path.exists(filename):
        model = joblib.load(filename)
        logger.info(f"Модель загружена: {filename}")
        return model
    return None

async def learn_from_errors():
    """
    Обучает модели на основе ошибок прошлых прогнозов
    """
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            df = await get_historical_data(symbol, timeframe)
            X, y, scaler = prepare_data_for_ml(df)
            if X is not None and y is not None:
                model = latest_analysis_results[(symbol, timeframe)]['ensemble']['model']
                
                # Получаем прогнозы модели
                y_pred = model.predict(X)
                
                # Вычисляем ошибки
                errors = y - y_pred
                
                # Создаем новый набор данных, включающий ошибки
                X_new = np.column_stack((X, errors[:-1]))  # Исключаем последнюю ошибку, так как для нее нет следующего значения
                y_new = y[1:]  # Сдвигаем целевую переменную на один шаг вперед
                
                # Обучаем модель на новом наборе данных
                model.fit(X_new, y_new)
                
                # Сохраняем обновленную модель
                save_model(model, symbol, timeframe)
                
                logger.info(f"Модель для {symbol} на {timeframe} обучена на ошибках")

async def error_recovery(func):
    async def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 5  # секунд
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Ошибка в {func.__name__}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Повторная попытка через {retry_delay} секунд...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Все попытки исчерпаны. Функция {func.__name__} не выполнена.")
    return wrapper

@error_recovery
async def analyze_crypto(symbol, timeframe):
    """
    Анализирует криптовалюту и возвращает результаты анализа
    """
    try:
        df = await get_historical_data(symbol, timeframe)
        if df.empty:
            logger.warning(f"Нет данных для {symbol} на {timeframe}")
            return None

        X, y, scaler = prepare_data_for_ml(df)
        if X is None or y is None:
            logger.warning(f"Недостаточно данных для анализа {symbol} на {timeframe}")
            return None

        # ARIMA прогноз
        try:
            arima_model = ARIMA(df['close'], order=(1,1,1))
            arima_results = arima_model.fit()
            arima_forecast = arima_results.forecast(steps=1).iloc[0]
        except Exception as e:
            logger.error(f"Ошибка при ARIMA прогнозе для {symbol} на {timeframe}: {e}")
            arima_forecast = None

        # Prophet прогноз
        try:
            prophet_df = pd.DataFrame({'ds': df.index.to_timestamp(), 'y': df['close']})
            prophet_model = Prophet(yearly_seasonality=True)
            prophet_model.fit(prophet_df)
            future_data = prophet_model.make_future_dataframe(periods=1, freq=timeframe)
            prophet_forecast = prophet_model.predict(future_data).iloc[-1]['yhat']
        except Exception as e:
            logger.error(f"Ошибка при Prophet прогнозе для {symbol} на {timeframe}: {e}")
            prophet_forecast = None

        # Анализ настроений новостей
        sentiment_score = await analyze_news_sentiment(symbol)

        # Анализ объема и ликвидности
        volume_ma, liquidity = analyze_volume_liquidity(df)

        # Обнаружение аномалий
        anomalies = detect_anomalies(df)

        model_results = await train_and_evaluate_models(X, y)
        ensemble_model = model_results['ensemble']['model']
        last_data_point = scaler.transform(df.iloc[-1][features].values.reshape(1, -1))
        prediction = ensemble_model.predict(last_data_point)[0]
        
        current_price = df['close'].iloc[-1]
        change = ((prediction - current_price) / current_price) * 100
        
        # Расчет волатильности и динамических стоп-лоссов/тейк-профитов
        volatility = calculate_volatility(df['close'])
        stop_loss, take_profit = calculate_dynamic_stops(current_price, volatility)
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'prediction': prediction,
            'arima_forecast': arima_forecast,
            'prophet_forecast': prophet_forecast,
            'change': change,
            'sentiment_score': sentiment_score,
            'volume_ma': volume_ma,
            'liquidity': liquidity,
            'anomalies': len(anomalies),
            'volatility': volatility,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'metrics': model_results['ensemble'],
            'validation_score': model_results['validation_score'],
            'validation_std': model_results['validation_std'],
            'backtest_profit': model_results['backtest_profit'],
            'naive_mse': model_results['naive_mse']
        }

        # Обновляем кэш результатов анализа
        cache_key = f"{symbol}_{timeframe}"
        ANALYSIS_RESULTS_CACHE[cache_key] = result

        # Если кэш превысил максимальный размер, удаляем старые записи
        if len(ANALYSIS_RESULTS_CACHE) > MAX_CACHE_SIZE:
            ANALYSIS_RESULTS_CACHE.popitem(last=False)

        logger.info(f"Выполнен анализ для {symbol} на {timeframe}")
        
        # Логирование результатов анализа
        crypto_ml_logger.info(f"Результаты анализа для {symbol} на {timeframe}:")
        crypto_ml_logger.info(f"Текущая цена: {current_price:.2f}")
        crypto_ml_logger.info(f"Прогноз (ансамбль): {prediction:.2f} ({change:.2f}%)")
        crypto_ml_logger.info(f"Прогноз (ARIMA): {arima_forecast:.2f}")
        crypto_ml_logger.info(f"Прогноз (Prophet): {prophet_forecast:.2f}")
        crypto_ml_logger.info(f"Настроение рынка: {sentiment_score:.2f}")
        crypto_ml_logger.info(f"Волатильность: {volatility:.4f}")
        crypto_ml_logger.info(f"Стоп-лосс: {stop_loss:.2f}")
        crypto_ml_logger.info(f"Тейк-профит: {take_profit:.2f}")
        
        return result

    except Exception as e:
        logger.error(f"Ошибка при анализе {symbol} на {timeframe}: {e}")
        return None

async def analyze_news_sentiment(symbol):
    """
    Анализирует настроения новостей для заданной криптовалюты
    """
    news = await get_crypto_news(symbol)
    if not news:
        return 0
    
    # Простой анализ настроений на основе ключевых слов
    positive_words = ['рост', 'увеличение', 'прибыль', 'успех', 'позитивный']
    negative_words = ['падение', 'снижение', 'убыток', 'проблема', 'негативный']
    
    sentiment_scores = []
    for item in news:
        score = 0
        for word in positive_words:
            if word in item['title'].lower():
                score += 1
        for word in negative_words:
            if word in item['title'].lower():
                score -= 1
        sentiment_scores.append(score)
    
    return np.mean(sentiment_scores) if sentiment_scores else 0

async def get_crypto_news(symbol):
    """
    Получает последние новости о криптовалюте
    """
    url = f"https://cryptonews-api.com/api/v1?tickers={symbol}&items=5&token=YOUR_API_TOKEN"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                news = await response.json()
                return news['data']
            else:
                logger.error(f"Не удалось получить новости для {symbol}")
                return []

def analyze_volume_liquidity(df):
    """
    Анализирует объем и ликвидность
    """
    volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
    liquidity = df['volume'].iloc[-1] / volume_ma
    return volume_ma, liquidity

def detect_anomalies(df):
    """
    Обнаруживает аномалии в данных
    """
    iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=num_cores)
    anomalies = iso_forest.fit_predict(df[['close', 'volume']])
    return np.where(anomalies == -1)[0]

def calculate_volatility(prices, window=20):
    """
    Рассчитывает волатильность
    """
    return np.std(np.log(prices[1:] / prices[:-1])) * np.sqrt(252)

def calculate_dynamic_stops(current_price, volatility, atr_multiplier=2):
    """
    Рассчитывает динамические стоп-лоссы и тейк-профиты
    """
    atr = volatility * current_price
    stop_loss = current_price - (atr * atr_multiplier)
    take_profit = current_price + (atr * atr_multiplier)
    return stop_loss, take_profit

async def apply_dynamic_stop_loss_take_profit(symbol, entry_price, amount, stop_loss, take_profit):
    """
    Применяет динамические стоп-лоссы и тейк-профиты к позиции
    """
    try:
        current_price = (await exchange.fetch_ticker(symbol))['last']
        
        if current_price <= stop_loss:
            order = await exchange.create_market_sell_order(symbol, amount)
            logger.info(f"Сработал динамический стоп-лосс для {symbol}: {order}")
            return 'stop_loss'
        elif current_price >= take_profit:
            order = await exchange.create_market_sell_order(symbol, amount)
            logger.info(f"Сработал динамический тейк-профит для {symbol}: {order}")
            return 'take_profit'
        return None
    except Exception as e:
        logger.error(f"Ошибка при применении динамического стоп-лосс/тейк-профит для {symbol}: {e}")
        return None

async def get_total_balance_usdt():
    """
    Получает общий баланс в USDT, включая спотовый и P2P баланс
    """
    try:
        spot_balance, _ = await get_spot_balance()
        p2p_balance = await get_p2p_balance()
        
        total_balance = spot_balance + p2p_balance
        return total_balance
    except Exception as e:
        logger.error(f"Ошибка при получении общего баланса: {e}")
        return 0

async def get_spot_balance():
    """
    Получает спотовый баланс
    """
    try:
        balance = await exchange.fetch_balance({'type': 'spot'})
        total_balance_usdt = 0
        crypto_balances = {}

        for currency, amount in balance['total'].items():
            if amount > 0:
                if currency != 'USDT':
                    try:
                        ticker = await exchange.fetch_ticker(f"{currency}/USDT")
                        price_usdt = ticker['last']
                        value_usdt = amount * price_usdt
                        total_balance_usdt += value_usdt
                        crypto_balances[currency] = {
                            'amount': amount,
                            'price_usdt': price_usdt,
                            'value_usdt': value_usdt
                        }
                    except Exception as e:
                        logger.error(f"Ошибка при получении цены для {currency}: {e}")
                else:
                    total_balance_usdt += amount
                    crypto_balances[currency] = {
                        'amount': amount,
                        'price_usdt': 1,
                        'value_usdt': amount
                    }

        return total_balance_usdt, crypto_balances
    except Exception as e:
        logger.error(f"Ошибка при получении спотового баланса: {e}")
        return 0, {}

async def get_p2p_balance():
    """
    Получает P2P баланс
    """
    try:
        # Здесь должна быть логика получения P2P баланса
        # Так как у CCXT нет прямого метода для этого, вам может потребоваться
        # использовать специфичный API Binance для P2P
        p2p_balance = 0  # Замените на реальное получение P2P баланса
        return p2p_balance
    except Exception as e:
        logger.error(f"Ошибка при получении P2P баланса: {e}")
        return 0

async def create_candlestick_chart(df, symbol, timeframe):
    """
    Создает свечной график и возвращает его как InputFile
    """
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=(f"{symbol} Price", "Volume", "MACD", "RSI"))

    # Свечной график
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    
    # График объема
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    fig.update_layout(height=1000, width=1200, title_text=f"{symbol} Analysis ({timeframe})")
    fig.update_xaxes(rangeslider_visible=False)

    img_bytes = fig.to_image(format="png")
    return InputFile(io.BytesIO(img_bytes), filename=f"{symbol}_{timeframe}_chart.png")

def clean_memory():
    gc.collect()

def log_performance():
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=1)
    logger.info(f"Использование памяти: {memory_info.rss / 1024 / 1024:.2f} MB")
    logger.info(f"Использование CPU: {cpu_percent}%")
    
async def continuous_analysis():
    """
    Непрерывно анализирует криптовалюты
    """
    while True:
        start_time = time.time()
        logger.info("Начало цикла непрерывного анализа")
        tasks = []
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                tasks.append(asyncio.create_task(analyze_crypto(symbol, timeframe)))
        
        try:
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if result:
                    symbol = result['symbol']
                    timeframe = result['timeframe']
                    latest_analysis_results[(symbol, timeframe)] = result
                    logger.info(f"Обновлены результаты анализа для {symbol} на {timeframe}")
                    
                    # Сохраняем графики
                    plt.figure(figsize=(12, 6))
                    plt.plot(result['metrics']['performance'])
                    plt.title(f"Производительность модели для {symbol} ({timeframe})")
                    plt.xlabel("Время")
                    plt.ylabel("MSE")
                    plt.savefig(f'model_performance_{symbol}_{timeframe}.png')
                    plt.close()
                    
                    # Загружаем график на Google Drive
                    asyncio.create_task(upload_to_drive(f'model_performance_{symbol}_{timeframe}.png', f'model_performance_{symbol}_{timeframe}.png'))
                    
                    logger.info(f"Сохранен график производительности для {symbol} на {timeframe}")
        
        except Exception as e:
            logger.error(f"Ошибка в цикле анализа: {e}")
        
        clean_memory()  # Очистка памяти после каждого цикла анализа
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Длительность цикла анализа: {duration:.2f} секунд")
        log_performance()
        
        # Динамическая корректировка интервала обновления
        if duration > UPDATE_INTERVAL:
            new_interval = min(UPDATE_INTERVAL * 2, 300)  # Максимум 5 минут
            logger.warning(f"Цикл анализа занял больше времени, чем интервал обновления. Увеличиваем интервал до {new_interval} секунд.")
            UPDATE_INTERVAL = new_interval
        elif duration < UPDATE_INTERVAL / 2:
            new_interval = max(UPDATE_INTERVAL / 2, 30)  # Минимум 30 секунд
            logger.info(f"Цикл анализа выполнился быстро. Уменьшаем интервал до {new_interval} секунд.")
            UPDATE_INTERVAL = new_interval
        
        await asyncio.sleep(UPDATE_INTERVAL)

async def trading_loop():
    """
    Основной торговый цикл
    """
    while True:
        if is_trading:
            logger.info("Начало торгового цикла")
            await make_trade_decision()
            await check_and_execute_opportunities()
            logger.info("Торговый цикл завершен")
        else:
            logger.info("Торговля приостановлена")
        await asyncio.sleep(60)  # Проверка каждую минуту

async def make_trade_decision():
    """
    Принимает решение о торговле на основе анализа
    """
    best_opportunity = None
    portfolio_weights = {}
    
    for symbol in SYMBOLS:
        analysis = latest_analysis_results.get((symbol, '5m'))  # Используем 5-минутный таймфрейм для быстрого реагирования
        if analysis:
            score = analysis['change'] + analysis['sentiment_score'] * 10
            if analysis['anomalies'] > 0:
                score *= 0.5
            
            if best_opportunity is None or score > best_opportunity['score']:
                best_opportunity = {'symbol': symbol, 'score': score, 'analysis': analysis}
    
    if best_opportunity:
        symbol = best_opportunity['symbol']
        analysis = best_opportunity['analysis']
        
        win_rate = 0.6
        win_loss_ratio = analysis['take_profit'] / analysis['stop_loss']
        kelly_fraction = kelly_criterion(win_rate, win_loss_ratio)
        
        returns = pd.DataFrame({s: (await get_historical_data(s, '1d'))['close'].pct_change() for s in SYMBOLS})
        target_return = 0.1
        weights = optimize_portfolio(returns, target_return)
        for symbol, weight in zip(SYMBOLS, weights):
            portfolio_weights[symbol] = weight
        
        trade_amount = PORTFOLIO_PERCENTAGE * portfolio_weights[symbol] * kelly_fraction
        
        # Улучшение: Проверка тренда перед принятием решения о торговле
        trend = analyze_trend(analysis)
        if trend == 'up' and analysis['change'] > 0:
            await execute_trade(symbol, 'buy', trade_amount, analysis['stop_loss'], analysis['take_profit'])
        elif trend == 'down' and analysis['change'] < 0:
            await execute_trade(symbol, 'sell', trade_amount, analysis['take_profit'], analysis['stop_loss'])
        else:
            logger.info(f"Нет четкого тренда для {symbol}, пропускаем торговлю")

async def execute_trade(symbol, side, amount, stop_loss, take_profit):
    """
    Выполняет торговую операцию с улучшенным управлением рисками
    """
    try:
        balance = await exchange.fetch_balance()
        total_balance = balance['total']['USDT']
        
        # Проверка на наличие достаточных средств
        if side == 'buy' and total_balance < amount:
            logger.warning(f"Недостаточно средств для покупки {symbol}. Требуется: {amount}, доступно: {total_balance}")
            return False

        # Улучшение: Динамическое определение размера позиции
        trade_amount = min(calculate_position_size(symbol, amount, stop_loss), total_balance)
        
        if side == 'buy':
            order = await exchange.create_market_buy_order(symbol, trade_amount)
        else:
            order = await exchange.create_market_sell_order(symbol, trade_amount)
        
        logger.info(f"Выполнен ордер: {order}")
        trade_history.append({
            'timestamp': datetime.now(),
            'type': 'trade',
            'symbol': symbol,
            'side': side,
            'amount': trade_amount,
            'order': order
        })
        
        entry_price = order['price']
        amount = order['amount']
        
        # Устанавливаем стоп-лосс и тейк-профит ордера
        await exchange.create_order(symbol, 'stop_loss', 'sell', amount, stop_loss)
        await exchange.create_order(symbol, 'take_profit', 'sell', amount, take_profit)
        
        # Запускаем асинхронную задачу для отслеживания позиции
        asyncio.create_task(monitor_position(symbol, entry_price, amount, stop_loss, take_profit))
        
        return True
    except ccxt.NetworkError as e:
        logger.error(f"Ошибка сети при выполнении сделки: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Ошибка биржи при выполнении сделки: {e}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при выполнении сделки: {e}")
    
    return False

async def transfer_between_currencies(from_currency, to_currency, amount):
    """
    Выполняет перевод между валютами
    """
    try:
        # Проверяем, есть ли прямая торговая пара
        try:
            await exchange.fetch_ticker(f"{from_currency}/{to_currency}")
            direct_pair = True
        except:
            direct_pair = False

        if direct_pair:
            if from_currency == to_currency.split('/')[0]:
                order = await exchange.create_market_sell_order(f"{from_currency}/{to_currency}", amount)
            else:
                order = await exchange.create_market_buy_order(f"{from_currency}/{to_currency}", amount)
        else:
            # Если нет прямой пары, используем USDT как промежуточную валюту
            usdt_amount = await exchange.create_market_sell_order(f"{from_currency}/USDT", amount)
            final_order = await exchange.create_market_buy_order(f"{to_currency}/USDT", usdt_amount['filled'])
            order = final_order

        logger.info(f"Выполнен перевод между валютами: {order}")
        trade_history.append({
            'timestamp': datetime.now(),
            'type': 'transfer',
            'from': from_currency,
            'to': to_currency,
            'amount': amount,
            'order': order
        })
        return True
    except Exception as e:
        logger.error(f"Ошибка при переводе между валютами: {e}")
        return False

def analyze_trend(analysis):
    """
    Анализирует тренд на основе технических индикаторов
    """
    if analysis['SMA20'] > analysis['SMA50'] and analysis['RSI'] > 50:
        return 'up'
    elif analysis['SMA20'] < analysis['SMA50'] and analysis['RSI'] < 50:
        return 'down'
    else:
        return 'sideways'

def calculate_position_size(symbol, suggested_amount, stop_loss):
    """
    Рассчитывает оптимальный размер позиции на основе риска
    """
    risk_per_trade = 0.01  # 1% от баланса на одну сделку
    balance = exchange.fetch_balance()['total']['USDT']
    risk_amount = balance * risk_per_trade
    
    current_price = exchange.fetch_ticker(symbol)['last']
    price_difference = abs(current_price - stop_loss)
    
    position_size = risk_amount / price_difference
    return min(position_size, suggested_amount)

async def monitor_position(symbol, entry_price, amount, initial_stop_loss, initial_take_profit):
    """
    Отслеживает открытую позицию и применяет трейлинг стоп-лосс
    """
    current_stop_loss = initial_stop_loss
    current_take_profit = initial_take_profit
    
    while True:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            if current_price <= current_stop_loss:
                await exchange.create_market_sell_order(symbol, amount)
                logger.info(f"Сработал стоп-лосс для {symbol} по цене {current_price}")
                break
            elif current_price >= current_take_profit:
                await exchange.create_market_sell_order(symbol, amount)
                logger.info(f"Сработал тейк-профит для {symbol} по цене {current_price}")
                break
            
            # Трейлинг стоп-лосс
            if current_price - current_stop_loss > 2 * (current_take_profit - current_price):
                new_stop_loss = current_price - (current_take_profit - current_price)
                if new_stop_loss > current_stop_loss:
                    current_stop_loss = new_stop_loss
                    await exchange.edit_order(symbol, 'stop_loss', 'sell', amount, current_stop_loss)
                    logger.info(f"Обновлен трейлинг стоп-лосс для {symbol}: {current_stop_loss}")
            
            await asyncio.sleep(10)  # Проверяем каждые 10 секунд
        except Exception as e:
            logger.error(f"Ошибка при мониторинге позиции {symbol}: {e}")
            await asyncio.sleep(60)  # Ждем минуту перед повторной попыткой

def kelly_criterion(win_rate, win_loss_ratio):
    """
    Рассчитывает размер ставки по критерию Келли
    """
    q = 1 - win_rate
    return (win_rate * win_loss_ratio - q) / win_loss_ratio
def optimize_portfolio(returns, target_return):
    """
    Оптимизирует портфель с использованием метода среднего отклонения Марковица
    """
    def objective(weights):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return portfolio_volatility

    def constraint(weights):
        return np.sum(weights) - 1

    n_assets = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': constraint})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(objective, np.array([1/n_assets] * n_assets), method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

async def check_and_execute_opportunities():
    """
    Проверяет и выполняет арбитражные возможности
    """
    tickers = await get_all_tickers()
    opportunities = await find_arbitrage_opportunities(tickers)
    
    for opportunity in opportunities:
        balance = await exchange.fetch_balance()
        base_balance = balance['free'].get(opportunity['base'], 0)
        quote_balance = balance['free'].get(opportunity['quote'], 0)
        
        if opportunity['action'] == 'buy' and quote_balance > 0:
            amount = min(quote_balance * PORTFOLIO_PERCENTAGE, quote_balance)
            success = await execute_arbitrage(opportunity, amount)
            if success:
                logger.info(f"Выполнен арбитраж: покупка {opportunity['base']} за {opportunity['quote']}")
        elif opportunity['action'] == 'sell' and base_balance > 0:
            amount = min(base_balance * PORTFOLIO_PERCENTAGE, base_balance)
            success = await execute_arbitrage(opportunity, amount)
            if success:
                logger.info(f"Выполнен арбитраж: продажа {opportunity['base']} за {opportunity['quote']}")

async def get_all_tickers():
    """
    Получает текущие цены для всех символов
    """
    try:
        tickers = await exchange.fetch_tickers(SYMBOLS)
        return {symbol: float(ticker['last']) for symbol, ticker in tickers.items()}
    except Exception as e:
        logger.error(f"Ошибка при получении тикеров: {e}")
        return {}

async def find_arbitrage_opportunities(tickers, min_profit_percent=0.5):
    """
    Находит возможности для арбитража
    """
    opportunities = []
    for base in SYMBOLS:
        for quote in SYMBOLS:
            if base != quote:
                base_price = tickers.get(f"{base}")
                quote_price = tickers.get(f"{quote}")
                if base_price and quote_price:
                    direct_rate = base_price / quote_price
                    market_rate = await get_market_rate(base, quote)
                    if market_rate:
                        profit_percent = (market_rate / direct_rate - 1) * 100
                        if profit_percent > min_profit_percent:
                            opportunities.append({
                                'base': base,
                                'quote': quote,
                                'profit_percent': profit_percent,
                                'action': 'buy' if market_rate > direct_rate else 'sell'
                            })
    return opportunities

async def get_market_rate(base, quote):
    """
    Получает рыночный курс для пары валют
    """
    try:
        ticker = await exchange.fetch_ticker(f"{base}/{quote}")
        return float(ticker['last'])
    except Exception as e:
        logger.error(f"Ошибка при получении рыночного курса {base}/{quote}: {e}")
        return None

async def execute_arbitrage(opportunity, amount):
    """
    Выполняет арбитражную сделку
    """
    base = opportunity['base']
    quote = opportunity['quote']
    action = opportunity['action']
    
    try:
        if action == 'buy':
            order = await exchange.create_market_buy_order(f"{base}/{quote}", amount)
        else:
            order = await exchange.create_market_sell_order(f"{base}/{quote}", amount)
        
        logger.info(f"Выполнен арбитражный ордер: {order}")
        trade_history.append({
            'timestamp': datetime.now(),
            'type': 'arbitrage',
            'base': base,
            'quote': quote,
            'action': action,
            'amount': amount,
            'order': order
        })
        return True
    except Exception as e:
        logger.error(f"Ошибка при выполнении арбитражной сделки: {e}")
        return False

async def monitor_positions():
    """
    Отслеживает все открытые позиции и обновляет их стоп-лоссы и тейк-профиты
    """
    while True:
        try:
            positions = await exchange.fetch_positions()
            for position in positions:
                if position['amount'] > 0:
                    symbol = position['symbol']
                    entry_price = position['entryPrice']
                    amount = position['amount']
                    
                    # Обновляем динамические стоп-лоссы и тейк-профиты
                    df = await get_historical_data(symbol, '1h')
                    volatility = calculate_volatility(df['close'])
                    stop_loss, take_profit = calculate_dynamic_stops(entry_price, volatility)
                    
                    # Применяем обновленные значения
                    await apply_dynamic_stop_loss_take_profit(symbol, entry_price, amount, stop_loss, take_profit)
            
            await asyncio.sleep(300)  # Проверяем каждые 5 минут
        except Exception as e:
            logger.error(f"Ошибка при мониторинге позиций: {e}")
            await asyncio.sleep(600)  # Ждем 10 минут перед повторной попыткой

async def risk_management():
    """
    Управляет рисками портфеля
    """
    while True:
        try:
            total_balance = await get_total_balance_usdt()
            positions = await exchange.fetch_positions()
            
            total_risk = sum(position['amount'] * position['entryPrice'] for position in positions if position['amount'] > 0)
            risk_percentage = (total_risk / total_balance) * 100
            
            if risk_percentage > MAX_PORTFOLIO_RISK:
                logger.warning(f"Превышен максимальный риск портфеля: {risk_percentage:.2f}%")
                # Уменьшаем риски, закрывая часть позиций
                for position in positions:
                    if position['amount'] > 0:
                        symbol = position['symbol']
                        amount_to_close = position['amount'] * 0.1  # Закрываем 10% позиции
                        try:
                            await exchange.create_market_sell_order(symbol, amount_to_close)
                            logger.info(f"Частично закрыта позиция по {symbol} для снижения риска")
                        except Exception as e:
                            logger.error(f"Ошибка при закрытии части позиции {symbol}: {e}")
            
            await asyncio.sleep(3600)  # Проверяем каждый час
        except Exception as e:
            logger.error(f"Ошибка при управлении рисками: {e}")
            await asyncio.sleep(1800)  # Ждем 30 минут перед повторной попыткой

# Обработчики команд
@dispatcher.message(Command("СТАРТ"))
async def cmd_start_trading(message: types.Message):
    """
    Обработчик команды СТАРТ
    """
    global is_trading
    is_trading = True
    await message.answer("Торговля и арбитраж запущены.")

@dispatcher.message(Command("СТОП"))
async def cmd_stop_trading(message: types.Message):
    """
    Обработчик команды СТОП
    """
    global is_trading
    is_trading = False
    await message.answer("Торговля и арбитраж остановлены.")

@dispatcher.message(Command("БАЛАНС 💰"))
async def cmd_balance(message: types.Message):
    """
    Обработчик команды БАЛАНС
    """
    await message.answer("Выберите тип баланса:", reply_markup=balance_keyboard)

@dispatcher.callback_query(lambda c: c.data == 'spot_balance')
async def callback_spot_balance(callback_query: types.CallbackQuery):
    """
    Обработчик callback для спотового баланса
    """
    try:
        total_balance_usdt, crypto_balances = await get_spot_balance()
        
        response = f"Спотовый баланс: {total_balance_usdt:.2f} USDT\n\n"
        response += "Балансы по валютам:\n"
        
        for currency, info in crypto_balances.items():
            response += f"{currency}: {info['amount']:.8f} "
            if currency != 'USDT':
                response += f"(Цена: {info['price_usdt']:.4f} USDT, "
            response += f"Стоимость: {info['value_usdt']:.2f} USDT)\n"

        await callback_query.message.answer(response)
    except Exception as e:
        await callback_query.message.answer(f"Ошибка при получении спотового баланса: {str(e)}")

@dispatcher.callback_query(lambda c: c.data == 'p2p_balance')
async def callback_p2p_balance(callback_query: types.CallbackQuery):
    """
    Обработчик callback для P2P баланса
    """
    try:
        p2p_balance = await get_p2p_balance()
        
        response = f"P2P баланс: {p2p_balance:.2f} USDT\n"
        # Здесь можно добавить дополнительную информацию о P2P балансе,
        # если она доступна через API Binance

        await callback_query.message.answer(response)
    except Exception as e:
        await callback_query.message.answer(f"Ошибка при получении P2P баланса: {str(e)}")

@dispatcher.message(Command("АНАЛИЗ 📊"))
async def cmd_analysis(message: types.Message):
    """
    Обработчик команды АНАЛИЗ
    """
    analysis_results = []
    for symbol in SYMBOLS:
        result = latest_analysis_results.get((symbol, '5m'))  # Используем 5-минутный таймфрейм для последнего прогноза
        if result:
            analysis_results.append(result)

    if not analysis_results:
        await message.answer("В данный момент нет доступных результатов анализа. Пожалуйста, попробуйте позже.")
        return

    for result in analysis_results:
        try:
            response = f"💹 {result['symbol']} (5m):\n"
            response += f"Текущая цена: {result['current_price']:.2f}\n"
            response += f"Прогноз (ансамбль): {result['prediction']:.2f} ({result['change']:.2f}%)\n"
            response += f"Прогноз (ARIMA): {result['arima_forecast']:.2f}\n"
            response += f"Прогноз (Prophet): {result['prophet_forecast']:.2f}\n"
            response += f"Настроение рынка: {result['sentiment_score']:.2f}\n"
            response += f"Объем (MA): {result['volume_ma']:.2f}\n"
            response += f"Ликвидность: {result['liquidity']:.2f}\n"
            response += f"Аномалии: {result['anomalies']}\n"
            response += f"Волатильность: {result['volatility']:.4f}\n"
            response += f"Рекомендуемый стоп-лосс: {result['stop_loss']:.2f}\n"
            response += f"Рекомендуемый тейк-профит: {result['take_profit']:.2f}\n"

            await message.answer(response)

            # Отправляем графики
            try:
                await message.answer_photo(InputFile(f'model_performance_{result["symbol"]}_5m.png'))
            except Exception as e:
                logger.error(f"Ошибка при отправке графика: {e}")
                await message.answer("Не удалось отправить график анализа.")

        except Exception as e:
            logger.error(f"Ошибка при формировании ответа для {result['symbol']}: {e}")
            await message.answer(f"Произошла ошибка при анализе {result['symbol']}.")

    await message.answer("Анализ завершен.")

@dispatcher.message(Command("ГРАФИК 📈"))
async def cmd_chart(message: types.Message):
    """
    Обработчик команды ГРАФИК
    """
    for symbol in SYMBOLS:
        try:
            df = await get_historical_data(symbol, '1h')
            if df.empty:
                await message.answer(f"Нет данных для {symbol}")
                continue
            
            df = add_features(df)
            chart_input_file = await create_candlestick_chart(df, symbol, '1h')
            
            await message.answer_photo(photo=chart_input_file, caption=f"График {symbol} (1h)")
        except Exception as e:
            logger.error(f"Ошибка при создании графика для {symbol}: {e}")
            await message.answer(f"Не удалось создать график для {symbol}")
        
        await asyncio.sleep(1)  # Добавляем небольшую задержку между отправкой графиков

@dispatcher.message(Command("СТАТИСТИКА 📉"))
async def cmd_stats(message: types.Message):
    """
    Обработчик команды СТАТИСТИКА
    """
    total_profit = sum(trade['order']['cost'] for trade in trade_history if trade['type'] in ['trade', 'arbitrage'])
    total_trades = len(trade_history)
    
    response = f"Статистика за последние 24 часа:\n"
    response += f"Всего сделок: {total_trades}\n"
    response += f"Общая прибыль: {total_profit:.2f} USDT\n"
    
    if daily_balance_start is not None:
        current_balance = await get_total_balance_usdt()
        daily_change = current_balance - daily_balance_start
        daily_change_percent = (daily_change / daily_balance_start) * 100
        response += f"Изменение баланса: {daily_change:.2f} USDT ({daily_change_percent:.2f}%)\n"
    
    await message.answer(response)

@dispatcher.message(Command("НОВОСТИ 📰"))
async def cmd_news(message: types.Message):
    """
    Обработчик команды НОВОСТИ
    """
    for symbol in SYMBOLS:
        news = await get_crypto_news(symbol)
        if news:
            response = f"Последние новости о {symbol}:\n\n"
            for item in news:
                response += f"📌 {item['title']}\n"
                response += f"Подробнее: {item['link']}\n\n"
            await message.answer(response)
        else:
            await message.answer(f"Не удалось получить новости о {symbol}")

@dispatcher.message(Command("НАСТРОЙКИ ⚙️"))
async def cmd_settings(message: types.Message):
    """
    Обработчик команды НАСТРОЙКИ
    """
    await message.answer("Выберите процент торгового портфеля:", reply_markup=settings_keyboard)

@dispatcher.callback_query(lambda c: c.data.startswith("set_percentage_"))
async def callback_set_percentage(callback_query: types.CallbackQuery):
    """
    Обработчик callback для установки процента торгового портфеля
    """
    global PORTFOLIO_PERCENTAGE
    percentage = int(callback_query.data.split("_")[-1])
    PORTFOLIO_PERCENTAGE = percentage / 100
    await callback_query.answer(f"Установлен процент торгового портфеля: {percentage}%")
    await callback_query.message.answer(f"Процент торгового портфеля установлен на {percentage}%")

@dispatcher.callback_query(lambda c: c.data == "set_custom_percentage")
async def callback_set_custom_percentage(callback_query: types.CallbackQuery):
    """
    Обработчик callback для установки пользовательского процента торгового портфеля
    """
    await callback_query.answer()
    await callback_query.message.answer("Введите желаемый процент торгового портфеля (от 1 до 100):")

@dispatcher.message(Command("ПЕРЕВОДЫ 💸"))
async def cmd_transfers(message: types.Message):
    """
    Обработчик команды ПЕРЕВОДЫ
    """
    await message.answer("Выберите тип перевода:", reply_markup=transfer_keyboard)

@dispatcher.callback_query(lambda c: c.data == "transfer_crypto")
async def callback_transfer_crypto(callback_query: types.CallbackQuery):
    """
    Обработчик callback для перевода криптовалюты
    """
    await callback_query.answer()
    await callback_query.message.answer("Введите перевод в формате: ПЕРЕВОД [FROM] [TO] [AMOUNT]\nНапример: ПЕРЕВОД BTC ETH 0.1")

@dispatcher.callback_query(lambda c: c.data == "transfer_p2p")
async def callback_transfer_p2p(callback_query: types.CallbackQuery):
    """
    Обработчик callback для P2P перевода
    """
    await callback_query.answer()
    await callback_query.message.answer("Введите перевод на P2P в формате: ПЕРЕВОД P2P [CURRENCY] [AMOUNT]\nНапример: ПЕРЕВОД P2P BTC 0.1")

@dispatcher.message(Command("ПЕРЕВОД"))
async def cmd_transfer(message: types.Message):
    """
    Обработчик команды ПЕРЕВОД
    """
    try:
        parts = message.text.split()
        if len(parts) < 4:
            await message.answer("Неверный формат команды. Используйте: ПЕРЕВОД [FROM] [TO] [AMOUNT] или ПЕРЕВОД P2P [CURRENCY] [AMOUNT]")
            return

        if parts[1].upper() == "P2P":
            if len(parts) != 4:
                await message.answer("Неверный формат команды для P2P перевода. Используйте: ПЕРЕВОД P2P [CURRENCY] [AMOUNT]")
                return
            currency, amount = parts[2], float(parts[3])
            # Здесь должна быть реализация перевода на P2P платформу
            await message.answer(f"Перевод {amount} {currency} на P2P платформу выполнен успешно.")
        else:
            from_currency, to_currency, amount = parts[1], parts[2], float(parts[3])
            if amount <= 0:
                await message.answer("Сумма перевода должна быть положительным числом.")
                return
            success = await transfer_between_currencies(from_currency, to_currency, amount)
            if success:
                await message.answer(f"Перевод {amount} {from_currency} в {to_currency} выполнен успешно.")
            else:
                await message.answer("Не удалось выполнить перевод. Проверьте баланс и попробуйте еще раз.")
    except ValueError:
        await message.answer("Неверный формат суммы. Пожалуйста, введите корректное числовое значение.")
    except Exception as e:
        await message.answer(f"Ошибка при выполнении перевода: {str(e)}")

@dispatcher.message(Command("ПОРТФЕЛЬ 💼"))
async def cmd_portfolio(message: types.Message):
    """
    Обработчик команды ПОРТФЕЛЬ
    """
    try:
        total_balance_usdt, crypto_balances = await get_spot_balance()
        
        response = "Текущий портфель:\n\n"
        for currency, info in crypto_balances.items():
            percentage = (info['value_usdt'] / total_balance_usdt) * 100
            response += f"{currency}: {info['amount']:.8f} ({percentage:.2f}% портфеля)\n"
        
        response += f"\nОбщая стоимость портфеля: {total_balance_usdt:.2f} USDT"
        
        await message.answer(response)
    except Exception as e:
        await message.answer(f"Ошибка при получении информации о портфеле: {str(e)}")

@dispatcher.message(Command("РИСК-МЕНЕДЖМЕНТ 🛡️"))
async def cmd_risk_management(message: types.Message):
    """
    Обработчик команды РИСК-МЕНЕДЖМЕНТ
    """
    try:
        total_balance = await get_total_balance_usdt()
        positions = await exchange.fetch_positions()
        
        total_risk = sum(position['amount'] * position['entryPrice'] for position in positions if position['amount'] > 0)
        risk_percentage = (total_risk / total_balance) * 100
        
        response = "Риск-менеджмент:\n\n"
        response += f"Общий баланс: {total_balance:.2f} USDT\n"
        response += f"Текущий риск: {risk_percentage:.2f}%\n"
        response += f"Максимальный допустимый риск: {MAX_PORTFOLIO_RISK}%\n\n"
        
        if risk_percentage > MAX_PORTFOLIO_RISK:
            response += "⚠️ Внимание: текущий риск превышает максимально допустимый!\n"
            response += "Рекомендуется уменьшить размер позиций или закрыть некоторые из них."
        else:
            response += "✅ Текущий уровень риска в пределах допустимого."
        
        await message.answer(response)
    except Exception as e:
        await message.answer(f"Ошибка при получении информации о риск-менеджменте: {str(e)}")

@dispatcher.message(Command("ОПТИМИЗАЦИЯ 🔧"))
async def cmd_optimization(message: types.Message):
    """
    Обработчик команды ОПТИМИЗАЦИЯ
    """
    await message.answer("Начинаю процесс оптимизации стратегий и моделей...")
    
    try:
        # Оптимизация параметров моделей
        await optimize_models()
        
        # Оптимизация торговых стратегий
        await optimize_trading_strategies()
        
        # Оптимизация распределения портфеля
        await optimize_portfolio_allocation()
        
        await message.answer("Оптимизация завершена. Стратегии и модели обновлены.")
    except Exception as e:
        await message.answer(f"Ошибка при выполнении оптимизации: {str(e)}")

async def optimize_models():
    """
    Оптимизирует параметры моделей машинного обучения
    """
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            df = await get_historical_data(symbol, timeframe)
            X, y, _ = prepare_data_for_ml(df)
            if X is not None and y is not None:
                # Оптимизация Random Forest
                rf_model = optimize_rf(X, y)
                
                # Оптимизация XGBoost
                xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=num_cores)
                xgb_model.fit(X, y)
                
                # Обновление моделей в latest_analysis_results
                latest_analysis_results[(symbol, timeframe)]['rf'] = {'model': rf_model}
                latest_analysis_results[(symbol, timeframe)]['xgb'] = {'model': xgb_model}
    
    logger.info("Оптимизация моделей завершена")

async def optimize_trading_strategies():
    """
    Оптимизирует торговые стратегии
    """
    global PORTFOLIO_PERCENTAGE, MAX_PORTFOLIO_RISK
    
    # Оптимизация процента торгового портфеля
    best_performance = float('-inf')
    best_percentage = PORTFOLIO_PERCENTAGE
    
    for percentage in [0.05, 0.1, 0.15, 0.2, 0.25]:
        PORTFOLIO_PERCENTAGE = percentage
        performance = await backtest_trading_strategy()
        if performance > best_performance:
            best_performance = performance
            best_percentage = percentage
    
    PORTFOLIO_PERCENTAGE = best_percentage
    
    # Оптимизация максимального риска портфеля
    best_performance = float('-inf')
    best_risk = MAX_PORTFOLIO_RISK
    
    for risk in [3, 5, 7, 10]:
        MAX_PORTFOLIO_RISK = risk
        performance = await backtest_trading_strategy()
        if performance > best_performance:
            best_performance = performance
            best_risk = risk
    
    MAX_PORTFOLIO_RISK = best_risk
    
    logger.info(f"Оптимизированные параметры: PORTFOLIO_PERCENTAGE={PORTFOLIO_PERCENTAGE}, MAX_PORTFOLIO_RISK={MAX_PORTFOLIO_RISK}")

async def optimize_portfolio_allocation():
    """
    Оптимизирует распределение портфеля
    """
    returns = pd.DataFrame()
    for symbol in SYMBOLS:
        df = await get_historical_data(symbol, '1d')
        returns[symbol] = df['close'].pct_change()
    
    target_return = 0.1  # Целевая доходность
    weights = optimize_portfolio(returns, target_return)
    
    # Обновляем веса в глобальной переменной
    global portfolio_weights
    portfolio_weights = {symbol: weight for symbol, weight in zip(SYMBOLS, weights)}
    
    logger.info(f"Оптимизированные веса портфеля: {portfolio_weights}")

async def backtest_trading_strategy():
    """
    Проводит бэктестинг торговой стратегии
    """
    initial_balance = 10000  # Начальный баланс для бэктестинга
    balance = initial_balance
    
    for symbol in SYMBOLS:
        df = await get_historical_data(symbol, '1h')
        X, y, scaler = prepare_data_for_ml(df)
        if X is not None and y is not None:
            model = latest_analysis_results[(symbol, '1h')]['ensemble']['model']
            balance += backtest_strategy(df, model, initial_balance=balance)
    
    return (balance - initial_balance) / initial_balance  # Возвращаем доходность

async def main():
    """
    Главная функция бота
    """
    logger.info("Бот запускается...")
    await bot.send_message(CHAT_ID, "Бот запущен. Используйте кнопки для управления.", reply_markup=main_keyboard)
    logger.info("Сообщение о запуске отправлено")
    
    logger.info("Запуск задач непрерывного анализа и торговли...")
    tasks = [
        asyncio.create_task(continuous_analysis()),
        asyncio.create_task(trading_loop()),
        asyncio.create_task(update_daily_balance()),
        asyncio.create_task(monitor_positions()),
        asyncio.create_task(risk_management()),
        asyncio.create_task(periodic_model_update())
    ]
    
    try:
        logger.info("Начало поллинга...")
        await dispatcher.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при поллинге: {e}")
    finally:
        logger.info("Завершение работы бота...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await exchange.close()
        await bot.session.close()
        # Закрыть все открытые сессии
        for session in aiohttp.ClientSession._sessions:
            await session.close()

async def periodic_model_update():
    """
    Периодически обновляет модели, обучая их на новых данных и ошибках
    """
    while True:
        await asyncio.sleep(3600)  # Обновляем модели раз в час
        logger.info("Начало периодического обновления моделей")
        await learn_from_errors()
        logger.info("Периодическое обновление моделей завершено")

async def update_daily_balance():
    """
    Обновляет ежедневный баланс
    """
    global daily_balance_start, daily_pnl
    while True:
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            daily_balance_start = await get_total_balance_usdt()
            daily_pnl = 0
            logger.info(f"Обновлен ежедневный начальный баланс: {daily_balance_start:.2f} USDT")
        await asyncio.sleep(60)  # Проверяем каждую минуту

if __name__ == "__main__":
    # Настройка для оптимальной производительности на AMD EPYC 7402
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    
    # Ограничение использования памяти для TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Настройка asyncio для Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Добавьте эту строку для логирования неперехваченных исключений
    logging.getLogger('asyncio').setLevel(logging.ERROR)
    
    # Функция для обработки неперехваченных исключений
    def handle_exception(loop, context):
        msg = context.get("exception", context["message"])
        logging.error(f"Необработанное исключение: {msg}")
        logging.info("Перезапуск бота...")
        asyncio.create_task(main())

    # Настройка обработчика исключений
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    
    # Запуск основного цикла бота
    asyncio.run(main())

# Дополнительные функции для работы с Google Drive

async def upload_log_to_drive():
    """
    Загружает лог-файлы на Google Drive
    """
    try:
        await upload_to_drive('bot_log.txt', 'bot_log.txt')
        await upload_to_drive('crypto_ml_log.txt', 'crypto_ml_log.txt')
        logger.info("Лог-файлы успешно загружены на Google Drive")
    except Exception as e:
        logger.error(f"Ошибка при загрузке лог-файлов на Google Drive: {e}")

# Дополнительные функции для анализа рынка

async def analyze_market_sentiment():
    """
    Анализирует общее настроение рынка
    """
    sentiment_scores = []
    for symbol in SYMBOLS:
        sentiment = await analyze_news_sentiment(symbol)
        sentiment_scores.append(sentiment)
    
    average_sentiment = np.mean(sentiment_scores)
    logger.info(f"Среднее настроение рынка: {average_sentiment:.2f}")
    return average_sentiment

async def detect_market_trends():
    """
    Определяет текущие тренды на рынке
    """
    trends = {}
    for symbol in SYMBOLS:
        df = await get_historical_data(symbol, '1d')
        if len(df) > 20:
            sma20 = ta.trend.sma_indicator(df['close'], window=20)
            sma50 = ta.trend.sma_indicator(df['close'], window=50)
            current_price = df['close'].iloc[-1]
            
            if current_price > sma20.iloc[-1] > sma50.iloc[-1]:
                trends[symbol] = "Восходящий"
            elif current_price < sma20.iloc[-1] < sma50.iloc[-1]:
                trends[symbol] = "Нисходящий"
            else:
                trends[symbol] = "Боковой"
    
    logger.info(f"Обнаруженные тренды на рынке: {trends}")
    return trends

# Функции для улучшенного управления рисками

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Рассчитывает коэффициент Шарпа
    """
    excess_returns = returns - risk_free_rate / 252  # Предполагаем дневные доходности
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

async def optimize_position_sizing():
    """
    Оптимизирует размеры позиций на основе риска и доходности
    """
    portfolio_values = []
    for symbol in SYMBOLS:
        df = await get_historical_data(symbol, '1d')
        returns = df['close'].pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)
        volatility = returns.std()
        
        # Используем коэффициент Шарпа и волатильность для определения размера позиции
        position_size = (sharpe / volatility) / len(SYMBOLS)
        portfolio_values.append((symbol, position_size))
    
    # Нормализуем размеры позиций
    total_size = sum(pv[1] for pv in portfolio_values)
    normalized_positions = {pv[0]: pv[1] / total_size for pv in portfolio_values}
    
    logger.info(f"Оптимизированные размеры позиций: {normalized_positions}")
    return normalized_positions

# Функции для расширенного анализа

async def perform_technical_analysis(symbol, timeframe):
    """
    Выполняет расширенный технический анализ
    """
    df = await get_historical_data(symbol, timeframe)
    
    # Рассчитываем дополнительные индикаторы
    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['EMA200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['MACD_diff'] = ta.trend.macd_diff(df['close'])
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
    
    # Анализ тренда
    last_close = df['close'].iloc[-1]
    trend = "Восходящий" if last_close > df['EMA50'].iloc[-1] > df['EMA200'].iloc[-1] else "Нисходящий"
    
    # Сила тренда
    adx = df['ADX'].iloc[-1]
    trend_strength = "Сильный" if adx > 25 else "Слабый"
    
    # Сигналы
    macd_signal = "Покупка" if df['MACD_diff'].iloc[-1] > 0 else "Продажа"
    
    analysis = {
        "symbol": symbol,
        "timeframe": timeframe,
        "trend": trend,
        "trend_strength": trend_strength,
        "macd_signal": macd_signal,
        "last_close": last_close,
        "adx": adx
    }
    
    logger.info(f"Технический анализ для {symbol} на {timeframe}: {analysis}")
    return analysis

# Функция для периодического сохранения состояния бота
async def save_bot_state():
    """
    Сохраняет текущее состояние бота
    """
    while True:
        try:
            state = {
                "latest_analysis_results": latest_analysis_results,
                "model_performance": model_performance,
                "user_portfolios": user_portfolios,
                "is_trading": is_trading,
                "daily_balance_start": daily_balance_start,
                "daily_pnl": daily_pnl,
                "trade_history": trade_history
            }
            with open("bot_state.pkl", "wb") as f:
                pickle.dump(state, f)
            logger.info("Состояние бота успешно сохранено")
            await upload_to_drive("bot_state.pkl", "bot_state.pkl")
        except Exception as e:
            logger.error(f"Ошибка при сохранении состояния бота: {e}")
        await asyncio.sleep(3600)  # Сохраняем состояние каждый час

# Функция для загрузки состояния бота при запуске
def load_bot_state():
    """
    Загружает сохраненное состояние бота
    """
    try:
        with open("bot_state.pkl", "rb") as f:
            state = pickle.load(f)
        global latest_analysis_results, model_performance, user_portfolios, is_trading, daily_balance_start, daily_pnl, trade_history
        latest_analysis_results = state["latest_analysis_results"]
        model_performance = state["model_performance"]
        user_portfolios = state["user_portfolios"]
        is_trading = state["is_trading"]
        daily_balance_start = state["daily_balance_start"]
        daily_pnl = state["daily_pnl"]
        trade_history = state["trade_history"]
        logger.info("Состояние бота успешно загружено")
    except FileNotFoundError:
        logger.info("Файл состояния бота не найден. Начинаем с чистого состояния.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке состояния бота: {e}")

# Обновляем функцию main для включения новых задач
async def main():
    """
    Главная функция бота
    """
    load_bot_state()  # Загружаем состояние бота при запуске
    
    logger.info("Бот запускается...")
    await bot.send_message(CHAT_ID, "Бот запущен. Используйте кнопки для управления.", reply_markup=main_keyboard)
    logger.info("Сообщение о запуске отправлено")
    
    logger.info("Запуск задач непрерывного анализа и торговли...")
    tasks = [
        asyncio.create_task(continuous_analysis()),
        asyncio.create_task(trading_loop()),
        asyncio.create_task(update_daily_balance()),
        asyncio.create_task(monitor_positions()),
        asyncio.create_task(risk_management()),
        asyncio.create_task(periodic_model_update()),
        asyncio.create_task(save_bot_state()),  # Добавляем задачу сохранения состояния
        asyncio.create_task(upload_log_to_drive())  # Добавляем задачу загрузки логов
    ]
    
    try:
        logger.info("Начало поллинга...")
        await dispatcher.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при поллинге: {e}")
    finally:
        logger.info("Завершение работы бота...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await exchange.close()
        await bot.session.close()
        # Закрыть все открытые сессии
        for session in aiohttp.ClientSession._sessions:
            await session.close()

if __name__ == "__main__":
    asyncio.run(main())