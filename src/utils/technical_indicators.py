import numpy as np
import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona indicadores técnicos clássicos ao DataFrame OHLCV.    Inclui: SMA, EMA, MACD, ATR, Bollinger Bands, RSI, Stochastic, ROC, ADX.
    """
    # Tendência
    df['sma_14'] = df['close'].rolling(window=14).mean()
    df['ema_21'] = df['close'].ewm(span=14, adjust=False).mean()
    df = adx(df, 14)
    #df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # Volatilidade
    df['std_14'] = df['close'].rolling(window=14).std()
    df['atr_14'] = atr(df, 14)
    # Bollinger Bands
    df['bb_upper'] = df['sma_14'] + 2 * df['std_14']
    df['bb_lower'] = df['sma_14'] - 2 * df['std_14']
    # Momento
    df['rsi_14'] = rsi(df['close'], 14)
    df['stoch_k'], df['stoch_d'] = stochastic(df, 14, 3)
    df['roc_14'] = roc(df['close'], 14)
    # Remove NaN iniciais
    df = df.dropna().reset_index(drop=True)
    return df
    #Volume médio (VWAP)
    #df['volume_ma_10'] = df['volume'].rolling(10).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def roc(series, period=14):
    return series.pct_change(periods=period) * 100

def adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    atr = tr1.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.abs().rolling(window=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100

    adx = dx.rolling(window=period).mean()
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    return df

def vwap(df):
    cumulative_volume = df['volume'].cumsum()
    cumulative_pv = (df['close'] * df['volume']).cumsum()
    return cumulative_pv / cumulative_volume

def stoch_rsi(series, period=14, k=3, d=3):
    rsi_val = rsi(series, period)
    stoch_k = 100 * (rsi_val - rsi_val.rolling(period).min()) / (rsi_val.rolling(period).max() - rsi_val.rolling(period).min())
    stoch_d = stoch_k.rolling(k).mean()
    return stoch_k, stoch_d