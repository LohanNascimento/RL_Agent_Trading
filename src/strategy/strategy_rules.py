# strategy_rules.py
from abc import ABC, abstractmethod
import numpy as np


class BaseStrategy(ABC):
    @abstractmethod
    def check_entry(self, env, current_step) -> bool:
        pass
    
    @abstractmethod
    def check_exit(self, env, current_step) -> bool:
        pass

class ReversalStrategy(BaseStrategy):
    def __init__(self, trend_lookback=50, volatility_lookback=20):
        self.trend_lookback = trend_lookback
        self.volatility_lookback = volatility_lookback
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def _get_market_structure(self, df):
        # Análise de tendência baseada em médias móveis
        sma = df['sma_14']
        ema = df['ema_21']
        uptrend = ema.iloc[-1] > sma.iloc[-1]
        downtrend = ema.iloc[-1] < sma.iloc[-1]
        return {'uptrend': uptrend, 'downtrend': downtrend}
    
    def check_entry(self, env, current_step):
        df_window = env.df.iloc[current_step - env.window_size:current_step]
        structure = self._get_market_structure(df_window)
        adx = df_window['adx'].iloc[-1]
        
        # Requisito de força da tendência: ADX > 25
        strong_trend = adx > 25
        
        return (
            strong_trend and
            ((structure['uptrend'] and env.df.iloc[current_step]['close'] < df_window['low'].min()) or
            (structure['downtrend'] and env.df.iloc[current_step]['close'] > df_window['high'].max()))
        )
    
    def check_exit(self, env, current_step):
        # Lógica de saída baseada em liquidity
        return env.next_liquidity_target is not None

class TrendFollowingStrategy(BaseStrategy):
    """
    Estratégia de tendência baseada apenas em indicadores técnicos clássicos.
    Exemplo: entrada quando EMA cruza SMA para cima e RSI < 30;
    saída quando EMA cruza SMA para baixo ou RSI > 70.
    """
    def __init__(self, rsi_oversold=30, rsi_overbought=70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def check_entry(self, env, current_step):
        df = env.df.iloc[current_step - env.window_size:current_step]
        ema = df['ema_21'].iloc[-1]
        sma = df['sma_14'].iloc[-1]
        ema_prev = df['ema_21'].iloc[-2]
        sma_prev = df['sma_14'].iloc[-2]
        rsi = df['rsi_14'].iloc[-1]
        adx = df['adx'].iloc[-1]

        cruzamento_alta = ema_prev < sma_prev and ema > sma
        strong_trend = adx > 25  # Valida que a tendência é forte
        
        return cruzamento_alta and rsi < self.rsi_oversold and strong_trend

    def check_exit(self, env, current_step):
        df = env.df.iloc[current_step - env.window_size:current_step]
        # Condição: EMA cruza SMA para baixo ou RSI > overbought
        ema = df['ema_21'].iloc[-1]
        sma = df['sma_14'].iloc[-1]
        ema_prev = df['ema_21'].iloc[-2]
        sma_prev = df['sma_14'].iloc[-2]
        rsi = df['rsi_14'].iloc[-1]
        cruzamento_baixa = ema_prev > sma_prev and ema < sma
        return cruzamento_baixa or rsi > self.rsi_overbought

class ScalpMomentumStrategy(BaseStrategy):
    def __init__(self, ema_fast=5, ema_slow=10, volume_threshold=1.5):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.volume_threshold = volume_threshold  # Volume > 150% da média

    def check_entry(self, env, current_step):
        df = env.df.iloc[current_step - env.window_size:current_step]
        # Condições de entrada
        ema_cross = df['ema_5'].iloc[-1] > df['ema_10'].iloc[-1]
        high_volume = df['volume'].iloc[-1] > (df['volume_ma_10'].iloc[-1] * self.volume_threshold)
        rsi_cond = df['rsi_14'].iloc[-1] < 40  # Evita sobrecompra
        return ema_cross and high_volume and rsi_cond

    def check_exit(self, env, current_step):
        df = env.df.iloc[current_step - env.window_size:current_step]
        # Saída quando EMA rápida inverte ou RSI > 60
        ema_cross_reverse = df['ema_5'].iloc[-1] < df['ema_10'].iloc[-1]
        rsi_exit = df['rsi_14'].iloc[-1] > 60
        return ema_cross_reverse or rsi_exit

class ScalpVWAPStrategy(BaseStrategy):
    def __init__(self, atr_multiplier=1.5):
        self.atr_multiplier = atr_multiplier

    def check_entry(self, env, current_step):
        df = env.df.iloc[current_step - env.window_size:current_step]
        # Entra se o preço está abaixo do VWAP (pullback) e ATR indica volatilidade
        vwap_val = vwap(df)
        atr_val = df['atr_14'].iloc[-1]
        price = df['close'].iloc[-1]
        return (price < vwap_val) and (atr_val > 0.005 * price)  # Ajuste conforme o ativo

    def check_exit(self, env, current_step):
        # Saída após atingir um take profit (1x ATR) ou stop loss (1.5x ATR)
        entry_price = env.current_position_price
        current_price = env.df.iloc[current_step]['close']
        atr_val = env.df.iloc[current_step - env.window_size:current_step]['atr_14'].iloc[-1]
        take_profit = entry_price + self.atr_multiplier * atr_val
        stop_loss = entry_price - self.atr_multiplier * atr_val
        return (current_price >= take_profit) or (current_price <= stop_loss)