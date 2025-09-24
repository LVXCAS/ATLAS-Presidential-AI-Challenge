"""Technical analysis utilities."""

import pandas as pd
import numpy as np
from typing import Tuple


def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Calculate simple moving average."""
    return prices.rolling(window=window).mean()


def exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Calculate exponential moving average."""
    return prices.ewm(span=window).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(
    prices: pd.Series, 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = exponential_moving_average(prices, fast)
    ema_slow = exponential_moving_average(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(
    prices: pd.Series, 
    window: int = 20, 
    num_std: float = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = simple_moving_average(prices, window)
    std = prices.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, sma, lower_band