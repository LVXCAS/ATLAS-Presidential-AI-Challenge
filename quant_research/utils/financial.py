"""Financial calculation utilities."""

import pandas as pd
import numpy as np
from typing import Union, Optional


def calculate_returns(
    prices: pd.Series,
    method: str = "simple",
    periods: int = 1
) -> pd.Series:
    """Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
        periods: Number of periods for return calculation
        
    Returns:
        Series of returns
    """
    if method == "log":
        return np.log(prices / prices.shift(periods))
    else:
        return prices.pct_change(periods)


def calculate_volatility(
    returns: pd.Series,
    annualize: bool = True,
    periods_per_year: int = 252
) -> float:
    """Calculate volatility from returns.
    
    Args:
        returns: Returns series
        annualize: Whether to annualize volatility
        periods_per_year: Trading periods per year
        
    Returns:
        Volatility
    """
    vol = returns.std()
    
    if annualize:
        vol *= np.sqrt(periods_per_year)
    
    return vol


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)


def calculate_max_drawdown(
    prices: pd.Series
) -> float:
    """Calculate maximum drawdown.
    
    Args:
        prices: Price series or cumulative returns
        
    Returns:
        Maximum drawdown as positive number
    """
    cumulative = (1 + prices).cumprod() if prices.min() < 0 else prices
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())