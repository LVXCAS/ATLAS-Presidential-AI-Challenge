"""Statistical analysis utilities."""

import pandas as pd
import numpy as np


def rolling_correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Calculate rolling correlation between two series."""
    return x.rolling(window=window).corr(y)


def rolling_beta(returns: pd.Series, market_returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling beta."""
    covariance = returns.rolling(window=window).cov(market_returns)
    market_variance = market_returns.rolling(window=window).var()
    return covariance / market_variance


def information_ratio(
    returns: pd.Series, 
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """Calculate information ratio."""
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    return (excess_returns.mean() * periods_per_year) / tracking_error


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """Calculate Calmar ratio."""
    from .financial import calculate_max_drawdown
    
    annual_return = returns.mean() * periods_per_year
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return np.inf
    
    return annual_return / max_dd