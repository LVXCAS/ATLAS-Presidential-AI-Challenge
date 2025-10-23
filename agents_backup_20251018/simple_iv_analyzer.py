"""
Simple IV Rank and IV Percentile Analyzer
Provides essential IV metrics for options trading decisions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleIVAnalyzer:
    """Simple IV Rank and IV Percentile calculator"""

    def __init__(self, lookback_days: int = 252):
        """
        Initialize IV analyzer

        Args:
            lookback_days: Number of days to look back for IV rank calculation (default 252 = 1 year)
        """
        self.lookback_days = lookback_days
        self.cache = {}  # Cache IV data

    def get_iv_metrics(self, symbol: str, current_iv: float) -> Dict[str, float]:
        """
        Calculate IV Rank and IV Percentile

        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility (as decimal, e.g., 0.35 for 35%)

        Returns:
            Dict with iv_rank, iv_percentile, historical_vol metrics
        """
        try:
            # Get historical volatility data
            historical_vol = self._get_historical_volatility(symbol)

            if historical_vol is None or len(historical_vol) < 50:
                logger.warning(f"Insufficient historical data for {symbol}")
                return {
                    'iv_rank': 50.0,  # Default to neutral
                    'iv_percentile': 50.0,
                    'current_iv': current_iv * 100,
                    'mean_iv': current_iv * 100,
                    'min_iv': current_iv * 100,
                    'max_iv': current_iv * 100,
                    'signal': 'NEUTRAL'
                }

            # Calculate IV Rank
            # IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
            min_vol = historical_vol.min()
            max_vol = historical_vol.max()

            if max_vol == min_vol:
                iv_rank = 50.0
            else:
                iv_rank = ((current_iv - min_vol) / (max_vol - min_vol)) * 100
                iv_rank = max(0, min(100, iv_rank))  # Clamp to 0-100

            # Calculate IV Percentile
            # Percentage of days in last year where IV was lower than current IV
            iv_percentile = (historical_vol < current_iv).sum() / len(historical_vol) * 100

            # Mean IV
            mean_iv = historical_vol.mean()

            # Determine signal
            if iv_rank < 25:
                signal = 'LOW'  # Bad for buying options, good for selling
            elif iv_rank < 50:
                signal = 'BELOW_AVERAGE'
            elif iv_rank < 75:
                signal = 'ABOVE_AVERAGE'  # Good for buying options
            else:
                signal = 'HIGH'  # Very good for buying options

            return {
                'iv_rank': round(iv_rank, 1),
                'iv_percentile': round(iv_percentile, 1),
                'current_iv': round(current_iv * 100, 1),
                'mean_iv': round(mean_iv * 100, 1),
                'min_iv': round(min_vol * 100, 1),
                'max_iv': round(max_vol * 100, 1),
                'signal': signal
            }

        except Exception as e:
            logger.error(f"Error calculating IV metrics for {symbol}: {e}")
            return {
                'iv_rank': 50.0,
                'iv_percentile': 50.0,
                'current_iv': current_iv * 100,
                'mean_iv': current_iv * 100,
                'min_iv': current_iv * 100,
                'max_iv': current_iv * 100,
                'signal': 'NEUTRAL'
            }

    def _get_historical_volatility(self, symbol: str) -> Optional[pd.Series]:
        """
        Get historical realized volatility for the symbol
        Uses cached data if available and recent
        """
        # Check cache
        cache_key = f"{symbol}_{datetime.now().date()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Download historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 30)  # Extra buffer

            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date.strftime('%Y-%m-%d'),
                                 end=end_date.strftime('%Y-%m-%d'))

            if hist.empty or len(hist) < 50:
                return None

            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()

            # Calculate 30-day rolling volatility (annualized)
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()

            # Cache the result
            self.cache[cache_key] = rolling_vol

            # Clean old cache entries (keep only today's)
            today_key_prefix = str(datetime.now().date())
            self.cache = {k: v for k, v in self.cache.items() if today_key_prefix in k}

            return rolling_vol

        except Exception as e:
            logger.error(f"Error fetching historical volatility for {symbol}: {e}")
            return None

    def should_buy_options(self, symbol: str, current_iv: float) -> Dict[str, any]:
        """
        Determine if IV environment is favorable for buying options

        Returns:
            Dict with recommendation, reasoning, and metrics
        """
        metrics = self.get_iv_metrics(symbol, current_iv)

        iv_rank = metrics['iv_rank']

        # Decision logic
        if iv_rank < 25:
            recommendation = 'AVOID'
            reasoning = f"IV Rank {iv_rank:.0f}% is too low - expensive to buy options"
            confidence_penalty = -0.20  # Reduce confidence by 20%
        elif iv_rank < 40:
            recommendation = 'CAUTION'
            reasoning = f"IV Rank {iv_rank:.0f}% is below average - not ideal for buying"
            confidence_penalty = -0.10  # Reduce confidence by 10%
        elif iv_rank < 60:
            recommendation = 'NEUTRAL'
            reasoning = f"IV Rank {iv_rank:.0f}% is average - acceptable conditions"
            confidence_penalty = 0.0
        elif iv_rank < 75:
            recommendation = 'FAVORABLE'
            reasoning = f"IV Rank {iv_rank:.0f}% is above average - good for buying options"
            confidence_penalty = 0.10  # Boost confidence by 10%
        else:
            recommendation = 'EXCELLENT'
            reasoning = f"IV Rank {iv_rank:.0f}% is high - excellent for buying options"
            confidence_penalty = 0.15  # Boost confidence by 15%

        return {
            'recommendation': recommendation,
            'reasoning': reasoning,
            'confidence_adjustment': confidence_penalty,
            'metrics': metrics
        }


# Singleton instance
_iv_analyzer = None

def get_iv_analyzer() -> SimpleIVAnalyzer:
    """Get singleton IV analyzer instance"""
    global _iv_analyzer
    if _iv_analyzer is None:
        _iv_analyzer = SimpleIVAnalyzer()
    return _iv_analyzer
