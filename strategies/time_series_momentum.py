"""
Time Series Momentum (TSMOM) Indicator
Academic implementation based on Moskowitz, Ooi, and Pedersen (2012)

TSMOM measures the sign and magnitude of an asset's own past returns
over multiple time horizons to predict future returns.

Key Features:
- Multiple lookback periods (1m, 3m, 6m, 12m)
- Volatility adjustment for risk-weighted signals
- Sign-based directional signals
- Rolling window calculations for stability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from strategies.technical_indicators import TechnicalIndicator, IndicatorResult
import warnings


@dataclass
class TSMOMResult(IndicatorResult):
    """Extended result with TSMOM-specific metadata"""
    signal_strength: float  # -1 to 1
    lookback_signals: Dict[str, float]  # Individual horizon signals
    volatility_adjusted: bool

    def get_direction(self) -> str:
        """Get human-readable direction"""
        if self.signal_strength > 0.3:
            return "STRONG_BULLISH"
        elif self.signal_strength > 0:
            return "BULLISH"
        elif self.signal_strength < -0.3:
            return "STRONG_BEARISH"
        elif self.signal_strength < 0:
            return "BEARISH"
        else:
            return "NEUTRAL"


class TimeSeriesMomentum(TechnicalIndicator):
    """
    Time Series Momentum (TSMOM) Indicator

    Calculates momentum signals based on an asset's own historical returns
    across multiple time horizons, with optional volatility adjustment.

    Reference:
    Moskowitz, Ooi, and Pedersen (2012)
    "Time Series Momentum" Journal of Financial Economics
    """

    def __init__(self):
        super().__init__("TimeSeriesMomentum")

        # Standard lookback periods (in trading days)
        self.LOOKBACK_PERIODS = {
            '1m': 21,    # 1 month (~21 trading days)
            '3m': 63,    # 3 months (~63 trading days)
            '6m': 126,   # 6 months (~126 trading days)
            '12m': 252   # 12 months (~252 trading days)
        }

    def calculate(self, data: Union[np.ndarray, pd.Series],
                 lookback_periods: Optional[Dict[str, int]] = None,
                 volatility_adjust: bool = True,
                 vol_window: int = 20,
                 weights: Optional[Dict[str, float]] = None) -> TSMOMResult:
        """
        Calculate Time Series Momentum signals

        Args:
            data: Price data (typically close prices)
            lookback_periods: Custom lookback periods dict (default: 1m, 3m, 6m, 12m)
            volatility_adjust: Whether to adjust signals by realized volatility
            vol_window: Window for volatility calculation (default: 20 days)
            weights: Custom weights for each horizon (default: equal weight)

        Returns:
            TSMOMResult with signals, metadata, and individual horizon signals
        """
        # Use default periods if not provided
        if lookback_periods is None:
            lookback_periods = self.LOOKBACK_PERIODS

        # Validate data
        max_period = max(lookback_periods.values())
        data = self.validate_data(data, min_periods=max_period + vol_window)

        # Calculate returns
        returns = pd.Series(data).pct_change().fillna(0).values

        # Calculate signals for each lookback horizon
        horizon_signals = {}
        tsmom_values = np.zeros(len(data))

        for horizon_name, lookback in lookback_periods.items():
            # Calculate cumulative returns over lookback period
            # Using simple sum of log returns for better numerical stability
            log_returns = np.log1p(returns)  # log(1 + r)
            cum_returns = pd.Series(log_returns).rolling(window=lookback).sum().values

            # Convert back to simple returns
            simple_returns = np.expm1(cum_returns)  # exp(x) - 1

            # Create directional signal (sign of past returns)
            signal = np.sign(simple_returns)

            # Volatility adjustment (if enabled)
            if volatility_adjust:
                # Calculate realized volatility
                realized_vol = pd.Series(returns).rolling(window=vol_window).std().values
                # Avoid division by zero
                realized_vol = np.where(realized_vol == 0, 1e-6, realized_vol)

                # Scale signal by inverse volatility (vol targeting)
                signal = signal / (realized_vol * np.sqrt(252))  # Annualized vol

            horizon_signals[horizon_name] = signal[-1] if len(signal) > 0 else 0
            tsmom_values += signal

        # Apply weights to combine horizons
        if weights is None:
            # Equal weight by default
            weights = {k: 1.0 / len(lookback_periods) for k in lookback_periods.keys()}

        # Weighted combination
        weighted_signal = np.zeros(len(data))
        for horizon_name, weight in weights.items():
            if horizon_name in horizon_signals:
                # Get signal for this horizon
                log_returns = np.log1p(returns)
                lookback = lookback_periods[horizon_name]
                cum_returns = pd.Series(log_returns).rolling(window=lookback).sum().values
                simple_returns = np.expm1(cum_returns)
                signal = np.sign(simple_returns)

                if volatility_adjust:
                    realized_vol = pd.Series(returns).rolling(window=vol_window).std().values
                    realized_vol = np.where(realized_vol == 0, 1e-6, realized_vol)
                    signal = signal / (realized_vol * np.sqrt(252))

                weighted_signal += weight * signal

        # Normalize to [-1, 1] range
        signal_strength = np.clip(weighted_signal / max(abs(weighted_signal[-1]), 1e-6), -1, 1)

        return TSMOMResult(
            values=weighted_signal,
            parameters={
                'lookback_periods': lookback_periods,
                'volatility_adjust': volatility_adjust,
                'vol_window': vol_window,
                'weights': weights
            },
            metadata={
                'min_periods': max_period + vol_window,
                'horizons': list(lookback_periods.keys()),
                'calculation_method': 'volatility_adjusted' if volatility_adjust else 'raw_returns'
            },
            name=self.name,
            signal_strength=float(signal_strength[-1]) if len(signal_strength) > 0 else 0.0,
            lookback_signals=horizon_signals,
            volatility_adjusted=volatility_adjust
        )

    def calculate_multi_asset_tsmom(self, price_data: Dict[str, pd.Series],
                                   lookback_periods: Optional[Dict[str, int]] = None,
                                   volatility_adjust: bool = True) -> Dict[str, TSMOMResult]:
        """
        Calculate TSMOM for multiple assets simultaneously

        Args:
            price_data: Dict of symbol -> price series
            lookback_periods: Custom lookback periods
            volatility_adjust: Whether to adjust by volatility

        Returns:
            Dict of symbol -> TSMOMResult
        """
        results = {}

        for symbol, prices in price_data.items():
            try:
                result = self.calculate(
                    prices,
                    lookback_periods=lookback_periods,
                    volatility_adjust=volatility_adjust
                )
                results[symbol] = result
            except Exception as e:
                warnings.warn(f"Failed to calculate TSMOM for {symbol}: {e}")
                results[symbol] = None

        return results

    def get_trading_signal(self, data: Union[np.ndarray, pd.Series],
                          threshold: float = 0.2) -> Tuple[str, float]:
        """
        Get a simple trading signal based on TSMOM

        Args:
            data: Price data
            threshold: Signal strength threshold for trading (default: 0.2)

        Returns:
            Tuple of (signal, confidence)
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: 0-1 confidence score
        """
        result = self.calculate(data)
        strength = result.signal_strength

        if abs(strength) < threshold:
            return 'HOLD', abs(strength) / threshold
        elif strength > 0:
            return 'BUY', min(abs(strength), 1.0)
        else:
            return 'SELL', min(abs(strength), 1.0)


# Convenience function for direct access
def calculate_tsmom(data: Union[np.ndarray, pd.Series],
                   lookback_periods: Optional[Dict[str, int]] = None,
                   volatility_adjust: bool = True) -> TSMOMResult:
    """Calculate TSMOM directly"""
    tsmom = TimeSeriesMomentum()
    return tsmom.calculate(data, lookback_periods=lookback_periods,
                          volatility_adjust=volatility_adjust)


def get_tsmom_signal(data: Union[np.ndarray, pd.Series],
                    threshold: float = 0.2) -> Tuple[str, float]:
    """Get trading signal from TSMOM"""
    tsmom = TimeSeriesMomentum()
    return tsmom.get_trading_signal(data, threshold=threshold)


# Example usage
if __name__ == "__main__":
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))

    # Calculate TSMOM
    tsmom = TimeSeriesMomentum()
    result = tsmom.calculate(prices, volatility_adjust=True)

    print("=" * 70)
    print("TIME SERIES MOMENTUM (TSMOM) EXAMPLE")
    print("=" * 70)
    print(f"\nCurrent Signal Strength: {result.signal_strength:.3f}")
    print(f"Direction: {result.get_direction()}")
    print(f"\nIndividual Horizon Signals:")
    for horizon, signal in result.lookback_signals.items():
        print(f"  {horizon:4s}: {signal:+.3f}")

    # Get trading signal
    signal, confidence = tsmom.get_trading_signal(prices, threshold=0.2)
    print(f"\nTrading Signal: {signal}")
    print(f"Confidence: {confidence:.1%}")
    print("=" * 70)
