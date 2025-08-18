"""
Technical Indicators Library for LangGraph Trading System

This module provides vectorized implementations of common technical indicators
including EMA, RSI, MACD, Bollinger Bands, and Z-score calculations.

All indicators are optimized for performance using NumPy vectorization
and include comprehensive parameter validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


@dataclass
class IndicatorResult:
    """Standard result structure for all technical indicators"""
    values: np.ndarray
    parameters: Dict
    metadata: Dict
    name: str
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary format"""
        return {
            'values': self.values.tolist() if isinstance(self.values, np.ndarray) else self.values,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'name': self.name
        }


class TechnicalIndicator(ABC):
    """Abstract base class for all technical indicators"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, data: Union[np.ndarray, pd.Series], **kwargs) -> IndicatorResult:
        """Calculate the indicator values"""
        pass
    
    def validate_data(self, data: Union[np.ndarray, pd.Series], min_periods: int = 1) -> np.ndarray:
        """Validate and convert input data to numpy array"""
        if isinstance(data, pd.Series):
            data = data.values
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if len(data) < min_periods:
            raise ValueError(f"Insufficient data: need at least {min_periods} periods, got {len(data)}")
        
        if np.any(np.isnan(data)):
            warnings.warn("Data contains NaN values, results may be affected")
        
        return data


class EMA(TechnicalIndicator):
    """Exponential Moving Average (EMA) indicator"""
    
    def __init__(self):
        super().__init__("EMA")
    
    def calculate(self, data: Union[np.ndarray, pd.Series], period: int = 20, 
                 alpha: Optional[float] = None) -> IndicatorResult:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price data (typically close prices)
            period: Number of periods for EMA calculation
            alpha: Smoothing factor (if None, calculated as 2/(period+1))
        
        Returns:
            IndicatorResult with EMA values
        """
        data = self.validate_data(data, min_periods=period)
        
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        if not 0 < alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        
        # Vectorized EMA calculation
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]  # Initialize with first value
        
        # Use pandas ewm for efficient calculation
        if isinstance(data, np.ndarray):
            data_series = pd.Series(data)
        else:
            data_series = data
            
        ema_values = data_series.ewm(alpha=alpha, adjust=False).mean().values
        
        return IndicatorResult(
            values=ema_values,
            parameters={'period': period, 'alpha': alpha},
            metadata={'min_periods': period, 'calculation_method': 'exponential_weighted'},
            name=self.name
        )


class RSI(TechnicalIndicator):
    """Relative Strength Index (RSI) indicator"""
    
    def __init__(self):
        super().__init__("RSI")
    
    def calculate(self, data: Union[np.ndarray, pd.Series], period: int = 14) -> IndicatorResult:
        """
        Calculate Relative Strength Index
        
        Args:
            data: Price data (typically close prices)
            period: Number of periods for RSI calculation
        
        Returns:
            IndicatorResult with RSI values (0-100 scale)
        """
        data = self.validate_data(data, min_periods=period + 1)
        
        # Calculate price changes
        delta = np.diff(data)
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate average gains and losses using EMA
        avg_gains = pd.Series(gains).ewm(span=period, adjust=False).mean().values
        avg_losses = pd.Series(losses).ewm(span=period, adjust=False).mean().values
        
        # Calculate RSI
        rs = np.divide(avg_gains, avg_losses, out=np.full_like(avg_gains, np.inf), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Prepend NaN for the first value (no change calculated)
        rsi = np.concatenate([np.array([np.nan]), rsi])
        
        return IndicatorResult(
            values=rsi,
            parameters={'period': period},
            metadata={'min_periods': period + 1, 'scale': '0-100'},
            name=self.name
        )


class MACD(TechnicalIndicator):
    """Moving Average Convergence Divergence (MACD) indicator"""
    
    def __init__(self):
        super().__init__("MACD")
    
    def calculate(self, data: Union[np.ndarray, pd.Series], 
                 fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9) -> IndicatorResult:
        """
        Calculate MACD indicator
        
        Args:
            data: Price data (typically close prices)
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        
        Returns:
            IndicatorResult with MACD line, signal line, and histogram
        """
        data = self.validate_data(data, min_periods=slow_period)
        
        if fast_period >= slow_period:
            raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
        
        # Calculate fast and slow EMAs
        ema_fast = EMA().calculate(data, period=fast_period).values
        ema_slow = EMA().calculate(data, period=slow_period).values
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD line)
        signal_line = EMA().calculate(macd_line, period=signal_period).values
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Combine results
        macd_result = np.column_stack([macd_line, signal_line, histogram])
        
        return IndicatorResult(
            values=macd_result,
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            },
            metadata={
                'min_periods': slow_period,
                'columns': ['macd_line', 'signal_line', 'histogram']
            },
            name=self.name
        )


class BollingerBands(TechnicalIndicator):
    """Bollinger Bands indicator"""
    
    def __init__(self):
        super().__init__("BollingerBands")
    
    def calculate(self, data: Union[np.ndarray, pd.Series], 
                 period: int = 20, std_dev: float = 2.0) -> IndicatorResult:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price data (typically close prices)
            period: Number of periods for moving average and standard deviation
            std_dev: Number of standard deviations for bands
        
        Returns:
            IndicatorResult with upper band, middle band (SMA), and lower band
        """
        data = self.validate_data(data, min_periods=period)
        
        if std_dev <= 0:
            raise ValueError(f"Standard deviation multiplier must be positive, got {std_dev}")
        
        # Calculate Simple Moving Average (middle band)
        sma = pd.Series(data).rolling(window=period).mean().values
        
        # Calculate rolling standard deviation
        rolling_std = pd.Series(data).rolling(window=period).std().values
        
        # Calculate upper and lower bands
        upper_band = sma + (std_dev * rolling_std)
        lower_band = sma - (std_dev * rolling_std)
        
        # Combine results
        bb_result = np.column_stack([upper_band, sma, lower_band])
        
        return IndicatorResult(
            values=bb_result,
            parameters={'period': period, 'std_dev': std_dev},
            metadata={
                'min_periods': period,
                'columns': ['upper_band', 'middle_band', 'lower_band']
            },
            name=self.name
        )


class ZScore(TechnicalIndicator):
    """Z-Score indicator for mean reversion analysis"""
    
    def __init__(self):
        super().__init__("ZScore")
    
    def calculate(self, data: Union[np.ndarray, pd.Series], 
                 period: int = 20) -> IndicatorResult:
        """
        Calculate Z-Score
        
        Args:
            data: Price data or any time series
            period: Number of periods for rolling mean and standard deviation
        
        Returns:
            IndicatorResult with Z-Score values
        """
        data = self.validate_data(data, min_periods=period)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = pd.Series(data).rolling(window=period).mean().values
        rolling_std = pd.Series(data).rolling(window=period).std().values
        
        # Calculate Z-Score
        z_score = np.divide(
            data - rolling_mean, 
            rolling_std, 
            out=np.zeros_like(data, dtype=float), 
            where=rolling_std!=0
        )
        
        return IndicatorResult(
            values=z_score,
            parameters={'period': period},
            metadata={
                'min_periods': period,
                'interpretation': 'Values > 2 or < -2 indicate potential mean reversion'
            },
            name=self.name
        )


class IndicatorLibrary:
    """Main library class for accessing all technical indicators"""
    
    def __init__(self):
        self.indicators = {
            'ema': EMA(),
            'rsi': RSI(),
            'macd': MACD(),
            'bollinger_bands': BollingerBands(),
            'z_score': ZScore()
        }
    
    def calculate_indicator(self, indicator_name: str, data: Union[np.ndarray, pd.Series], 
                          **kwargs) -> IndicatorResult:
        """
        Calculate any indicator by name
        
        Args:
            indicator_name: Name of the indicator ('ema', 'rsi', 'macd', 'bollinger_bands', 'z_score')
            data: Price data
            **kwargs: Indicator-specific parameters
        
        Returns:
            IndicatorResult with calculated values
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}. Available: {list(self.indicators.keys())}")
        
        return self.indicators[indicator_name].calculate(data, **kwargs)
    
    def calculate_multiple(self, data: Union[np.ndarray, pd.Series], 
                          indicators_config: Dict[str, Dict]) -> Dict[str, IndicatorResult]:
        """
        Calculate multiple indicators at once
        
        Args:
            data: Price data
            indicators_config: Dict with indicator names as keys and parameters as values
        
        Returns:
            Dict of IndicatorResults
        """
        results = {}
        
        for indicator_name, params in indicators_config.items():
            try:
                results[indicator_name] = self.calculate_indicator(indicator_name, data, **params)
            except Exception as e:
                warnings.warn(f"Failed to calculate {indicator_name}: {str(e)}")
                results[indicator_name] = None
        
        return results
    
    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators"""
        return list(self.indicators.keys())


# Convenience functions for direct access
def calculate_ema(data: Union[np.ndarray, pd.Series], period: int = 20, 
                 alpha: Optional[float] = None) -> IndicatorResult:
    """Calculate EMA directly"""
    return EMA().calculate(data, period=period, alpha=alpha)


def calculate_rsi(data: Union[np.ndarray, pd.Series], period: int = 14) -> IndicatorResult:
    """Calculate RSI directly"""
    return RSI().calculate(data, period=period)


def calculate_macd(data: Union[np.ndarray, pd.Series], fast_period: int = 12, 
                  slow_period: int = 26, signal_period: int = 9) -> IndicatorResult:
    """Calculate MACD directly"""
    return MACD().calculate(data, fast_period=fast_period, slow_period=slow_period, 
                           signal_period=signal_period)


def calculate_bollinger_bands(data: Union[np.ndarray, pd.Series], period: int = 20, 
                             std_dev: float = 2.0) -> IndicatorResult:
    """Calculate Bollinger Bands directly"""
    return BollingerBands().calculate(data, period=period, std_dev=std_dev)


def calculate_z_score(data: Union[np.ndarray, pd.Series], period: int = 20) -> IndicatorResult:
    """Calculate Z-Score directly"""
    return ZScore().calculate(data, period=period)