"""
Fibonacci Analysis Library for LangGraph Trading System

This module provides comprehensive Fibonacci analysis tools including:
- Fibonacci retracement and extension calculations
- Confluence zone detection algorithm
- Support/resistance level identification
- Integration with technical indicators for signal enhancement

All calculations are optimized for performance using NumPy vectorization
and include comprehensive parameter validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from enum import Enum


class FibonacciLevel(Enum):
    """Standard Fibonacci retracement and extension levels"""
    # Retracement levels
    RETRACEMENT_236 = 0.236
    RETRACEMENT_382 = 0.382
    RETRACEMENT_500 = 0.500
    RETRACEMENT_618 = 0.618
    RETRACEMENT_786 = 0.786
    
    # Extension levels
    EXTENSION_1272 = 1.272
    EXTENSION_1414 = 1.414
    EXTENSION_1618 = 1.618
    EXTENSION_2618 = 2.618


@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    index: int
    price: float
    timestamp: Optional[pd.Timestamp] = None
    swing_type: str = 'high'  # 'high' or 'low'


@dataclass
class FibonacciLevels:
    """Container for Fibonacci retracement levels"""
    swing_high: SwingPoint
    swing_low: SwingPoint
    levels: Dict[str, float]
    level_prices: Dict[str, float]
    swing_range: float
    direction: str  # 'bullish' or 'bearish'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'swing_high': {'index': self.swing_high.index, 'price': self.swing_high.price},
            'swing_low': {'index': self.swing_low.index, 'price': self.swing_low.price},
            'levels': self.levels,
            'level_prices': self.level_prices,
            'swing_range': self.swing_range,
            'direction': self.direction
        }


@dataclass
class FibonacciExtensions:
    """Container for Fibonacci extension levels"""
    swing_high: SwingPoint
    swing_low: SwingPoint
    retracement_point: SwingPoint
    extension_levels: Dict[str, float]
    direction: str  # 'bullish' or 'bearish'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'swing_high': {'index': self.swing_high.index, 'price': self.swing_high.price},
            'swing_low': {'index': self.swing_low.index, 'price': self.swing_low.price},
            'retracement_point': {'index': self.retracement_point.index, 'price': self.retracement_point.price},
            'extension_levels': self.extension_levels,
            'direction': self.direction
        }


@dataclass
class ConfluenceZone:
    """Represents a confluence zone where multiple levels align"""
    price: float
    strength: float
    components: List[str]
    tolerance: float
    level_count: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'price': self.price,
            'strength': self.strength,
            'components': self.components,
            'tolerance': self.tolerance,
            'level_count': self.level_count
        }


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float
    touch_count: int
    first_touch_index: int
    last_touch_index: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'price': self.price,
            'level_type': self.level_type,
            'strength': self.strength,
            'touch_count': self.touch_count,
            'first_touch_index': self.first_touch_index,
            'last_touch_index': self.last_touch_index
        }


class SwingPointDetector:
    """Detects swing highs and lows in price data"""
    
    def __init__(self, lookback_periods: int = 5):
        """
        Initialize swing point detector
        
        Args:
            lookback_periods: Number of periods to look back/forward for swing detection
        """
        self.lookback_periods = lookback_periods
    
    def detect_swing_points(self, high_data: np.ndarray, low_data: np.ndarray, 
                           timestamps: Optional[np.ndarray] = None) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect swing highs and lows in price data
        
        Args:
            high_data: Array of high prices
            low_data: Array of low prices
            timestamps: Optional array of timestamps
        
        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        if len(high_data) != len(low_data):
            raise ValueError("High and low data arrays must have the same length")
        
        if len(high_data) < 2 * self.lookback_periods + 1:
            raise ValueError(f"Insufficient data: need at least {2 * self.lookback_periods + 1} periods")
        
        # Handle NaN values by skipping them
        if np.any(np.isnan(high_data)) or np.any(np.isnan(low_data)):
            warnings.warn("Data contains NaN values, swing detection may be affected")
        
        swing_highs = []
        swing_lows = []
        
        # Detect swing highs
        for i in range(self.lookback_periods, len(high_data) - self.lookback_periods):
            current_high = high_data[i]
            
            # Skip NaN values
            if np.isnan(current_high):
                continue
            
            is_swing_high = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - self.lookback_periods, i + self.lookback_periods + 1):
                if j != i:
                    if np.isnan(high_data[j]):
                        continue  # Skip NaN values in comparison
                    if high_data[j] >= current_high:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                timestamp = timestamps[i] if timestamps is not None else None
                swing_highs.append(SwingPoint(
                    index=i,
                    price=current_high,
                    timestamp=timestamp,
                    swing_type='high'
                ))
        
        # Detect swing lows
        for i in range(self.lookback_periods, len(low_data) - self.lookback_periods):
            current_low = low_data[i]
            
            # Skip NaN values
            if np.isnan(current_low):
                continue
            
            is_swing_low = True
            
            # Check if current point is lower than surrounding points
            for j in range(i - self.lookback_periods, i + self.lookback_periods + 1):
                if j != i:
                    if np.isnan(low_data[j]):
                        continue  # Skip NaN values in comparison
                    if low_data[j] <= current_low:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                timestamp = timestamps[i] if timestamps is not None else None
                swing_lows.append(SwingPoint(
                    index=i,
                    price=current_low,
                    timestamp=timestamp,
                    swing_type='low'
                ))
        
        return swing_highs, swing_lows


class FibonacciCalculator:
    """Core Fibonacci calculations for retracements and extensions"""
    
    @staticmethod
    def calculate_retracement_levels(swing_high: SwingPoint, swing_low: SwingPoint,
                                   custom_levels: Optional[List[float]] = None) -> FibonacciLevels:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            swing_high: Swing high point
            swing_low: Swing low point
            custom_levels: Optional custom Fibonacci levels (default: standard levels)
        
        Returns:
            FibonacciLevels object with calculated levels
        """
        if custom_levels is None:
            custom_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
        
        swing_range = swing_high.price - swing_low.price
        direction = 'bullish' if swing_high.index > swing_low.index else 'bearish'
        
        levels = {}
        level_prices = {}
        
        for level in custom_levels:
            level_name = f"fib_{int(level * 1000)}"
            levels[level_name] = level
            
            if direction == 'bullish':
                # For bullish retracement, levels are below the high
                level_prices[level_name] = swing_high.price - (swing_range * level)
            else:
                # For bearish retracement, levels are above the low
                level_prices[level_name] = swing_low.price + (swing_range * level)
        
        return FibonacciLevels(
            swing_high=swing_high,
            swing_low=swing_low,
            levels=levels,
            level_prices=level_prices,
            swing_range=swing_range,
            direction=direction
        )
    
    @staticmethod
    def calculate_extension_levels(swing_high: SwingPoint, swing_low: SwingPoint,
                                 retracement_point: SwingPoint,
                                 custom_levels: Optional[List[float]] = None) -> FibonacciExtensions:
        """
        Calculate Fibonacci extension levels
        
        Args:
            swing_high: Initial swing high
            swing_low: Initial swing low
            retracement_point: Point where price retraced to
            custom_levels: Optional custom extension levels
        
        Returns:
            FibonacciExtensions object with calculated levels
        """
        if custom_levels is None:
            custom_levels = [1.272, 1.414, 1.618, 2.618]
        
        swing_range = abs(swing_high.price - swing_low.price)
        direction = 'bullish' if retracement_point.price > swing_low.price else 'bearish'
        
        extension_levels = {}
        
        for level in custom_levels:
            level_name = f"ext_{int(level * 1000)}"
            
            if direction == 'bullish':
                # Bullish extensions project upward from retracement point
                extension_levels[level_name] = retracement_point.price + (swing_range * level)
            else:
                # Bearish extensions project downward from retracement point
                extension_levels[level_name] = retracement_point.price - (swing_range * level)
        
        return FibonacciExtensions(
            swing_high=swing_high,
            swing_low=swing_low,
            retracement_point=retracement_point,
            extension_levels=extension_levels,
            direction=direction
        )


class SupportResistanceDetector:
    """Detects support and resistance levels in price data"""
    
    def __init__(self, min_touches: int = 2, tolerance_pct: float = 0.5):
        """
        Initialize support/resistance detector
        
        Args:
            min_touches: Minimum number of touches to confirm a level
            tolerance_pct: Price tolerance as percentage for level detection
        """
        self.min_touches = min_touches
        self.tolerance_pct = tolerance_pct / 100.0
    
    def detect_levels(self, high_data: np.ndarray, low_data: np.ndarray,
                     close_data: np.ndarray) -> List[SupportResistanceLevel]:
        """
        Detect support and resistance levels
        
        Args:
            high_data: Array of high prices
            low_data: Array of low prices
            close_data: Array of close prices
        
        Returns:
            List of SupportResistanceLevel objects
        """
        levels = []
        
        # Combine all price points for level detection
        all_prices = np.concatenate([high_data, low_data])
        price_indices = np.concatenate([
            np.arange(len(high_data)),
            np.arange(len(low_data))
        ])
        price_types = ['high'] * len(high_data) + ['low'] * len(low_data)
        
        # Sort by price
        sorted_indices = np.argsort(all_prices)
        sorted_prices = all_prices[sorted_indices]
        sorted_original_indices = price_indices[sorted_indices]
        sorted_types = [price_types[i] for i in sorted_indices]
        
        # Group similar prices
        i = 0
        while i < len(sorted_prices):
            current_price = sorted_prices[i]
            group_prices = [current_price]
            group_indices = [sorted_original_indices[i]]
            group_types = [sorted_types[i]]
            
            j = i + 1
            while j < len(sorted_prices):
                if abs(sorted_prices[j] - current_price) / current_price <= self.tolerance_pct:
                    group_prices.append(sorted_prices[j])
                    group_indices.append(sorted_original_indices[j])
                    group_types.append(sorted_types[j])
                    j += 1
                else:
                    break
            
            # Check if this group forms a valid support/resistance level
            if len(group_prices) >= self.min_touches:
                avg_price = np.mean(group_prices)
                
                # Determine if it's support or resistance based on price action around it
                level_type = self._determine_level_type(avg_price, close_data, group_indices)
                
                # Calculate strength based on number of touches and price action
                strength = self._calculate_level_strength(group_prices, group_indices, close_data)
                
                levels.append(SupportResistanceLevel(
                    price=avg_price,
                    level_type=level_type,
                    strength=strength,
                    touch_count=len(group_prices),
                    first_touch_index=min(group_indices),
                    last_touch_index=max(group_indices)
                ))
            
            i = j
        
        return levels
    
    def _determine_level_type(self, level_price: float, close_data: np.ndarray,
                            touch_indices: List[int]) -> str:
        """Determine if a level is support or resistance"""
        # Look at price action around the level
        nearby_closes = []
        for idx in touch_indices:
            start_idx = max(0, idx - 5)
            end_idx = min(len(close_data), idx + 6)
            nearby_closes.extend(close_data[start_idx:end_idx])
        
        if not nearby_closes:
            return 'support'  # Default
        
        # If most nearby closes are above the level, it's likely support
        above_count = sum(1 for price in nearby_closes if price > level_price)
        below_count = sum(1 for price in nearby_closes if price < level_price)
        
        return 'support' if above_count > below_count else 'resistance'
    
    def _calculate_level_strength(self, prices: List[float], indices: List[int],
                                close_data: np.ndarray) -> float:
        """Calculate the strength of a support/resistance level"""
        # Base strength from number of touches
        touch_strength = min(len(prices) / 10.0, 1.0)  # Max 1.0 for 10+ touches
        
        # Time span strength (longer time span = stronger level)
        if len(indices) > 1:
            time_span = max(indices) - min(indices)
            time_strength = min(time_span / len(close_data), 1.0)
        else:
            time_strength = 0.1
        
        # Price consistency strength (tighter clustering = stronger)
        if len(prices) > 1:
            price_std = np.std(prices)
            avg_price = np.mean(prices)
            consistency_strength = max(0, 1.0 - (price_std / avg_price) * 100)
        else:
            consistency_strength = 1.0
        
        # Combine strengths
        total_strength = (touch_strength * 0.4 + time_strength * 0.3 + 
                         consistency_strength * 0.3)
        
        return min(total_strength, 1.0)


class ConfluenceDetector:
    """Detects confluence zones where multiple Fibonacci levels and S/R levels align"""
    
    def __init__(self, tolerance_pct: float = 0.5):
        """
        Initialize confluence detector
        
        Args:
            tolerance_pct: Price tolerance as percentage for confluence detection
        """
        self.tolerance_pct = tolerance_pct / 100.0
    
    def detect_confluence_zones(self, fibonacci_levels: List[FibonacciLevels],
                              fibonacci_extensions: List[FibonacciExtensions],
                              support_resistance: List[SupportResistanceLevel]) -> List[ConfluenceZone]:
        """
        Detect confluence zones where multiple levels align
        
        Args:
            fibonacci_levels: List of Fibonacci retracement levels
            fibonacci_extensions: List of Fibonacci extension levels
            support_resistance: List of support/resistance levels
        
        Returns:
            List of ConfluenceZone objects
        """
        all_levels = []
        
        # Collect all Fibonacci retracement levels
        for fib_levels in fibonacci_levels:
            for level_name, price in fib_levels.level_prices.items():
                all_levels.append({
                    'price': price,
                    'type': f'fib_retracement_{level_name}',
                    'strength': 1.0,
                    'source': 'fibonacci_retracement'
                })
        
        # Collect all Fibonacci extension levels
        for fib_ext in fibonacci_extensions:
            for level_name, price in fib_ext.extension_levels.items():
                all_levels.append({
                    'price': price,
                    'type': f'fib_extension_{level_name}',
                    'strength': 1.0,
                    'source': 'fibonacci_extension'
                })
        
        # Collect all support/resistance levels
        for sr_level in support_resistance:
            all_levels.append({
                'price': sr_level.price,
                'type': f'sr_{sr_level.level_type}',
                'strength': sr_level.strength,
                'source': 'support_resistance'
            })
        
        if not all_levels:
            return []
        
        # Sort levels by price
        all_levels.sort(key=lambda x: x['price'])
        
        confluence_zones = []
        i = 0
        
        while i < len(all_levels):
            current_level = all_levels[i]
            confluent_levels = [current_level]
            
            # Find all levels within tolerance
            j = i + 1
            while j < len(all_levels):
                if (abs(all_levels[j]['price'] - current_level['price']) / 
                    current_level['price'] <= self.tolerance_pct):
                    confluent_levels.append(all_levels[j])
                    j += 1
                else:
                    break
            
            # Create confluence zone if multiple levels align
            if len(confluent_levels) >= 2:
                avg_price = np.mean([level['price'] for level in confluent_levels])
                total_strength = sum(level['strength'] for level in confluent_levels)
                components = [level['type'] for level in confluent_levels]
                
                confluence_zones.append(ConfluenceZone(
                    price=avg_price,
                    strength=total_strength,
                    components=components,
                    tolerance=self.tolerance_pct,
                    level_count=len(confluent_levels)
                ))
            
            i = j if j > i + 1 else i + 1
        
        # Sort by strength (strongest first)
        confluence_zones.sort(key=lambda x: x.strength, reverse=True)
        
        return confluence_zones


class FibonacciAnalyzer:
    """Main class for comprehensive Fibonacci analysis"""
    
    def __init__(self, lookback_periods: int = 5, min_touches: int = 2,
                 tolerance_pct: float = 0.5):
        """
        Initialize Fibonacci analyzer
        
        Args:
            lookback_periods: Periods for swing point detection
            min_touches: Minimum touches for S/R level confirmation
            tolerance_pct: Price tolerance percentage for confluence detection
        """
        self.swing_detector = SwingPointDetector(lookback_periods)
        self.sr_detector = SupportResistanceDetector(min_touches, tolerance_pct)
        self.confluence_detector = ConfluenceDetector(tolerance_pct)
        self.fibonacci_calculator = FibonacciCalculator()
    
    def analyze(self, high_data: np.ndarray, low_data: np.ndarray,
               close_data: np.ndarray, timestamps: Optional[np.ndarray] = None,
               max_swings: int = 10) -> Dict:
        """
        Perform comprehensive Fibonacci analysis
        
        Args:
            high_data: Array of high prices
            low_data: Array of low prices
            close_data: Array of close prices
            timestamps: Optional timestamps
            max_swings: Maximum number of recent swings to analyze
        
        Returns:
            Dictionary containing all analysis results
        """
        # Detect swing points
        swing_highs, swing_lows = self.swing_detector.detect_swing_points(
            high_data, low_data, timestamps
        )
        
        # Limit to most recent swings
        swing_highs = swing_highs[-max_swings:] if len(swing_highs) > max_swings else swing_highs
        swing_lows = swing_lows[-max_swings:] if len(swing_lows) > max_swings else swing_lows
        
        # Calculate Fibonacci retracement levels for all swing combinations
        fibonacci_levels = []
        for high in swing_highs:
            for low in swing_lows:
                if abs(high.index - low.index) >= 5:  # Minimum swing distance
                    fib_levels = self.fibonacci_calculator.calculate_retracement_levels(high, low)
                    fibonacci_levels.append(fib_levels)
        
        # Calculate Fibonacci extensions (simplified - using recent swings)
        fibonacci_extensions = []
        if len(swing_highs) >= 1 and len(swing_lows) >= 2:
            # Example: Use most recent high and two recent lows
            recent_high = swing_highs[-1]
            if len(swing_lows) >= 2:
                low1, low2 = swing_lows[-2], swing_lows[-1]
                if recent_high.index > low1.index:
                    fib_ext = self.fibonacci_calculator.calculate_extension_levels(
                        recent_high, low1, low2
                    )
                    fibonacci_extensions.append(fib_ext)
        
        # Detect support and resistance levels
        support_resistance = self.sr_detector.detect_levels(high_data, low_data, close_data)
        
        # Detect confluence zones
        confluence_zones = self.confluence_detector.detect_confluence_zones(
            fibonacci_levels, fibonacci_extensions, support_resistance
        )
        
        return {
            'swing_highs': [swing.to_dict() if hasattr(swing, 'to_dict') else {
                'index': swing.index, 'price': swing.price, 'swing_type': swing.swing_type
            } for swing in swing_highs],
            'swing_lows': [swing.to_dict() if hasattr(swing, 'to_dict') else {
                'index': swing.index, 'price': swing.price, 'swing_type': swing.swing_type
            } for swing in swing_lows],
            'fibonacci_levels': [fib.to_dict() for fib in fibonacci_levels],
            'fibonacci_extensions': [ext.to_dict() for ext in fibonacci_extensions],
            'support_resistance': [sr.to_dict() for sr in support_resistance],
            'confluence_zones': [zone.to_dict() for zone in confluence_zones],
            'analysis_metadata': {
                'total_swings_analyzed': len(swing_highs) + len(swing_lows),
                'fibonacci_level_count': len(fibonacci_levels),
                'extension_count': len(fibonacci_extensions),
                'sr_level_count': len(support_resistance),
                'confluence_zone_count': len(confluence_zones),
                'strongest_confluence': confluence_zones[0].to_dict() if confluence_zones else None
            }
        }


# Convenience functions for direct access
def analyze_fibonacci_levels(high_data: Union[np.ndarray, pd.Series],
                           low_data: Union[np.ndarray, pd.Series],
                           close_data: Union[np.ndarray, pd.Series],
                           timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
                           **kwargs) -> Dict:
    """
    Perform comprehensive Fibonacci analysis on price data
    
    Args:
        high_data: High prices
        low_data: Low prices
        close_data: Close prices
        timestamps: Optional timestamps
        **kwargs: Additional parameters for FibonacciAnalyzer
    
    Returns:
        Dictionary with complete Fibonacci analysis results
    """
    # Convert to numpy arrays
    if isinstance(high_data, pd.Series):
        high_data = high_data.values
    if isinstance(low_data, pd.Series):
        low_data = low_data.values
    if isinstance(close_data, pd.Series):
        close_data = close_data.values
    if timestamps is not None and isinstance(timestamps, pd.Series):
        timestamps = timestamps.values
    
    analyzer = FibonacciAnalyzer(**kwargs)
    return analyzer.analyze(high_data, low_data, close_data, timestamps)


def calculate_fibonacci_retracements(swing_high_price: float, swing_low_price: float,
                                   swing_high_index: int = 0, swing_low_index: int = 1,
                                   custom_levels: Optional[List[float]] = None) -> Dict:
    """
    Calculate Fibonacci retracement levels for a single swing
    
    Args:
        swing_high_price: Price of swing high
        swing_low_price: Price of swing low
        swing_high_index: Index of swing high
        swing_low_index: Index of swing low
        custom_levels: Optional custom Fibonacci levels
    
    Returns:
        Dictionary with retracement levels
    """
    swing_high = SwingPoint(swing_high_index, swing_high_price, swing_type='high')
    swing_low = SwingPoint(swing_low_index, swing_low_price, swing_type='low')
    
    fib_levels = FibonacciCalculator.calculate_retracement_levels(
        swing_high, swing_low, custom_levels
    )
    
    return fib_levels.to_dict()


def detect_confluence_zones_simple(price_levels: List[float], 
                                 tolerance_pct: float = 0.5) -> List[Dict]:
    """
    Simple confluence zone detection for a list of price levels
    
    Args:
        price_levels: List of price levels to analyze
        tolerance_pct: Price tolerance percentage
    
    Returns:
        List of confluence zones
    """
    if not price_levels:
        return []
    
    # Create mock support/resistance levels
    sr_levels = [
        SupportResistanceLevel(
            price=price,
            level_type='support',
            strength=1.0,
            touch_count=1,
            first_touch_index=0,
            last_touch_index=0
        ) for price in price_levels
    ]
    
    detector = ConfluenceDetector(tolerance_pct)
    confluence_zones = detector.detect_confluence_zones([], [], sr_levels)
    
    return [zone.to_dict() for zone in confluence_zones]