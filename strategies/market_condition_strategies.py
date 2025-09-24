"""
Market Condition Strategy Modules
================================

This module implements adaptive strategy modules that automatically adjust
trading behavior based on detected market conditions:

1. Bull Market Strategy - Optimized for uptrending markets
2. Bear Market Strategy - Optimized for downtrending markets
3. Sideways Market Strategy - Optimized for range-bound markets

Each strategy module includes:
- Market regime detection algorithms
- Condition-specific signal generation
- Dynamic parameter adjustment
- Risk management adapted to market conditions
- Performance monitoring and feedback
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market condition types"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class SignalDirection(Enum):
    """Signal direction types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class MarketContext:
    """Market context information"""
    condition: MarketCondition
    trend_strength: float  # -1 (strong bear) to +1 (strong bull)
    volatility: float
    volume_profile: float
    momentum: float
    support_resistance: Dict[str, float]
    confidence: float
    timestamp: datetime


@dataclass
class TradingSignal:
    """Trading signal with market context"""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size: float
    market_context: MarketContext
    strategy_name: str
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketRegimeDetector:
    """Detects current market regime using multiple indicators"""

    def __init__(self):
        self.lookback_periods = {
            'short': 20,
            'medium': 50,
            'long': 200
        }
        self.volatility_window = 20
        self.volume_window = 20
        self.regime_history: List[MarketContext] = []

    def detect_market_condition(self, data: pd.DataFrame) -> MarketContext:
        """
        Detect current market condition using comprehensive analysis

        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            MarketContext with detected condition and metrics
        """
        try:
            if len(data) < self.lookback_periods['long']:
                return self._create_unknown_context()

            # Calculate trend indicators
            trend_metrics = self._calculate_trend_metrics(data)

            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(data)

            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(data)

            # Calculate momentum metrics
            momentum_metrics = self._calculate_momentum_metrics(data)

            # Detect support and resistance
            support_resistance = self._detect_support_resistance(data)

            # Classify market condition
            condition, confidence = self._classify_market_condition(
                trend_metrics, volatility_metrics, volume_metrics, momentum_metrics
            )

            market_context = MarketContext(
                condition=condition,
                trend_strength=trend_metrics['trend_strength'],
                volatility=volatility_metrics['current_volatility'],
                volume_profile=volume_metrics['volume_profile'],
                momentum=momentum_metrics['momentum_score'],
                support_resistance=support_resistance,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )

            # Store in history
            self.regime_history.append(market_context)
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]

            return market_context

        except Exception as e:
            logger.error(f"Error detecting market condition: {e}")
            return self._create_unknown_context()

    def _calculate_trend_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend-related metrics"""
        try:
            close = data['close']

            # Moving averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)

            # Current values
            current_price = close.iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            current_sma_200 = sma_200.iloc[-1]

            # Trend strength calculation
            if current_price > current_sma_20 > current_sma_50 > current_sma_200:
                trend_strength = 1.0  # Strong bullish
            elif current_price < current_sma_20 < current_sma_50 < current_sma_200:
                trend_strength = -1.0  # Strong bearish
            else:
                # Calculate relative position
                price_vs_ma = (current_price - current_sma_50) / current_sma_50
                trend_strength = np.clip(price_vs_ma * 10, -1.0, 1.0)

            # ADX for trend strength
            adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
            adx_strength = min(adx.iloc[-1] / 100.0, 1.0) if not np.isnan(adx.iloc[-1]) else 0.5

            # Slope of moving averages
            ma_slope = (current_sma_20 - sma_20.iloc[-5]) / sma_20.iloc[-5] if len(sma_20) > 5 else 0

            return {
                'trend_strength': trend_strength,
                'adx_strength': adx_strength,
                'ma_slope': ma_slope,
                'price_vs_ma_20': (current_price - current_sma_20) / current_sma_20,
                'price_vs_ma_50': (current_price - current_sma_50) / current_sma_50,
                'price_vs_ma_200': (current_price - current_sma_200) / current_sma_200
            }

        except Exception as e:
            logger.error(f"Error calculating trend metrics: {e}")
            return {'trend_strength': 0.0, 'adx_strength': 0.5, 'ma_slope': 0.0,
                   'price_vs_ma_20': 0.0, 'price_vs_ma_50': 0.0, 'price_vs_ma_200': 0.0}

    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-related metrics"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']

            # ATR (Average True Range)
            atr = talib.ATR(high, low, close, timeperiod=14)
            current_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.01

            # Historical volatility
            returns = close.pct_change().dropna()
            current_volatility = returns.iloc[-self.volatility_window:].std() * np.sqrt(252)

            # Bollinger Bands width
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]

            # VIX-like calculation (simplified)
            vix_like = current_volatility * 100

            # Volatility percentile (current vs historical)
            vol_percentile = np.percentile(returns.iloc[-252:].std() * np.sqrt(252), 50) if len(returns) > 252 else 0.5

            return {
                'current_volatility': current_volatility,
                'atr': current_atr,
                'bb_width': bb_width,
                'vix_like': vix_like,
                'volatility_percentile': vol_percentile
            }

        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {'current_volatility': 0.2, 'atr': 0.01, 'bb_width': 0.1,
                   'vix_like': 20.0, 'volatility_percentile': 0.5}

    def _calculate_volume_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-related metrics"""
        try:
            volume = data['volume']
            close = data['close']

            # Volume moving average
            volume_ma = volume.rolling(window=self.volume_window).mean()
            current_volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0

            # On-Balance Volume
            obv = talib.OBV(close, volume)
            obv_trend = (obv.iloc[-1] - obv.iloc[-10]) / obv.iloc[-10] if len(obv) > 10 and obv.iloc[-10] != 0 else 0

            # Volume profile (simplified)
            volume_profile = np.percentile(volume.iloc[-self.volume_window:], 75) / np.percentile(volume.iloc[-self.volume_window:], 25) if len(volume) >= self.volume_window else 1.0

            # Accumulation/Distribution Line
            ad_line = talib.AD(data['high'], data['low'], data['close'], data['volume'])
            ad_trend = (ad_line.iloc[-1] - ad_line.iloc[-10]) / abs(ad_line.iloc[-10]) if len(ad_line) > 10 and ad_line.iloc[-10] != 0 else 0

            return {
                'volume_ratio': current_volume_ratio,
                'obv_trend': obv_trend,
                'volume_profile': volume_profile,
                'ad_trend': ad_trend
            }

        except Exception as e:
            logger.error(f"Error calculating volume metrics: {e}")
            return {'volume_ratio': 1.0, 'obv_trend': 0.0, 'volume_profile': 1.0, 'ad_trend': 0.0}

    def _calculate_momentum_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum-related metrics"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']

            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            macd_value = macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0.0
            macd_signal_value = macd_signal.iloc[-1] if not np.isnan(macd_signal.iloc[-1]) else 0.0

            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            stoch_value = stoch_k.iloc[-1] if not np.isnan(stoch_k.iloc[-1]) else 50.0

            # Williams %R
            willr = talib.WILLR(high, low, close, timeperiod=14)
            willr_value = willr.iloc[-1] if not np.isnan(willr.iloc[-1]) else -50.0

            # Momentum score (composite)
            momentum_score = (
                (current_rsi - 50) / 50 * 0.3 +  # RSI contribution
                (1 if macd_value > macd_signal_value else -1) * 0.3 +  # MACD contribution
                (stoch_value - 50) / 50 * 0.2 +  # Stochastic contribution
                (willr_value + 50) / 50 * 0.2  # Williams %R contribution
            )

            return {
                'momentum_score': momentum_score,
                'rsi': current_rsi,
                'macd': macd_value,
                'macd_signal': macd_signal_value,
                'stochastic': stoch_value,
                'williams_r': willr_value
            }

        except Exception as e:
            logger.error(f"Error calculating momentum metrics: {e}")
            return {'momentum_score': 0.0, 'rsi': 50.0, 'macd': 0.0,
                   'macd_signal': 0.0, 'stochastic': 50.0, 'williams_r': -50.0}

    def _detect_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Detect key support and resistance levels"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']

            # Find recent highs and lows
            recent_high = high.iloc[-20:].max()
            recent_low = low.iloc[-20:].min()
            current_price = close.iloc[-1]

            # Pivot points (simplified)
            pivot = (recent_high + recent_low + current_price) / 3
            r1 = 2 * pivot - recent_low
            s1 = 2 * pivot - recent_high
            r2 = pivot + (recent_high - recent_low)
            s2 = pivot - (recent_high - recent_low)

            return {
                'pivot': pivot,
                'resistance_1': r1,
                'resistance_2': r2,
                'support_1': s1,
                'support_2': s2,
                'recent_high': recent_high,
                'recent_low': recent_low
            }

        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            current_price = data['close'].iloc[-1]
            return {
                'pivot': current_price,
                'resistance_1': current_price * 1.02,
                'resistance_2': current_price * 1.04,
                'support_1': current_price * 0.98,
                'support_2': current_price * 0.96,
                'recent_high': current_price * 1.05,
                'recent_low': current_price * 0.95
            }

    def _classify_market_condition(self,
                                 trend_metrics: Dict[str, float],
                                 volatility_metrics: Dict[str, float],
                                 volume_metrics: Dict[str, float],
                                 momentum_metrics: Dict[str, float]) -> Tuple[MarketCondition, float]:
        """Classify market condition based on all metrics"""
        try:
            trend_strength = trend_metrics['trend_strength']
            volatility = volatility_metrics['current_volatility']
            momentum = momentum_metrics['momentum_score']
            adx = trend_metrics['adx_strength']

            confidence = 0.5

            # High volatility check
            if volatility > 0.4:
                return MarketCondition.HIGH_VOLATILITY, min(0.8, volatility)

            # Low volatility check
            if volatility < 0.1:
                return MarketCondition.LOW_VOLATILITY, min(0.8, 1 - volatility * 5)

            # Strong trend conditions
            if trend_strength > 0.5 and momentum > 0.3 and adx > 0.3:
                confidence = min(0.9, (trend_strength + momentum + adx) / 3)
                return MarketCondition.BULL_MARKET, confidence

            if trend_strength < -0.5 and momentum < -0.3 and adx > 0.3:
                confidence = min(0.9, (-trend_strength - momentum + adx) / 3)
                return MarketCondition.BEAR_MARKET, confidence

            # Sideways market (low ADX, neutral trend)
            if abs(trend_strength) < 0.2 and adx < 0.25:
                confidence = min(0.8, (0.2 - abs(trend_strength)) * 5)
                return MarketCondition.SIDEWAYS_MARKET, confidence

            # Breakout detection (high volatility + strong momentum)
            if volatility > 0.25 and abs(momentum) > 0.4:
                confidence = min(0.8, (volatility * 2 + abs(momentum)) / 3)
                return MarketCondition.BREAKOUT, confidence

            # Default to trend-based classification
            if trend_strength > 0.1:
                return MarketCondition.BULL_MARKET, abs(trend_strength)
            elif trend_strength < -0.1:
                return MarketCondition.BEAR_MARKET, abs(trend_strength)
            else:
                return MarketCondition.SIDEWAYS_MARKET, 0.5

        except Exception as e:
            logger.error(f"Error classifying market condition: {e}")
            return MarketCondition.UNKNOWN, 0.0

    def _create_unknown_context(self) -> MarketContext:
        """Create unknown market context for error cases"""
        return MarketContext(
            condition=MarketCondition.UNKNOWN,
            trend_strength=0.0,
            volatility=0.2,
            volume_profile=1.0,
            momentum=0.0,
            support_resistance={},
            confidence=0.0,
            timestamp=datetime.now(timezone.utc)
        )


class BaseMarketStrategy(ABC):
    """Abstract base class for market condition strategies"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.regime_detector = MarketRegimeDetector()

        # Strategy state
        self.active = True
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

        # Signal history
        self.signal_history: List[TradingSignal] = []

    @abstractmethod
    async def generate_signal(self, data: pd.DataFrame, market_context: MarketContext) -> Optional[TradingSignal]:
        """Generate trading signal based on market data and context"""
        pass

    @abstractmethod
    def is_suitable_condition(self, market_context: MarketContext) -> bool:
        """Check if current market condition is suitable for this strategy"""
        pass

    async def analyze_market_and_signal(self, data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Analyze market condition and generate signal if suitable"""
        try:
            # Detect market condition
            market_context = self.regime_detector.detect_market_condition(data)

            # Check if strategy is suitable for current conditions
            if not self.is_suitable_condition(market_context):
                return None

            # Generate signal
            signal = await self.generate_signal(data, market_context)

            if signal:
                signal.symbol = symbol
                signal.strategy_name = self.name
                self.signal_history.append(signal)
                self.performance_metrics['total_signals'] += 1

                # Limit history size
                if len(self.signal_history) > 1000:
                    self.signal_history = self.signal_history[-1000:]

            return signal

        except Exception as e:
            logger.error(f"Error in {self.name} strategy analysis: {e}")
            return None

    def calculate_position_size(self,
                              signal_strength: SignalStrength,
                              market_context: MarketContext,
                              account_balance: float,
                              risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on signal strength and risk management"""
        try:
            # Base position size from risk management
            base_size = account_balance * risk_per_trade

            # Adjust for signal strength
            strength_multiplier = {
                SignalStrength.VERY_WEAK: 0.2,
                SignalStrength.WEAK: 0.5,
                SignalStrength.MODERATE: 1.0,
                SignalStrength.STRONG: 1.5,
                SignalStrength.VERY_STRONG: 2.0
            }

            # Adjust for market volatility
            volatility_adjustment = max(0.5, min(2.0, 1.0 / market_context.volatility))

            # Adjust for confidence
            confidence_adjustment = market_context.confidence

            position_size = (base_size *
                           strength_multiplier[signal_strength] *
                           volatility_adjustment *
                           confidence_adjustment)

            return max(0.1, min(position_size, account_balance * 0.1))  # Cap at 10% of account

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return account_balance * 0.01  # 1% fallback

    def update_performance(self, signal: TradingSignal, actual_return: float) -> None:
        """Update strategy performance metrics"""
        try:
            if actual_return > 0:
                self.performance_metrics['successful_signals'] += 1

            # Update win rate
            total = self.performance_metrics['total_signals']
            if total > 0:
                self.performance_metrics['win_rate'] = self.performance_metrics['successful_signals'] / total

            # Update average return (simple moving average)
            current_avg = self.performance_metrics['avg_return']
            self.performance_metrics['avg_return'] = (current_avg * (total - 1) + actual_return) / total

        except Exception as e:
            logger.error(f"Error updating performance: {e}")


class BullMarketStrategy(BaseMarketStrategy):
    """Strategy optimized for bull market conditions"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Bull Market Strategy", config)

        # Bull market specific parameters
        self.momentum_threshold = config.get('momentum_threshold', 0.3) if config else 0.3
        self.trend_strength_threshold = config.get('trend_strength_threshold', 0.4) if config else 0.4
        self.rsi_oversold_level = config.get('rsi_oversold_level', 35) if config else 35
        self.breakout_volume_multiplier = config.get('breakout_volume_multiplier', 1.5) if config else 1.5

    def is_suitable_condition(self, market_context: MarketContext) -> bool:
        """Check if current conditions favor bull market strategy"""
        return (market_context.condition in [MarketCondition.BULL_MARKET, MarketCondition.BREAKOUT] or
                (market_context.trend_strength > 0.2 and market_context.momentum > 0.1))

    async def generate_signal(self, data: pd.DataFrame, market_context: MarketContext) -> Optional[TradingSignal]:
        """Generate bull market trading signals"""
        try:
            close = data['close']
            volume = data['volume']
            high = data['high']
            low = data['low']

            current_price = close.iloc[-1]

            # Technical indicators
            rsi = talib.RSI(close, timeperiod=14).iloc[-1]
            macd, macd_signal, _ = talib.MACD(close)
            sma_20 = talib.SMA(close, timeperiod=20).iloc[-1]
            sma_50 = talib.SMA(close, timeperiod=50).iloc[-1]

            # Volume analysis
            volume_ma = volume.rolling(window=20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_ma

            signal_strength = SignalStrength.MODERATE
            confidence = 0.5
            reasoning = []

            # Bull market signals

            # 1. Momentum breakout with volume
            if (current_price > sma_20 > sma_50 and
                market_context.momentum > self.momentum_threshold and
                volume_ratio > self.breakout_volume_multiplier):

                signal_strength = SignalStrength.STRONG
                confidence = min(0.9, market_context.confidence + 0.2)
                reasoning.append("Momentum breakout with strong volume")

            # 2. Pullback to moving average (buy the dip)
            elif (market_context.trend_strength > self.trend_strength_threshold and
                  current_price < sma_20 and current_price > sma_50 and
                  rsi < self.rsi_oversold_level):

                signal_strength = SignalStrength.MODERATE
                confidence = min(0.8, market_context.confidence)
                reasoning.append("Pullback to support in uptrend")

            # 3. MACD bullish crossover
            elif (macd.iloc[-1] > macd_signal.iloc[-1] and
                  macd.iloc[-2] <= macd_signal.iloc[-2] and
                  market_context.trend_strength > 0.1):

                signal_strength = SignalStrength.MODERATE
                confidence = min(0.7, market_context.confidence)
                reasoning.append("MACD bullish crossover")

            # 4. Strong momentum continuation
            elif (market_context.momentum > self.momentum_threshold and
                  current_price > sma_20 and
                  rsi < 80):  # Not overbought

                signal_strength = SignalStrength.WEAK
                confidence = min(0.6, market_context.confidence)
                reasoning.append("Momentum continuation")

            else:
                return None  # No suitable signal

            # Calculate targets and stops
            target_price = self._calculate_bull_target(current_price, market_context)
            stop_loss = self._calculate_bull_stop(current_price, market_context, sma_20)

            # Position sizing
            position_size = self.calculate_position_size(
                signal_strength, market_context, 100000  # Assume $100k account
            )

            return TradingSignal(
                symbol="",  # Will be set by caller
                direction=SignalDirection.BUY,
                strength=signal_strength,
                confidence=confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                position_size=position_size,
                market_context=market_context,
                strategy_name=self.name,
                reasoning="; ".join(reasoning),
                timestamp=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Error generating bull market signal: {e}")
            return None

    def _calculate_bull_target(self, current_price: float, market_context: MarketContext) -> float:
        """Calculate target price for bull market trades"""
        try:
            # Base target based on volatility
            base_target = current_price * (1 + market_context.volatility * 2)

            # Adjust for trend strength
            trend_adjustment = 1 + (market_context.trend_strength * 0.1)

            # Check resistance levels
            resistance = market_context.support_resistance.get('resistance_1', current_price * 1.05)

            target = min(base_target * trend_adjustment, resistance * 0.98)  # Stop before resistance

            return max(target, current_price * 1.01)  # Minimum 1% target

        except Exception as e:
            logger.error(f"Error calculating bull target: {e}")
            return current_price * 1.03

    def _calculate_bull_stop(self, current_price: float, market_context: MarketContext, sma_20: float) -> float:
        """Calculate stop loss for bull market trades"""
        try:
            # Base stop below moving average
            ma_stop = sma_20 * 0.98

            # Volatility-based stop
            vol_stop = current_price * (1 - market_context.volatility * 1.5)

            # Support level stop
            support = market_context.support_resistance.get('support_1', current_price * 0.95)

            # Use the highest stop (least aggressive)
            stop = max(ma_stop, vol_stop, support * 0.98)

            return min(stop, current_price * 0.97)  # Maximum 3% stop

        except Exception as e:
            logger.error(f"Error calculating bull stop: {e}")
            return current_price * 0.95


class BearMarketStrategy(BaseMarketStrategy):
    """Strategy optimized for bear market conditions"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Bear Market Strategy", config)

        # Bear market specific parameters
        self.momentum_threshold = config.get('momentum_threshold', -0.3) if config else -0.3
        self.trend_strength_threshold = config.get('trend_strength_threshold', -0.4) if config else -0.4
        self.rsi_overbought_level = config.get('rsi_overbought_level', 65) if config else 65
        self.short_enabled = config.get('short_enabled', True) if config else True

    def is_suitable_condition(self, market_context: MarketContext) -> bool:
        """Check if current conditions favor bear market strategy"""
        return (market_context.condition == MarketCondition.BEAR_MARKET or
                (market_context.trend_strength < -0.2 and market_context.momentum < -0.1))

    async def generate_signal(self, data: pd.DataFrame, market_context: MarketContext) -> Optional[TradingSignal]:
        """Generate bear market trading signals"""
        try:
            close = data['close']
            volume = data['volume']
            high = data['high']
            low = data['low']

            current_price = close.iloc[-1]

            # Technical indicators
            rsi = talib.RSI(close, timeperiod=14).iloc[-1]
            macd, macd_signal, _ = talib.MACD(close)
            sma_20 = talib.SMA(close, timeperiod=20).iloc[-1]
            sma_50 = talib.SMA(close, timeperiod=50).iloc[-1]

            signal_strength = SignalStrength.MODERATE
            confidence = 0.5
            reasoning = []
            direction = SignalDirection.SELL if self.short_enabled else SignalDirection.HOLD

            # Bear market signals

            # 1. Breakdown with volume
            if (current_price < sma_20 < sma_50 and
                market_context.momentum < self.momentum_threshold and
                volume.iloc[-1] > volume.rolling(window=20).mean().iloc[-1] * 1.3):

                signal_strength = SignalStrength.STRONG
                confidence = min(0.9, market_context.confidence + 0.2)
                reasoning.append("Breakdown with strong volume")

            # 2. Bear market rally (dead cat bounce)
            elif (market_context.trend_strength < self.trend_strength_threshold and
                  current_price > sma_20 and current_price < sma_50 and
                  rsi > self.rsi_overbought_level):

                signal_strength = SignalStrength.MODERATE
                confidence = min(0.8, market_context.confidence)
                reasoning.append("Bear market rally exhaustion")

            # 3. MACD bearish crossover
            elif (macd.iloc[-1] < macd_signal.iloc[-1] and
                  macd.iloc[-2] >= macd_signal.iloc[-2] and
                  market_context.trend_strength < -0.1):

                signal_strength = SignalStrength.MODERATE
                confidence = min(0.7, market_context.confidence)
                reasoning.append("MACD bearish crossover")

            # 4. Momentum continuation to downside
            elif (market_context.momentum < self.momentum_threshold and
                  current_price < sma_20 and
                  rsi > 20):  # Not oversold yet

                signal_strength = SignalStrength.WEAK
                confidence = min(0.6, market_context.confidence)
                reasoning.append("Downward momentum continuation")

            else:
                return None  # No suitable signal

            # In bear markets, if shorting not enabled, return hold signals
            if not self.short_enabled:
                direction = SignalDirection.HOLD
                reasoning.append("(Cash position - shorting disabled)")

            # Calculate targets and stops for short positions
            if direction == SignalDirection.SELL:
                target_price = self._calculate_bear_target(current_price, market_context)
                stop_loss = self._calculate_bear_stop(current_price, market_context, sma_20)
            else:
                target_price = None
                stop_loss = None

            # Position sizing
            position_size = self.calculate_position_size(
                signal_strength, market_context, 100000  # Assume $100k account
            )

            return TradingSignal(
                symbol="",  # Will be set by caller
                direction=direction,
                strength=signal_strength,
                confidence=confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                position_size=position_size,
                market_context=market_context,
                strategy_name=self.name,
                reasoning="; ".join(reasoning),
                timestamp=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Error generating bear market signal: {e}")
            return None

    def _calculate_bear_target(self, current_price: float, market_context: MarketContext) -> float:
        """Calculate target price for bear market short trades"""
        try:
            # Base target based on volatility
            base_target = current_price * (1 - market_context.volatility * 2)

            # Adjust for trend strength
            trend_adjustment = 1 + (abs(market_context.trend_strength) * 0.1)

            # Check support levels
            support = market_context.support_resistance.get('support_1', current_price * 0.95)

            target = max(base_target / trend_adjustment, support * 1.02)  # Stop before support

            return min(target, current_price * 0.99)  # Minimum 1% target

        except Exception as e:
            logger.error(f"Error calculating bear target: {e}")
            return current_price * 0.97

    def _calculate_bear_stop(self, current_price: float, market_context: MarketContext, sma_20: float) -> float:
        """Calculate stop loss for bear market short trades"""
        try:
            # Base stop above moving average
            ma_stop = sma_20 * 1.02

            # Volatility-based stop
            vol_stop = current_price * (1 + market_context.volatility * 1.5)

            # Resistance level stop
            resistance = market_context.support_resistance.get('resistance_1', current_price * 1.05)

            # Use the lowest stop (least aggressive)
            stop = min(ma_stop, vol_stop, resistance * 1.02)

            return max(stop, current_price * 1.03)  # Maximum 3% stop

        except Exception as e:
            logger.error(f"Error calculating bear stop: {e}")
            return current_price * 1.05


class SidewaysMarketStrategy(BaseMarketStrategy):
    """Strategy optimized for sideways/range-bound markets"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Sideways Market Strategy", config)

        # Sideways market specific parameters
        self.range_threshold = config.get('range_threshold', 0.05) if config else 0.05  # 5% range
        self.rsi_oversold = config.get('rsi_oversold', 30) if config else 30
        self.rsi_overbought = config.get('rsi_overbought', 70) if config else 70
        self.bollinger_position_threshold = config.get('bollinger_position_threshold', 0.2) if config else 0.2

    def is_suitable_condition(self, market_context: MarketContext) -> bool:
        """Check if current conditions favor sideways market strategy"""
        return (market_context.condition == MarketCondition.SIDEWAYS_MARKET or
                (abs(market_context.trend_strength) < 0.3 and market_context.volatility < 0.3))

    async def generate_signal(self, data: pd.DataFrame, market_context: MarketContext) -> Optional[TradingSignal]:
        """Generate mean reversion signals for sideways markets"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']

            current_price = close.iloc[-1]

            # Technical indicators for mean reversion
            rsi = talib.RSI(close, timeperiod=14).iloc[-1]
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            stoch_k, stoch_d = talib.STOCH(high, low, close)

            # Current values
            bb_upper_val = bb_upper.iloc[-1]
            bb_lower_val = bb_lower.iloc[-1]
            bb_middle_val = bb_middle.iloc[-1]
            stoch_val = stoch_k.iloc[-1]

            # Calculate position within Bollinger Bands
            bb_position = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)

            # Identify range boundaries
            range_high = market_context.support_resistance.get('resistance_1', bb_upper_val)
            range_low = market_context.support_resistance.get('support_1', bb_lower_val)

            signal_strength = SignalStrength.MODERATE
            confidence = 0.5
            reasoning = []
            direction = SignalDirection.HOLD

            # Mean reversion signals

            # 1. Oversold at range bottom
            if (rsi < self.rsi_oversold and
                bb_position < self.bollinger_position_threshold and
                stoch_val < 20 and
                current_price <= range_low * 1.02):

                direction = SignalDirection.BUY
                signal_strength = SignalStrength.STRONG
                confidence = min(0.9, market_context.confidence + 0.3)
                reasoning.append("Oversold at range support")

            # 2. Overbought at range top
            elif (rsi > self.rsi_overbought and
                  bb_position > (1 - self.bollinger_position_threshold) and
                  stoch_val > 80 and
                  current_price >= range_high * 0.98):

                direction = SignalDirection.SELL
                signal_strength = SignalStrength.STRONG
                confidence = min(0.9, market_context.confidence + 0.3)
                reasoning.append("Overbought at range resistance")

            # 3. Moderate oversold
            elif (rsi < 40 and bb_position < 0.3 and stoch_val < 30):
                direction = SignalDirection.BUY
                signal_strength = SignalStrength.MODERATE
                confidence = min(0.7, market_context.confidence + 0.1)
                reasoning.append("Moderate oversold condition")

            # 4. Moderate overbought
            elif (rsi > 60 and bb_position > 0.7 and stoch_val > 70):
                direction = SignalDirection.SELL
                signal_strength = SignalStrength.MODERATE
                confidence = min(0.7, market_context.confidence + 0.1)
                reasoning.append("Moderate overbought condition")

            # 5. Return to mean
            elif abs(current_price - bb_middle_val) / bb_middle_val > 0.02:
                if current_price < bb_middle_val:
                    direction = SignalDirection.BUY
                    reasoning.append("Price below mean - buy")
                else:
                    direction = SignalDirection.SELL
                    reasoning.append("Price above mean - sell")

                signal_strength = SignalStrength.WEAK
                confidence = min(0.6, market_context.confidence)

            else:
                return None  # No suitable signal

            # Calculate targets and stops for range trading
            if direction == SignalDirection.BUY:
                target_price = min(range_high * 0.98, bb_upper_val * 0.95)
                stop_loss = max(range_low * 0.98, current_price * 0.97)
            elif direction == SignalDirection.SELL:
                target_price = max(range_low * 1.02, bb_lower_val * 1.05)
                stop_loss = min(range_high * 1.02, current_price * 1.03)
            else:
                target_price = None
                stop_loss = None

            # Position sizing (smaller for mean reversion)
            position_size = self.calculate_position_size(
                signal_strength, market_context, 100000  # Assume $100k account
            ) * 0.8  # Reduce size for range trading

            return TradingSignal(
                symbol="",  # Will be set by caller
                direction=direction,
                strength=signal_strength,
                confidence=confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                position_size=position_size,
                market_context=market_context,
                strategy_name=self.name,
                reasoning="; ".join(reasoning),
                timestamp=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Error generating sideways market signal: {e}")
            return None


class AdaptiveMarketStrategyOrchestrator:
    """
    Orchestrates multiple market condition strategies and selects the most
    appropriate one based on current market conditions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize strategies
        self.strategies = {
            MarketCondition.BULL_MARKET: BullMarketStrategy(config.get('bull_config')),
            MarketCondition.BEAR_MARKET: BearMarketStrategy(config.get('bear_config')),
            MarketCondition.SIDEWAYS_MARKET: SidewaysMarketStrategy(config.get('sideways_config'))
        }

        # Strategy performance tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.active_strategy: Optional[BaseMarketStrategy] = None
        self.strategy_switch_history: List[Dict[str, Any]] = []

        # Configuration
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.strategy_switch_cooldown = timedelta(minutes=config.get('switch_cooldown_minutes', 30))
        self.last_strategy_switch = datetime.now(timezone.utc) - self.strategy_switch_cooldown

    async def generate_adaptive_signal(self, data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal using the most appropriate strategy
        for current market conditions.
        """
        try:
            # Find the best strategy for current conditions
            best_strategy = await self._select_best_strategy(data)

            if not best_strategy:
                return None

            # Switch strategy if needed
            if best_strategy != self.active_strategy:
                await self._switch_strategy(best_strategy, symbol)

            # Generate signal with active strategy
            signal = await self.active_strategy.analyze_market_and_signal(data, symbol)

            if signal:
                # Add orchestrator metadata
                signal.metadata.update({
                    'orchestrator': 'AdaptiveMarketStrategyOrchestrator',
                    'strategy_confidence': signal.market_context.confidence,
                    'strategies_evaluated': list(self.strategies.keys())
                })

            return signal

        except Exception as e:
            logger.error(f"Error generating adaptive signal: {e}")
            return None

    async def _select_best_strategy(self, data: pd.DataFrame) -> Optional[BaseMarketStrategy]:
        """Select the best strategy based on current market conditions"""
        try:
            # Detect market condition using first available strategy
            regime_detector = list(self.strategies.values())[0].regime_detector
            market_context = regime_detector.detect_market_condition(data)

            # Find suitable strategies
            suitable_strategies = []

            for condition, strategy in self.strategies.items():
                if strategy.is_suitable_condition(market_context):
                    suitability_score = self._calculate_strategy_suitability(strategy, market_context)
                    suitable_strategies.append((strategy, suitability_score))

            if not suitable_strategies:
                return None

            # Sort by suitability score
            suitable_strategies.sort(key=lambda x: x[1], reverse=True)

            # Return best strategy if confidence is sufficient
            best_strategy, best_score = suitable_strategies[0]

            if best_score >= self.confidence_threshold:
                return best_strategy

            return None

        except Exception as e:
            logger.error(f"Error selecting best strategy: {e}")
            return None

    def _calculate_strategy_suitability(self,
                                      strategy: BaseMarketStrategy,
                                      market_context: MarketContext) -> float:
        """Calculate strategy suitability score"""
        try:
            base_score = market_context.confidence

            # Adjust based on strategy performance history
            strategy_name = strategy.name
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
                performance_score = (perf.get('win_rate', 0.5) +
                                   min(1.0, max(0.0, perf.get('sharpe_ratio', 0.0) / 2 + 0.5)))
                base_score = (base_score + performance_score) / 2

            # Adjust based on market condition match
            if strategy.name == "Bull Market Strategy" and market_context.condition == MarketCondition.BULL_MARKET:
                base_score += 0.2
            elif strategy.name == "Bear Market Strategy" and market_context.condition == MarketCondition.BEAR_MARKET:
                base_score += 0.2
            elif strategy.name == "Sideways Market Strategy" and market_context.condition == MarketCondition.SIDEWAYS_MARKET:
                base_score += 0.2

            return min(1.0, base_score)

        except Exception as e:
            logger.error(f"Error calculating strategy suitability: {e}")
            return 0.0

    async def _switch_strategy(self, new_strategy: BaseMarketStrategy, symbol: str) -> None:
        """Switch to a new strategy"""
        try:
            # Check cooldown period
            time_since_switch = datetime.now(timezone.utc) - self.last_strategy_switch
            if time_since_switch < self.strategy_switch_cooldown:
                return

            old_strategy = self.active_strategy.name if self.active_strategy else "None"
            new_strategy_name = new_strategy.name

            # Log strategy switch
            switch_record = {
                'timestamp': datetime.now(timezone.utc),
                'symbol': symbol,
                'from_strategy': old_strategy,
                'to_strategy': new_strategy_name,
                'reason': 'market_condition_change'
            }

            self.strategy_switch_history.append(switch_record)
            self.active_strategy = new_strategy
            self.last_strategy_switch = datetime.now(timezone.utc)

            # Limit history size
            if len(self.strategy_switch_history) > 100:
                self.strategy_switch_history = self.strategy_switch_history[-100:]

            logger.info(f"Strategy switch: {old_strategy} -> {new_strategy_name} for {symbol}")

        except Exception as e:
            logger.error(f"Error switching strategy: {e}")

    def update_strategy_performance(self,
                                  strategy_name: str,
                                  signal: TradingSignal,
                                  actual_return: float) -> None:
        """Update performance metrics for a specific strategy"""
        try:
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    'total_signals': 0,
                    'successful_signals': 0,
                    'win_rate': 0.0,
                    'total_return': 0.0,
                    'avg_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'returns_history': []
                }

            perf = self.strategy_performance[strategy_name]
            perf['total_signals'] += 1
            perf['total_return'] += actual_return

            if actual_return > 0:
                perf['successful_signals'] += 1

            # Update derived metrics
            perf['win_rate'] = perf['successful_signals'] / perf['total_signals']
            perf['avg_return'] = perf['total_return'] / perf['total_signals']

            # Update returns history for Sharpe calculation
            perf['returns_history'].append(actual_return)
            if len(perf['returns_history']) > 252:  # Keep last year
                perf['returns_history'] = perf['returns_history'][-252:]

            # Calculate Sharpe ratio
            if len(perf['returns_history']) > 10:
                returns_array = np.array(perf['returns_history'])
                sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
                perf['sharpe_ratio'] = sharpe

            # Update strategy object performance
            for strategy in self.strategies.values():
                if strategy.name == strategy_name:
                    strategy.update_performance(signal, actual_return)
                    break

        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        try:
            return {
                'active_strategy': self.active_strategy.name if self.active_strategy else None,
                'strategy_performance': self.strategy_performance,
                'recent_switches': self.strategy_switch_history[-10:],
                'strategies_available': list(self.strategies.keys()),
                'last_strategy_switch': self.last_strategy_switch.isoformat(),
                'confidence_threshold': self.confidence_threshold
            }

        except Exception as e:
            logger.error(f"Error getting orchestrator status: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    async def test_market_strategies():
        """Test the market condition strategies"""

        # Generate sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)

        # Simulate different market conditions
        price_data = []
        base_price = 100.0

        for i in range(300):
            if i < 100:  # Bull market
                trend = 0.001
                volatility = 0.015
            elif i < 200:  # Bear market
                trend = -0.002
                volatility = 0.025
            else:  # Sideways market
                trend = 0.0
                volatility = 0.012

            # Generate OHLCV
            change = np.random.normal(trend, volatility)
            base_price *= (1 + change)

            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = base_price + np.random.normal(0, 0.005)
            volume = np.random.randint(100000, 1000000)

            price_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': base_price,
                'volume': volume
            })

        data = pd.DataFrame(price_data, index=dates)

        try:
            # Test orchestrator
            config = {
                'confidence_threshold': 0.6,
                'switch_cooldown_minutes': 1,  # Short for testing
                'bull_config': {'momentum_threshold': 0.2},
                'bear_config': {'momentum_threshold': -0.2},
                'sideways_config': {'range_threshold': 0.04}
            }

            orchestrator = AdaptiveMarketStrategyOrchestrator(config)

            # Test signals at different periods
            test_periods = [50, 150, 250]  # Bull, bear, sideways

            for period in test_periods:
                test_data = data.iloc[:period+50]  # Include enough history
                signal = await orchestrator.generate_adaptive_signal(test_data, "TEST")

                if signal:
                    print(f"\nPeriod {period} (Day {test_data.index[-1].date()}):")
                    print(f"Strategy: {signal.strategy_name}")
                    print(f"Market Condition: {signal.market_context.condition.value}")
                    print(f"Direction: {signal.direction.value}")
                    print(f"Strength: {signal.strength.value}")
                    print(f"Confidence: {signal.confidence:.2f}")
                    print(f"Entry: ${signal.entry_price:.2f}")
                    print(f"Target: ${signal.target_price:.2f}" if signal.target_price else "Target: None")
                    print(f"Stop: ${signal.stop_loss:.2f}" if signal.stop_loss else "Stop: None")
                    print(f"Position Size: ${signal.position_size:.0f}")
                    print(f"Reasoning: {signal.reasoning}")
                    print(f"Trend Strength: {signal.market_context.trend_strength:.2f}")
                    print(f"Volatility: {signal.market_context.volatility:.2f}")
                else:
                    print(f"\nPeriod {period}: No signal generated")

            # Test strategy switching
            print(f"\nOrchestrator Status:")
            status = orchestrator.get_orchestrator_status()
            print(f"Active Strategy: {status['active_strategy']}")
            print(f"Strategy Switches: {len(status['recent_switches'])}")

        except Exception as e:
            print(f"Error testing strategies: {e}")

    # Run test
    asyncio.run(test_market_strategies())