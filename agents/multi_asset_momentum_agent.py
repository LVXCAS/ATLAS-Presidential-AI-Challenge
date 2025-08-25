#!/usr/bin/env python3
"""
Multi-Asset Momentum Agent - HIVE TRADE
Cross-asset momentum trading with regime detection and dynamic position sizing
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Asset class categorization"""
    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"
    COMMODITY = "COMMODITY"
    BOND = "BOND"
    CURRENCY = "CURRENCY"

class MomentumRegime(Enum):
    """Market momentum regimes"""
    STRONG_BULL = "STRONG_BULL"      # Strong upward momentum across assets
    MILD_BULL = "MILD_BULL"          # Moderate upward momentum
    SIDEWAYS = "SIDEWAYS"            # No clear momentum direction
    MILD_BEAR = "MILD_BEAR"          # Moderate downward momentum
    STRONG_BEAR = "STRONG_BEAR"      # Strong downward momentum
    VOLATILE = "VOLATILE"            # High volatility, unclear direction

@dataclass
class AssetMetrics:
    """Comprehensive asset momentum metrics"""
    symbol: str
    asset_class: AssetClass
    
    # Price metrics
    current_price: float = 0.0
    
    # Momentum indicators
    momentum_1d: float = 0.0      # 1-day momentum
    momentum_5d: float = 0.0      # 5-day momentum
    momentum_10d: float = 0.0     # 10-day momentum
    momentum_20d: float = 0.0     # 20-day momentum
    momentum_60d: float = 0.0     # 60-day momentum
    
    # Technical indicators
    rsi: float = 50.0             # Relative Strength Index
    macd: float = 0.0             # MACD signal
    bb_position: float = 0.5      # Position within Bollinger Bands (0-1)
    volume_ratio: float = 1.0     # Current volume / avg volume
    
    # Cross-asset metrics
    correlation_to_market: float = 0.0    # Correlation to broad market
    relative_strength: float = 0.0        # Relative strength vs market
    sector_momentum: float = 0.0          # Average momentum of asset class
    
    # Risk metrics
    volatility: float = 0.2
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Signal strength
    momentum_score: float = 0.0    # Combined momentum score (-1 to +1)
    signal_strength: float = 0.0   # Overall signal strength (0-1)
    confidence: float = 0.0        # Signal confidence (0-1)

@dataclass
class MomentumSignal:
    """Multi-asset momentum signal"""
    timestamp: datetime
    symbol: str
    asset_class: AssetClass
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    position_size: float  # Suggested position size as % of capital
    
    # Signal metrics
    momentum_score: float
    signal_strength: float
    confidence: float
    expected_return: float
    risk_score: float
    
    # Context
    regime: MomentumRegime
    cross_asset_confirmation: bool
    sector_support: bool
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'asset_class': self.asset_class.value,
            'signal_type': self.signal_type,
            'position_size': self.position_size,
            'momentum_score': self.momentum_score,
            'signal_strength': self.signal_strength,
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'risk_score': self.risk_score,
            'regime': self.regime.value,
            'cross_asset_confirmation': bool(self.cross_asset_confirmation),
            'sector_support': bool(self.sector_support)
        }

class MultiAssetMomentumAgent:
    """Advanced multi-asset momentum trading agent"""
    
    def __init__(self, asset_universe: Dict[str, AssetClass], lookback_window: int = 60):
        self.asset_universe = asset_universe  # symbol -> asset_class mapping
        self.lookback_window = lookback_window
        
        # Trading parameters
        self.momentum_threshold = 0.3        # Minimum momentum score for signal
        self.min_signal_strength = 0.6      # Minimum signal strength
        self.min_confidence = 0.7           # Minimum confidence for trade
        self.max_position_size = 0.15       # Max position size per asset (15%)
        self.max_asset_class_allocation = 0.4  # Max allocation per asset class (40%)
        self.risk_budget_per_trade = 0.02   # Max risk per trade (2%)
        
        # Cross-asset parameters
        self.regime_lookback = 30           # Periods for regime detection
        self.correlation_window = 20        # Window for correlation calculation
        self.rebalance_threshold = 0.1      # Position drift threshold for rebalancing
        
        # Data storage
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.asset_metrics: Dict[str, AssetMetrics] = {}
        self.current_regime: MomentumRegime = MomentumRegime.SIDEWAYS
        self.regime_confidence: float = 0.0
        
        # Portfolio state
        self.current_positions: Dict[str, float] = {}  # symbol -> position_size
        self.target_positions: Dict[str, float] = {}   # symbol -> target_size
        self.active_signals: List[MomentumSignal] = []
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.portfolio_value_history: deque = deque(maxlen=252)
        self.total_return: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        
        # Initialize data structures
        for symbol in asset_universe.keys():
            self.price_history[symbol] = deque(maxlen=lookback_window * 2)
            self.volume_history[symbol] = deque(maxlen=lookback_window * 2)
            self.asset_metrics[symbol] = AssetMetrics(symbol, asset_universe[symbol])
            self.current_positions[symbol] = 0.0
            self.target_positions[symbol] = 0.0
        
        # Market proxy for relative calculations
        self.market_proxy = 'SPY'  # Assume SPY as market proxy
        if self.market_proxy not in asset_universe:
            # Find an equity symbol as proxy
            equity_symbols = [s for s, ac in asset_universe.items() if ac == AssetClass.EQUITY]
            self.market_proxy = equity_symbols[0] if equity_symbols else list(asset_universe.keys())[0]
        
        logger.info(f"Multi-Asset Momentum Agent initialized")
        logger.info(f"Asset universe: {len(asset_universe)} symbols across asset classes")
        logger.info(f"Lookback window: {lookback_window} periods")

    def add_market_data(self, symbol: str, price: float, volume: float = None, timestamp: datetime = None):
        """Add market data for an asset"""
        if symbol not in self.asset_universe:
            return
        
        timestamp = timestamp or datetime.now()
        
        # Store price data
        self.price_history[symbol].append((timestamp, price))
        
        # Store volume data if provided
        if volume is not None:
            self.volume_history[symbol].append((timestamp, volume))
        
        # Update current price
        self.asset_metrics[symbol].current_price = price

    def calculate_momentum_indicators(self, symbol: str) -> AssetMetrics:
        """Calculate comprehensive momentum indicators for an asset"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return self.asset_metrics[symbol]
        
        prices = np.array([p[1] for p in list(self.price_history[symbol])])
        
        if len(prices) < 20:
            return self.asset_metrics[symbol]
        
        metrics = self.asset_metrics[symbol]
        
        # Calculate momentum over different periods
        if len(prices) >= 2:
            metrics.momentum_1d = (prices[-1] / prices[-2] - 1)
        if len(prices) >= 6:
            metrics.momentum_5d = (prices[-1] / prices[-6] - 1)
        if len(prices) >= 11:
            metrics.momentum_10d = (prices[-1] / prices[-11] - 1)
        if len(prices) >= 21:
            metrics.momentum_20d = (prices[-1] / prices[-21] - 1)
        if len(prices) >= 60:
            metrics.momentum_60d = (prices[-1] / prices[-60] - 1)
        
        # Calculate RSI (simplified)
        if len(prices) >= 15:
            changes = np.diff(prices[-15:])
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
            
            rs = avg_gain / avg_loss
            metrics.rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD (simplified)
        if len(prices) >= 26:
            ema12 = self._calculate_ema(prices, 12)
            ema26 = self._calculate_ema(prices, 26)
            macd_line = ema12 - ema26
            
            if len(prices) >= 35:
                # Get MACD signal line (9-period EMA of MACD)
                macd_values = [macd_line]  # Simplified - should be calculated over time
                signal_line = macd_line * 0.9  # Simplified signal
                metrics.macd = macd_line - signal_line
            else:
                metrics.macd = macd_line
        
        # Calculate Bollinger Band position
        if len(prices) >= 20:
            sma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])
            upper_band = sma20 + (2 * std20)
            lower_band = sma20 - (2 * std20)
            
            if upper_band > lower_band:
                metrics.bb_position = (prices[-1] - lower_band) / (upper_band - lower_band)
            else:
                metrics.bb_position = 0.5
        
        # Calculate volume ratio
        if symbol in self.volume_history and len(self.volume_history[symbol]) >= 10:
            volumes = np.array([v[1] for v in list(self.volume_history[symbol])])
            if len(volumes) >= 10:
                current_volume = volumes[-1]
                avg_volume = np.mean(volumes[-10:])
                metrics.volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate volatility
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / prices[-20:-1]
            metrics.volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio (simplified)
        if len(prices) >= 30:
            returns = np.diff(prices[-30:]) / prices[-30:-1]
            avg_return = np.mean(returns) * 252  # Annualized
            vol = np.std(returns) * np.sqrt(252)
            metrics.sharpe_ratio = avg_return / vol if vol > 0 else 0
        
        # Calculate max drawdown
        if len(prices) >= 20:
            cumulative = prices / prices[0]
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            metrics.max_drawdown = np.min(drawdowns)
        
        return metrics

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema

    def calculate_cross_asset_metrics(self):
        """Calculate cross-asset momentum and correlation metrics"""
        # Update correlation to market proxy
        if (self.market_proxy in self.price_history and 
            len(self.price_history[self.market_proxy]) >= 20):
            
            market_prices = np.array([p[1] for p in list(self.price_history[self.market_proxy])[-20:]])
            market_returns = np.diff(market_prices) / market_prices[:-1]
            
            for symbol, metrics in self.asset_metrics.items():
                if (symbol != self.market_proxy and 
                    len(self.price_history[symbol]) >= 20):
                    
                    asset_prices = np.array([p[1] for p in list(self.price_history[symbol])[-20:]])
                    asset_returns = np.diff(asset_prices) / asset_prices[:-1]
                    
                    if len(asset_returns) == len(market_returns):
                        correlation = np.corrcoef(asset_returns, market_returns)[0, 1]
                        metrics.correlation_to_market = correlation if not np.isnan(correlation) else 0
                        
                        # Calculate relative strength
                        asset_perf = (asset_prices[-1] / asset_prices[0] - 1)
                        market_perf = (market_prices[-1] / market_prices[0] - 1)
                        metrics.relative_strength = asset_perf - market_perf
        
        # Calculate sector momentum
        asset_class_returns = {}
        for asset_class in AssetClass:
            class_symbols = [s for s, ac in self.asset_universe.items() if ac == asset_class]
            
            if class_symbols:
                class_momentums = []
                for symbol in class_symbols:
                    if symbol in self.asset_metrics:
                        class_momentums.append(self.asset_metrics[symbol].momentum_20d)
                
                if class_momentums:
                    asset_class_returns[asset_class] = np.mean(class_momentums)
        
        # Update sector momentum for each asset
        for symbol, metrics in self.asset_metrics.items():
            asset_class = self.asset_universe[symbol]
            if asset_class in asset_class_returns:
                metrics.sector_momentum = asset_class_returns[asset_class]

    def detect_market_regime(self) -> MomentumRegime:
        """Detect current market momentum regime"""
        if len(self.asset_metrics) < 3:
            return MomentumRegime.SIDEWAYS
        
        # Collect momentum scores across assets
        momentums = []
        volatilities = []
        
        for metrics in self.asset_metrics.values():
            # Calculate composite momentum score
            momentum_score = (
                metrics.momentum_5d * 0.3 +
                metrics.momentum_10d * 0.4 +
                metrics.momentum_20d * 0.3
            )
            momentums.append(momentum_score)
            volatilities.append(metrics.volatility)
        
        avg_momentum = np.mean(momentums)
        momentum_consistency = 1.0 - np.std(momentums)  # Higher = more consistent
        avg_volatility = np.mean(volatilities)
        
        # Regime classification
        if avg_momentum > 0.05 and momentum_consistency > 0.5:
            if avg_momentum > 0.1:
                regime = MomentumRegime.STRONG_BULL
                confidence = min(0.95, momentum_consistency + 0.2)
            else:
                regime = MomentumRegime.MILD_BULL
                confidence = momentum_consistency
                
        elif avg_momentum < -0.05 and momentum_consistency > 0.5:
            if avg_momentum < -0.1:
                regime = MomentumRegime.STRONG_BEAR
                confidence = min(0.95, momentum_consistency + 0.2)
            else:
                regime = MomentumRegime.MILD_BEAR
                confidence = momentum_consistency
                
        elif avg_volatility > 0.4:
            regime = MomentumRegime.VOLATILE
            confidence = min(avg_volatility, 0.8)
            
        else:
            regime = MomentumRegime.SIDEWAYS
            confidence = 0.6
        
        self.current_regime = regime
        self.regime_confidence = confidence
        
        return regime

    def calculate_momentum_score(self, metrics: AssetMetrics) -> float:
        """Calculate composite momentum score for an asset"""
        # Weighted momentum components
        momentum_components = {
            'short_term': metrics.momentum_5d * 0.2,
            'medium_term': metrics.momentum_10d * 0.3,
            'long_term': metrics.momentum_20d * 0.3,
            'rsi_momentum': (metrics.rsi - 50) / 50 * 0.1,  # RSI deviation from neutral
            'macd': np.tanh(metrics.macd * 10) * 0.1  # MACD signal (normalized)
        }
        
        base_score = sum(momentum_components.values())
        
        # Apply regime adjustments
        regime_multipliers = {
            MomentumRegime.STRONG_BULL: 1.2,
            MomentumRegime.MILD_BULL: 1.1,
            MomentumRegime.SIDEWAYS: 1.0,
            MomentumRegime.MILD_BEAR: 0.9,
            MomentumRegime.STRONG_BEAR: 0.8,
            MomentumRegime.VOLATILE: 0.7
        }
        
        regime_multiplier = regime_multipliers.get(self.current_regime, 1.0)
        adjusted_score = base_score * regime_multiplier
        
        # Apply cross-asset confirmation
        sector_confirmation = 1.0
        if abs(metrics.sector_momentum) > 0.02:  # Significant sector momentum
            if np.sign(adjusted_score) == np.sign(metrics.sector_momentum):
                sector_confirmation = 1.2  # Sector supports signal
            else:
                sector_confirmation = 0.8  # Sector opposes signal
        
        final_score = np.tanh(adjusted_score * sector_confirmation)  # Bound between -1 and 1
        metrics.momentum_score = final_score
        
        return final_score

    def calculate_signal_strength(self, metrics: AssetMetrics) -> float:
        """Calculate signal strength based on multiple factors"""
        # Base strength from momentum score
        base_strength = abs(metrics.momentum_score)
        
        # Volume confirmation
        volume_confirmation = min(1.2, metrics.volume_ratio) if metrics.volume_ratio > 1.0 else 1.0
        
        # Technical indicator alignment
        tech_alignment = 0.0
        
        # RSI alignment
        if metrics.momentum_score > 0 and metrics.rsi > 50:
            tech_alignment += 0.3
        elif metrics.momentum_score < 0 and metrics.rsi < 50:
            tech_alignment += 0.3
        
        # MACD alignment
        if np.sign(metrics.momentum_score) == np.sign(metrics.macd):
            tech_alignment += 0.3
        
        # Bollinger Band position
        if metrics.momentum_score > 0 and metrics.bb_position > 0.7:
            tech_alignment += 0.2
        elif metrics.momentum_score < 0 and metrics.bb_position < 0.3:
            tech_alignment += 0.2
        
        # Relative strength factor
        rel_strength_factor = 1.0 + (metrics.relative_strength * 0.5)
        
        # Combine all factors
        signal_strength = base_strength * volume_confirmation * (1 + tech_alignment) * rel_strength_factor
        
        # Normalize to 0-1 range
        final_strength = min(1.0, signal_strength)
        metrics.signal_strength = final_strength
        
        return final_strength

    def calculate_position_size(self, metrics: AssetMetrics, signal_strength: float) -> float:
        """Calculate optimal position size for an asset"""
        # Base position size from signal strength
        base_size = signal_strength * self.max_position_size
        
        # Risk adjustment based on volatility
        vol_adjustment = min(1.0, 0.2 / max(metrics.volatility, 0.1))  # Scale down for high vol
        
        # Sharpe ratio adjustment
        sharpe_adjustment = max(0.5, min(1.5, 1.0 + metrics.sharpe_ratio * 0.2))
        
        # Regime adjustment
        regime_adjustments = {
            MomentumRegime.STRONG_BULL: 1.3,
            MomentumRegime.MILD_BULL: 1.1,
            MomentumRegime.SIDEWAYS: 0.7,
            MomentumRegime.MILD_BEAR: 1.1,
            MomentumRegime.STRONG_BEAR: 1.3,
            MomentumRegime.VOLATILE: 0.5
        }
        
        regime_adj = regime_adjustments.get(self.current_regime, 1.0)
        
        # Calculate final position size
        position_size = base_size * vol_adjustment * sharpe_adjustment * regime_adj
        
        # Apply maximum limits
        position_size = min(position_size, self.max_position_size)
        
        # Check asset class limits
        asset_class = self.asset_universe[metrics.symbol]
        current_class_allocation = sum(
            self.current_positions.get(s, 0) for s, ac in self.asset_universe.items() 
            if ac == asset_class
        )
        
        remaining_class_budget = self.max_asset_class_allocation - abs(current_class_allocation)
        position_size = min(position_size, remaining_class_budget)
        
        return max(0, position_size)

    def generate_momentum_signals(self) -> List[MomentumSignal]:
        """Generate momentum signals for all assets"""
        signals = []
        current_time = datetime.now()
        
        # Update all metrics
        for symbol in self.asset_universe.keys():
            self.calculate_momentum_indicators(symbol)
        
        # Calculate cross-asset metrics
        self.calculate_cross_asset_metrics()
        
        # Detect current regime
        regime = self.detect_market_regime()
        
        # Generate signals for each asset
        for symbol, metrics in self.asset_metrics.items():
            # Calculate momentum score and signal strength
            momentum_score = self.calculate_momentum_score(metrics)
            signal_strength = self.calculate_signal_strength(metrics)
            
            # Skip weak signals
            if signal_strength < self.min_signal_strength:
                continue
            
            # Determine signal type
            if momentum_score > self.momentum_threshold:
                signal_type = 'BUY'
                expected_return = momentum_score * 0.5  # Rough estimate
            elif momentum_score < -self.momentum_threshold:
                signal_type = 'SELL'
                expected_return = abs(momentum_score) * 0.5
            else:
                continue  # No signal
            
            # Calculate position size
            position_size = self.calculate_position_size(metrics, signal_strength)
            
            if position_size < 0.01:  # Skip very small positions
                continue
            
            # Calculate confidence
            confidence = (signal_strength + self.regime_confidence) / 2
            
            if confidence < self.min_confidence:
                continue
            
            # Check for cross-asset confirmation
            cross_asset_confirmation = abs(metrics.correlation_to_market) > 0.3
            sector_support = abs(metrics.sector_momentum) > 0.02 and np.sign(momentum_score) == np.sign(metrics.sector_momentum)
            
            # Calculate risk score
            risk_score = metrics.volatility * position_size
            
            # Create signal
            signal = MomentumSignal(
                timestamp=current_time,
                symbol=symbol,
                asset_class=self.asset_universe[symbol],
                signal_type=signal_type,
                position_size=position_size,
                momentum_score=momentum_score,
                signal_strength=signal_strength,
                confidence=confidence,
                expected_return=expected_return,
                risk_score=risk_score,
                regime=regime,
                cross_asset_confirmation=cross_asset_confirmation,
                sector_support=sector_support
            )
            
            signals.append(signal)
            
            logger.info(f"Generated {signal_type} signal for {symbol}: "
                       f"score={momentum_score:.3f}, strength={signal_strength:.3f}, "
                       f"size={position_size:.3f}, confidence={confidence:.3f}")
        
        self.active_signals = signals
        return signals

    def get_momentum_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive momentum trading dashboard"""
        current_time = datetime.now()
        
        # Asset metrics summary
        assets_summary = []
        for symbol, metrics in self.asset_metrics.items():
            assets_summary.append({
                'symbol': symbol,
                'asset_class': self.asset_universe[symbol].value,
                'price': round(metrics.current_price, 4),
                'momentum_1d': round(metrics.momentum_1d * 100, 2),
                'momentum_5d': round(metrics.momentum_5d * 100, 2),
                'momentum_20d': round(metrics.momentum_20d * 100, 2),
                'momentum_score': round(metrics.momentum_score, 3),
                'signal_strength': round(metrics.signal_strength, 3),
                'rsi': round(metrics.rsi, 1),
                'volatility': round(metrics.volatility * 100, 1),
                'relative_strength': round(metrics.relative_strength * 100, 2),
                'sector_momentum': round(metrics.sector_momentum * 100, 2),
                'current_position': round(self.current_positions.get(symbol, 0), 3)
            })
        
        # Active signals summary
        signals_summary = [signal.to_dict() for signal in self.active_signals]
        
        # Regime analysis
        regime_info = {
            'current_regime': self.current_regime.value,
            'confidence': round(self.regime_confidence, 3),
            'description': self._get_regime_description(self.current_regime)
        }
        
        # Portfolio allocation by asset class
        allocation_by_class = {}
        for asset_class in AssetClass:
            class_allocation = sum(
                self.current_positions.get(s, 0) for s, ac in self.asset_universe.items() 
                if ac == asset_class
            )
            if class_allocation != 0:
                allocation_by_class[asset_class.value] = round(class_allocation, 3)
        
        return {
            'timestamp': current_time.isoformat(),
            'strategy': 'Multi-Asset Momentum',
            'regime_analysis': regime_info,
            'performance_metrics': {
                'total_return': round(self.total_return, 4),
                'sharpe_ratio': round(self.sharpe_ratio, 3),
                'max_drawdown': round(self.max_drawdown, 4),
                'active_positions': len([p for p in self.current_positions.values() if abs(p) > 0.01])
            },
            'portfolio_allocation': allocation_by_class,
            'asset_metrics': assets_summary,
            'active_signals': signals_summary,
            'parameters': {
                'momentum_threshold': self.momentum_threshold,
                'min_signal_strength': self.min_signal_strength,
                'min_confidence': self.min_confidence,
                'max_position_size': self.max_position_size,
                'max_asset_class_allocation': self.max_asset_class_allocation,
                'lookback_window': self.lookback_window
            }
        }

    def _get_regime_description(self, regime: MomentumRegime) -> str:
        """Get description of market regime"""
        descriptions = {
            MomentumRegime.STRONG_BULL: "Strong upward momentum across multiple assets - favorable for momentum strategies",
            MomentumRegime.MILD_BULL: "Moderate upward momentum - selective momentum opportunities",
            MomentumRegime.SIDEWAYS: "No clear directional momentum - range-bound market conditions",
            MomentumRegime.MILD_BEAR: "Moderate downward momentum - short momentum opportunities",
            MomentumRegime.STRONG_BEAR: "Strong downward momentum - risk management priority",
            MomentumRegime.VOLATILE: "High volatility environment - reduced position sizes recommended"
        }
        return descriptions.get(regime, "Unknown regime")

def main():
    """Demonstrate multi-asset momentum agent"""
    print("HIVE TRADE - Multi-Asset Momentum Agent Demo")
    print("=" * 55)
    
    # Define asset universe
    asset_universe = {
        # Equities
        'AAPL': AssetClass.EQUITY,
        'MSFT': AssetClass.EQUITY,
        'GOOGL': AssetClass.EQUITY,
        'TSLA': AssetClass.EQUITY,
        'NVDA': AssetClass.EQUITY,
        'SPY': AssetClass.EQUITY,
        
        # Crypto
        'BTC-USD': AssetClass.CRYPTO,
        'ETH-USD': AssetClass.CRYPTO,
        
        # Commodities (simulated)
        'GLD': AssetClass.COMMODITY,  # Gold ETF
        'USO': AssetClass.COMMODITY,  # Oil ETF
    }
    
    # Initialize agent
    agent = MultiAssetMomentumAgent(asset_universe, lookback_window=60)
    
    # Simulate market data
    print(f"\nSimulating market data for {len(asset_universe)} assets...")
    
    base_prices = {
        'AAPL': 180, 'MSFT': 340, 'GOOGL': 2800, 'TSLA': 220, 'NVDA': 450, 'SPY': 420,
        'BTC-USD': 45000, 'ETH-USD': 4800, 'GLD': 180, 'USO': 75
    }
    
    # Generate trending data to show momentum
    np.random.seed(42)
    n_periods = 80
    
    # Create different momentum patterns for different assets
    momentum_factors = {
        'AAPL': 0.002, 'MSFT': 0.001, 'GOOGL': -0.001, 'TSLA': 0.003, 'NVDA': 0.0025,
        'SPY': 0.0015, 'BTC-USD': 0.005, 'ETH-USD': 0.004, 'GLD': -0.0005, 'USO': 0.001
    }
    
    for i in range(n_periods):
        timestamp = datetime.now() - timedelta(days=n_periods-i)
        
        for symbol, base_price in base_prices.items():
            # Generate trending price with momentum + noise
            trend = momentum_factors[symbol] * i
            noise = np.random.normal(0, 0.02)
            
            if i == 0:
                price = base_price
            else:
                # Get previous price
                prev_data = list(agent.price_history[symbol])
                prev_price = prev_data[-1][1] if prev_data else base_price
                price = prev_price * (1 + trend + noise)
            
            volume = np.random.uniform(50000, 200000)
            agent.add_market_data(symbol, price, volume, timestamp)
    
    print(f"Generated {n_periods} periods of trending data")
    
    # Generate momentum signals
    print("\nGenerating multi-asset momentum signals...")
    signals = agent.generate_momentum_signals()
    
    # Get dashboard
    dashboard = agent.get_momentum_dashboard()
    
    # Display results
    print(f"\nMULTI-ASSET MOMENTUM ANALYSIS:")
    regime = dashboard['regime_analysis']
    print(f"  Market Regime: {regime['current_regime']} (confidence: {regime['confidence']:.3f})")
    print(f"  Description: {regime['description']}")
    
    print(f"\nACTIVE SIGNALS ({len(signals)}):")
    for signal in signals[:5]:  # Show top 5 signals
        print(f"  {signal.signal_type} {signal.symbol} ({signal.asset_class.value}):")
        print(f"    Position Size: {signal.position_size:.1%}")
        print(f"    Momentum Score: {signal.momentum_score:+.3f}")
        print(f"    Signal Strength: {signal.signal_strength:.3f}")
        print(f"    Confidence: {signal.confidence:.3f}")
        print(f"    Expected Return: {signal.expected_return:.2%}")
        print(f"    Cross-Asset Support: {signal.cross_asset_confirmation}")
    
    print(f"\nTOP MOMENTUM ASSETS:")
    sorted_assets = sorted(dashboard['asset_metrics'], 
                          key=lambda x: abs(x['momentum_score']), reverse=True)
    
    for asset in sorted_assets[:5]:
        print(f"  {asset['symbol']} ({asset['asset_class']}):")
        print(f"    20d Momentum: {asset['momentum_20d']:+.2f}%")
        print(f"    Momentum Score: {asset['momentum_score']:+.3f}")
        print(f"    Signal Strength: {asset['signal_strength']:.3f}")
        print(f"    Relative Strength: {asset['relative_strength']:+.2f}%")
    
    if dashboard['portfolio_allocation']:
        print(f"\nPORTFOLIO ALLOCATION BY ASSET CLASS:")
        for asset_class, allocation in dashboard['portfolio_allocation'].items():
            print(f"  {asset_class}: {allocation:.1%}")
    
    # Save results
    results_file = 'momentum_trading_results.json'
    with open(results_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("Multi-Asset Momentum Agent demonstration completed!")

if __name__ == "__main__":
    main()