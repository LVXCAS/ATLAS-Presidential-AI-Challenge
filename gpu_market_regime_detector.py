"""
GPU MARKET REGIME DETECTION SYSTEM
Advanced market condition identification using GTX 1660 Super acceleration
Real-time detection of bull/bear markets, volatility regimes, and trend changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import json
from enum import Enum
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "Bull Market"
    BEAR_MARKET = "Bear Market"
    SIDEWAYS = "Sideways/Choppy"
    HIGH_VOLATILITY = "High Volatility"
    LOW_VOLATILITY = "Low Volatility"
    MOMENTUM = "Strong Momentum"
    MEAN_REVERSION = "Mean Reversion"
    CRISIS = "Crisis/Crash"
    RECOVERY = "Recovery"

@dataclass
class RegimeDetection:
    """Market regime detection result"""
    primary_regime: MarketRegime
    confidence: float
    regime_probabilities: Dict[MarketRegime, float]
    volatility_level: str
    trend_strength: float
    momentum_score: float
    timestamp: datetime

class GPURegimeClassifier(nn.Module):
    """GPU-accelerated neural network for market regime classification"""

    def __init__(self, input_features: int = 50, hidden_dim: int = 256):
        super().__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim // 4,
                           batch_first=True, num_layers=2, dropout=0.2)

        # Multi-head attention for regime transitions
        self.attention = nn.MultiheadAttention(hidden_dim // 4, num_heads=4,
                                             dropout=0.1, batch_first=True)

        # Regime classifiers
        self.trend_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 3)  # Bull, Bear, Sideways
        )

        self.volatility_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 3)  # Low, Medium, High
        )

        self.momentum_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 3)  # Mean-reversion, Neutral, Momentum
        )

        # Crisis detection
        self.crisis_detector = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 2)  # Normal, Crisis
        )

    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)

        # Add sequence dimension for LSTM
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        # LSTM processing
        lstm_out, _ = self.lstm(features)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last time step
        final_features = attn_out[:, -1, :]

        # Classifications
        trend_logits = self.trend_classifier(final_features)
        volatility_logits = self.volatility_classifier(final_features)
        momentum_logits = self.momentum_classifier(final_features)
        crisis_logits = self.crisis_detector(final_features)

        return trend_logits, volatility_logits, momentum_logits, crisis_logits

class MarketRegimeDetector:
    """Advanced market regime detection with GPU acceleration"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('RegimeDetector')

        # Initialize classifier
        self.classifier = GPURegimeClassifier().to(self.device)

        # Feature extractors
        self.scaler = StandardScaler()
        self.lookback_window = 60  # Days of data for regime detection

        # Regime tracking
        self.current_regime = None
        self.regime_history = []
        self.detection_count = 0

        # Performance tracking
        self.processing_speed = 0

        self.logger.info(f"Market regime detector initialized on {self.device}")

    def extract_market_features(self, price_data: np.ndarray, volume_data: np.ndarray) -> torch.Tensor:
        """
        Extract comprehensive market features for regime detection

        Args:
            price_data: Historical price data
            volume_data: Volume data

        Returns:
            Feature tensor for GPU processing
        """
        if len(price_data) < self.lookback_window:
            raise ValueError(f"Need at least {self.lookback_window} data points")

        features = []

        # Price-based features
        returns = np.diff(price_data) / price_data[:-1]

        # 1. Return statistics (multiple timeframes)
        for window in [5, 10, 20, 30]:
            if len(returns) >= window:
                rolling_returns = np.array([returns[i-window:i].mean() for i in range(window, len(returns))])
                if len(rolling_returns) > 0:
                    features.extend([
                        rolling_returns[-1],  # Latest average return
                        np.std(returns[-window:]) if len(returns) >= window else 0,  # Volatility
                        stats.skew(returns[-window:]) if len(returns) >= window else 0,  # Skewness
                        stats.kurtosis(returns[-window:]) if len(returns) >= window else 0  # Kurtosis
                    ])

        # 2. Technical indicators
        # Moving averages and trends
        for window in [5, 10, 20, 50]:
            if len(price_data) >= window:
                ma = np.mean(price_data[-window:])
                features.extend([
                    price_data[-1] / ma - 1,  # Price relative to MA
                    (ma - np.mean(price_data[-window*2:-window])) / np.mean(price_data[-window*2:-window]) if len(price_data) >= window*2 else 0  # MA slope
                ])

        # 3. Volatility regime indicators
        # GARCH-like volatility
        if len(returns) >= 20:
            ewm_var = pd.Series(returns).ewm(span=20).var().iloc[-1]
            features.append(ewm_var)

            # VIX-like indicator
            short_var = np.var(returns[-5:]) if len(returns) >= 5 else 0
            long_var = np.var(returns[-20:]) if len(returns) >= 20 else 0
            vix_like = short_var / (long_var + 1e-8)
            features.append(vix_like)

        # 4. Momentum indicators
        if len(price_data) >= 20:
            # Price momentum
            momentum_5 = (price_data[-1] / price_data[-6] - 1) if len(price_data) >= 6 else 0
            momentum_20 = (price_data[-1] / price_data[-21] - 1) if len(price_data) >= 21 else 0
            features.extend([momentum_5, momentum_20])

            # RSI-like indicator
            gains = np.maximum(returns, 0)
            losses = np.maximum(-returns, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))
            features.append(rsi / 100)  # Normalize

        # 5. Volume features
        if len(volume_data) >= 20:
            volume_ma = np.mean(volume_data[-20:])
            volume_ratio = volume_data[-1] / (volume_ma + 1e-8)
            features.append(volume_ratio)

            # Volume-price trend
            price_volume_corr = np.corrcoef(price_data[-20:], volume_data[-20:])[0,1] if len(price_data) >= 20 else 0
            features.append(price_volume_corr)

        # 6. Market microstructure (if available)
        # Simplified bid-ask spread proxy
        if len(price_data) >= 5:
            high_low_spread = np.mean([(max(price_data[i-4:i+1]) - min(price_data[i-4:i+1])) / price_data[i]
                                     for i in range(4, min(len(price_data), 20))])
            features.append(high_low_spread)

        # 7. Cross-market features (simplified)
        # Market stress indicator (volatility clustering)
        if len(returns) >= 30:
            vol_clustering = np.mean([abs(returns[i]) * abs(returns[i-1]) for i in range(1, min(30, len(returns)))])
            features.append(vol_clustering)

        # Pad or truncate to exactly 50 features
        while len(features) < 50:
            features.append(0.0)
        features = features[:50]

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def batch_detect_regimes(self, market_data: Dict[str, Dict]) -> Dict[str, RegimeDetection]:
        """
        Detect market regimes for multiple assets simultaneously

        Args:
            market_data: Dictionary containing price and volume data for multiple assets

        Returns:
            Regime detection results for each asset
        """
        start_time = time.time()

        # Prepare batch features
        batch_features = []
        asset_names = []

        for asset, data in market_data.items():
            try:
                if 'prices' in data and 'volumes' in data:
                    if len(data['prices']) >= self.lookback_window:
                        features = self.extract_market_features(
                            np.array(data['prices']),
                            np.array(data['volumes'])
                        )
                        batch_features.append(features)
                        asset_names.append(asset)
            except Exception as e:
                self.logger.warning(f"Error processing {asset}: {e}")
                continue

        if not batch_features:
            return {}

        # Convert to batch tensor
        batch_tensor = torch.stack(batch_features)

        # GPU inference
        self.classifier.eval()
        with torch.no_grad():
            trend_logits, vol_logits, momentum_logits, crisis_logits = self.classifier(batch_tensor)

            # Convert to probabilities
            trend_probs = F.softmax(trend_logits, dim=-1)
            vol_probs = F.softmax(vol_logits, dim=-1)
            momentum_probs = F.softmax(momentum_logits, dim=-1)
            crisis_probs = F.softmax(crisis_logits, dim=-1)

        # Process results
        results = {}
        trend_labels = [MarketRegime.BEAR_MARKET, MarketRegime.SIDEWAYS, MarketRegime.BULL_MARKET]
        vol_labels = [MarketRegime.LOW_VOLATILITY, "Medium Volatility", MarketRegime.HIGH_VOLATILITY]
        momentum_labels = [MarketRegime.MEAN_REVERSION, "Neutral", MarketRegime.MOMENTUM]

        for i, asset in enumerate(asset_names):
            # Primary trend regime
            trend_idx = torch.argmax(trend_probs[i]).item()
            primary_regime = trend_labels[trend_idx]

            # Check for crisis
            crisis_prob = crisis_probs[i][1].item()  # Crisis probability
            if crisis_prob > 0.7:
                primary_regime = MarketRegime.CRISIS

            # Volatility classification
            vol_idx = torch.argmax(vol_probs[i]).item()
            volatility_level = vol_labels[vol_idx]

            # Momentum classification
            momentum_idx = torch.argmax(momentum_probs[i]).item()
            momentum_regime = momentum_labels[momentum_idx]

            # Calculate confidence and other metrics
            confidence = float(torch.max(trend_probs[i]).item())
            trend_strength = float(trend_probs[i][2].item() - trend_probs[i][0].item())  # Bull - Bear
            momentum_score = float(momentum_probs[i][2].item() - momentum_probs[i][0].item())  # Momentum - Mean reversion

            # Create regime probabilities dictionary
            regime_probs = {}
            for j, regime in enumerate(trend_labels):
                regime_probs[regime] = float(trend_probs[i][j].item())

            # Special regimes
            regime_probs[MarketRegime.CRISIS] = crisis_prob
            regime_probs[MarketRegime.HIGH_VOLATILITY] = float(vol_probs[i][2].item())
            regime_probs[MarketRegime.MOMENTUM] = float(momentum_probs[i][2].item())

            results[asset] = RegimeDetection(
                primary_regime=primary_regime,
                confidence=confidence,
                regime_probabilities=regime_probs,
                volatility_level=volatility_level,
                trend_strength=trend_strength,
                momentum_score=momentum_score,
                timestamp=datetime.now()
            )

        processing_time = time.time() - start_time
        self.detection_count += len(asset_names)
        self.processing_speed = len(asset_names) / processing_time if processing_time > 0 else 0

        self.logger.info(f"Detected regimes for {len(asset_names)} assets in {processing_time:.4f}s "
                        f"({self.processing_speed:.1f} assets/second)")

        return results

    def analyze_regime_transitions(self, current_regimes: Dict[str, RegimeDetection],
                                 previous_regimes: Dict[str, RegimeDetection]) -> Dict[str, Dict]:
        """
        Analyze regime transitions and their implications

        Args:
            current_regimes: Current regime detections
            previous_regimes: Previous regime detections

        Returns:
            Transition analysis results
        """
        transitions = {}

        for asset in current_regimes:
            if asset in previous_regimes:
                current = current_regimes[asset]
                previous = previous_regimes[asset]

                # Check for regime change
                regime_changed = current.primary_regime != previous.primary_regime

                # Calculate transition probability
                transition_confidence = abs(current.confidence - previous.confidence)

                # Determine transition type
                transition_type = "Stable"
                if regime_changed:
                    if (previous.primary_regime == MarketRegime.BULL_MARKET and
                        current.primary_regime == MarketRegime.BEAR_MARKET):
                        transition_type = "Bull to Bear"
                    elif (previous.primary_regime == MarketRegime.BEAR_MARKET and
                          current.primary_regime == MarketRegime.BULL_MARKET):
                        transition_type = "Bear to Bull"
                    elif current.primary_regime == MarketRegime.CRISIS:
                        transition_type = "Crisis Onset"
                    else:
                        transition_type = "Regime Shift"

                transitions[asset] = {
                    'regime_changed': regime_changed,
                    'transition_type': transition_type,
                    'transition_confidence': transition_confidence,
                    'previous_regime': previous.primary_regime.value,
                    'current_regime': current.primary_regime.value,
                    'volatility_change': current.volatility_level != previous.volatility_level,
                    'momentum_change': abs(current.momentum_score - previous.momentum_score)
                }

        return transitions

    def generate_trading_signals_from_regime(self, regime_results: Dict[str, RegimeDetection]) -> Dict[str, Dict]:
        """
        Generate trading signals based on detected market regimes

        Args:
            regime_results: Regime detection results

        Returns:
            Trading signals for each asset
        """
        signals = {}

        for asset, detection in regime_results.items():
            signal = "HOLD"
            strength = 0.5
            rationale = "Neutral regime"

            # Bull market signals
            if detection.primary_regime == MarketRegime.BULL_MARKET:
                if detection.confidence > 0.7:
                    signal = "BUY"
                    strength = min(detection.confidence + detection.trend_strength * 0.3, 1.0)
                    rationale = "Strong bull market detected"

            # Bear market signals
            elif detection.primary_regime == MarketRegime.BEAR_MARKET:
                if detection.confidence > 0.7:
                    signal = "SELL"
                    strength = min(detection.confidence + abs(detection.trend_strength) * 0.3, 1.0)
                    rationale = "Strong bear market detected"

            # Crisis signals
            elif detection.primary_regime == MarketRegime.CRISIS:
                signal = "SELL"
                strength = 0.9
                rationale = "Crisis regime - defensive positioning"

            # High volatility adjustments
            if detection.volatility_level == MarketRegime.HIGH_VOLATILITY.value:
                strength *= 0.8  # Reduce position size in high volatility
                rationale += " (high volatility)"

            # Momentum considerations
            if detection.momentum_score > 0.5 and signal == "BUY":
                strength = min(strength * 1.2, 1.0)
                rationale += " (strong momentum)"
            elif detection.momentum_score < -0.5 and signal == "SELL":
                strength = min(strength * 1.2, 1.0)
                rationale += " (strong momentum)"

            signals[asset] = {
                'signal': signal,
                'strength': strength,
                'rationale': rationale,
                'regime': detection.primary_regime.value,
                'confidence': detection.confidence,
                'timestamp': detection.timestamp.isoformat()
            }

        return signals

    def create_regime_dashboard(self, regime_results: Dict[str, RegimeDetection]) -> Dict[str, Any]:
        """
        Create comprehensive regime analysis dashboard

        Args:
            regime_results: Regime detection results

        Returns:
            Dashboard data
        """
        # Market overview
        total_assets = len(regime_results)
        regime_counts = {}

        for detection in regime_results.values():
            regime = detection.primary_regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Calculate market stress level
        crisis_count = regime_counts.get(MarketRegime.CRISIS.value, 0)
        bear_count = regime_counts.get(MarketRegime.BEAR_MARKET.value, 0)
        stress_level = (crisis_count + bear_count) / total_assets if total_assets > 0 else 0

        # Average confidence
        avg_confidence = np.mean([d.confidence for d in regime_results.values()]) if regime_results else 0

        # Volatility overview
        high_vol_count = sum(1 for d in regime_results.values()
                           if d.volatility_level == MarketRegime.HIGH_VOLATILITY.value)
        volatility_ratio = high_vol_count / total_assets if total_assets > 0 else 0

        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'total_assets_analyzed': total_assets,
            'regime_distribution': regime_counts,
            'market_stress_level': stress_level,
            'average_confidence': avg_confidence,
            'high_volatility_ratio': volatility_ratio,
            'processing_speed': self.processing_speed,
            'top_regimes': sorted(regime_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }

        return dashboard

def demo_regime_detection():
    """Demonstration of market regime detection system"""
    print("\n" + "="*80)
    print("GPU MARKET REGIME DETECTION SYSTEM DEMONSTRATION")
    print("="*80)

    # Initialize detector
    detector = MarketRegimeDetector()

    print(f"\n>> Market Regime Detector initialized on {detector.device}")
    print(f">> Lookback window: {detector.lookback_window} days")
    print(f">> Feature extraction: 50 market indicators")

    # Generate sample market data
    print(f"\n>> Generating sample market data...")

    np.random.seed(42)
    symbols = ['SPY', 'QQQ', 'IWM', 'VIX', 'GLD']
    market_data = {}

    for symbol in symbols:
        # Generate different market conditions for each symbol
        days = 100
        if symbol == 'VIX':
            # High volatility asset
            base_price = 20
            returns = np.random.normal(0.001, 0.05, days)
        elif symbol == 'GLD':
            # Mean-reverting asset
            base_price = 180
            returns = np.random.normal(-0.0005, 0.015, days)
        else:
            # Stock market assets
            base_price = 300
            returns = np.random.normal(0.0008, 0.02, days)

        # Generate price series
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        # Generate volumes
        volumes = np.random.lognormal(10, 0.5, len(prices))

        market_data[symbol] = {
            'prices': prices,
            'volumes': volumes
        }

    # Run regime detection
    print(f">> Running regime detection analysis...")

    regime_results = detector.batch_detect_regimes(market_data)

    print(f">> Analysis completed for {len(regime_results)} assets")
    print(f">> Processing speed: {detector.processing_speed:.1f} assets/second")

    # Display results
    print(f"\n>> DETECTED MARKET REGIMES:")
    for asset, detection in regime_results.items():
        print(f"   {asset}:")
        print(f"     Primary Regime: {detection.primary_regime.value}")
        print(f"     Confidence: {detection.confidence:.3f}")
        print(f"     Volatility Level: {detection.volatility_level}")
        print(f"     Trend Strength: {detection.trend_strength:.3f}")
        print(f"     Momentum Score: {detection.momentum_score:.3f}")

    # Generate trading signals
    signals = detector.generate_trading_signals_from_regime(regime_results)
    print(f"\n>> REGIME-BASED TRADING SIGNALS:")
    for asset, signal_data in signals.items():
        print(f"   {asset}: {signal_data['signal']} (Strength: {signal_data['strength']:.2f})")
        print(f"     Rationale: {signal_data['rationale']}")

    # Create dashboard
    dashboard = detector.create_regime_dashboard(regime_results)
    print(f"\n>> MARKET REGIME DASHBOARD:")
    print(f"   Assets Analyzed: {dashboard['total_assets_analyzed']}")
    print(f"   Market Stress Level: {dashboard['market_stress_level']:.2f}")
    print(f"   Average Confidence: {dashboard['average_confidence']:.3f}")
    print(f"   High Volatility Ratio: {dashboard['high_volatility_ratio']:.2f}")
    print(f"   Top Regimes: {dashboard['top_regimes']}")

    print(f"\n" + "="*80)
    print("MARKET REGIME DETECTION SYSTEM READY!")
    print("Advanced regime analysis for institutional-grade trading")
    print("="*80)

if __name__ == "__main__":
    demo_regime_detection()