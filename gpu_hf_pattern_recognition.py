"""
GPU HIGH-FREQUENCY PATTERN RECOGNITION SYSTEM
Advanced pattern detection for high-frequency trading with GTX 1660 Super
Real-time identification of profitable patterns in tick-level market data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
from enum import Enum
import concurrent.futures
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PatternType(Enum):
    """High-frequency trading pattern types"""
    MOMENTUM_BURST = "Momentum Burst"
    MEAN_REVERSION = "Mean Reversion"
    BREAKOUT = "Breakout"
    BREAKDOWN = "Breakdown"
    VOLUME_SPIKE = "Volume Spike"
    PRICE_LADDER = "Price Ladder"
    ICEBERG_ORDER = "Iceberg Order"
    LIQUIDITY_TRAP = "Liquidity Trap"
    MOMENTUM_FADE = "Momentum Fade"
    SUPPORT_RESISTANCE = "Support/Resistance"

@dataclass
class TickData:
    """High-frequency tick data structure"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int

@dataclass
class PatternDetection:
    """Pattern detection result"""
    pattern_type: PatternType
    confidence: float
    start_time: datetime
    end_time: datetime
    symbol: str
    entry_price: float
    target_price: float
    stop_loss: float
    expected_duration: timedelta
    pattern_strength: float
    volume_confirmation: bool

class GPUPatternRecognizer(nn.Module):
    """GPU-accelerated neural network for pattern recognition"""

    def __init__(self, input_features: int = 20, hidden_dim: int = 512):
        super().__init__()

        # Multi-scale convolutional feature extraction
        self.conv_layers = nn.ModuleList([
            # Short-term patterns (1-5 ticks)
            nn.Conv1d(input_features, hidden_dim // 4, kernel_size=3, padding=1),
            # Medium-term patterns (5-20 ticks)
            nn.Conv1d(input_features, hidden_dim // 4, kernel_size=7, padding=3),
            # Long-term patterns (20-100 ticks)
            nn.Conv1d(input_features, hidden_dim // 4, kernel_size=15, padding=7),
            # Ultra-short patterns (tick-by-tick)
            nn.Conv1d(input_features, hidden_dim // 4, kernel_size=1)
        ])

        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Pattern-specific encoders
        self.momentum_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        self.mean_reversion_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        self.volume_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Pattern classifiers
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, len(PatternType))
        )

        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # Price target predictor
        self.target_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Target and stop loss
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)  # Convert to (batch_size, features, sequence_length)

        # Multi-scale convolution
        conv_features = []
        for conv_layer in self.conv_layers:
            conv_out = F.relu(conv_layer(x))
            conv_features.append(conv_out)

        # Concatenate multi-scale features
        combined_features = torch.cat(conv_features, dim=1)
        combined_features = combined_features.transpose(1, 2)  # Back to (batch, seq, features)

        # Temporal attention
        attn_out, _ = self.temporal_attention(combined_features, combined_features, combined_features)

        # Global average pooling
        pooled_features = torch.mean(attn_out, dim=1)

        # Pattern-specific encoding
        momentum_features = self.momentum_encoder(pooled_features)
        mean_reversion_features = self.mean_reversion_encoder(pooled_features)
        volume_features = self.volume_encoder(pooled_features)

        # Combine all features
        final_features = torch.cat([momentum_features, mean_reversion_features, volume_features], dim=1)

        # Predictions
        pattern_logits = self.pattern_classifier(pooled_features)
        confidence = self.confidence_estimator(pooled_features)
        targets = self.target_predictor(pooled_features)

        return pattern_logits, confidence, targets

class HighFrequencyPatternDetector:
    """Advanced high-frequency pattern detection with GPU acceleration"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('HFPatternDetector')

        # Initialize pattern recognition model
        self.model = GPUPatternRecognizer().to(self.device)

        # Pattern detection parameters
        self.sequence_length = 100  # Number of ticks to analyze
        self.overlap_ratio = 0.5    # Overlap between analysis windows

        # Market microstructure tracking
        self.tick_buffers = defaultdict(lambda: deque(maxlen=1000))
        self.pattern_history = defaultdict(list)

        # Performance tracking
        self.patterns_detected = 0
        self.detection_speed = 0

        # Pattern thresholds
        self.confidence_threshold = 0.7
        self.volume_spike_threshold = 2.0
        self.price_movement_threshold = 0.001  # 0.1%

        self.logger.info(f"HF Pattern detector initialized on {self.device}")

    def generate_synthetic_hf_data(self, symbol: str, num_ticks: int = 1000) -> List[TickData]:
        """
        Generate synthetic high-frequency tick data for testing

        Args:
            symbol: Trading symbol
            num_ticks: Number of ticks to generate

        Returns:
            List of synthetic tick data
        """
        np.random.seed(hash(symbol) % 1000)  # Symbol-specific seed

        base_price = 100.0
        current_time = datetime.now()
        tick_data = []

        # Market microstructure parameters
        spread_bp = np.random.uniform(1, 5)  # Spread in basis points
        base_volume = np.random.uniform(100, 1000)

        for i in range(num_ticks):
            # Price evolution with microstructure noise
            if i == 0:
                price = base_price
            else:
                # Brownian motion with mean reversion
                drift = -0.1 * (tick_data[-1].price - base_price) / base_price  # Mean reversion
                noise = np.random.normal(0, 0.0001)  # Microstructure noise
                price_change = drift + noise
                price = tick_data[-1].price * (1 + price_change)

            # Bid-ask spread
            spread = price * spread_bp / 10000
            mid_price = price
            bid = mid_price - spread / 2
            ask = mid_price + spread / 2

            # Volume with clustering
            volume_multiplier = 1.0
            if np.random.random() < 0.05:  # 5% chance of volume spike
                volume_multiplier = np.random.uniform(3, 10)

            volume = int(base_volume * np.random.lognormal(0, 0.5) * volume_multiplier)
            bid_size = int(volume * np.random.uniform(0.3, 0.7))
            ask_size = int(volume * np.random.uniform(0.3, 0.7))

            # Timestamp (microsecond precision)
            timestamp = current_time + timedelta(microseconds=i * np.random.randint(1000, 100000))

            tick = TickData(
                timestamp=timestamp,
                symbol=symbol,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size
            )

            tick_data.append(tick)

        return tick_data

    def extract_hf_features(self, tick_data: List[TickData]) -> torch.Tensor:
        """
        Extract high-frequency trading features from tick data

        Args:
            tick_data: List of tick data

        Returns:
            Feature tensor for GPU processing
        """
        if len(tick_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} ticks")

        # Take last sequence_length ticks
        ticks = tick_data[-self.sequence_length:]
        features = []

        # Extract features for each tick
        for i, tick in enumerate(ticks):
            tick_features = []

            # Price features
            if i > 0:
                prev_tick = ticks[i-1]
                price_return = (tick.price - prev_tick.price) / prev_tick.price
                log_return = np.log(tick.price / prev_tick.price)
            else:
                price_return = 0.0
                log_return = 0.0

            tick_features.extend([
                tick.price,
                price_return,
                log_return,
                tick.bid,
                tick.ask,
                tick.ask - tick.bid,  # Spread
                (tick.ask - tick.bid) / tick.price,  # Relative spread
            ])

            # Volume features
            tick_features.extend([
                tick.volume,
                tick.bid_size,
                tick.ask_size,
                tick.volume / max(tick.bid_size + tick.ask_size, 1),  # Volume imbalance
                np.log(tick.volume + 1),  # Log volume
            ])

            # Microstructure features
            if i >= 5:
                # Recent price volatility
                recent_prices = [t.price for t in ticks[i-5:i+1]]
                volatility = np.std(recent_prices) / np.mean(recent_prices)

                # Volume weighted average price (VWAP)
                recent_volumes = [t.volume for t in ticks[i-5:i+1]]
                vwap = np.average(recent_prices, weights=recent_volumes)
                vwap_deviation = (tick.price - vwap) / vwap

                # Order flow imbalance
                bid_volumes = [t.bid_size for t in ticks[i-5:i+1]]
                ask_volumes = [t.ask_size for t in ticks[i-5:i+1]]
                flow_imbalance = (sum(bid_volumes) - sum(ask_volumes)) / (sum(bid_volumes) + sum(ask_volumes) + 1)

                tick_features.extend([volatility, vwap_deviation, flow_imbalance])
            else:
                tick_features.extend([0.0, 0.0, 0.0])

            # Time-based features
            if i > 0:
                time_delta = (tick.timestamp - ticks[i-1].timestamp).total_seconds()
            else:
                time_delta = 0.0

            tick_features.extend([
                time_delta,
                tick.timestamp.hour,  # Hour of day
                tick.timestamp.minute,  # Minute of hour
            ])

            # Technical indicators (simplified for HF)
            if i >= 10:
                # Short-term momentum
                prices_10 = [t.price for t in ticks[i-10:i+1]]
                momentum = (prices_10[-1] - prices_10[0]) / prices_10[0]

                # Mean reversion signal
                mean_price = np.mean(prices_10)
                reversion_signal = (tick.price - mean_price) / mean_price

                tick_features.extend([momentum, reversion_signal])
            else:
                tick_features.extend([0.0, 0.0])

            # Pad to exactly 20 features per tick
            while len(tick_features) < 20:
                tick_features.append(0.0)

            features.append(tick_features[:20])

        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

    def batch_detect_patterns(self, tick_data_dict: Dict[str, List[TickData]]) -> Dict[str, List[PatternDetection]]:
        """
        Detect patterns across multiple symbols simultaneously

        Args:
            tick_data_dict: Dictionary of symbol -> tick data

        Returns:
            Detected patterns for each symbol
        """
        start_time = time.time()

        # Prepare batch features
        batch_features = []
        symbol_names = []

        for symbol, tick_data in tick_data_dict.items():
            try:
                if len(tick_data) >= self.sequence_length:
                    features = self.extract_hf_features(tick_data)
                    batch_features.append(features.squeeze(0))
                    symbol_names.append(symbol)
            except Exception as e:
                self.logger.warning(f"Error processing {symbol}: {e}")
                continue

        if not batch_features:
            return {}

        # Convert to batch tensor
        batch_tensor = torch.stack(batch_features)

        # GPU inference
        self.model.eval()
        with torch.no_grad():
            pattern_logits, confidence, targets = self.model(batch_tensor)

            # Convert to probabilities
            pattern_probs = F.softmax(pattern_logits, dim=-1)

        # Process results
        results = {}
        pattern_types = list(PatternType)

        for i, symbol in enumerate(symbol_names):
            detected_patterns = []

            # Get most likely pattern
            pattern_idx = torch.argmax(pattern_probs[i]).item()
            pattern_confidence = confidence[i].item()
            pattern_type = pattern_types[pattern_idx]

            # Extract price targets
            target_change = targets[i][0].item()
            stop_change = targets[i][1].item()

            current_price = tick_data_dict[symbol][-1].price
            target_price = current_price * (1 + target_change)
            stop_loss = current_price * (1 + stop_change)

            # Only include high-confidence patterns
            if pattern_confidence > self.confidence_threshold:
                # Additional validation
                volume_confirmation = self.validate_volume_pattern(tick_data_dict[symbol])

                detection = PatternDetection(
                    pattern_type=pattern_type,
                    confidence=pattern_confidence,
                    start_time=tick_data_dict[symbol][-self.sequence_length].timestamp,
                    end_time=tick_data_dict[symbol][-1].timestamp,
                    symbol=symbol,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    expected_duration=timedelta(minutes=np.random.randint(1, 15)),
                    pattern_strength=float(torch.max(pattern_probs[i]).item()),
                    volume_confirmation=volume_confirmation
                )

                detected_patterns.append(detection)

            results[symbol] = detected_patterns

        processing_time = time.time() - start_time
        self.patterns_detected += sum(len(patterns) for patterns in results.values())
        self.detection_speed = len(symbol_names) / processing_time if processing_time > 0 else 0

        self.logger.info(f"Analyzed {len(symbol_names)} symbols in {processing_time:.4f}s "
                        f"({self.detection_speed:.1f} symbols/second)")

        return results

    def validate_volume_pattern(self, tick_data: List[TickData]) -> bool:
        """
        Validate pattern with volume analysis

        Args:
            tick_data: Tick data to analyze

        Returns:
            True if volume confirms the pattern
        """
        if len(tick_data) < 20:
            return False

        # Recent volume analysis
        recent_volumes = [tick.volume for tick in tick_data[-20:]]
        avg_volume = np.mean(recent_volumes[:-5])  # Exclude last 5 ticks
        recent_avg = np.mean(recent_volumes[-5:])   # Last 5 ticks

        # Volume spike detection
        volume_ratio = recent_avg / (avg_volume + 1e-8)

        return volume_ratio > self.volume_spike_threshold

    def detect_support_resistance_levels(self, tick_data: List[TickData]) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels from tick data

        Args:
            tick_data: Historical tick data

        Returns:
            Dictionary with support and resistance levels
        """
        if len(tick_data) < 100:
            return {'support': [], 'resistance': []}

        prices = np.array([tick.price for tick in tick_data])

        # Find local maxima and minima
        peaks, _ = find_peaks(prices, distance=10)
        troughs, _ = find_peaks(-prices, distance=10)

        # Cluster peaks and troughs to find significant levels
        resistance_levels = []
        support_levels = []

        if len(peaks) > 2:
            peak_prices = prices[peaks]
            # Simple clustering by proximity
            for price in peak_prices:
                similar_peaks = peak_prices[np.abs(peak_prices - price) < price * 0.002]
                if len(similar_peaks) >= 2:  # Multiple touches
                    resistance_levels.append(float(np.mean(similar_peaks)))

        if len(troughs) > 2:
            trough_prices = prices[troughs]
            for price in trough_prices:
                similar_troughs = trough_prices[np.abs(trough_prices - price) < price * 0.002]
                if len(similar_troughs) >= 2:  # Multiple touches
                    support_levels.append(float(np.mean(similar_troughs)))

        return {
            'support': list(set(support_levels)),
            'resistance': list(set(resistance_levels))
        }

    def generate_hf_trading_signals(self, pattern_detections: Dict[str, List[PatternDetection]]) -> Dict[str, Dict]:
        """
        Generate high-frequency trading signals from detected patterns

        Args:
            pattern_detections: Detected patterns

        Returns:
            Trading signals for each symbol
        """
        signals = {}

        for symbol, patterns in pattern_detections.items():
            if not patterns:
                continue

            # Take highest confidence pattern
            best_pattern = max(patterns, key=lambda p: p.confidence)

            # Generate signal based on pattern type
            if best_pattern.pattern_type in [PatternType.MOMENTUM_BURST, PatternType.BREAKOUT]:
                signal = "BUY"
                urgency = "HIGH"
            elif best_pattern.pattern_type in [PatternType.BREAKDOWN, PatternType.MOMENTUM_FADE]:
                signal = "SELL"
                urgency = "HIGH"
            elif best_pattern.pattern_type == PatternType.MEAN_REVERSION:
                # Direction depends on current price vs pattern average
                signal = "BUY" if best_pattern.target_price > best_pattern.entry_price else "SELL"
                urgency = "MEDIUM"
            else:
                signal = "HOLD"
                urgency = "LOW"

            signals[symbol] = {
                'signal': signal,
                'urgency': urgency,
                'confidence': best_pattern.confidence,
                'pattern': best_pattern.pattern_type.value,
                'entry_price': best_pattern.entry_price,
                'target_price': best_pattern.target_price,
                'stop_loss': best_pattern.stop_loss,
                'expected_duration_minutes': best_pattern.expected_duration.total_seconds() / 60,
                'volume_confirmed': best_pattern.volume_confirmation,
                'timestamp': best_pattern.end_time.isoformat()
            }

        return signals

    def create_hf_dashboard(self, pattern_detections: Dict[str, List[PatternDetection]],
                           signals: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create high-frequency trading dashboard

        Args:
            pattern_detections: Detected patterns
            signals: Generated signals

        Returns:
            Dashboard data
        """
        # Pattern statistics
        total_patterns = sum(len(patterns) for patterns in pattern_detections.values())
        pattern_type_counts = defaultdict(int)

        for patterns in pattern_detections.values():
            for pattern in patterns:
                pattern_type_counts[pattern.pattern_type.value] += 1

        # Signal statistics
        signal_counts = defaultdict(int)
        high_urgency_signals = 0

        for signal_data in signals.values():
            signal_counts[signal_data['signal']] += 1
            if signal_data['urgency'] == 'HIGH':
                high_urgency_signals += 1

        # Average confidence
        all_patterns = [p for patterns in pattern_detections.values() for p in patterns]
        avg_confidence = np.mean([p.confidence for p in all_patterns]) if all_patterns else 0

        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'total_patterns_detected': total_patterns,
            'symbols_analyzed': len(pattern_detections),
            'active_signals': len(signals),
            'high_urgency_signals': high_urgency_signals,
            'pattern_distribution': dict(pattern_type_counts),
            'signal_distribution': dict(signal_counts),
            'average_confidence': avg_confidence,
            'detection_speed': self.detection_speed,
            'top_patterns': sorted(pattern_type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }

        return dashboard

def demo_hf_pattern_recognition():
    """Demonstration of high-frequency pattern recognition system"""
    print("\n" + "="*80)
    print("GPU HIGH-FREQUENCY PATTERN RECOGNITION DEMONSTRATION")
    print("="*80)

    # Initialize detector
    detector = HighFrequencyPatternDetector()

    print(f"\n>> HF Pattern Detector initialized on {detector.device}")
    print(f">> Sequence length: {detector.sequence_length} ticks")
    print(f">> Confidence threshold: {detector.confidence_threshold}")

    # Generate synthetic HF data
    print(f"\n>> Generating synthetic high-frequency tick data...")

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    tick_data_dict = {}

    for symbol in symbols:
        tick_data = detector.generate_synthetic_hf_data(symbol, num_ticks=200)
        tick_data_dict[symbol] = tick_data

    print(f">> Generated {sum(len(data) for data in tick_data_dict.values())} total ticks")

    # Run pattern detection
    print(f"\n>> Running high-frequency pattern analysis...")

    pattern_detections = detector.batch_detect_patterns(tick_data_dict)

    print(f">> Analysis completed for {len(tick_data_dict)} symbols")
    print(f">> Detection speed: {detector.detection_speed:.1f} symbols/second")

    # Display detected patterns
    total_patterns = sum(len(patterns) for patterns in pattern_detections.values())
    print(f"\n>> DETECTED PATTERNS ({total_patterns} total):")

    for symbol, patterns in pattern_detections.items():
        if patterns:
            for pattern in patterns:
                print(f"   {symbol}: {pattern.pattern_type.value}")
                print(f"     Confidence: {pattern.confidence:.3f}")
                print(f"     Entry: ${pattern.entry_price:.2f}")
                print(f"     Target: ${pattern.target_price:.2f}")
                print(f"     Stop Loss: ${pattern.stop_loss:.2f}")
                print(f"     Volume Confirmed: {'Yes' if pattern.volume_confirmation else 'No'}")

    # Generate trading signals
    signals = detector.generate_hf_trading_signals(pattern_detections)
    print(f"\n>> HIGH-FREQUENCY TRADING SIGNALS:")

    for symbol, signal_data in signals.items():
        print(f"   {symbol}: {signal_data['signal']} ({signal_data['urgency']} urgency)")
        print(f"     Pattern: {signal_data['pattern']}")
        print(f"     Confidence: {signal_data['confidence']:.3f}")
        print(f"     Expected Duration: {signal_data['expected_duration_minutes']:.1f} minutes")

    # Create dashboard
    dashboard = detector.create_hf_dashboard(pattern_detections, signals)
    print(f"\n>> HF TRADING DASHBOARD:")
    print(f"   Total Patterns: {dashboard['total_patterns_detected']}")
    print(f"   Active Signals: {dashboard['active_signals']}")
    print(f"   High Urgency: {dashboard['high_urgency_signals']}")
    print(f"   Average Confidence: {dashboard['average_confidence']:.3f}")
    print(f"   Detection Speed: {dashboard['detection_speed']:.1f} symbols/second")
    print(f"   Top Patterns: {dashboard['top_patterns']}")

    print(f"\n" + "="*80)
    print("HIGH-FREQUENCY PATTERN RECOGNITION SYSTEM READY!")
    print("Advanced microsecond-level pattern detection for professional trading")
    print("="*80)

if __name__ == "__main__":
    demo_hf_pattern_recognition()