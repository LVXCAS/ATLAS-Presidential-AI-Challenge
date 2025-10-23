"""
Mean Reversion Trading Agent Demo

This script demonstrates the capabilities of the Mean Reversion Trading Agent,
including Bollinger Band reversions, Z-score analysis, pairs trading,
Fibonacci extension targets, and sentiment divergence detection.
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mean_reversion_agent import (
    MeanReversionTradingAgent,
    MarketData,
    SentimentData,
    SignalType,
    MarketRegime,
    analyze_mean_reversion_sync
)


class MeanReversionDemo:
    """Demo class for Mean Reversion Trading Agent"""
    
    def __init__(self):
        self.agent = MeanReversionTradingAgent()
        print("[BOT] Mean Reversion Trading Agent Demo")
        print("=" * 50)
    
    def generate_mean_reverting_data(self, symbol: str, length: int = 100, 
                                   base_price: float = 100.0, 
                                   mean_reversion_strength: float = 0.1) -> List[MarketData]:
        """Generate synthetic mean-reverting market data"""
        np.random.seed(42)  # For reproducible results
        data = []
        current_price = base_price
        
        print(f"[CHART] Generating {length} days of mean-reverting data for {symbol}")
        print(f"   Base price: ${base_price:.2f}")
        print(f"   Mean reversion strength: {mean_reversion_strength}")
        
        for i in range(length):
            # Mean reversion component (pulls price back to base_price)
            mean_reversion = (base_price - current_price) * mean_reversion_strength
            
            # Random shock
            random_shock = np.random.normal(0, 1.5)
            
            # Combine components
            price_change = mean_reversion + random_shock
            current_price = max(50, min(150, current_price + price_change))
            
            # Generate OHLC data
            open_price = current_price + np.random.normal(0, 0.3)
            high_price = current_price + abs(np.random.normal(0, 1.0))
            low_price = current_price - abs(np.random.normal(0, 1.0))
            close_price = current_price
            volume = np.random.randint(1000000, 5000000)
            
            data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(days=length-i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            ))
        
        return data
    
    def generate_cointegrated_pair(self, symbol_a: str, symbol_b: str, 
                                 length: int = 100) -> Tuple[List[MarketData], List[MarketData]]:
        """Generate cointegrated pair for pairs trading demo"""
        np.random.seed(123)
        
        # Generate common trend
        common_trend = np.cumsum(np.random.normal(0, 0.2, length))
        
        # Generate cointegrated series
        base_a, base_b = 100.0, 50.0
        prices_a = base_a + common_trend + np.random.normal(0, 1.0, length)
        prices_b = base_b + 0.5 * common_trend + np.random.normal(0, 0.8, length)
        
        data_a, data_b = [], []
        
        for i in range(length):
            # Data for symbol A
            price_a = prices_a[i]
            data_a.append(MarketData(
                symbol=symbol_a,
                timestamp=datetime.now() - timedelta(days=length-i),
                open=price_a + np.random.normal(0, 0.2),
                high=price_a + abs(np.random.normal(0, 0.5)),
                low=price_a - abs(np.random.normal(0, 0.5)),
                close=price_a,
                volume=np.random.randint(1000000, 3000000)
            ))
            
            # Data for symbol B
            price_b = prices_b[i]
            data_b.append(MarketData(
                symbol=symbol_b,
                timestamp=datetime.now() - timedelta(days=length-i),
                open=price_b + np.random.normal(0, 0.2),
                high=price_b + abs(np.random.normal(0, 0.5)),
                low=price_b - abs(np.random.normal(0, 0.5)),
                close=price_b,
                volume=np.random.randint(800000, 2000000)
            ))
        
        return data_a, data_b
    
    def generate_sentiment_data(self, symbol: str, sentiment_bias: float = 0.0) -> SentimentData:
        """Generate synthetic sentiment data"""
        return SentimentData(
            symbol=symbol,
            overall_sentiment=sentiment_bias + np.random.normal(0, 0.3),
            confidence=np.random.uniform(0.6, 0.9),
            news_count=np.random.randint(10, 30),
            social_sentiment=sentiment_bias + np.random.normal(0, 0.4),
            timestamp=datetime.now()
        )
    
    def demo_bollinger_band_reversion(self):
        """Demonstrate Bollinger Band reversion detection"""
        print("\n[TARGET] Demo 1: Bollinger Band Reversion Detection")
        print("-" * 45)
        
        # Generate data with extreme moves (touches bands)
        market_data = []
        base_price = 100.0
        
        # Create pattern that touches Bollinger Bands
        for i in range(60):
            if 20 <= i <= 25:  # Spike up (touch upper band)
                price = base_price + 8 + np.random.normal(0, 0.5)
            elif 40 <= i <= 45:  # Spike down (touch lower band)
                price = base_price - 8 + np.random.normal(0, 0.5)
            else:
                price = base_price + np.random.normal(0, 2)
            
            market_data.append(MarketData(
                symbol="AAPL",
                timestamp=datetime.now() - timedelta(days=60-i),
                open=price + np.random.normal(0, 0.2),
                high=price + abs(np.random.normal(0, 0.5)),
                low=price - abs(np.random.normal(0, 0.5)),
                close=price,
                volume=np.random.randint(2000000, 4000000)
            ))
        
        # Generate signal
        signal = self.agent.generate_signal_sync("AAPL", market_data)
        
        if signal:
            print(f"[OK] Signal Generated: {signal.signal_type.value.upper()}")
            print(f"   Signal Value: {signal.value:.3f}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Bollinger Signals: {len(signal.bollinger_signals)}")
            
            if signal.bollinger_signals:
                print("   Bollinger Band Analysis:")
                for bb_signal in signal.bollinger_signals[:3]:
                    print(f"     • {bb_signal.indicator}: {bb_signal.explanation}")
        else:
            print("[X] No signal generated")
    
    def demo_zscore_analysis(self):
        """Demonstrate Z-score mean reversion analysis"""
        print("\n[UP] Demo 2: Z-Score Mean Reversion Analysis")
        print("-" * 42)
        
        # Generate data with extreme Z-scores
        market_data = []
        base_price = 100.0
        
        # Create stable period followed by extreme moves
        for i in range(80):
            if i < 40:  # Stable period
                price = base_price + np.random.normal(0, 1)
            elif 40 <= i <= 50:  # Extreme high period
                price = base_price + 15 + np.random.normal(0, 1)
            elif 60 <= i <= 70:  # Extreme low period
                price = base_price - 12 + np.random.normal(0, 1)
            else:  # Return to normal
                price = base_price + np.random.normal(0, 2)
            
            market_data.append(MarketData(
                symbol="TSLA",
                timestamp=datetime.now() - timedelta(days=80-i),
                open=price + np.random.normal(0, 0.3),
                high=price + abs(np.random.normal(0, 0.8)),
                low=price - abs(np.random.normal(0, 0.8)),
                close=price,
                volume=np.random.randint(3000000, 8000000)
            ))
        
        signal = self.agent.generate_signal_sync("TSLA", market_data)
        
        if signal:
            print(f"[OK] Signal Generated: {signal.signal_type.value.upper()}")
            print(f"   Signal Value: {signal.value:.3f}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Z-Score Signals: {len(signal.zscore_signals)}")
            
            if signal.zscore_signals:
                print("   Z-Score Analysis:")
                for z_signal in signal.zscore_signals[:3]:
                    print(f"     • {z_signal.indicator}: {z_signal.explanation}")
        else:
            print("[X] No signal generated")
    
    def demo_pairs_trading(self):
        """Demonstrate pairs trading with cointegration"""
        print("\n[INFO] Demo 3: Pairs Trading with Cointegration")
        print("-" * 41)
        
        # Generate cointegrated pair
        aapl_data, msft_data = self.generate_cointegrated_pair("AAPL", "MSFT", 80)
        
        # Add extreme spread at the end
        for i in range(-10, 0):
            aapl_data[i] = MarketData(
                symbol="AAPL",
                timestamp=aapl_data[i].timestamp,
                open=aapl_data[i].open + 10,  # Increase AAPL prices
                high=aapl_data[i].high + 10,
                low=aapl_data[i].low + 10,
                close=aapl_data[i].close + 10,
                volume=aapl_data[i].volume
            )
        
        pairs_data = {"MSFT": msft_data}
        
        signal = self.agent.generate_signal_sync("AAPL", aapl_data, pairs_data=pairs_data)
        
        if signal:
            print(f"[OK] Signal Generated: {signal.signal_type.value.upper()}")
            print(f"   Signal Value: {signal.value:.3f}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Pairs Signals: {len(signal.pairs_signals)}")
            
            if signal.pairs_signals:
                print("   Pairs Trading Analysis:")
                for pair_signal in signal.pairs_signals:
                    print(f"     • {pair_signal.symbol_a}/{pair_signal.symbol_b}:")
                    print(f"       Spread Z-Score: {pair_signal.z_score:.2f}")
                    print(f"       Cointegration p-value: {pair_signal.cointegration_pvalue:.3f}")
                    print(f"       Recommendation: {pair_signal.explanation}")
        else:
            print("[X] No signal generated")
    
    def demo_fibonacci_targets(self):
        """Demonstrate Fibonacci extension targets"""
        print("\n[INFO] Demo 4: Fibonacci Extension Targets")
        print("-" * 35)
        
        # Generate data with clear swings for Fibonacci analysis
        market_data = []
        base_price = 100.0
        
        # Create swing pattern
        swing_pattern = [
            (0, 20, 0),      # Base to high
            (20, 35, -15),   # High to low (retracement)
            (35, 50, 8),     # Low to target (extension)
            (50, 80, -3)     # Consolidation
        ]
        
        for start, end, move in swing_pattern:
            segment_length = end - start
            for i in range(segment_length):
                progress = i / segment_length
                price = base_price + move * progress + np.random.normal(0, 0.8)
                
                market_data.append(MarketData(
                    symbol="NVDA",
                    timestamp=datetime.now() - timedelta(days=80-(start+i)),
                    open=price + np.random.normal(0, 0.3),
                    high=price + abs(np.random.normal(0, 0.6)),
                    low=price - abs(np.random.normal(0, 0.6)),
                    close=price,
                    volume=np.random.randint(2000000, 6000000)
                ))
        
        signal = self.agent.generate_signal_sync("NVDA", market_data)
        
        if signal:
            print(f"[OK] Signal Generated: {signal.signal_type.value.upper()}")
            print(f"   Signal Value: {signal.value:.3f}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Fibonacci Targets: {len(signal.fibonacci_targets)}")
            
            if signal.fibonacci_targets:
                print("   Fibonacci Extension Targets:")
                for fib_target in signal.fibonacci_targets:
                    print(f"     • {fib_target.level_name}: ${fib_target.target_price:.2f}")
                    print(f"       Distance: {fib_target.distance_pct:.1f}%")
                    print(f"       Confidence: {fib_target.confidence:.1%}")
        else:
            print("[X] No signal generated")
    
    def demo_sentiment_divergence(self):
        """Demonstrate sentiment divergence detection"""
        print("\n[INFO] Demo 5: Sentiment Divergence Detection")
        print("-" * 38)
        
        # Generate data with price-sentiment divergence
        market_data = self.generate_mean_reverting_data("GOOGL", 60, 150.0, 0.05)
        
        # Create bearish divergence: price up, sentiment down
        sentiment_data = SentimentData(
            symbol="GOOGL",
            overall_sentiment=-0.4,  # Negative sentiment
            confidence=0.85,
            news_count=25,
            social_sentiment=-0.3,
            timestamp=datetime.now()
        )
        
        signal = self.agent.generate_signal_sync("GOOGL", market_data, sentiment_data)
        
        if signal:
            print(f"[OK] Signal Generated: {signal.signal_type.value.upper()}")
            print(f"   Signal Value: {signal.value:.3f}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Sentiment Divergence: {signal.sentiment_divergence}")
            print(f"   Market Regime: {signal.market_regime.value if signal.market_regime else 'Unknown'}")
            
            if signal.sentiment_divergence is not None:
                divergence_type = "Bullish" if signal.sentiment_divergence > 0 else "Bearish"
                print(f"   Divergence Type: {divergence_type}")
                print(f"   Divergence Strength: {abs(signal.sentiment_divergence):.2f}")
        else:
            print("[X] No signal generated")
    
    def demo_complete_analysis(self):
        """Demonstrate complete mean reversion analysis"""
        print("\n[TARGET] Demo 6: Complete Mean Reversion Analysis")
        print("-" * 42)
        
        # Generate comprehensive test data
        market_data = self.generate_mean_reverting_data("SPY", 100, 400.0, 0.08)
        sentiment_data = self.generate_sentiment_data("SPY", sentiment_bias=0.2)
        
        # Add pairs data
        qqq_data = self.generate_mean_reverting_data("QQQ", 100, 350.0, 0.07)
        pairs_data = {"QQQ": qqq_data}
        
        print("[CHART] Analyzing SPY with complete mean reversion strategy...")
        print(f"   Market data points: {len(market_data)}")
        print(f"   Sentiment score: {sentiment_data.overall_sentiment:.2f}")
        print(f"   Pairs symbols: {list(pairs_data.keys())}")
        
        signal = self.agent.generate_signal_sync("SPY", market_data, sentiment_data, pairs_data)
        
        if signal:
            print(f"\n[OK] FINAL SIGNAL: {signal.signal_type.value.upper()}")
            print(f"   Signal Value: {signal.value:.3f}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Market Regime: {signal.market_regime.value if signal.market_regime else 'Unknown'}")
            
            print(f"\n[UP] Technical Analysis Summary:")
            print(f"   Bollinger Band Signals: {len(signal.bollinger_signals)}")
            print(f"   Z-Score Signals: {len(signal.zscore_signals)}")
            print(f"   Pairs Trading Signals: {len(signal.pairs_signals)}")
            print(f"   Fibonacci Targets: {len(signal.fibonacci_targets)}")
            
            print(f"\n[TARGET] Risk Management:")
            print(f"   Position Size: {signal.position_size_pct:.1%}")
            print(f"   Stop Loss: {signal.stop_loss_pct:.1%}")
            print(f"   Take Profit Targets: {signal.take_profit_targets}")
            print(f"   Max Holding Period: {signal.max_holding_period} days")
            
            print(f"\n[AI] Top 3 Reasons for Decision:")
            for reason in signal.top_3_reasons:
                print(f"   {reason.rank}. {reason.factor}")
                print(f"      {reason.explanation}")
                print(f"      Contribution: {reason.contribution:.2f}, Confidence: {reason.confidence:.1%}")
            
            # Show signal as JSON
            print(f"\n[INFO] Complete Signal Data:")
            signal_dict = signal.to_dict()
            print(json.dumps({
                'symbol': signal_dict['symbol'],
                'signal_type': signal_dict['signal_type'],
                'value': round(signal_dict['value'], 3),
                'confidence': round(signal_dict['confidence'], 3),
                'market_regime': signal_dict['market_regime'],
                'position_size_pct': signal_dict['position_size_pct'],
                'top_reasons': [r['factor'] for r in signal_dict['top_3_reasons']]
            }, indent=2))
        else:
            print("[X] No signal generated")
    
    def demo_market_regimes(self):
        """Demonstrate market regime detection"""
        print("\n[INFO] Demo 7: Market Regime Detection")
        print("-" * 32)
        
        regimes_data = {
            "High Volatility": self._generate_high_vol_data("VIX", 50),
            "Low Volatility": self._generate_low_vol_data("BOND", 50),
            "Trending": self._generate_trending_data("TREND", 50),
            "Mean Reverting": self.generate_mean_reverting_data("MEAN", 50, 100.0, 0.15)
        }
        
        for regime_name, data in regimes_data.items():
            signal = self.agent.generate_signal_sync(regime_name, data)
            
            if signal and signal.market_regime:
                print(f"   {regime_name:15} → {signal.market_regime.value}")
            else:
                print(f"   {regime_name:15} → Unknown")
    
    def _generate_high_vol_data(self, symbol: str, length: int) -> List[MarketData]:
        """Generate high volatility data"""
        np.random.seed(100)
        data = []
        price = 100.0
        
        for i in range(length):
            price += np.random.normal(0, 5)  # High volatility
            price = max(50, min(200, price))
            
            data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(days=length-i),
                open=price + np.random.normal(0, 1),
                high=price + abs(np.random.normal(0, 2)),
                low=price - abs(np.random.normal(0, 2)),
                close=price,
                volume=np.random.randint(1000000, 3000000)
            ))
        
        return data
    
    def _generate_low_vol_data(self, symbol: str, length: int) -> List[MarketData]:
        """Generate low volatility data"""
        np.random.seed(200)
        data = []
        price = 100.0
        
        for i in range(length):
            price += np.random.normal(0, 0.2)  # Low volatility
            
            data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(days=length-i),
                open=price + np.random.normal(0, 0.1),
                high=price + abs(np.random.normal(0, 0.1)),
                low=price - abs(np.random.normal(0, 0.1)),
                close=price,
                volume=np.random.randint(1000000, 2000000)
            ))
        
        return data
    
    def _generate_trending_data(self, symbol: str, length: int) -> List[MarketData]:
        """Generate trending data"""
        np.random.seed(300)
        data = []
        price = 100.0
        
        for i in range(length):
            price += 0.5 + np.random.normal(0, 0.8)  # Clear uptrend
            
            data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(days=length-i),
                open=price + np.random.normal(0, 0.3),
                high=price + abs(np.random.normal(0, 0.5)),
                low=price - abs(np.random.normal(0, 0.3)),
                close=price,
                volume=np.random.randint(1500000, 3500000)
            ))
        
        return data
    
    def run_all_demos(self):
        """Run all demonstration scenarios"""
        print("[LAUNCH] Running All Mean Reversion Trading Agent Demos")
        print("=" * 55)
        
        try:
            self.demo_bollinger_band_reversion()
            self.demo_zscore_analysis()
            self.demo_pairs_trading()
            self.demo_fibonacci_targets()
            self.demo_sentiment_divergence()
            self.demo_market_regimes()
            self.demo_complete_analysis()
            
            print("\n" + "=" * 55)
            print("[OK] All demos completed successfully!")
            print("\n[TARGET] Key Features Demonstrated:")
            print("   • Bollinger Band reversion detection")
            print("   • Z-score mean reversion analysis")
            print("   • Pairs trading with cointegration")
            print("   • Fibonacci extension targets")
            print("   • Sentiment divergence detection")
            print("   • Market regime identification")
            print("   • Complete explainable AI workflow")
            print("   • Risk management integration")
            
        except Exception as e:
            print(f"\n[X] Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the demo"""
    demo = MeanReversionDemo()
    demo.run_all_demos()


if __name__ == "__main__":
    main()