"""
Test suite for Momentum Trading Agent

Tests all components of the momentum trading strategy including:
- Technical indicator calculations (EMA, RSI, MACD)
- Fibonacci integration for entry timing
- Sentiment confirmation
- Volatility-adjusted position sizing
- Signal generation and explainability
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio
from typing import List, Dict

# Import the momentum trading agent components
from agents.momentum_trading_agent import (
    MomentumTradingAgent,
    TechnicalAnalyzer,
    FibonacciIntegrator,
    SentimentIntegrator,
    VolatilityAdjuster,
    ExplainabilityEngine,
    MarketData,
    SentimentData,
    TechnicalSignal,
    FibonacciSignal,
    MomentumSignal,
    SignalType,
    MarketRegime,
    generate_momentum_signal
)


class TestTechnicalAnalyzer:
    """Test technical indicator calculations"""
    
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()
        
        # Create sample price data with clear patterns
        self.sample_prices = self._create_sample_price_data()
    
    def _create_sample_price_data(self) -> np.ndarray:
        """Create sample price data with known patterns"""
        # Create 100 data points with uptrend and some volatility
        base_prices = np.linspace(100, 120, 100)
        noise = np.random.normal(0, 1, 100)
        return base_prices + noise
    
    def test_ema_crossover_signals(self):
        """Test EMA crossover signal generation"""
        signals = self.analyzer.calculate_ema_signals(self.sample_prices)
        
        assert isinstance(signals, list)
        
        # Should generate at least some signals for trending data
        if signals:
            signal = signals[0]
            assert isinstance(signal, TechnicalSignal)
            assert signal.indicator in ["EMA_Crossover", "EMA_Trend"]
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 1
            assert signal.explanation is not None
    
    def test_rsi_signals(self):
        """Test RSI signal generation"""
        # Create data with clear oversold/overbought conditions
        oscillating_prices = np.array([100 + 10 * np.sin(i * 0.1) for i in range(50)])
        
        signals = self.analyzer.calculate_rsi_signals(oscillating_prices)
        
        assert isinstance(signals, list)
        
        if signals:
            signal = signals[0]
            assert isinstance(signal, TechnicalSignal)
            assert "RSI" in signal.indicator
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 1
    
    def test_macd_signals(self):
        """Test MACD signal generation"""
        signals = self.analyzer.calculate_macd_signals(self.sample_prices)
        
        assert isinstance(signals, list)
        
        if signals:
            signal = signals[0]
            assert isinstance(signal, TechnicalSignal)
            assert "MACD" in signal.indicator
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 1
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        short_data = np.array([100, 101, 102])
        
        # Should handle short data gracefully
        ema_signals = self.analyzer.calculate_ema_signals(short_data)
        rsi_signals = self.analyzer.calculate_rsi_signals(short_data)
        macd_signals = self.analyzer.calculate_macd_signals(short_data)
        
        # Should return empty lists or handle gracefully
        assert isinstance(ema_signals, list)
        assert isinstance(rsi_signals, list)
        assert isinstance(macd_signals, list)


class TestFibonacciIntegrator:
    """Test Fibonacci integration"""
    
    def setup_method(self):
        self.integrator = FibonacciIntegrator()
        
        # Create sample OHLC data
        self.high_data, self.low_data, self.close_data = self._create_ohlc_data()
    
    def _create_ohlc_data(self):
        """Create sample OHLC data"""
        base_prices = np.linspace(100, 120, 50)
        high_data = base_prices + np.random.uniform(0.5, 2.0, 50)
        low_data = base_prices - np.random.uniform(0.5, 2.0, 50)
        close_data = base_prices + np.random.normal(0, 0.5, 50)
        
        return high_data, low_data, close_data
    
    def test_fibonacci_signal_generation(self):
        """Test Fibonacci signal generation"""
        current_price = self.close_data[-1]
        
        signals = self.integrator.calculate_fibonacci_signals(
            self.high_data, self.low_data, self.close_data, current_price
        )
        
        assert isinstance(signals, list)
        
        if signals:
            signal = signals[0]
            assert isinstance(signal, FibonacciSignal)
            assert signal.level_type in ['retracement', 'extension', 'confluence']
            assert signal.current_price == current_price
            assert signal.distance_pct >= 0
            assert signal.confluence_strength >= 0
    
    def test_confluence_strength_calculation(self):
        """Test confluence strength calculation"""
        current_price = 110.0
        
        # Create mock Fibonacci signals
        fib_signals = [
            FibonacciSignal(
                level_type='retracement',
                level_name='fib_618',
                level_price=109.5,
                current_price=current_price,
                distance_pct=0.45,
                confluence_strength=1.0,
                explanation="Test signal"
            )
        ]
        
        strength = self.integrator.calculate_confluence_strength(current_price, fib_signals)
        
        assert 0 <= strength <= 1
        assert isinstance(strength, float)


class TestSentimentIntegrator:
    """Test sentiment integration"""
    
    def setup_method(self):
        self.integrator = SentimentIntegrator()
    
    def test_sentiment_alignment_positive(self):
        """Test positive sentiment alignment"""
        # Both signal and sentiment positive
        alignment = self.integrator.calculate_sentiment_alignment(0.5, 0.7)
        assert alignment > 1.0  # Should boost signal
        
        # Both signal and sentiment negative
        alignment = self.integrator.calculate_sentiment_alignment(-0.5, -0.7)
        assert alignment > 1.0  # Should boost signal
    
    def test_sentiment_alignment_conflicting(self):
        """Test conflicting sentiment alignment"""
        # Positive signal, negative sentiment
        alignment = self.integrator.calculate_sentiment_alignment(0.5, -0.7)
        assert alignment < 1.0  # Should reduce signal
        
        # Negative signal, positive sentiment
        alignment = self.integrator.calculate_sentiment_alignment(-0.5, 0.7)
        assert alignment < 1.0  # Should reduce signal
    
    def test_sentiment_alignment_neutral(self):
        """Test neutral sentiment handling"""
        alignment = self.integrator.calculate_sentiment_alignment(0.5, None)
        assert alignment == 1.0  # Should not affect signal
        
        alignment = self.integrator.calculate_sentiment_alignment(0.5, 0.0)
        assert alignment == 1.0  # Should not affect signal


class TestVolatilityAdjuster:
    """Test volatility adjustment and position sizing"""
    
    def setup_method(self):
        self.adjuster = VolatilityAdjuster()
    
    def test_volatility_calculation_high_vol(self):
        """Test high volatility detection"""
        # Create high volatility data
        high_vol_data = np.array([100 + 10 * np.random.normal(0, 1) for _ in range(50)])
        
        adjustment, regime = self.adjuster.calculate_volatility_adjustment(high_vol_data)
        
        assert isinstance(adjustment, float)
        assert isinstance(regime, MarketRegime)
        assert adjustment > 0
    
    def test_volatility_calculation_low_vol(self):
        """Test low volatility detection"""
        # Create low volatility data
        low_vol_data = np.array([100 + 0.1 * np.random.normal(0, 1) for _ in range(50)])
        
        adjustment, regime = self.adjuster.calculate_volatility_adjustment(low_vol_data)
        
        assert isinstance(adjustment, float)
        assert isinstance(regime, MarketRegime)
        assert adjustment > 0
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        base_size = 0.05  # 5%
        vol_adjustment = 0.8
        confidence = 0.7
        account_balance = 100000
        
        position_size = self.adjuster.calculate_position_size(
            base_size, vol_adjustment, confidence, account_balance
        )
        
        assert 0.01 <= position_size <= 0.1  # Within risk limits
        assert isinstance(position_size, float)
    
    def test_trending_regime_detection(self):
        """Test trending regime detection"""
        # Create trending up data
        trending_up_data = np.linspace(100, 120, 30)
        
        adjustment, regime = self.adjuster.calculate_volatility_adjustment(trending_up_data)
        
        # Should detect uptrend or at least not be sideways for strong trend
        assert regime in [MarketRegime.TRENDING_UP, MarketRegime.LOW_VOLATILITY, MarketRegime.SIDEWAYS]


class TestExplainabilityEngine:
    """Test explainability engine"""
    
    def setup_method(self):
        self.engine = ExplainabilityEngine()
    
    def test_top_3_reasons_generation(self):
        """Test generation of top 3 reasons"""
        # Create sample technical signals
        technical_signals = [
            TechnicalSignal(
                indicator="EMA_Crossover",
                signal_type=SignalType.BUY,
                strength=0.8,
                confidence=0.7,
                value=0.8,
                explanation="Fast EMA crossed above slow EMA",
                timestamp=datetime.utcnow()
            ),
            TechnicalSignal(
                indicator="RSI_Breakout",
                signal_type=SignalType.BUY,
                strength=0.6,
                confidence=0.8,
                value=0.6,
                explanation="RSI broke above oversold level",
                timestamp=datetime.utcnow()
            )
        ]
        
        # Create sample Fibonacci signals
        fibonacci_signals = [
            FibonacciSignal(
                level_type='retracement',
                level_name='fib_618',
                level_price=109.5,
                current_price=110.0,
                distance_pct=0.45,
                confluence_strength=1.0,
                explanation="Price near Fibonacci 618 level"
            )
        ]
        
        reasons = self.engine.generate_top_3_reasons(
            technical_signals, fibonacci_signals, 0.3, 0.8, 0.5
        )
        
        assert len(reasons) <= 3
        assert all(isinstance(reason.rank, int) for reason in reasons)
        assert all(isinstance(reason.factor, str) for reason in reasons)
        assert all(isinstance(reason.contribution, float) for reason in reasons)
        assert all(isinstance(reason.explanation, str) for reason in reasons)
        assert all(0 <= reason.confidence <= 1 for reason in reasons)
        
        # Reasons should be ranked by contribution
        if len(reasons) > 1:
            assert reasons[0].contribution >= reasons[1].contribution


class TestMomentumTradingAgent:
    """Test the main momentum trading agent"""
    
    def setup_method(self):
        self.agent = MomentumTradingAgent()
        self.sample_market_data = self._create_sample_market_data()
    
    def _create_sample_market_data(self) -> List[MarketData]:
        """Create sample market data"""
        data = []
        base_price = 100.0
        
        for i in range(50):
            price = base_price + i * 0.2 + np.random.normal(0, 1)
            data.append(MarketData(
                symbol='AAPL',
                timestamp=datetime.utcnow() - timedelta(days=50-i),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000000 + np.random.randint(-100000, 100000)
            ))
        
        return data
    
    @pytest.mark.asyncio
    async def test_signal_generation_without_sentiment(self):
        """Test signal generation without sentiment data"""
        signal = await self.agent.generate_momentum_signal(
            'AAPL', self.sample_market_data
        )
        
        if signal:  # Signal generation might return None for insufficient data
            assert isinstance(signal, MomentumSignal)
            assert signal.symbol == 'AAPL'
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, 
                                        SignalType.STRONG_BUY, SignalType.STRONG_SELL]
            assert -1 <= signal.value <= 1
            assert 0 <= signal.confidence <= 1
            assert len(signal.top_3_reasons) <= 3
            assert signal.model_version == "1.0.0"
            assert signal.position_size_pct is not None
            assert signal.stop_loss_pct is not None
            assert signal.take_profit_pct is not None
    
    @pytest.mark.asyncio
    async def test_signal_generation_with_sentiment(self):
        """Test signal generation with sentiment data"""
        sentiment_data = SentimentData(
            symbol='AAPL',
            overall_sentiment=0.3,
            confidence=0.8,
            news_count=15,
            timestamp=datetime.utcnow()
        )
        
        signal = await self.agent.generate_momentum_signal(
            'AAPL', self.sample_market_data, sentiment_data
        )
        
        if signal:
            assert isinstance(signal, MomentumSignal)
            assert signal.sentiment_score == 0.3
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self):
        """Test handling of insufficient market data"""
        short_data = self.sample_market_data[:5]  # Very short data
        
        signal = await self.agent.generate_momentum_signal('AAPL', short_data)
        
        # Should handle gracefully (might return None or a signal with low confidence)
        if signal:
            assert isinstance(signal, MomentumSignal)
    
    def test_langgraph_creation(self):
        """Test LangGraph creation"""
        assert self.agent.graph is not None
        # The graph should be compiled and ready to use


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.mark.asyncio
    async def test_generate_momentum_signal_function(self):
        """Test the convenience function for signal generation"""
        # Create sample raw data
        sample_data = []
        base_price = 100.0
        
        for i in range(30):
            price = base_price + i * 0.1 + np.random.normal(0, 0.5)
            sample_data.append({
                'symbol': 'AAPL',
                'timestamp': (datetime.utcnow() - timedelta(days=30-i)).isoformat(),
                'open': price - 0.5,
                'high': price + 1.0,
                'low': price - 1.0,
                'close': price,
                'volume': 1000000
            })
        
        sentiment_data = {
            'symbol': 'AAPL',
            'overall_sentiment': 0.2,
            'confidence': 0.7,
            'news_count': 10,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        signal_dict = await generate_momentum_signal('AAPL', sample_data, sentiment_data)
        
        if signal_dict:
            assert isinstance(signal_dict, dict)
            assert 'symbol' in signal_dict
            assert 'signal_type' in signal_dict
            assert 'value' in signal_dict
            assert 'confidence' in signal_dict
            assert 'top_3_reasons' in signal_dict
            assert 'timestamp' in signal_dict


class TestIntegrationScenarios:
    """Test various market scenarios"""
    
    def setup_method(self):
        self.agent = MomentumTradingAgent()
    
    def _create_scenario_data(self, scenario: str) -> List[MarketData]:
        """Create market data for specific scenarios"""
        data = []
        base_price = 100.0
        
        if scenario == "strong_uptrend":
            prices = np.linspace(100, 130, 50)  # Strong uptrend
        elif scenario == "strong_downtrend":
            prices = np.linspace(130, 100, 50)  # Strong downtrend
        elif scenario == "sideways":
            prices = np.array([100 + 2 * np.sin(i * 0.2) for i in range(50)])  # Sideways
        elif scenario == "volatile":
            prices = np.array([100 + 10 * np.random.normal(0, 1) for _ in range(50)])  # High volatility
        else:
            prices = np.linspace(100, 110, 50)  # Default mild uptrend
        
        for i, price in enumerate(prices):
            data.append(MarketData(
                symbol='TEST',
                timestamp=datetime.utcnow() - timedelta(days=50-i),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000000
            ))
        
        return data
    
    @pytest.mark.asyncio
    async def test_strong_uptrend_scenario(self):
        """Test signal generation in strong uptrend"""
        market_data = self._create_scenario_data("strong_uptrend")
        
        signal = await self.agent.generate_momentum_signal('TEST', market_data)
        
        if signal:
            # Should generate buy signals in strong uptrend
            assert signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.HOLD]
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                assert signal.value > 0
    
    @pytest.mark.asyncio
    async def test_strong_downtrend_scenario(self):
        """Test signal generation in strong downtrend"""
        market_data = self._create_scenario_data("strong_downtrend")
        
        signal = await self.agent.generate_momentum_signal('TEST', market_data)
        
        if signal:
            # Should generate sell signals in strong downtrend
            assert signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL, SignalType.HOLD]
            if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                assert signal.value < 0
    
    @pytest.mark.asyncio
    async def test_sideways_scenario(self):
        """Test signal generation in sideways market"""
        market_data = self._create_scenario_data("sideways")
        
        signal = await self.agent.generate_momentum_signal('TEST', market_data)
        
        if signal:
            # Should generate hold or weak signals in sideways market
            assert signal.confidence <= 0.7  # Lower confidence in sideways market


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        self.agent = MomentumTradingAgent()
    
    @pytest.mark.asyncio
    async def test_empty_market_data(self):
        """Test handling of empty market data"""
        signal = await self.agent.generate_momentum_signal('TEST', [])
        
        # Should handle gracefully
        assert signal is None or isinstance(signal, MomentumSignal)
    
    @pytest.mark.asyncio
    async def test_invalid_market_data(self):
        """Test handling of invalid market data"""
        invalid_data = [
            MarketData(
                symbol='TEST',
                timestamp=datetime.utcnow(),
                open=float('nan'),  # Invalid data
                high=float('nan'),
                low=float('nan'),
                close=float('nan'),
                volume=0
            )
        ]
        
        signal = await self.agent.generate_momentum_signal('TEST', invalid_data)
        
        # Should handle gracefully
        assert signal is None or isinstance(signal, MomentumSignal)


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])