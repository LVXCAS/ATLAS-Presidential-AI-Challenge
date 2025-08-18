#!/usr/bin/env python3
"""
Momentum Trading Agent Validation Script

This script validates the momentum trading agent implementation by:
1. Testing technical indicator calculations
2. Validating Fibonacci integration
3. Checking sentiment confirmation
4. Verifying volatility adjustment
5. Testing signal generation and explainability
6. Running backtesting scenarios
"""

import sys
import os
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.momentum_trading_agent import (
    MomentumTradingAgent,
    TechnicalAnalyzer,
    FibonacciIntegrator,
    SentimentIntegrator,
    VolatilityAdjuster,
    ExplainabilityEngine,
    generate_momentum_signal,
    MarketData,
    SentimentData,
    SignalType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MomentumAgentValidator:
    """Validates momentum trading agent functionality"""
    
    def __init__(self):
        self.agent = MomentumTradingAgent()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fibonacci_integrator = FibonacciIntegrator()
        self.sentiment_integrator = SentimentIntegrator()
        self.volatility_adjuster = VolatilityAdjuster()
        self.explainability_engine = ExplainabilityEngine()
        
        self.validation_results = {
            'technical_indicators': False,
            'fibonacci_integration': False,
            'sentiment_confirmation': False,
            'volatility_adjustment': False,
            'signal_generation': False,
            'explainability': False,
            'backtesting': False,
            'error_handling': False
        }
    
    def create_test_data(self, scenario: str = "uptrend", days: int = 50) -> Tuple[List[MarketData], SentimentData]:
        """Create test market data and sentiment data"""
        base_price = 100.0
        
        if scenario == "uptrend":
            trend = np.linspace(0, 20, days)
            noise = np.random.normal(0, 1, days)
            prices = base_price + trend + noise
        elif scenario == "downtrend":
            trend = np.linspace(0, -15, days)
            noise = np.random.normal(0, 1, days)
            prices = base_price + trend + noise
        elif scenario == "sideways":
            prices = np.array([base_price + 2 * np.sin(i * 0.3) + np.random.normal(0, 0.5) for i in range(days)])
        elif scenario == "volatile":
            prices = np.array([base_price + np.random.normal(0, 5) for _ in range(days)])
        else:
            prices = np.linspace(base_price, base_price + 10, days)
        
        # Ensure positive prices
        prices = np.maximum(prices, 10.0)
        
        market_data = []
        for i, price in enumerate(prices):
            market_data.append(MarketData(
                symbol='TEST',
                timestamp=datetime.utcnow() - timedelta(days=days-i),
                open=price - np.random.uniform(0.2, 0.8),
                high=price + np.random.uniform(0.5, 2.0),
                low=price - np.random.uniform(0.5, 2.0),
                close=price,
                volume=int(1000000 + np.random.randint(-200000, 200000))
            ))
        
        sentiment_data = SentimentData(
            symbol='TEST',
            overall_sentiment=np.random.uniform(-0.5, 0.5),
            confidence=0.8,
            news_count=15,
            timestamp=datetime.utcnow()
        )
        
        return market_data, sentiment_data
    
    def validate_technical_indicators(self) -> bool:
        """Validate technical indicator calculations"""
        logger.info("Validating technical indicators...")
        
        try:
            # Create test data
            prices = np.array([100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(50)])
            
            # Test EMA signals
            ema_signals = self.technical_analyzer.calculate_ema_signals(prices)
            assert isinstance(ema_signals, list), "EMA signals should be a list"
            
            # Test RSI signals
            rsi_signals = self.technical_analyzer.calculate_rsi_signals(prices)
            assert isinstance(rsi_signals, list), "RSI signals should be a list"
            
            # Test MACD signals
            macd_signals = self.technical_analyzer.calculate_macd_signals(prices)
            assert isinstance(macd_signals, list), "MACD signals should be a list"
            
            # Validate signal structure if signals exist
            all_signals = ema_signals + rsi_signals + macd_signals
            for signal in all_signals:
                assert hasattr(signal, 'indicator'), "Signal should have indicator"
                assert hasattr(signal, 'signal_type'), "Signal should have signal_type"
                assert hasattr(signal, 'strength'), "Signal should have strength"
                assert hasattr(signal, 'confidence'), "Signal should have confidence"
                assert 0 <= signal.confidence <= 1, "Confidence should be between 0 and 1"
                assert signal.signal_type in [SignalType.BUY, SignalType.SELL], "Valid signal type"
            
            logger.info(f"✓ Technical indicators validated - Generated {len(all_signals)} signals")
            return True
            
        except Exception as e:
            logger.error(f"✗ Technical indicators validation failed: {e}")
            return False
    
    def validate_fibonacci_integration(self) -> bool:
        """Validate Fibonacci integration"""
        logger.info("Validating Fibonacci integration...")
        
        try:
            # Create test OHLC data with clear swings
            high_data = np.array([100, 105, 110, 115, 120, 118, 116, 114, 112, 115, 118, 122])
            low_data = np.array([98, 103, 108, 113, 118, 116, 114, 112, 110, 113, 116, 120])
            close_data = np.array([99, 104, 109, 114, 119, 117, 115, 113, 111, 114, 117, 121])
            current_price = close_data[-1]
            
            # Test Fibonacci signal generation
            fib_signals = self.fibonacci_integrator.calculate_fibonacci_signals(
                high_data, low_data, close_data, current_price
            )
            
            assert isinstance(fib_signals, list), "Fibonacci signals should be a list"
            
            # Test confluence strength calculation
            confluence_strength = self.fibonacci_integrator.calculate_confluence_strength(
                current_price, fib_signals
            )
            
            assert isinstance(confluence_strength, float), "Confluence strength should be float"
            assert 0 <= confluence_strength <= 1, "Confluence strength should be between 0 and 1"
            
            # Validate signal structure if signals exist
            for signal in fib_signals:
                assert hasattr(signal, 'level_type'), "Fibonacci signal should have level_type"
                assert hasattr(signal, 'level_price'), "Fibonacci signal should have level_price"
                assert hasattr(signal, 'distance_pct'), "Fibonacci signal should have distance_pct"
                assert signal.distance_pct >= 0, "Distance should be non-negative"
            
            logger.info(f"✓ Fibonacci integration validated - Generated {len(fib_signals)} signals")
            return True
            
        except Exception as e:
            logger.error(f"✗ Fibonacci integration validation failed: {e}")
            return False
    
    def validate_sentiment_confirmation(self) -> bool:
        """Validate sentiment confirmation"""
        logger.info("Validating sentiment confirmation...")
        
        try:
            # Test positive alignment
            alignment1 = self.sentiment_integrator.calculate_sentiment_alignment(0.5, 0.7)
            assert alignment1 > 1.0, "Positive alignment should boost signal"
            
            # Test negative alignment
            alignment2 = self.sentiment_integrator.calculate_sentiment_alignment(-0.5, -0.7)
            assert alignment2 > 1.0, "Negative alignment should boost signal"
            
            # Test conflicting alignment
            alignment3 = self.sentiment_integrator.calculate_sentiment_alignment(0.5, -0.7)
            assert alignment3 < 1.0, "Conflicting alignment should reduce signal"
            
            # Test neutral alignment
            alignment4 = self.sentiment_integrator.calculate_sentiment_alignment(0.5, None)
            assert alignment4 == 1.0, "Neutral alignment should not affect signal"
            
            logger.info("✓ Sentiment confirmation validated")
            return True
            
        except Exception as e:
            logger.error(f"✗ Sentiment confirmation validation failed: {e}")
            return False
    
    def validate_volatility_adjustment(self) -> bool:
        """Validate volatility adjustment"""
        logger.info("Validating volatility adjustment...")
        
        try:
            # Test high volatility data
            high_vol_data = np.array([100 + 10 * np.random.normal(0, 1) for _ in range(30)])
            vol_adj1, regime1 = self.volatility_adjuster.calculate_volatility_adjustment(high_vol_data)
            
            assert isinstance(vol_adj1, float), "Volatility adjustment should be float"
            assert vol_adj1 > 0, "Volatility adjustment should be positive"
            
            # Test low volatility data
            low_vol_data = np.array([100 + 0.1 * np.random.normal(0, 1) for _ in range(30)])
            vol_adj2, regime2 = self.volatility_adjuster.calculate_volatility_adjustment(low_vol_data)
            
            assert isinstance(vol_adj2, float), "Volatility adjustment should be float"
            assert vol_adj2 > 0, "Volatility adjustment should be positive"
            
            # Test position sizing
            position_size = self.volatility_adjuster.calculate_position_size(0.05, vol_adj1, 0.7, 100000)
            
            assert isinstance(position_size, float), "Position size should be float"
            assert 0.01 <= position_size <= 0.1, "Position size should be within risk limits"
            
            logger.info("✓ Volatility adjustment validated")
            return True
            
        except Exception as e:
            logger.error(f"✗ Volatility adjustment validation failed: {e}")
            return False
    
    async def validate_signal_generation(self) -> bool:
        """Validate signal generation"""
        logger.info("Validating signal generation...")
        
        try:
            # Test different scenarios
            scenarios = ['uptrend', 'downtrend', 'sideways', 'volatile']
            
            for scenario in scenarios:
                market_data, sentiment_data = self.create_test_data(scenario)
                
                signal = await self.agent.generate_momentum_signal(
                    'TEST', market_data, sentiment_data
                )
                
                if signal:  # Signal might be None for insufficient data
                    assert hasattr(signal, 'symbol'), "Signal should have symbol"
                    assert hasattr(signal, 'signal_type'), "Signal should have signal_type"
                    assert hasattr(signal, 'value'), "Signal should have value"
                    assert hasattr(signal, 'confidence'), "Signal should have confidence"
                    assert hasattr(signal, 'top_3_reasons'), "Signal should have top_3_reasons"
                    
                    assert -1 <= signal.value <= 1, "Signal value should be between -1 and 1"
                    assert 0 <= signal.confidence <= 1, "Confidence should be between 0 and 1"
                    assert len(signal.top_3_reasons) <= 3, "Should have at most 3 reasons"
                    
                    # Validate signal components
                    assert isinstance(signal.ema_signals, list), "EMA signals should be list"
                    assert isinstance(signal.rsi_signals, list), "RSI signals should be list"
                    assert isinstance(signal.macd_signals, list), "MACD signals should be list"
                    assert isinstance(signal.fibonacci_signals, list), "Fibonacci signals should be list"
            
            logger.info("✓ Signal generation validated")
            return True
            
        except Exception as e:
            logger.error(f"✗ Signal generation validation failed: {e}")
            return False
    
    def validate_explainability(self) -> bool:
        """Validate explainability engine"""
        logger.info("Validating explainability...")
        
        try:
            # Create mock signals for testing
            from agents.momentum_trading_agent import TechnicalSignal, FibonacciSignal, Reason
            
            technical_signals = [
                TechnicalSignal(
                    indicator="EMA_Crossover",
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    confidence=0.7,
                    value=0.8,
                    explanation="Test EMA signal",
                    timestamp=datetime.utcnow()
                )
            ]
            
            fibonacci_signals = [
                FibonacciSignal(
                    level_type='retracement',
                    level_name='fib_618',
                    level_price=110.0,
                    current_price=111.0,
                    distance_pct=0.9,
                    confluence_strength=1.0,
                    explanation="Test Fibonacci signal"
                )
            ]
            
            reasons = self.explainability_engine.generate_top_3_reasons(
                technical_signals, fibonacci_signals, 0.3, 0.8, 0.5
            )
            
            assert isinstance(reasons, list), "Reasons should be a list"
            assert len(reasons) <= 3, "Should have at most 3 reasons"
            
            for reason in reasons:
                assert isinstance(reason, Reason), "Should be Reason object"
                assert hasattr(reason, 'rank'), "Reason should have rank"
                assert hasattr(reason, 'factor'), "Reason should have factor"
                assert hasattr(reason, 'contribution'), "Reason should have contribution"
                assert hasattr(reason, 'explanation'), "Reason should have explanation"
                assert hasattr(reason, 'confidence'), "Reason should have confidence"
                assert 0 <= reason.confidence <= 1, "Confidence should be between 0 and 1"
            
            logger.info("✓ Explainability validated")
            return True
            
        except Exception as e:
            logger.error(f"✗ Explainability validation failed: {e}")
            return False
    
    async def validate_backtesting_scenario(self) -> bool:
        """Validate backtesting scenario"""
        logger.info("Validating backtesting scenario...")
        
        try:
            # Create 1 year of test data
            market_data, sentiment_data = self.create_test_data("uptrend", days=252)
            
            # Generate signals for different time windows
            signals = []
            window_size = 50
            
            for i in range(window_size, len(market_data), 10):  # Every 10 days
                window_data = market_data[i-window_size:i]
                
                signal = await self.agent.generate_momentum_signal(
                    'BACKTEST', window_data, sentiment_data
                )
                
                if signal:
                    signals.append({
                        'date': window_data[-1].timestamp,
                        'signal_type': signal.signal_type.value,
                        'value': signal.value,
                        'confidence': signal.confidence,
                        'price': window_data[-1].close
                    })
            
            assert len(signals) > 0, "Should generate some signals during backtesting"
            
            # Basic performance metrics
            buy_signals = [s for s in signals if s['signal_type'] in ['buy', 'strong_buy']]
            sell_signals = [s for s in signals if s['signal_type'] in ['sell', 'strong_sell']]
            
            logger.info(f"✓ Backtesting validated - Generated {len(signals)} signals "
                       f"({len(buy_signals)} buy, {len(sell_signals)} sell)")
            return True
            
        except Exception as e:
            logger.error(f"✗ Backtesting validation failed: {e}")
            return False
    
    async def validate_error_handling(self) -> bool:
        """Validate error handling"""
        logger.info("Validating error handling...")
        
        try:
            # Test empty data
            empty_signal = await self.agent.generate_momentum_signal('TEST', [])
            assert empty_signal is None, "Should handle empty data gracefully"
            
            # Test insufficient data
            short_data = [MarketData(
                symbol='TEST',
                timestamp=datetime.utcnow(),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000000
            )]
            
            short_signal = await self.agent.generate_momentum_signal('TEST', short_data)
            # Should handle gracefully (might return None or low confidence signal)
            
            # Test invalid data
            invalid_data = [MarketData(
                symbol='TEST',
                timestamp=datetime.utcnow(),
                open=float('nan'),
                high=float('nan'),
                low=float('nan'),
                close=float('nan'),
                volume=0
            )]
            
            invalid_signal = await self.agent.generate_momentum_signal('TEST', invalid_data)
            # Should handle gracefully
            
            logger.info("✓ Error handling validated")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error handling validation failed: {e}")
            return False
    
    async def run_validation(self) -> Dict[str, bool]:
        """Run all validation tests"""
        logger.info("Starting Momentum Trading Agent validation...")
        
        # Run all validation tests
        self.validation_results['technical_indicators'] = self.validate_technical_indicators()
        self.validation_results['fibonacci_integration'] = self.validate_fibonacci_integration()
        self.validation_results['sentiment_confirmation'] = self.validate_sentiment_confirmation()
        self.validation_results['volatility_adjustment'] = self.validate_volatility_adjustment()
        self.validation_results['signal_generation'] = await self.validate_signal_generation()
        self.validation_results['explainability'] = self.validate_explainability()
        self.validation_results['backtesting'] = await self.validate_backtesting_scenario()
        self.validation_results['error_handling'] = await self.validate_error_handling()
        
        return self.validation_results
    
    def print_validation_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("MOMENTUM TRADING AGENT VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = sum(self.validation_results.values())
        total = len(self.validation_results)
        
        for test_name, result in self.validation_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name.replace('_', ' ').title():<30} {status}")
        
        print("-" * 60)
        print(f"Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL VALIDATIONS PASSED - Momentum Trading Agent is ready!")
        else:
            print("⚠️  Some validations failed - Please review the issues above")
        
        print("=" * 60)


async def main():
    """Main validation function"""
    validator = MomentumAgentValidator()
    
    try:
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Run validation
        results = await validator.run_validation()
        
        # Print summary
        validator.print_validation_summary()
        
        # Return appropriate exit code
        if all(results.values()):
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)