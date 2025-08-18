"""
Test suite for Options Volatility Agent

Tests all major functionality including:
- IV surface analysis and skew detection
- Earnings calendar integration
- Greeks calculation and risk management
- Volatility regime detection
- Signal generation with explainability
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.options_volatility_agent import (
    OptionsVolatilityAgent,
    OptionsData,
    IVSurfacePoint,
    VolatilitySkew,
    EarningsEvent,
    GreeksRisk,
    OptionsSignal,
    VolatilityRegime,
    OptionsStrategy,
    BlackScholesCalculator,
    options_volatility_agent_node
)

class TestBlackScholesCalculator:
    """Test Black-Scholes calculator functionality"""
    
    def test_option_price_calculation(self):
        """Test option price calculation"""
        calculator = BlackScholesCalculator()
        
        # Test call option
        call_price = calculator.calculate_option_price(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call'
        )
        assert call_price > 0
        assert call_price < 100  # Sanity check
        
        # Test put option
        put_price = calculator.calculate_option_price(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='put'
        )
        assert put_price > 0
        assert put_price < 100  # Sanity check
        
        # Test put-call parity approximately
        forward_price = 100 * np.exp(0.05 * 0.25)
        parity_diff = abs(call_price - put_price - (forward_price - 100))
        assert parity_diff < 0.01  # Should be very close
    
    def test_greeks_calculation(self):
        """Test Greeks calculation"""
        calculator = BlackScholesCalculator()
        
        greeks = calculator.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call'
        )
        
        # Check all Greeks are present
        required_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in required_greeks:
            assert greek in greeks
            assert isinstance(greeks[greek], (int, float))
        
        # ATM call delta should be around 0.5
        assert 0.4 < greeks['delta'] < 0.6
        
        # Gamma should be positive
        assert greeks['gamma'] > 0
        
        # Theta should be negative for long options
        assert greeks['theta'] < 0
        
        # Vega should be positive
        assert greeks['vega'] > 0
    
    def test_implied_volatility_calculation(self):
        """Test implied volatility calculation"""
        calculator = BlackScholesCalculator()
        
        # Calculate theoretical price first
        true_iv = 0.25
        theoretical_price = calculator.calculate_option_price(
            S=100, K=100, T=0.25, r=0.05, sigma=true_iv, option_type='call'
        )
        
        # Calculate implied volatility from the price
        calculated_iv = calculator.calculate_implied_volatility(
            market_price=theoretical_price, S=100, K=100, T=0.25, r=0.05, option_type='call'
        )
        
        # Should be very close to the original IV
        assert abs(calculated_iv - true_iv) < 0.01

class TestOptionsVolatilityAgent:
    """Test Options Volatility Agent functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        return OptionsVolatilityAgent()
    
    @pytest.fixture
    def sample_options_data(self):
        """Create sample options data for testing"""
        current_price = 150.0
        options_data = []
        
        strikes = [140, 145, 150, 155, 160]
        expirations = [datetime.now() + timedelta(days=d) for d in [7, 30]]
        
        for exp in expirations:
            for strike in strikes:
                tte = (exp - datetime.now()).days / 365.0
                
                for option_type in ['call', 'put']:
                    # Mock implied volatility with skew
                    moneyness = strike / current_price
                    if moneyness < 0.95:  # OTM puts
                        iv = 0.30
                    elif moneyness > 1.05:  # OTM calls
                        iv = 0.22
                    else:  # ATM
                        iv = 0.25
                    
                    options_data.append(OptionsData(
                        symbol="AAPL",
                        expiration=exp,
                        strike=strike,
                        option_type=option_type,
                        bid=5.0,
                        ask=5.2,
                        last_price=5.1,
                        volume=100,
                        open_interest=500,
                        implied_volatility=iv,
                        delta=0.5 if option_type == 'call' else -0.5,
                        gamma=0.02,
                        theta=-0.05,
                        vega=0.15,
                        rho=0.08,
                        underlying_price=current_price,
                        time_to_expiration=tte
                    ))
        
        return options_data
    
    @pytest.mark.asyncio
    async def test_iv_surface_analysis(self, agent, sample_options_data):
        """Test IV surface analysis"""
        result = await agent.analyze_iv_surface("AAPL", sample_options_data)
        
        assert 'symbol' in result
        assert result['symbol'] == "AAPL"
        assert 'surface_points' in result
        assert result['surface_points'] > 0
        assert 'skew_analysis' in result
        assert 'surface_metrics' in result
        
        # Check surface metrics
        metrics = result['surface_metrics']
        assert 'average_iv' in metrics
        assert 'iv_std' in metrics
        assert 'min_iv' in metrics
        assert 'max_iv' in metrics
        assert metrics['average_iv'] > 0
    
    @pytest.mark.asyncio
    async def test_volatility_skew_analysis(self, agent, sample_options_data):
        """Test volatility skew analysis"""
        # Convert to surface points
        surface_points = []
        for option in sample_options_data:
            surface_points.append(IVSurfacePoint(
                strike=option.strike,
                expiration=option.expiration,
                time_to_expiration=option.time_to_expiration,
                moneyness=option.strike / option.underlying_price,
                implied_volatility=option.implied_volatility,
                delta=option.delta,
                volume=option.volume,
                open_interest=option.open_interest
            ))
        
        skew_results = await agent._analyze_volatility_skew("AAPL", surface_points)
        
        assert len(skew_results) > 0
        for skew in skew_results:
            assert isinstance(skew, VolatilitySkew)
            assert skew.symbol == "AAPL"
            assert isinstance(skew.skew_slope, (int, float))
            assert isinstance(skew.is_anomalous, bool)
    
    @pytest.mark.asyncio
    async def test_volatility_arbitrage_detection(self, agent, sample_options_data):
        """Test volatility arbitrage detection"""
        # Convert to surface points
        surface_points = []
        for option in sample_options_data:
            surface_points.append(IVSurfacePoint(
                strike=option.strike,
                expiration=option.expiration,
                time_to_expiration=option.time_to_expiration,
                moneyness=option.strike / option.underlying_price,
                implied_volatility=option.implied_volatility,
                delta=option.delta,
                volume=option.volume,
                open_interest=option.open_interest
            ))
        
        opportunities = await agent._detect_volatility_arbitrage(surface_points)
        
        # Should return a list (may be empty)
        assert isinstance(opportunities, list)
        
        # If opportunities exist, check structure
        for opp in opportunities:
            assert 'type' in opp
            assert 'confidence' in opp
            assert 0 <= opp['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_earnings_calendar_integration(self, agent, sample_options_data):
        """Test earnings calendar integration"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock earnings calendar
            mock_calendar = Mock()
            mock_calendar.empty = False
            mock_calendar.index = [datetime.now() + timedelta(days=5)]
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.calendar = mock_calendar
            mock_ticker.return_value = mock_ticker_instance
            
            earnings_event = await agent.integrate_earnings_calendar("AAPL", sample_options_data)
            
            if earnings_event:  # May be None if no earnings
                assert isinstance(earnings_event, EarningsEvent)
                assert earnings_event.symbol == "AAPL"
                assert earnings_event.days_to_earnings >= 0
                assert 0 <= earnings_event.iv_rank <= 1
                assert isinstance(earnings_event.strategy_recommendation, OptionsStrategy)
    
    @pytest.mark.asyncio
    async def test_greeks_risk_calculation(self, agent, sample_options_data):
        """Test Greeks risk calculation"""
        greeks_risk = await agent.calculate_greeks_risk(sample_options_data)
        
        assert isinstance(greeks_risk, GreeksRisk)
        assert isinstance(greeks_risk.total_delta, (int, float))
        assert isinstance(greeks_risk.total_gamma, (int, float))
        assert isinstance(greeks_risk.total_theta, (int, float))
        assert isinstance(greeks_risk.total_vega, (int, float))
        assert isinstance(greeks_risk.total_rho, (int, float))
        assert isinstance(greeks_risk.delta_neutral, bool)
        assert greeks_risk.gamma_risk_level in ['low', 'medium', 'high', 'unknown']
    
    @pytest.mark.asyncio
    async def test_volatility_regime_detection(self, agent):
        """Test volatility regime detection"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock historical data
            mock_hist = Mock()
            mock_hist.empty = False
            dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
            prices = 100 + np.cumsum(np.random.randn(30) * 0.02)  # Random walk
            mock_hist.__getitem__ = Mock(return_value=pd.Series(prices, index=dates))
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_hist
            mock_ticker.return_value = mock_ticker_instance
            
            regime = await agent.detect_volatility_regime("AAPL")
            
            assert isinstance(regime, VolatilityRegime)
            assert regime in [VolatilityRegime.LOW_VOL, VolatilityRegime.NORMAL_VOL, 
                            VolatilityRegime.HIGH_VOL, VolatilityRegime.EXTREME_VOL]
    
    @pytest.mark.asyncio
    async def test_options_signal_generation(self, agent):
        """Test options signal generation"""
        with patch.object(agent, '_get_options_data') as mock_get_options:
            # Mock options data
            mock_get_options.return_value = [
                OptionsData(
                    symbol="AAPL",
                    expiration=datetime.now() + timedelta(days=30),
                    strike=150.0,
                    option_type='call',
                    bid=5.0,
                    ask=5.2,
                    last_price=5.1,
                    volume=100,
                    open_interest=500,
                    implied_volatility=0.25,
                    delta=0.5,
                    gamma=0.02,
                    theta=-0.05,
                    vega=0.15,
                    rho=0.08,
                    underlying_price=150.0,
                    time_to_expiration=30/365.0
                )
            ]
            
            market_data = {'current_price': 150.0, 'volume': 1000000}
            signals = await agent.generate_options_signals("AAPL", market_data)
            
            assert isinstance(signals, list)
            
            # Check signal structure if any signals generated
            for signal in signals:
                assert isinstance(signal, OptionsSignal)
                assert signal.symbol == "AAPL"
                assert -1 <= signal.value <= 1
                assert 0 <= signal.confidence <= 1
                assert len(signal.top_3_reasons) <= 3
                assert isinstance(signal.strategy, OptionsStrategy)
                assert isinstance(signal.volatility_regime, VolatilityRegime)
                
                # Check top 3 reasons structure
                for reason in signal.top_3_reasons:
                    assert 'rank' in reason
                    assert 'factor' in reason
                    assert 'explanation' in reason
                    assert 'confidence' in reason
    
    def test_strategy_recommendation(self, agent):
        """Test earnings strategy recommendation"""
        # Test high IV scenario
        strategy = agent._recommend_earnings_strategy(
            days_to_earnings=10,
            expected_move=0.05,
            iv_rank=0.8,
            historical_moves=[0.08, 0.06, 0.09, 0.07]
        )
        assert isinstance(strategy, OptionsStrategy)
        
        # Test low IV scenario
        strategy = agent._recommend_earnings_strategy(
            days_to_earnings=5,
            expected_move=0.12,
            iv_rank=0.2,
            historical_moves=[0.08, 0.06, 0.09, 0.07]
        )
        assert isinstance(strategy, OptionsStrategy)

class TestLangGraphIntegration:
    """Test LangGraph integration functionality"""
    
    @pytest.mark.asyncio
    async def test_options_volatility_agent_node(self):
        """Test LangGraph node function"""
        # Mock state
        state = {
            'market_data': {
                'AAPL': {'current_price': 150.0, 'volume': 1000000}
            },
            'signals': {}
        }
        
        with patch('agents.options_volatility_agent.OptionsVolatilityAgent') as mock_agent_class:
            # Mock agent instance
            mock_agent = Mock()
            mock_agent.generate_options_signals = AsyncMock(return_value=[
                OptionsSignal(
                    signal_type='test',
                    symbol='AAPL',
                    strategy=OptionsStrategy.LONG_CALL,
                    value=0.7,
                    confidence=0.8,
                    top_3_reasons=[],
                    timestamp=datetime.utcnow(),
                    model_version='1.0.0',
                    expiration=datetime.now() + timedelta(days=30),
                    strike=150.0,
                    option_type='call',
                    entry_price=5.0,
                    target_profit=10.0,
                    stop_loss=2.5,
                    max_risk=5.0,
                    expected_return=7.5,
                    greeks=GreeksRisk(0.5, 0.02, -0.05, 0.15, 0.08, True, 'low', 0.15, -0.05),
                    iv_analysis={},
                    volatility_regime=VolatilityRegime.NORMAL_VOL
                )
            ])
            mock_agent_class.return_value = mock_agent
            
            result = await options_volatility_agent_node(state)
            
            assert 'signals' in result
            assert 'options_volatility' in result['signals']
            assert 'options_analysis' in result
            assert result['options_analysis']['agent'] == 'options_volatility'
            assert result['options_analysis']['signals_generated'] >= 0

class TestDataStructures:
    """Test data structure functionality"""
    
    def test_options_data_creation(self):
        """Test OptionsData creation"""
        option = OptionsData(
            symbol="AAPL",
            expiration=datetime.now() + timedelta(days=30),
            strike=150.0,
            option_type='call',
            bid=5.0,
            ask=5.2,
            last_price=5.1,
            volume=100,
            open_interest=500,
            implied_volatility=0.25,
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            rho=0.08,
            underlying_price=150.0,
            time_to_expiration=30/365.0
        )
        
        assert option.symbol == "AAPL"
        assert option.strike == 150.0
        assert option.option_type == 'call'
        assert option.implied_volatility == 0.25
    
    def test_volatility_regime_enum(self):
        """Test VolatilityRegime enum"""
        assert VolatilityRegime.LOW_VOL.value == "low_volatility"
        assert VolatilityRegime.NORMAL_VOL.value == "normal_volatility"
        assert VolatilityRegime.HIGH_VOL.value == "high_volatility"
        assert VolatilityRegime.EXTREME_VOL.value == "extreme_volatility"
    
    def test_options_strategy_enum(self):
        """Test OptionsStrategy enum"""
        assert OptionsStrategy.LONG_CALL.value == "long_call"
        assert OptionsStrategy.STRADDLE.value == "straddle"
        assert OptionsStrategy.IRON_CONDOR.value == "iron_condor"

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete options analysis workflow"""
        agent = OptionsVolatilityAgent()
        
        # Mock external dependencies
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock ticker data
            mock_hist = Mock()
            mock_hist.empty = False
            dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
            prices = 150 + np.cumsum(np.random.randn(30) * 0.02)
            mock_hist.__getitem__ = Mock(return_value=pd.Series(prices, index=dates))
            mock_hist['Close'] = pd.Series(prices, index=dates)
            
            mock_calendar = Mock()
            mock_calendar.empty = True
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_hist
            mock_ticker_instance.calendar = mock_calendar
            mock_ticker.return_value = mock_ticker_instance
            
            # Test the workflow
            market_data = {'current_price': 150.0, 'volume': 1000000}
            signals = await agent.generate_options_signals("AAPL", market_data)
            
            # Should complete without errors
            assert isinstance(signals, list)
            
            # If signals generated, verify structure
            for signal in signals:
                assert hasattr(signal, 'signal_type')
                assert hasattr(signal, 'confidence')
                assert hasattr(signal, 'top_3_reasons')

if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])