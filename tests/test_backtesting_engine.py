"""
Tests for the Backtesting Engine

This module tests all aspects of the event-driven backtesting framework
including order execution, slippage modeling, commission calculation,
performance metrics, and walk-forward analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strategies.backtesting_engine import (
    BacktestingEngine, MarketData, Order, Trade, Position, Portfolio,
    OrderType, OrderSide, OrderStatus, PerformanceMetrics,
    LinearSlippageModel, PerShareCommissionModel, PercentageCommissionModel,
    simple_momentum_strategy, buy_and_hold_strategy
)


class TestMarketData:
    """Test MarketData class"""
    
    def test_market_data_creation(self):
        """Test MarketData object creation"""
        timestamp = datetime.now()
        data = MarketData(
            timestamp=timestamp,
            symbol="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        
        assert data.timestamp == timestamp
        assert data.symbol == "AAPL"
        assert data.open == 150.0
        assert data.high == 152.0
        assert data.low == 149.0
        assert data.close == 151.0
        assert data.volume == 1000000


class TestSlippageModels:
    """Test slippage calculation models"""
    
    def test_linear_slippage_model(self):
        """Test linear slippage model"""
        model = LinearSlippageModel(base_slippage=0.001, volume_impact=0.00001)
        
        order = Order(
            id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000
        )
        
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=100000,
            spread=0.01
        )
        
        slippage = model.calculate_slippage(order, market_data)
        
        # Should include base slippage + volume impact + spread impact
        expected_slippage = 0.001 + (1000/100000) * 0.00001 + 0.01 * 0.5
        assert abs(slippage - expected_slippage) < 1e-6


class TestCommissionModels:
    """Test commission calculation models"""
    
    def test_per_share_commission_model(self):
        """Test per-share commission model"""
        model = PerShareCommissionModel(per_share=0.005, minimum=1.0)
        
        order = Order(
            id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000
        )
        
        commission = model.calculate_commission(order, 150.0)
        assert commission == 5.0  # 1000 * 0.005
        
        # Test minimum commission
        order.quantity = 100
        commission = model.calculate_commission(order, 150.0)
        assert commission == 1.0  # Minimum applies
    
    def test_percentage_commission_model(self):
        """Test percentage-based commission model"""
        model = PercentageCommissionModel(percentage=0.001, minimum=1.0)
        
        order = Order(
            id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000
        )
        
        commission = model.calculate_commission(order, 150.0)
        assert commission == 150.0  # 1000 * 150.0 * 0.001


class TestBacktestingEngine:
    """Test the main BacktestingEngine class"""
    
    @pytest.fixture
    def engine(self):
        """Create a backtesting engine for testing"""
        return BacktestingEngine(
            initial_capital=100000.0,
            random_seed=42
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        data = []
        base_time = datetime(2023, 1, 1, 9, 30)
        base_price = 100.0
        
        for i in range(100):
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.01)  # 1% daily volatility
            new_price = base_price * (1 + price_change)
            
            data.append(MarketData(
                timestamp=base_time + timedelta(days=i),
                symbol="TEST",
                open=base_price,
                high=max(base_price, new_price) * 1.005,
                low=min(base_price, new_price) * 0.995,
                close=new_price,
                volume=1000000,
                spread=0.01
            ))
            
            base_price = new_price
        
        return data
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.initial_capital == 100000.0
        assert engine.portfolio.cash == 100000.0
        assert engine.portfolio.total_value == 100000.0
        assert len(engine.orders) == 0
        assert len(engine.trades) == 0
    
    def test_submit_order(self, engine):
        """Test order submission"""
        order_id = engine.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            strategy="test"
        )
        
        assert order_id.startswith("ORDER_")
        assert len(engine.orders) == 1
        
        order = engine.orders[0]
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.strategy == "test"
        assert order.status == OrderStatus.PENDING
    
    def test_market_order_execution(self, engine):
        """Test market order execution"""
        # Submit a buy order
        order_id = engine.submit_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Process market data
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
            spread=0.01
        )
        
        engine.process_market_data(market_data)
        
        # Check that order was executed
        assert len(engine.trades) == 1
        assert len([o for o in engine.orders if o.status == OrderStatus.FILLED]) == 1
        
        # Check position was created
        assert "TEST" in engine.portfolio.positions
        position = engine.portfolio.positions["TEST"]
        assert position.quantity == 100
        assert position.avg_price > 100.0  # Should include slippage
    
    def test_limit_order_execution(self, engine):
        """Test limit order execution"""
        # Submit a buy limit order below current price
        order_id = engine.submit_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            price=99.0
        )
        
        # Process market data where low touches limit price
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=98.5,  # Touches limit price
            close=100.5,
            volume=1000000
        )
        
        engine.process_market_data(market_data)
        
        # Check that order was executed
        assert len(engine.trades) == 1
        trade = engine.trades[0]
        assert trade.price <= 99.0  # Should execute at or below limit price
    
    def test_insufficient_cash_rejection(self, engine):
        """Test order rejection due to insufficient cash"""
        # Try to buy more than we can afford
        order_id = engine.submit_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=10000,  # Very large quantity
            order_type=OrderType.MARKET
        )
        
        # Process market data
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000000
        )
        
        engine.process_market_data(market_data)
        
        # Check that order was rejected
        rejected_orders = [o for o in engine.orders if o.status == OrderStatus.REJECTED]
        assert len(rejected_orders) == 1
        assert len(engine.trades) == 0
    
    def test_position_tracking(self, engine):
        """Test position tracking with multiple trades"""
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000000
        )
        
        # First buy
        engine.submit_order("TEST", OrderSide.BUY, 100, OrderType.MARKET)
        engine.process_market_data(market_data)
        
        # Second buy at different price
        market_data.close = 105.0
        engine.submit_order("TEST", OrderSide.BUY, 50, OrderType.MARKET)
        engine.process_market_data(market_data)
        
        # Check position
        position = engine.portfolio.positions["TEST"]
        assert position.quantity == 150
        # Average price should be weighted average
        assert 100.0 < position.avg_price < 105.0
    
    def test_performance_metrics_calculation(self, engine, sample_market_data):
        """Test performance metrics calculation"""
        # Run a simple backtest
        results = engine.run_backtest(
            sample_market_data,
            buy_and_hold_strategy,
            {}
        )
        
        metrics = results['performance_metrics']
        
        # Check that all metrics are calculated
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.annualized_return, float)
        assert isinstance(metrics.volatility, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.total_trades >= 0
        
        # Check that portfolio value changed
        assert engine.portfolio.total_value != engine.initial_capital
    
    def test_walk_forward_analysis(self, engine, sample_market_data):
        """Test walk-forward analysis"""
        # Use smaller periods for testing
        results = engine.walk_forward_analysis(
            sample_market_data,
            buy_and_hold_strategy,
            training_period=20,
            testing_period=10,
            step_size=5
        )
        
        assert 'periods' in results
        assert 'aggregate_metrics' in results
        assert len(results['periods']) > 0
        
        # Check aggregate metrics
        agg = results['aggregate_metrics']
        assert 'avg_return' in agg
        assert 'avg_sharpe' in agg
        assert 'consistency_ratio' in agg
    
    def test_synthetic_scenario_testing(self, engine, sample_market_data):
        """Test synthetic scenario testing"""
        scenarios = ['trending_up', 'trending_down', 'high_volatility']
        
        results = engine.synthetic_scenario_testing(
            sample_market_data[:50],  # Use smaller dataset
            buy_and_hold_strategy,
            scenarios
        )
        
        assert len(results) == len(scenarios)
        
        for scenario in scenarios:
            assert scenario in results
            assert 'performance' in results[scenario]
            assert 'final_value' in results[scenario]
    
    def test_report_generation(self, engine, sample_market_data):
        """Test report generation"""
        results = engine.run_backtest(
            sample_market_data,
            buy_and_hold_strategy,
            {}
        )
        
        report = engine.generate_report(results)
        
        assert "Backtesting Report" in report
        assert "Total Return" in report
        assert "Sharpe Ratio" in report
        assert "Maximum Drawdown" in report
        assert "Total Trades" in report
    
    def test_reproducibility(self, sample_market_data):
        """Test that backtests are reproducible with same random seed"""
        # Run same backtest twice with same seed
        engine1 = BacktestingEngine(initial_capital=100000.0, random_seed=42)
        engine2 = BacktestingEngine(initial_capital=100000.0, random_seed=42)
        
        results1 = engine1.run_backtest(sample_market_data, simple_momentum_strategy, {})
        results2 = engine2.run_backtest(sample_market_data, simple_momentum_strategy, {})
        
        # Results should be identical
        assert abs(results1['performance_metrics'].total_return - 
                  results2['performance_metrics'].total_return) < 1e-10
        assert results1['performance_metrics'].total_trades == results2['performance_metrics'].total_trades


class TestStrategies:
    """Test the example strategy functions"""
    
    def test_buy_and_hold_strategy(self):
        """Test buy and hold strategy"""
        engine = BacktestingEngine(initial_capital=100000.0)
        
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000000
        )
        
        # First call should place buy order
        buy_and_hold_strategy(engine, market_data, {})
        assert len(engine.orders) == 1
        
        # Second call should not place another order
        buy_and_hold_strategy(engine, market_data, {})
        assert len(engine.orders) == 1
    
    def test_simple_momentum_strategy(self):
        """Test simple momentum strategy"""
        engine = BacktestingEngine(initial_capital=100000.0)
        
        # Create price data for moving average calculation
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]
        
        params = {'short_window': 5, 'long_window': 10}
        
        for i, price in enumerate(prices):
            market_data = MarketData(
                timestamp=datetime.now() + timedelta(days=i),
                symbol="TEST",
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000
            )
            
            simple_momentum_strategy(engine, market_data, params)
            engine.process_market_data(market_data)
        
        # Should have generated some trades due to momentum
        assert len(engine.trades) > 0


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_market_data(self):
        """Test backtesting with empty market data"""
        engine = BacktestingEngine()
        
        results = engine.run_backtest([], buy_and_hold_strategy, {})
        
        # Should handle gracefully
        assert results['performance_metrics'].total_return == 0.0
        assert len(results['trades']) == 0
    
    def test_single_data_point(self):
        """Test backtesting with single data point"""
        engine = BacktestingEngine()
        
        market_data = [MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000000
        )]
        
        results = engine.run_backtest(market_data, buy_and_hold_strategy, {})
        
        # Should handle gracefully
        assert isinstance(results['performance_metrics'].total_return, float)
    
    def test_zero_volume_data(self):
        """Test handling of zero volume data"""
        engine = BacktestingEngine()
        
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=0  # Zero volume
        )
        
        order_id = engine.submit_order("TEST", OrderSide.BUY, 100, OrderType.MARKET)
        engine.process_market_data(market_data)
        
        # Should still execute (slippage model should handle zero volume)
        assert len(engine.trades) == 1
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements"""
        engine = BacktestingEngine()
        
        # Extreme price jump
        market_data = MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=1000.0,  # 10x price jump
            low=100.0,
            close=1000.0,
            volume=1000000
        )
        
        order_id = engine.submit_order("TEST", OrderSide.BUY, 100, OrderType.MARKET)
        engine.process_market_data(market_data)
        
        # Should handle extreme movements
        assert len(engine.trades) == 1
        assert engine.trades[0].price > 100.0


if __name__ == "__main__":
    pytest.main([__file__])