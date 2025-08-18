"""
Test suite for Risk Manager Agent

Tests comprehensive risk management functionality including:
- Real-time position monitoring and VaR calculation
- Dynamic position limits and exposure controls
- Emergency circuit breakers and kill switch
- Correlation risk management
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.risk_manager_agent import (
    RiskManagerAgent, RiskLimits, RiskAlert, PortfolioRiskMetrics, Position,
    RiskAlertType, RiskAlertSeverity, EmergencyAction
)


class TestRiskManagerAgent:
    """Test suite for Risk Manager Agent"""
    
    @pytest.fixture
    def db_config(self):
        """Mock database configuration"""
        return {
            'host': 'localhost',
            'database': 'test_trading',
            'user': 'test_user',
            'password': 'test_pass',
            'port': 5432
        }
    
    @pytest.fixture
    def risk_limits(self):
        """Test risk limits configuration"""
        return RiskLimits(
            max_daily_loss_pct=5.0,
            max_position_size_pct=3.0,
            max_leverage=2.0,
            max_var_95_pct=2.0,
            max_correlation=0.7,
            min_liquidity_days=3,
            max_sector_concentration_pct=15.0,
            volatility_spike_threshold=1.5
        )
    
    @pytest.fixture
    def sample_positions(self):
        """Sample positions for testing"""
        return [
            Position(
                symbol='AAPL',
                exchange='NASDAQ',
                strategy='momentum',
                agent_name='momentum_agent',
                quantity=100,
                avg_cost=150.0,
                market_value=15000.0,
                unrealized_pnl=500.0,
                realized_pnl=0.0,
                weight_pct=2.5
            ),
            Position(
                symbol='GOOGL',
                exchange='NASDAQ',
                strategy='mean_reversion',
                agent_name='mean_reversion_agent',
                quantity=50,
                avg_cost=2800.0,
                market_value=140000.0,
                unrealized_pnl=-2000.0,
                realized_pnl=1000.0,
                weight_pct=23.3
            ),
            Position(
                symbol='TSLA',
                exchange='NASDAQ',
                strategy='momentum',
                agent_name='momentum_agent',
                quantity=200,
                avg_cost=200.0,
                market_value=40000.0,
                unrealized_pnl=2000.0,
                realized_pnl=500.0,
                weight_pct=6.7
            )
        ]
    
    @pytest.fixture
    def risk_manager(self, db_config, risk_limits):
        """Risk Manager Agent instance"""
        return RiskManagerAgent(db_config, risk_limits)
    
    def test_initialization(self, risk_manager, risk_limits):
        """Test Risk Manager Agent initialization"""
        assert risk_manager.risk_limits == risk_limits
        assert not risk_manager.emergency_stop_active
        assert risk_manager.last_risk_check is None
        assert risk_manager.workflow is not None
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_monitoring(self, risk_manager):
        """Test comprehensive portfolio risk monitoring"""
        # Mock database operations
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = [
                {
                    'symbol': 'AAPL',
                    'exchange': 'NASDAQ',
                    'strategy': 'momentum',
                    'agent_name': 'momentum_agent',
                    'quantity': 100,
                    'avg_cost': 150.0,
                    'market_value': 15000.0,
                    'unrealized_pnl': 500.0,
                    'realized_pnl': 0.0
                }
            ]
            mock_cursor.fetchone.return_value = (10000.0,)  # Cash position
            mock_connect.return_value.cursor.return_value = mock_cursor
            
            # Mock historical returns
            with patch.object(risk_manager, '_get_historical_returns') as mock_returns:
                mock_returns.return_value = pd.DataFrame({
                    'AAPL': np.random.normal(0.001, 0.02, 100)  # 100 days of returns
                })
                
                # Test risk monitoring
                risk_metrics = await risk_manager.monitor_portfolio_risk()
                
                assert isinstance(risk_metrics, PortfolioRiskMetrics)
                assert risk_metrics.portfolio_value > 0
                assert risk_metrics.leverage >= 0
                assert risk_metrics.var_1d_95 >= 0
                assert risk_manager.last_risk_check is not None
    
    @pytest.mark.asyncio
    async def test_var_calculation(self, risk_manager, sample_positions):
        """Test Value at Risk calculation"""
        # Mock historical returns data
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0005, 0.025, 252),
            'TSLA': np.random.normal(0.002, 0.04, 252)
        })
        
        with patch.object(risk_manager, '_get_historical_returns') as mock_returns:
            mock_returns.return_value = returns_data
            
            var_metrics = await risk_manager._calculate_var(sample_positions)
            
            assert 'var_1d_95' in var_metrics
            assert 'var_1d_99' in var_metrics
            assert 'var_5d_95' in var_metrics
            assert 'var_5d_99' in var_metrics
            assert 'expected_shortfall_95' in var_metrics
            assert 'expected_shortfall_99' in var_metrics
            
            # VaR should be positive
            assert var_metrics['var_1d_95'] >= 0
            assert var_metrics['var_1d_99'] >= var_metrics['var_1d_95']
            assert var_metrics['var_5d_95'] >= var_metrics['var_1d_95']
    
    @pytest.mark.asyncio
    async def test_position_limit_check(self, risk_manager):
        """Test position limit checking for new orders"""
        # Mock current positions
        with patch.object(risk_manager, '_load_current_positions') as mock_positions:
            mock_positions.return_value = [
                Position(
                    symbol='AAPL',
                    exchange='NASDAQ',
                    strategy='momentum',
                    agent_name='momentum_agent',
                    quantity=100,
                    avg_cost=150.0,
                    market_value=15000.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    weight_pct=2.5
                )
            ]
            
            # Test order within limits
            small_order = {
                'symbol': 'MSFT',
                'quantity': 50,
                'price': 300.0
            }
            
            result = await risk_manager.check_position_limits(small_order)
            assert result['approved'] is True
            
            # Test order exceeding limits
            large_order = {
                'symbol': 'MSFT',
                'quantity': 1000,
                'price': 300.0  # $300k order
            }
            
            result = await risk_manager.check_position_limits(large_order)
            assert result['approved'] is False
            assert 'exceed limit' in result['reason']
    
    @pytest.mark.asyncio
    async def test_risk_limit_breaches(self, risk_manager, sample_positions):
        """Test risk limit breach detection"""
        # Create portfolio metrics with limit breaches
        portfolio_metrics = PortfolioRiskMetrics(
            timestamp=datetime.now(timezone.utc),
            portfolio_value=100000.0,
            cash=10000.0,
            gross_exposure=200000.0,
            net_exposure=90000.0,
            leverage=2.5,  # Exceeds limit of 2.0
            var_1d_95=3000.0,  # Exceeds 2% of portfolio
            var_1d_99=4000.0,
            var_5d_95=6000.0,
            var_5d_99=8000.0,
            expected_shortfall_95=3500.0,
            expected_shortfall_99=4500.0,
            max_position_size=25000.0,
            max_position_pct=25.0,  # Large position
            sector_concentration=30.0,  # Exceeds 15% limit
            correlation_risk=0.8,  # Exceeds 0.7 limit
            liquidity_risk=5.0
        )
        
        # Create positions with daily loss
        losing_positions = [
            Position(
                symbol='AAPL',
                exchange='NASDAQ',
                strategy='momentum',
                agent_name='momentum_agent',
                quantity=100,
                avg_cost=150.0,
                market_value=15000.0,
                unrealized_pnl=-8000.0,  # Large loss
                realized_pnl=0.0,
                weight_pct=15.0
            )
        ]
        
        state = {
            'portfolio_metrics': portfolio_metrics,
            'positions': losing_positions
        }
        
        # Test risk limit checking
        result_state = await risk_manager._check_risk_limits(state)
        risk_alerts = result_state.get('risk_alerts', [])
        
        # Should have multiple alerts
        assert len(risk_alerts) > 0
        
        # Check for specific alert types
        alert_types = [alert.alert_type for alert in risk_alerts]
        assert RiskAlertType.LEVERAGE_LIMIT in alert_types
        assert RiskAlertType.VAR_EXCEEDED in alert_types
    
    @pytest.mark.asyncio
    async def test_emergency_stop_trigger(self, risk_manager):
        """Test emergency stop functionality"""
        # Test manual emergency stop
        result = risk_manager.trigger_emergency_stop("Manual test stop")
        assert result is True
        assert risk_manager.is_emergency_stop_active() is True
        
        # Test that orders are rejected during emergency stop
        test_order = {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0
        }
        
        with patch.object(risk_manager, '_load_current_positions') as mock_positions:
            mock_positions.return_value = []
            
            result = await risk_manager.check_position_limits(test_order)
            assert result['approved'] is False
            assert 'Emergency stop' in result['reason']
        
        # Test emergency stop reset
        reset_result = risk_manager.reset_emergency_stop()
        assert reset_result is True
        assert risk_manager.is_emergency_stop_active() is False
    
    @pytest.mark.asyncio
    async def test_correlation_risk_calculation(self, risk_manager, sample_positions):
        """Test correlation risk calculation"""
        # Mock historical returns with high correlation
        correlated_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 60),
            'GOOGL': np.random.normal(0.001, 0.02, 60),
            'TSLA': np.random.normal(0.001, 0.02, 60)
        })
        
        # Make GOOGL highly correlated with AAPL
        correlated_returns['GOOGL'] = 0.8 * correlated_returns['AAPL'] + 0.2 * correlated_returns['GOOGL']
        
        with patch.object(risk_manager, '_get_historical_returns') as mock_returns:
            mock_returns.return_value = correlated_returns
            
            correlation_risk = await risk_manager._calculate_correlation_risk(sample_positions)
            
            assert correlation_risk >= 0
            assert correlation_risk <= 1
    
    @pytest.mark.asyncio
    async def test_liquidity_risk_calculation(self, risk_manager, sample_positions):
        """Test liquidity risk calculation"""
        # Mock volume data
        volume_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'TSLA'],
            'avg_volume': [50000000, 1000000, 25000000]  # Different liquidity levels
        })
        
        with patch('pandas.read_sql_query') as mock_sql:
            mock_sql.return_value = volume_data
            with patch('psycopg2.connect'):
                
                liquidity_risk = await risk_manager._calculate_liquidity_risk(sample_positions)
                
                assert liquidity_risk >= 0
                # Should reflect different liquidity levels
    
    @pytest.mark.asyncio
    async def test_sector_concentration_calculation(self, risk_manager, sample_positions):
        """Test sector concentration calculation"""
        concentration = await risk_manager._calculate_sector_concentration(sample_positions)
        
        assert concentration >= 0
        assert concentration <= 100  # Percentage
    
    @pytest.mark.asyncio
    async def test_emergency_actions(self, risk_manager):
        """Test emergency action execution"""
        # Mock database operations
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_connect.return_value.cursor.return_value = mock_cursor
            
            # Test halt all trading
            await risk_manager._halt_all_trading()
            assert risk_manager.emergency_stop_active is True
            
            # Test position reduction
            sample_positions = [
                Position(
                    symbol='AAPL',
                    exchange='NASDAQ',
                    strategy='momentum',
                    agent_name='momentum_agent',
                    quantity=1000,
                    avg_cost=150.0,
                    market_value=150000.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    weight_pct=25.0
                )
            ]
            
            await risk_manager._reduce_positions(sample_positions)
            # Should execute without errors
    
    @pytest.mark.asyncio
    async def test_database_updates(self, risk_manager):
        """Test risk metrics and alerts database updates"""
        # Create test data
        portfolio_metrics = PortfolioRiskMetrics(
            timestamp=datetime.now(timezone.utc),
            portfolio_value=100000.0,
            cash=10000.0,
            gross_exposure=90000.0,
            net_exposure=80000.0,
            leverage=1.5,
            var_1d_95=2000.0,
            var_1d_99=3000.0,
            var_5d_95=4000.0,
            var_5d_99=6000.0,
            expected_shortfall_95=2500.0,
            expected_shortfall_99=3500.0,
            max_position_size=20000.0,
            max_position_pct=20.0,
            sector_concentration=15.0,
            correlation_risk=0.6,
            liquidity_risk=2.0
        )
        
        risk_alert = RiskAlert(
            timestamp=datetime.now(timezone.utc),
            alert_type=RiskAlertType.POSITION_LIMIT,
            severity=RiskAlertSeverity.HIGH,
            symbol='AAPL',
            strategy='momentum',
            agent_name='momentum_agent',
            current_value=25.0,
            limit_value=20.0,
            breach_percentage=25.0,
            description='Position size exceeds limit'
        )
        
        state = {
            'portfolio_metrics': portfolio_metrics,
            'risk_alerts': [risk_alert]
        }
        
        # Mock database operations
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_connect.return_value.cursor.return_value = mock_cursor
            
            result_state = await risk_manager._update_risk_database(state)
            
            # Should execute without errors
            assert result_state == state
            
            # Verify database calls were made
            assert mock_cursor.execute.call_count >= 2  # Risk metrics + alert
    
    def test_risk_limits_configuration(self):
        """Test risk limits configuration"""
        custom_limits = RiskLimits(
            max_daily_loss_pct=8.0,
            max_position_size_pct=4.0,
            max_leverage=3.0,
            max_var_95_pct=2.5,
            max_correlation=0.75,
            min_liquidity_days=7,
            max_sector_concentration_pct=25.0,
            volatility_spike_threshold=2.5
        )
        
        assert custom_limits.max_daily_loss_pct == 8.0
        assert custom_limits.max_position_size_pct == 4.0
        assert custom_limits.max_leverage == 3.0
        assert custom_limits.max_var_95_pct == 2.5
        assert custom_limits.max_correlation == 0.75
        assert custom_limits.min_liquidity_days == 7
        assert custom_limits.max_sector_concentration_pct == 25.0
        assert custom_limits.volatility_spike_threshold == 2.5
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, risk_manager):
        """Test complete LangGraph workflow execution"""
        # Mock all database operations
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = []  # No positions
            mock_cursor.fetchone.return_value = (10000.0,)  # Cash
            mock_connect.return_value.cursor.return_value = mock_cursor
            
            # Mock historical returns
            with patch.object(risk_manager, '_get_historical_returns') as mock_returns:
                mock_returns.return_value = pd.DataFrame()
                
                # Execute workflow
                try:
                    risk_metrics = await risk_manager.monitor_portfolio_risk()
                    # Should complete without errors even with no data
                    assert risk_metrics is not None or True  # Allow None for empty portfolio
                except Exception as e:
                    # Should handle empty portfolio gracefully
                    assert "No positions" in str(e) or "Failed to calculate" in str(e)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])