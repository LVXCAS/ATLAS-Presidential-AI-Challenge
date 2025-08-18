"""
Tests for Execution Engine Agent

Tests cover:
- LangGraph workflow execution
- Smart order routing
- Market impact estimation
- Execution algorithms (TWAP, VWAP, Implementation Shortfall)
- Order size optimization
- Slippage minimization
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from agents.execution_engine_agent import (
    ExecutionEngineAgent, ExecutionOrder, ExecutionSlice, ExecutionState,
    SmartOrderRouter, ExecutionAlgorithmEngine, MarketImpactModel,
    ExecutionAlgorithm, OrderSide, VenueType, LiquidityProfile
)
from agents.broker_integration import AlpacaBrokerIntegration, OrderResponse, OrderStatus
from agents.market_data_ingestor import MarketDataIngestorAgent


class TestMarketImpactModel:
    """Test market impact estimation"""
    
    def test_market_impact_estimation(self):
        """Test market impact calculation"""
        model = MarketImpactModel(
            symbol="AAPL",
            avg_daily_volume=Decimal('50000000'),
            volatility=Decimal('0.02'),
            bid_ask_spread=Decimal('0.01'),
            liquidity_profile=LiquidityProfile.HIGH
        )
        
        # Test small order
        small_impact = model.estimate_impact(Decimal('1000'), urgency=0.5)
        assert small_impact < Decimal('0.01')  # Less than 1%
        
        # Test large order
        large_impact = model.estimate_impact(Decimal('100000'), urgency=0.5)
        assert large_impact > small_impact
        
        # Test urgency effect
        urgent_impact = model.estimate_impact(Decimal('10000'), urgency=0.9)
        normal_impact = model.estimate_impact(Decimal('10000'), urgency=0.5)
        assert urgent_impact > normal_impact
    
    def test_liquidity_profile_impact(self):
        """Test impact of liquidity profile on market impact"""
        base_params = {
            'symbol': 'TEST',
            'avg_daily_volume': Decimal('10000000'),
            'volatility': Decimal('0.02'),
            'bid_ask_spread': Decimal('0.01')
        }
        
        high_liquidity = MarketImpactModel(**base_params, liquidity_profile=LiquidityProfile.HIGH)
        low_liquidity = MarketImpactModel(**base_params, liquidity_profile=LiquidityProfile.LOW)
        
        order_size = Decimal('5000')
        high_impact = high_liquidity.estimate_impact(order_size)
        low_impact = low_liquidity.estimate_impact(order_size)
        
        assert low_impact > high_impact


class TestSmartOrderRouter:
    """Test smart order routing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.router = SmartOrderRouter()
        self.test_order = ExecutionOrder(
            symbol="AAPL",
            total_quantity=Decimal('5000'),
            side=OrderSide.BUY,
            urgency=0.5,
            allow_dark_pools=True
        )
        self.market_data = {
            'current_price': 150.0,
            'spread': 0.02,
            'volume': 1000000
        }
    
    def test_venue_scoring(self):
        """Test venue scoring algorithm"""
        scores = self.router._score_venues(self.test_order, self.market_data)
        
        assert len(scores) > 0
        assert all(0 <= score <= 2 for score in scores.values())
        
        # NASDAQ should have a high score
        assert 'NASDAQ' in scores
        assert scores['NASDAQ'] > 0.5
    
    def test_order_routing(self):
        """Test order routing across venues"""
        slices = self.router.route_order(self.test_order, self.market_data)
        
        assert len(slices) > 0
        
        # Total quantity should be preserved
        total_routed = sum(slice_order.quantity for slice_order in slices)
        assert total_routed <= self.test_order.total_quantity
        
        # Each slice should have valid venue
        for slice_order in slices:
            assert slice_order.venue.venue_id in self.router.venues
            assert slice_order.quantity > 0
    
    def test_dark_pool_routing(self):
        """Test dark pool routing for large orders"""
        large_order = ExecutionOrder(
            symbol="AAPL",
            total_quantity=Decimal('10000'),
            side=OrderSide.BUY,
            urgency=0.3,  # Low urgency favors dark pools
            allow_dark_pools=True
        )
        
        slices = self.router.route_order(large_order, self.market_data)
        
        # Should include dark pool venues for large orders
        dark_pool_slices = [s for s in slices if s.venue.venue_type == VenueType.DARK_POOL]
        assert len(dark_pool_slices) > 0
    
    def test_small_order_routing(self):
        """Test routing for small orders"""
        small_order = ExecutionOrder(
            symbol="AAPL",
            total_quantity=Decimal('100'),
            side=OrderSide.BUY,
            urgency=0.8,  # High urgency
            allow_dark_pools=False
        )
        
        slices = self.router.route_order(small_order, self.market_data)
        
        # Should not route to dark pools for small orders
        dark_pool_slices = [s for s in slices if s.venue.venue_type == VenueType.DARK_POOL]
        assert len(dark_pool_slices) == 0


class TestExecutionAlgorithmEngine:
    """Test execution algorithms"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.market_data_ingestor = Mock(spec=MarketDataIngestorAgent)
        self.engine = ExecutionAlgorithmEngine(self.market_data_ingestor)
        
        self.test_order = ExecutionOrder(
            symbol="AAPL",
            total_quantity=Decimal('2000'),
            side=OrderSide.BUY,
            time_horizon_minutes=60
        )
    
    @pytest.mark.asyncio
    async def test_twap_execution(self):
        """Test TWAP algorithm"""
        slices = await self.engine.execute_twap(self.test_order, 30)
        
        assert len(slices) > 0
        
        # Total quantity should be preserved
        total_quantity = sum(slice_order.quantity for slice_order in slices)
        assert total_quantity == self.test_order.total_quantity
        
        # Slices should be roughly equal in size
        slice_sizes = [slice_order.quantity for slice_order in slices]
        avg_size = sum(slice_sizes) / len(slice_sizes)
        
        for size in slice_sizes:
            assert abs(size - avg_size) / avg_size < 0.1  # Within 10%
    
    @pytest.mark.asyncio
    async def test_vwap_execution(self):
        """Test VWAP algorithm"""
        # Mock volume profile
        volume_profile = [
            {'timestamp': datetime.now(timezone.utc), 'volume': 1000000, 'interval': 0},
            {'timestamp': datetime.now(timezone.utc), 'volume': 1500000, 'interval': 1},
            {'timestamp': datetime.now(timezone.utc), 'volume': 800000, 'interval': 2}
        ]
        
        slices = await self.engine.execute_vwap(self.test_order, volume_profile)
        
        assert len(slices) > 0
        
        # Total quantity should be preserved
        total_quantity = sum(slice_order.quantity for slice_order in slices)
        assert abs(total_quantity - self.test_order.total_quantity) < Decimal('1')
        
        # Slice sizes should be proportional to volume
        slice_quantities = [slice_order.quantity for slice_order in slices]
        volumes = [period['volume'] for period in volume_profile]
        
        # Higher volume periods should have larger slices
        max_volume_idx = volumes.index(max(volumes))
        max_slice_qty = max(slice_quantities)
        assert slice_quantities[max_volume_idx] == max_slice_qty
    
    @pytest.mark.asyncio
    async def test_implementation_shortfall(self):
        """Test Implementation Shortfall algorithm"""
        market_impact_model = MarketImpactModel(
            symbol="AAPL",
            avg_daily_volume=Decimal('50000000'),
            volatility=Decimal('0.02'),
            bid_ask_spread=Decimal('0.01')
        )
        
        slices = await self.engine.execute_implementation_shortfall(
            self.test_order, market_impact_model
        )
        
        assert len(slices) > 0
        assert len(slices) <= 20  # Max slices limit
        
        # Total quantity should be preserved
        total_quantity = sum(slice_order.quantity for slice_order in slices)
        assert total_quantity <= self.test_order.total_quantity


class TestExecutionEngineAgent:
    """Test main execution engine agent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_broker = Mock(spec=AlpacaBrokerIntegration)
        self.mock_market_data = Mock(spec=MarketDataIngestorAgent)
        
        # Mock market data response
        self.mock_market_data.get_latest_data = AsyncMock(return_value={
            'close': 150.0,
            'volume': 1000000,
            'symbol': 'AAPL'
        })
        
        self.agent = ExecutionEngineAgent(
            broker_integration=self.mock_broker,
            market_data_ingestor=self.mock_market_data
        )
        
        self.test_order = ExecutionOrder(
            symbol="AAPL",
            total_quantity=Decimal('1000'),
            side=OrderSide.BUY,
            algorithm=ExecutionAlgorithm.TWAP,
            client_order_id="test_001"
        )
    
    @pytest.mark.asyncio
    async def test_order_analysis(self):
        """Test order analysis step"""
        market_data = {
            'current_price': 150.0,
            'avg_daily_volume': 50000000,
            'volume': 1000000
        }
        
        market_impact_model = MarketImpactModel(
            symbol="AAPL",
            avg_daily_volume=Decimal('50000000'),
            volatility=Decimal('0.02'),
            bid_ask_spread=Decimal('0.01')
        )
        
        state = ExecutionState(
            execution_order=self.test_order,
            market_data=market_data,
            venue_data={},
            market_impact_model=market_impact_model,
            execution_slices=[],
            completed_slices=[],
            failed_slices=[],
            current_slice=None,
            execution_metrics={},
            error_log=[],
            next_action="analyze_order"
        )
        
        result_state = await self.agent._analyze_order(state)
        
        assert 'order_classification' in result_state.execution_metrics
        assert 'participation_rate' in result_state.execution_metrics
        assert result_state.next_action == "estimate_market_impact"
    
    @pytest.mark.asyncio
    async def test_market_impact_estimation(self):
        """Test market impact estimation step"""
        market_impact_model = MarketImpactModel(
            symbol="AAPL",
            avg_daily_volume=Decimal('50000000'),
            volatility=Decimal('0.02'),
            bid_ask_spread=Decimal('0.01')
        )
        
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={},
            venue_data={},
            market_impact_model=market_impact_model,
            execution_slices=[],
            completed_slices=[],
            failed_slices=[],
            current_slice=None,
            execution_metrics={},
            error_log=[],
            next_action="estimate_market_impact"
        )
        
        result_state = await self.agent._estimate_market_impact(state)
        
        assert 'impact_estimates' in result_state.execution_metrics
        assert 'selected_impact_bps' in result_state.execution_metrics
        assert result_state.next_action == "select_algorithm"
    
    @pytest.mark.asyncio
    async def test_algorithm_selection(self):
        """Test algorithm selection step"""
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={},
            venue_data={},
            market_impact_model=Mock(),
            execution_slices=[],
            completed_slices=[],
            failed_slices=[],
            current_slice=None,
            execution_metrics={
                'participation_rate': 0.02,
                'selected_impact_bps': 25
            },
            error_log=[],
            next_action="select_algorithm"
        )
        
        result_state = await self.agent._select_algorithm(state)
        
        assert 'selected_algorithm' in result_state.execution_metrics
        assert result_state.next_action == "create_execution_plan"
    
    @pytest.mark.asyncio
    async def test_execution_plan_creation(self):
        """Test execution plan creation"""
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={'current_price': 150.0},
            venue_data={},
            market_impact_model=Mock(),
            execution_slices=[],
            completed_slices=[],
            failed_slices=[],
            current_slice=None,
            execution_metrics={'selected_algorithm': 'twap'},
            error_log=[],
            next_action="create_execution_plan"
        )
        
        result_state = await self.agent._create_execution_plan(state)
        
        assert len(result_state.execution_slices) > 0
        assert 'total_slices' in result_state.execution_metrics
        assert result_state.next_action == "route_orders"
    
    @pytest.mark.asyncio
    async def test_order_routing(self):
        """Test order routing step"""
        # Create mock execution slices
        mock_slice = ExecutionSlice(
            slice_id="test_slice_1",
            symbol="AAPL",
            quantity=Decimal('500'),
            side=OrderSide.BUY,
            venue=Mock(),
            order_type=Mock()
        )
        
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={'current_price': 150.0, 'spread': 0.02},
            venue_data={},
            market_impact_model=Mock(),
            execution_slices=[mock_slice],
            completed_slices=[],
            failed_slices=[],
            current_slice=None,
            execution_metrics={'selected_algorithm': 'smart_routing'},
            error_log=[],
            next_action="route_orders"
        )
        
        result_state = await self.agent._route_orders(state)
        
        assert len(result_state.execution_slices) > 0
        assert result_state.next_action == "execute_slice"
    
    @pytest.mark.asyncio
    async def test_slice_execution_success(self):
        """Test successful slice execution"""
        # Mock successful order submission
        mock_order_response = OrderResponse(
            id="order_123",
            client_order_id="test_slice_1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            submitted_at=datetime.now(timezone.utc),
            filled_at=None,
            expired_at=None,
            canceled_at=None,
            failed_at=None,
            replaced_at=None,
            symbol="AAPL",
            asset_id="asset_123",
            asset_class="us_equity",
            qty=Decimal('500'),
            filled_qty=Decimal('0'),
            type=Mock(),
            side=OrderSide.BUY,
            time_in_force=Mock(),
            limit_price=Decimal('150.00'),
            stop_price=None,
            status=OrderStatus.NEW,
            extended_hours=False
        )
        
        self.mock_broker.submit_order = AsyncMock(return_value=mock_order_response)
        
        mock_slice = ExecutionSlice(
            slice_id="test_slice_1",
            symbol="AAPL",
            quantity=Decimal('500'),
            side=OrderSide.BUY,
            venue=Mock(),
            order_type=Mock(),
            limit_price=Decimal('150.00')
        )
        
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={},
            venue_data={},
            market_impact_model=Mock(),
            execution_slices=[mock_slice],
            completed_slices=[],
            failed_slices=[],
            current_slice=None,
            execution_metrics={},
            error_log=[],
            next_action="execute_slice"
        )
        
        result_state = await self.agent._execute_slice(state)
        
        assert len(result_state.execution_order.child_orders) == 1
        assert result_state.current_slice is not None
        assert result_state.next_action == "continue"
        
        # Verify broker was called
        self.mock_broker.submit_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_slice_execution_failure(self):
        """Test failed slice execution"""
        # Mock failed order submission
        self.mock_broker.submit_order = AsyncMock(side_effect=Exception("Order failed"))
        
        mock_slice = ExecutionSlice(
            slice_id="test_slice_1",
            symbol="AAPL",
            quantity=Decimal('500'),
            side=OrderSide.BUY,
            venue=Mock(),
            order_type=Mock()
        )
        
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={},
            venue_data={},
            market_impact_model=Mock(),
            execution_slices=[mock_slice],
            completed_slices=[],
            failed_slices=[],
            current_slice=None,
            execution_metrics={},
            error_log=[],
            next_action="execute_slice"
        )
        
        result_state = await self.agent._execute_slice(state)
        
        assert len(result_state.failed_slices) == 1
        assert len(result_state.error_log) > 0
        assert result_state.next_action == "optimize"
    
    @pytest.mark.asyncio
    async def test_execution_monitoring(self):
        """Test execution monitoring"""
        # Mock filled order status
        mock_order_status = OrderResponse(
            id="order_123",
            client_order_id="test_slice_1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            submitted_at=datetime.now(timezone.utc),
            filled_at=datetime.now(timezone.utc),
            expired_at=None,
            canceled_at=None,
            failed_at=None,
            replaced_at=None,
            symbol="AAPL",
            asset_id="asset_123",
            asset_class="us_equity",
            qty=Decimal('500'),
            filled_qty=Decimal('500'),
            type=Mock(),
            side=OrderSide.BUY,
            time_in_force=Mock(),
            limit_price=Decimal('150.00'),
            stop_price=None,
            status=OrderStatus.FILLED,
            extended_hours=False
        )
        
        self.mock_broker.get_order_status = AsyncMock(return_value=mock_order_status)
        
        # Create mock child order
        child_order = OrderResponse(
            id="order_123",
            client_order_id="test_slice_1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            submitted_at=datetime.now(timezone.utc),
            filled_at=None,
            expired_at=None,
            canceled_at=None,
            failed_at=None,
            replaced_at=None,
            symbol="AAPL",
            asset_id="asset_123",
            asset_class="us_equity",
            qty=Decimal('500'),
            filled_qty=Decimal('0'),
            type=Mock(),
            side=OrderSide.BUY,
            time_in_force=Mock(),
            limit_price=Decimal('150.00'),
            stop_price=None,
            status=OrderStatus.NEW,
            extended_hours=False
        )
        
        self.test_order.child_orders = [child_order]
        
        mock_slice = ExecutionSlice(
            slice_id="test_slice_1",
            symbol="AAPL",
            quantity=Decimal('500'),
            side=OrderSide.BUY,
            venue=Mock(),
            order_type=Mock()
        )
        
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={},
            venue_data={},
            market_impact_model=Mock(),
            execution_slices=[],
            completed_slices=[],
            failed_slices=[],
            current_slice=mock_slice,
            execution_metrics={},
            error_log=[],
            next_action="monitor_execution"
        )
        
        result_state = await self.agent._monitor_execution(state)
        
        assert len(result_state.completed_slices) == 1
        assert result_state.execution_order.executed_quantity == Decimal('500')
        assert result_state.current_slice is None
    
    @pytest.mark.asyncio
    async def test_execution_optimization(self):
        """Test execution optimization"""
        # Create failed slice
        failed_slice = ExecutionSlice(
            slice_id="failed_slice",
            symbol="AAPL",
            quantity=Decimal('500'),
            side=OrderSide.BUY,
            venue=Mock(),
            order_type=Mock()
        )
        
        # Create remaining slice
        remaining_slice = ExecutionSlice(
            slice_id="remaining_slice",
            symbol="AAPL",
            quantity=Decimal('500'),
            side=OrderSide.BUY,
            venue=Mock(),
            order_type=Mock(),
            limit_price=Decimal('150.00')
        )
        
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={'current_price': 150.0, 'spread': 0.02},
            venue_data={},
            market_impact_model=Mock(),
            execution_slices=[remaining_slice],
            completed_slices=[],
            failed_slices=[failed_slice],
            current_slice=None,
            execution_metrics={},
            error_log=[],
            next_action="optimize_execution"
        )
        
        result_state = await self.agent._optimize_execution(state)
        
        # Should have optimized remaining slices
        assert result_state.next_action == "continue"
        
        # Slice quantity should be reduced
        optimized_slice = result_state.execution_slices[0]
        assert optimized_slice.quantity < Decimal('500')
    
    @pytest.mark.asyncio
    async def test_execution_completion(self):
        """Test execution completion"""
        self.test_order.executed_quantity = Decimal('800')
        self.test_order.remaining_quantity = Decimal('200')
        self.test_order.avg_execution_price = Decimal('150.50')
        
        completed_slice = ExecutionSlice(
            slice_id="completed_slice",
            symbol="AAPL",
            quantity=Decimal('800'),
            side=OrderSide.BUY,
            venue=Mock(),
            order_type=Mock()
        )
        
        state = ExecutionState(
            execution_order=self.test_order,
            market_data={'current_price': 150.0},
            venue_data={},
            market_impact_model=Mock(),
            execution_slices=[],
            completed_slices=[completed_slice],
            failed_slices=[],
            current_slice=None,
            execution_metrics={},
            error_log=[],
            next_action="complete_execution"
        )
        
        result_state = await self.agent._complete_execution(state)
        
        assert 'completion_timestamp' in result_state.execution_metrics
        assert 'executed_quantity' in result_state.execution_metrics
        assert 'total_slippage_bps' in result_state.execution_metrics
        assert 'execution_rate' in result_state.execution_metrics
        assert result_state.next_action == "end"
    
    @pytest.mark.asyncio
    async def test_full_execution_workflow(self):
        """Test complete execution workflow"""
        # Mock successful order execution
        mock_order_response = OrderResponse(
            id="order_123",
            client_order_id="test_slice_1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            submitted_at=datetime.now(timezone.utc),
            filled_at=datetime.now(timezone.utc),
            expired_at=None,
            canceled_at=None,
            failed_at=None,
            replaced_at=None,
            symbol="AAPL",
            asset_id="asset_123",
            asset_class="us_equity",
            qty=Decimal('1000'),
            filled_qty=Decimal('1000'),
            type=Mock(),
            side=OrderSide.BUY,
            time_in_force=Mock(),
            limit_price=Decimal('150.00'),
            stop_price=None,
            status=OrderStatus.FILLED,
            extended_hours=False
        )
        
        self.mock_broker.submit_order = AsyncMock(return_value=mock_order_response)
        self.mock_broker.get_order_status = AsyncMock(return_value=mock_order_response)
        
        # Execute order
        final_state = await self.agent.execute_order(self.test_order)
        
        # Verify execution completed
        assert final_state.execution_order.symbol == "AAPL"
        assert 'completion_timestamp' in final_state.execution_metrics
        
        # Verify order was tracked
        assert "test_001" in self.agent.completed_executions
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        metrics = self.agent.get_performance_metrics()
        
        assert 'total_orders' in metrics
        assert 'successful_executions' in metrics
        assert 'avg_slippage_bps' in metrics
        assert 'success_rate' in metrics
        assert 'active_executions' in metrics
        assert 'completed_executions' in metrics
    
    @pytest.mark.asyncio
    async def test_execution_status_retrieval(self):
        """Test execution status retrieval"""
        # Add mock execution to active executions
        mock_state = Mock(spec=ExecutionState)
        self.agent.active_executions["test_execution"] = mock_state
        
        # Test active execution retrieval
        status = await self.agent.get_execution_status("test_execution")
        assert status == mock_state
        
        # Test non-existent execution
        status = await self.agent.get_execution_status("non_existent")
        assert status is None


class TestExecutionIntegration:
    """Integration tests for execution engine"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_execution(self):
        """Test end-to-end execution flow"""
        # Create mocks
        mock_broker = Mock(spec=AlpacaBrokerIntegration)
        mock_market_data = Mock(spec=MarketDataIngestorAgent)
        
        # Mock responses
        mock_market_data.get_latest_data = AsyncMock(return_value={
            'close': 150.0,
            'volume': 1000000
        })
        
        mock_order_response = Mock(spec=OrderResponse)
        mock_order_response.id = "order_123"
        mock_order_response.client_order_id = "test_slice"
        mock_order_response.status = OrderStatus.FILLED
        mock_order_response.filled_qty = Decimal('1000')
        mock_order_response.limit_price = Decimal('150.00')
        
        mock_broker.submit_order = AsyncMock(return_value=mock_order_response)
        mock_broker.get_order_status = AsyncMock(return_value=mock_order_response)
        
        # Create agent
        agent = ExecutionEngineAgent(mock_broker, mock_market_data)
        
        # Create test order
        test_order = ExecutionOrder(
            symbol="AAPL",
            total_quantity=Decimal('1000'),
            side=OrderSide.BUY,
            algorithm=ExecutionAlgorithm.MARKET,
            client_order_id="integration_test"
        )
        
        # Execute order
        final_state = await agent.execute_order(test_order)
        
        # Verify execution
        assert final_state.execution_order.symbol == "AAPL"
        assert len(final_state.execution_metrics) > 0
        
        # Verify broker interactions
        assert mock_broker.submit_order.called
        
        # Verify metrics updated
        metrics = agent.get_performance_metrics()
        assert metrics['total_orders'] == 1


if __name__ == "__main__":
    pytest.main([__file__])