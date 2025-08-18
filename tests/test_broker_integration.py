"""
Tests for Alpaca Broker Integration

This module contains comprehensive tests for the broker integration including:
- Order lifecycle management
- Position reconciliation
- Trade reporting
- Error handling
"""

import pytest
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from agents.broker_integration import (
    AlpacaBrokerIntegration,
    OrderRequest,
    OrderResponse,
    PositionInfo,
    TradeReport,
    BrokerError,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    create_market_order,
    create_limit_order,
    create_stop_loss_order
)


class MockAlpacaOrder:
    """Mock Alpaca Order object for testing"""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 'test-order-123')
        self.client_order_id = kwargs.get('client_order_id', None)
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))
        self.submitted_at = kwargs.get('submitted_at', datetime.now(timezone.utc))
        self.filled_at = kwargs.get('filled_at', None)
        self.expired_at = kwargs.get('expired_at', None)
        self.canceled_at = kwargs.get('canceled_at', None)
        self.failed_at = kwargs.get('failed_at', None)
        self.replaced_at = kwargs.get('replaced_at', None)
        self.symbol = kwargs.get('symbol', 'AAPL')
        self.asset_id = kwargs.get('asset_id', 'test-asset-123')
        self.asset_class = kwargs.get('asset_class', 'us_equity')
        self.qty = kwargs.get('qty', 100)
        self.filled_qty = kwargs.get('filled_qty', 0)
        self.order_type = kwargs.get('order_type', 'market')
        self.side = kwargs.get('side', 'buy')
        self.time_in_force = kwargs.get('time_in_force', 'day')
        self.limit_price = kwargs.get('limit_price', None)
        self.stop_price = kwargs.get('stop_price', None)
        self.status = kwargs.get('status', 'new')
        self.extended_hours = kwargs.get('extended_hours', False)
        self.legs = kwargs.get('legs', None)
        self.trail_price = kwargs.get('trail_price', None)
        self.trail_percent = kwargs.get('trail_percent', None)
        self.hwm = kwargs.get('hwm', None)


class MockAlpacaPosition:
    """Mock Alpaca Position object for testing"""
    
    def __init__(self, **kwargs):
        self.asset_id = kwargs.get('asset_id', 'test-asset-123')
        self.symbol = kwargs.get('symbol', 'AAPL')
        self.exchange = kwargs.get('exchange', 'NASDAQ')
        self.asset_class = kwargs.get('asset_class', 'us_equity')
        self.avg_entry_price = kwargs.get('avg_entry_price', 150.00)
        self.qty = kwargs.get('qty', 100)
        self.side = kwargs.get('side', 'long')
        self.market_value = kwargs.get('market_value', 15000.00)
        self.cost_basis = kwargs.get('cost_basis', 15000.00)
        self.unrealized_pl = kwargs.get('unrealized_pl', 0.00)
        self.unrealized_plpc = kwargs.get('unrealized_plpc', 0.00)
        self.unrealized_intraday_pl = kwargs.get('unrealized_intraday_pl', 0.00)
        self.unrealized_intraday_plpc = kwargs.get('unrealized_intraday_plpc', 0.00)
        self.current_price = kwargs.get('current_price', 150.00)
        self.lastday_price = kwargs.get('lastday_price', 150.00)
        self.change_today = kwargs.get('change_today', 0.00)


class MockAlpacaAccount:
    """Mock Alpaca Account object for testing"""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 'test-account-123')
        self.account_number = kwargs.get('account_number', '123456789')
        self.status = kwargs.get('status', 'ACTIVE')
        self.currency = kwargs.get('currency', 'USD')
        self.buying_power = kwargs.get('buying_power', 50000.00)
        self.regt_buying_power = kwargs.get('regt_buying_power', 50000.00)
        self.daytrading_buying_power = kwargs.get('daytrading_buying_power', 100000.00)
        self.cash = kwargs.get('cash', 25000.00)
        self.portfolio_value = kwargs.get('portfolio_value', 50000.00)
        self.pattern_day_trader = kwargs.get('pattern_day_trader', False)
        self.trading_blocked = kwargs.get('trading_blocked', False)
        self.transfers_blocked = kwargs.get('transfers_blocked', False)
        self.account_blocked = kwargs.get('account_blocked', False)
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.trade_suspended_by_user = kwargs.get('trade_suspended_by_user', False)
        self.multiplier = kwargs.get('multiplier', 2.0)
        self.shorting_enabled = kwargs.get('shorting_enabled', True)
        self.equity = kwargs.get('equity', 50000.00)
        self.last_equity = kwargs.get('last_equity', 50000.00)
        self.long_market_value = kwargs.get('long_market_value', 25000.00)
        self.short_market_value = kwargs.get('short_market_value', 0.00)
        self.initial_margin = kwargs.get('initial_margin', 0.00)
        self.maintenance_margin = kwargs.get('maintenance_margin', 0.00)
        self.last_maintenance_margin = kwargs.get('last_maintenance_margin', 0.00)
        self.sma = kwargs.get('sma', 50000.00)
        self.daytrade_count = kwargs.get('daytrade_count', 0)


@pytest.fixture
def mock_alpaca_api():
    """Mock Alpaca API for testing"""
    with patch('agents.broker_integration.tradeapi.REST') as mock_api:
        api_instance = Mock()
        mock_api.return_value = api_instance
        yield api_instance


@pytest.fixture
def broker_integration(mock_alpaca_api):
    """Create broker integration instance for testing"""
    with patch('agents.broker_integration.get_settings') as mock_settings:
        mock_settings.return_value.ALPACA_API_KEY = 'test-api-key'
        mock_settings.return_value.ALPACA_SECRET_KEY = 'test-secret-key'
        mock_settings.return_value.ALPACA_PAPER_BASE_URL = 'https://paper-api.alpaca.markets'
        mock_settings.return_value.ALPACA_LIVE_BASE_URL = 'https://api.alpaca.markets'
        
        broker = AlpacaBrokerIntegration(
            api_key='test-api-key',
            secret_key='test-secret-key',
            paper_trading=True
        )
        return broker


class TestOrderRequest:
    """Test OrderRequest data structure"""
    
    def test_valid_market_order(self):
        """Test creating a valid market order request"""
        order = OrderRequest(
            symbol='AAPL',
            qty=100,
            side=OrderSide.BUY,
            type=OrderType.MARKET
        )
        
        assert order.symbol == 'AAPL'
        assert order.qty == 100
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.time_in_force == TimeInForce.DAY
    
    def test_valid_limit_order(self):
        """Test creating a valid limit order request"""
        order = OrderRequest(
            symbol='AAPL',
            qty=100,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            limit_price=150.00
        )
        
        assert order.limit_price == 150.00
    
    def test_invalid_limit_order_no_price(self):
        """Test that limit order without price raises error"""
        with pytest.raises(ValueError, match="limit_price required"):
            OrderRequest(
                symbol='AAPL',
                qty=100,
                side=OrderSide.BUY,
                type=OrderType.LIMIT
            )
    
    def test_invalid_stop_order_no_price(self):
        """Test that stop order without price raises error"""
        with pytest.raises(ValueError, match="stop_price required"):
            OrderRequest(
                symbol='AAPL',
                qty=100,
                side=OrderSide.SELL,
                type=OrderType.STOP
            )
    
    def test_invalid_trailing_stop_no_trail(self):
        """Test that trailing stop without trail parameters raises error"""
        with pytest.raises(ValueError, match="trail_price or trail_percent required"):
            OrderRequest(
                symbol='AAPL',
                qty=100,
                side=OrderSide.SELL,
                type=OrderType.TRAILING_STOP
            )


class TestOrderResponse:
    """Test OrderResponse data structure"""
    
    def test_from_alpaca_order(self):
        """Test creating OrderResponse from Alpaca order"""
        mock_order = MockAlpacaOrder(
            id='test-123',
            symbol='AAPL',
            qty=100,
            filled_qty=50,
            order_type='market',
            side='buy',
            status='partially_filled'
        )
        
        order_response = OrderResponse.from_alpaca_order(mock_order)
        
        assert order_response.id == 'test-123'
        assert order_response.symbol == 'AAPL'
        assert order_response.qty == Decimal('100')
        assert order_response.filled_qty == Decimal('50')
        assert order_response.type == OrderType.MARKET
        assert order_response.side == OrderSide.BUY
        assert order_response.status == OrderStatus.PARTIALLY_FILLED


class TestPositionInfo:
    """Test PositionInfo data structure"""
    
    def test_from_alpaca_position(self):
        """Test creating PositionInfo from Alpaca position"""
        mock_position = MockAlpacaPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=150.00,
            market_value=15000.00,
            unrealized_pl=500.00
        )
        
        position_info = PositionInfo.from_alpaca_position(mock_position)
        
        assert position_info.symbol == 'AAPL'
        assert position_info.qty == Decimal('100')
        assert position_info.avg_entry_price == Decimal('150.00')
        assert position_info.market_value == Decimal('15000.00')
        assert position_info.unrealized_pl == Decimal('500.00')


class TestAlpacaBrokerIntegration:
    """Test AlpacaBrokerIntegration class"""
    
    @pytest.mark.asyncio
    async def test_submit_market_order_success(self, broker_integration, mock_alpaca_api):
        """Test successful market order submission"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='test-order-123',
            symbol='AAPL',
            qty=100,
            order_type='market',
            side='buy',
            status='new'
        )
        mock_alpaca_api.submit_order.return_value = mock_order
        
        # Create order request
        order_request = OrderRequest(
            symbol='AAPL',
            qty=100,
            side=OrderSide.BUY,
            type=OrderType.MARKET
        )
        
        # Submit order
        order_response = await broker_integration.submit_order(order_request)
        
        # Verify response
        assert order_response.id == 'test-order-123'
        assert order_response.symbol == 'AAPL'
        assert order_response.qty == Decimal('100')
        assert order_response.side == OrderSide.BUY
        assert order_response.type == OrderType.MARKET
        assert order_response.status == OrderStatus.NEW
        
        # Verify order is tracked
        assert 'test-order-123' in broker_integration.active_orders
        
        # Verify API was called correctly
        mock_alpaca_api.submit_order.assert_called_once()
        call_args = mock_alpaca_api.submit_order.call_args[1]
        assert call_args['symbol'] == 'AAPL'
        assert call_args['qty'] == '100'
        assert call_args['side'] == 'buy'
        assert call_args['type'] == 'market'
    
    @pytest.mark.asyncio
    async def test_submit_limit_order_success(self, broker_integration, mock_alpaca_api):
        """Test successful limit order submission"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='test-order-456',
            symbol='AAPL',
            qty=100,
            order_type='limit',
            side='buy',
            status='new',
            limit_price=150.00
        )
        mock_alpaca_api.submit_order.return_value = mock_order
        
        # Create order request
        order_request = OrderRequest(
            symbol='AAPL',
            qty=100,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            limit_price=150.00
        )
        
        # Submit order
        order_response = await broker_integration.submit_order(order_request)
        
        # Verify response
        assert order_response.limit_price == Decimal('150.00')
        
        # Verify API was called with limit price
        call_args = mock_alpaca_api.submit_order.call_args[1]
        assert call_args['limit_price'] == '150.0'
    
    @pytest.mark.asyncio
    async def test_submit_order_api_error_retry(self, broker_integration, mock_alpaca_api):
        """Test order submission with API error and retry"""
        from alpaca_trade_api.rest import APIError
        
        # Setup mock to fail first time, succeed second time
        mock_order = MockAlpacaOrder(id='test-order-789')
        mock_alpaca_api.submit_order.side_effect = [
            APIError("Rate limit exceeded", 429),
            mock_order
        ]
        
        # Create order request
        order_request = OrderRequest(
            symbol='AAPL',
            qty=100,
            side=OrderSide.BUY,
            type=OrderType.MARKET
        )
        
        # Submit order (should succeed after retry)
        order_response = await broker_integration.submit_order(order_request)
        
        # Verify response
        assert order_response.id == 'test-order-789'
        
        # Verify retry occurred
        assert mock_alpaca_api.submit_order.call_count == 2
        
        # Verify error was logged
        assert len(broker_integration.error_log) == 1
        assert broker_integration.error_log[0].error_code == '429'
        assert broker_integration.error_log[0].is_retryable == True
    
    @pytest.mark.asyncio
    async def test_submit_order_non_retryable_error(self, broker_integration, mock_alpaca_api):
        """Test order submission with non-retryable error"""
        from alpaca_trade_api.rest import APIError
        
        # Setup mock to fail with non-retryable error
        mock_alpaca_api.submit_order.side_effect = APIError("Invalid symbol", 400)
        
        # Create order request
        order_request = OrderRequest(
            symbol='INVALID',
            qty=100,
            side=OrderSide.BUY,
            type=OrderType.MARKET
        )
        
        # Submit order (should fail)
        with pytest.raises(Exception, match="Order submission failed"):
            await broker_integration.submit_order(order_request)
        
        # Verify only one attempt was made
        assert mock_alpaca_api.submit_order.call_count == 1
        
        # Verify error was logged
        assert len(broker_integration.error_log) == 1
        assert broker_integration.error_log[0].error_code == '400'
        assert broker_integration.error_log[0].is_retryable == False
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, broker_integration, mock_alpaca_api):
        """Test successful order cancellation"""
        # Setup active order
        order_id = 'test-order-123'
        mock_order_response = OrderResponse(
            id=order_id,
            client_order_id=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            submitted_at=datetime.now(timezone.utc),
            filled_at=None,
            expired_at=None,
            canceled_at=None,
            failed_at=None,
            replaced_at=None,
            symbol='AAPL',
            asset_id='test-asset',
            asset_class='us_equity',
            qty=Decimal('100'),
            filled_qty=Decimal('0'),
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=None,
            stop_price=None,
            status=OrderStatus.NEW,
            extended_hours=False
        )
        broker_integration.active_orders[order_id] = mock_order_response
        
        # Mock successful cancellation
        mock_alpaca_api.cancel_order.return_value = None
        
        # Cancel order
        result = await broker_integration.cancel_order(order_id)
        
        # Verify success
        assert result == True
        
        # Verify order status updated
        assert broker_integration.active_orders[order_id].status == OrderStatus.CANCELED
        assert broker_integration.active_orders[order_id].canceled_at is not None
        
        # Verify API was called
        mock_alpaca_api.cancel_order.assert_called_once_with(order_id)
    
    @pytest.mark.asyncio
    async def test_get_order_status(self, broker_integration, mock_alpaca_api):
        """Test getting order status"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='test-order-123',
            symbol='AAPL',
            status='filled',
            filled_qty=100
        )
        mock_alpaca_api.get_order.return_value = mock_order
        
        # Get order status
        order_response = await broker_integration.get_order_status('test-order-123')
        
        # Verify response
        assert order_response is not None
        assert order_response.id == 'test-order-123'
        assert order_response.status == OrderStatus.FILLED
        assert order_response.filled_qty == Decimal('100')
        
        # Verify order moved to completed orders
        assert 'test-order-123' in broker_integration.completed_orders
        assert 'test-order-123' not in broker_integration.active_orders
    
    @pytest.mark.asyncio
    async def test_get_positions(self, broker_integration, mock_alpaca_api):
        """Test getting all positions"""
        # Setup mock response
        mock_positions = [
            MockAlpacaPosition(symbol='AAPL', qty=100, market_value=15000.00),
            MockAlpacaPosition(symbol='GOOGL', qty=50, market_value=10000.00)
        ]
        mock_alpaca_api.list_positions.return_value = mock_positions
        
        # Get positions
        positions = await broker_integration.get_positions()
        
        # Verify response
        assert len(positions) == 2
        assert positions[0].symbol == 'AAPL'
        assert positions[0].qty == Decimal('100')
        assert positions[1].symbol == 'GOOGL'
        assert positions[1].qty == Decimal('50')
    
    @pytest.mark.asyncio
    async def test_get_position_single(self, broker_integration, mock_alpaca_api):
        """Test getting single position"""
        # Setup mock response
        mock_position = MockAlpacaPosition(
            symbol='AAPL',
            qty=100,
            avg_entry_price=150.00,
            market_value=15000.00
        )
        mock_alpaca_api.get_position.return_value = mock_position
        
        # Get position
        position = await broker_integration.get_position('AAPL')
        
        # Verify response
        assert position is not None
        assert position.symbol == 'AAPL'
        assert position.qty == Decimal('100')
        assert position.avg_entry_price == Decimal('150.00')
    
    @pytest.mark.asyncio
    async def test_get_position_not_found(self, broker_integration, mock_alpaca_api):
        """Test getting position that doesn't exist"""
        from alpaca_trade_api.rest import APIError
        
        # Setup mock to raise position not found error
        mock_alpaca_api.get_position.side_effect = APIError("position does not exist", 404)
        
        # Get position
        position = await broker_integration.get_position('NONEXISTENT')
        
        # Verify no position returned
        assert position is None
    
    @pytest.mark.asyncio
    async def test_close_position(self, broker_integration, mock_alpaca_api):
        """Test closing a position"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='close-order-123',
            symbol='AAPL',
            side='sell',
            qty=100,
            order_type='market'
        )
        mock_alpaca_api.close_position.return_value = mock_order
        
        # Close position
        order_response = await broker_integration.close_position('AAPL')
        
        # Verify response
        assert order_response is not None
        assert order_response.id == 'close-order-123'
        assert order_response.symbol == 'AAPL'
        assert order_response.side == OrderSide.SELL
        
        # Verify order is tracked
        assert 'close-order-123' in broker_integration.active_orders
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, broker_integration, mock_alpaca_api):
        """Test getting account information"""
        # Setup mock response
        mock_account = MockAlpacaAccount(
            account_number='123456789',
            status='ACTIVE',
            buying_power=50000.00,
            portfolio_value=75000.00
        )
        mock_alpaca_api.get_account.return_value = mock_account
        
        # Get account info
        account_info = await broker_integration.get_account_info()
        
        # Verify response
        assert account_info is not None
        assert account_info['account_number'] == '123456789'
        assert account_info['status'] == 'ACTIVE'
        assert account_info['buying_power'] == 50000.00
        assert account_info['portfolio_value'] == 75000.00
    
    @pytest.mark.asyncio
    async def test_reconcile_positions(self, broker_integration, mock_alpaca_api):
        """Test position reconciliation"""
        # Setup mock positions
        mock_positions = [
            MockAlpacaPosition(
                symbol='AAPL',
                qty=100,
                market_value=15000.00,
                unrealized_pl=500.00
            ),
            MockAlpacaPosition(
                symbol='GOOGL',
                qty=50,
                market_value=10000.00,
                unrealized_pl=-200.00
            )
        ]
        mock_alpaca_api.list_positions.return_value = mock_positions
        
        # Reconcile positions
        report = await broker_integration.reconcile_positions()
        
        # Verify report
        assert 'timestamp' in report
        assert report['broker_positions_count'] == 2
        assert len(report['positions']) == 2
        assert report['total_market_value'] == Decimal('25000.00')
        assert report['total_unrealized_pl'] == Decimal('300.00')
        
        # Verify position details
        aapl_position = next(p for p in report['positions'] if p['symbol'] == 'AAPL')
        assert aapl_position['qty'] == Decimal('100')
        assert aapl_position['market_value'] == Decimal('15000.00')
    
    @pytest.mark.asyncio
    async def test_generate_trade_report(self, broker_integration, mock_alpaca_api):
        """Test generating trade report"""
        # Setup mock orders
        mock_orders = [
            MockAlpacaOrder(
                id='order-1',
                symbol='AAPL',
                side='buy',
                qty=100,
                filled_qty=100,
                status='filled',
                limit_price=150.00
            ),
            MockAlpacaOrder(
                id='order-2',
                symbol='AAPL',
                side='sell',
                qty=50,
                filled_qty=50,
                status='filled',
                limit_price=155.00
            ),
            MockAlpacaOrder(
                id='order-3',
                symbol='GOOGL',
                side='buy',
                qty=25,
                filled_qty=25,
                status='filled',
                limit_price=2000.00
            )
        ]
        mock_alpaca_api.list_orders.return_value = mock_orders
        
        # Generate trade report
        report = await broker_integration.generate_trade_report()
        
        # Verify report
        assert 'timestamp' in report
        assert report['summary']['total_trades'] == 3
        assert report['summary']['buy_orders'] == 2
        assert report['summary']['sell_orders'] == 1
        assert report['summary']['unique_symbols'] == 2
        
        # Verify by-symbol breakdown
        assert 'AAPL' in report['by_symbol']
        assert 'GOOGL' in report['by_symbol']
        assert report['by_symbol']['AAPL']['total_trades'] == 2
        assert report['by_symbol']['GOOGL']['total_trades'] == 1
        
        # Verify order details
        assert len(report['orders']) == 3
        order_1 = next(o for o in report['orders'] if o['id'] == 'order-1')
        assert order_1['symbol'] == 'AAPL'
        assert order_1['side'] == 'buy'
        assert order_1['qty'] == 100.0
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, broker_integration, mock_alpaca_api):
        """Test health check when all systems are healthy"""
        # Setup mock responses
        mock_account = MockAlpacaAccount()
        mock_alpaca_api.get_account.return_value = mock_account
        mock_alpaca_api.list_orders.return_value = []
        mock_alpaca_api.list_positions.return_value = []
        
        # Perform health check
        health_status = await broker_integration.health_check()
        
        # Verify health status
        assert health_status['connection_status'] == 'healthy'
        assert health_status['account_accessible'] == True
        assert health_status['orders_accessible'] == True
        assert health_status['positions_accessible'] == True
        assert health_status['broker'] == 'alpaca'
        assert health_status['paper_trading'] == True
        assert len(health_status['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, broker_integration, mock_alpaca_api):
        """Test health check when systems are unhealthy"""
        from alpaca_trade_api.rest import APIError
        
        # Setup mock to fail
        mock_alpaca_api.get_account.side_effect = APIError("Connection failed", 500)
        
        # Perform health check
        health_status = await broker_integration.health_check()
        
        # Verify health status
        assert health_status['connection_status'] == 'unhealthy'
        assert health_status['account_accessible'] == False
        assert len(health_status['errors']) > 0


class TestConvenienceFunctions:
    """Test convenience functions for common operations"""
    
    @pytest.mark.asyncio
    async def test_create_market_order(self, broker_integration, mock_alpaca_api):
        """Test create_market_order convenience function"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='market-order-123',
            symbol='AAPL',
            qty=100,
            order_type='market',
            side='buy'
        )
        mock_alpaca_api.submit_order.return_value = mock_order
        
        # Create market order
        order_response = await create_market_order(
            broker_integration,
            'AAPL',
            100,
            OrderSide.BUY
        )
        
        # Verify response
        assert order_response.id == 'market-order-123'
        assert order_response.type == OrderType.MARKET
        assert order_response.side == OrderSide.BUY
    
    @pytest.mark.asyncio
    async def test_create_limit_order(self, broker_integration, mock_alpaca_api):
        """Test create_limit_order convenience function"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='limit-order-123',
            symbol='AAPL',
            qty=100,
            order_type='limit',
            side='buy',
            limit_price=150.00
        )
        mock_alpaca_api.submit_order.return_value = mock_order
        
        # Create limit order
        order_response = await create_limit_order(
            broker_integration,
            'AAPL',
            100,
            OrderSide.BUY,
            150.00
        )
        
        # Verify response
        assert order_response.id == 'limit-order-123'
        assert order_response.type == OrderType.LIMIT
        assert order_response.limit_price == Decimal('150.00')
    
    @pytest.mark.asyncio
    async def test_create_stop_loss_order(self, broker_integration, mock_alpaca_api):
        """Test create_stop_loss_order convenience function"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='stop-order-123',
            symbol='AAPL',
            qty=100,
            order_type='stop',
            side='sell',
            stop_price=140.00
        )
        mock_alpaca_api.submit_order.return_value = mock_order
        
        # Create stop loss order
        order_response = await create_stop_loss_order(
            broker_integration,
            'AAPL',
            100,
            OrderSide.SELL,
            140.00
        )
        
        # Verify response
        assert order_response.id == 'stop-order-123'
        assert order_response.type == OrderType.STOP
        assert order_response.stop_price == Decimal('140.00')
        assert order_response.time_in_force == TimeInForce.GTC


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_is_retryable_error_rate_limit(self, broker_integration):
        """Test that rate limit errors are retryable"""
        from alpaca_trade_api.rest import APIError
        
        error = APIError("Rate limit exceeded", 429)
        assert broker_integration._is_retryable_error(error) == True
    
    def test_is_retryable_error_server_error(self, broker_integration):
        """Test that server errors are retryable"""
        from alpaca_trade_api.rest import APIError
        
        error = APIError("Internal server error", 500)
        assert broker_integration._is_retryable_error(error) == True
    
    def test_is_retryable_error_client_error(self, broker_integration):
        """Test that client errors are not retryable"""
        from alpaca_trade_api.rest import APIError
        
        error = APIError("Invalid symbol", 400)
        assert broker_integration._is_retryable_error(error) == False
    
    def test_is_retryable_error_timeout(self, broker_integration):
        """Test that timeout errors are retryable"""
        from alpaca_trade_api.rest import APIError
        
        error = APIError("Request timeout", 408)
        assert broker_integration._is_retryable_error(error) == True


if __name__ == "__main__":
    pytest.main([__file__])