"""
Test suite for Market Data Ingestor Agent

This test suite validates the core functionality of the Market Data Ingestor Agent
including data fetching, validation, storage, and failover mechanisms.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from agents.market_data_ingestor import (
    MarketDataIngestorAgent,
    MarketData,
    DataProvider,
    DataQuality,
    ValidationResult,
    FailoverResult,
    IngestionState,
    DataProviderClient,
    DataValidator,
    DatabaseManager,
    create_market_data_ingestor
)


class TestMarketData:
    """Test MarketData dataclass"""
    
    def test_market_data_creation(self):
        """Test MarketData object creation"""
        data = MarketData(
            symbol="AAPL",
            exchange="NASDAQ",
            timestamp=datetime.now(),
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
            vwap=Decimal("150.50"),
            provider="alpaca"
        )
        
        assert data.symbol == "AAPL"
        assert data.exchange == "NASDAQ"
        assert data.open == Decimal("150.00")
        assert data.volume == 1000000
        assert data.provider == "alpaca"
    
    def test_market_data_to_dict(self):
        """Test MarketData to_dict conversion"""
        timestamp = datetime.now()
        data = MarketData(
            symbol="AAPL",
            exchange="NASDAQ",
            timestamp=timestamp,
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
            provider="alpaca"
        )
        
        result = data.to_dict()
        
        assert result['symbol'] == "AAPL"
        assert result['exchange'] == "NASDAQ"
        assert result['timestamp'] == timestamp
        assert result['open'] == 150.00
        assert result['high'] == 152.00
        assert result['low'] == 149.00
        assert result['close'] == 151.00
        assert result['volume'] == 1000000
        assert result['provider'] == "alpaca"


class TestDataValidator:
    """Test DataValidator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.validator = DataValidator()
    
    def test_valid_data_validation(self):
        """Test validation of valid market data"""
        valid_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'timestamp': datetime.now(),
            'open': 150.00,
            'high': 152.00,
            'low': 149.00,
            'close': 151.00,
            'volume': 1000000,
            'provider': 'alpaca'
        }
        
        result = self.validator.validate_market_data(valid_data)
        
        assert result.is_valid is True
        assert result.quality == DataQuality.VALID
        assert len(result.issues) == 0
        assert result.corrected_data is not None
        assert result.corrected_data.symbol == 'AAPL'
    
    def test_invalid_ohlc_relationship(self):
        """Test validation with invalid OHLC relationship"""
        invalid_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'timestamp': datetime.now(),
            'open': 150.00,
            'high': 148.00,  # High less than open - invalid
            'low': 149.00,
            'close': 151.00,
            'volume': 1000000,
            'provider': 'alpaca'
        }
        
        result = self.validator.validate_market_data(invalid_data)
        
        assert result.quality == DataQuality.SUSPICIOUS
        assert "Invalid OHLC relationship" in result.issues
    
    def test_extreme_price_change(self):
        """Test validation with extreme price change"""
        extreme_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'timestamp': datetime.now(),
            'open': 100.00,
            'high': 130.00,
            'low': 95.00,
            'close': 125.00,  # 25% price change - suspicious
            'volume': 1000000,
            'provider': 'alpaca'
        }
        
        result = self.validator.validate_market_data(extreme_data)
        
        assert result.quality == DataQuality.SUSPICIOUS
        assert any("Extreme price change" in issue for issue in result.issues)
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        incomplete_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            # Missing timestamp, OHLC, volume
            'provider': 'alpaca'
        }
        
        result = self.validator.validate_market_data(incomplete_data)
        
        assert result.is_valid is False
        assert result.quality == DataQuality.INVALID
        assert len(result.issues) > 0
    
    def test_negative_volume(self):
        """Test validation with negative volume"""
        negative_volume_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'timestamp': datetime.now(),
            'open': 150.00,
            'high': 152.00,
            'low': 149.00,
            'close': 151.00,
            'volume': -1000,  # Negative volume - invalid
            'provider': 'alpaca'
        }
        
        result = self.validator.validate_market_data(negative_volume_data)
        
        assert result.is_valid is False
        assert result.quality == DataQuality.INVALID
        assert "Negative volume" in result.issues


class TestDataProviderClient:
    """Test DataProviderClient class"""
    
    @pytest.fixture
    def mock_api_keys(self):
        """Mock API keys for testing"""
        return {
            'ALPACA_API_KEY': 'test_alpaca_key',
            'ALPACA_SECRET_KEY': 'test_alpaca_secret',
            'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
            'POLYGON_API_KEY': 'test_polygon_key'
        }
    
    @patch('agents.market_data_ingestor.get_api_keys')
    @patch('agents.market_data_ingestor.get_settings')
    def test_client_initialization(self, mock_settings, mock_api_keys):
        """Test DataProviderClient initialization"""
        mock_api_keys.return_value = {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret',
            'POLYGON_API_KEY': 'test_polygon_key'
        }
        mock_settings.return_value = Mock()
        
        with patch('agents.market_data_ingestor.tradeapi.REST'), \
             patch('agents.market_data_ingestor.PolygonClient'):
            client = DataProviderClient()
            assert client is not None
    
    def test_timeframe_conversion(self):
        """Test timeframe conversion to Polygon format"""
        with patch('agents.market_data_ingestor.get_api_keys'), \
             patch('agents.market_data_ingestor.get_settings'), \
             patch('agents.market_data_ingestor.tradeapi.REST'), \
             patch('agents.market_data_ingestor.PolygonClient'):
            
            client = DataProviderClient()
            
            # Test various timeframe conversions
            assert client._convert_timeframe_to_polygon('1Min') == (1, 'minute')
            assert client._convert_timeframe_to_polygon('5Min') == (5, 'minute')
            assert client._convert_timeframe_to_polygon('1Hour') == (1, 'hour')
            assert client._convert_timeframe_to_polygon('1Day') == (1, 'day')
            assert client._convert_timeframe_to_polygon('unknown') == (1, 'day')  # Default


@pytest.mark.asyncio
class TestDatabaseManager:
    """Test DatabaseManager class"""
    
    @patch('agents.market_data_ingestor.create_async_engine')
    @patch('agents.market_data_ingestor.create_engine')
    @patch('agents.market_data_ingestor.get_settings')
    async def test_database_initialization(self, mock_settings, mock_sync_engine, mock_async_engine):
        """Test database manager initialization"""
        mock_settings.return_value = Mock(
            database_user='test_user',
            database_password='test_pass',
            database_host='localhost',
            database_port=5432,
            database_name='test_db'
        )
        
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        assert mock_async_engine.called
        assert mock_sync_engine.called
    
    @patch('agents.market_data_ingestor.get_settings')
    async def test_create_market_data_table(self, mock_settings):
        """Test market data table creation"""
        mock_settings.return_value = Mock()
        
        # Mock async engine and connection
        mock_conn = AsyncMock()
        mock_engine = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        db_manager = DatabaseManager()
        db_manager.async_engine = mock_engine
        
        await db_manager.create_market_data_table()
        
        # Verify that execute was called with SQL
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS market_data" in str(call_args)
    
    @patch('agents.market_data_ingestor.get_settings')
    async def test_bulk_insert_market_data(self, mock_settings):
        """Test bulk insert of market data"""
        mock_settings.return_value = Mock()
        
        # Create test data
        test_data = [
            MarketData(
                symbol="AAPL",
                exchange="NASDAQ",
                timestamp=datetime.now(),
                open=Decimal("150.00"),
                high=Decimal("152.00"),
                low=Decimal("149.00"),
                close=Decimal("151.00"),
                volume=1000000,
                provider="alpaca"
            )
        ]
        
        # Mock async engine and connection
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        mock_engine = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        db_manager = DatabaseManager()
        db_manager.async_engine = mock_engine
        
        result = await db_manager.bulk_insert_market_data(test_data)
        
        assert result == 1
        mock_conn.execute.assert_called_once()


@pytest.mark.asyncio
class TestMarketDataIngestorAgent:
    """Test MarketDataIngestorAgent class"""
    
    @patch('agents.market_data_ingestor.DatabaseManager')
    @patch('agents.market_data_ingestor.DataProviderClient')
    async def test_agent_initialization(self, mock_client, mock_db):
        """Test agent initialization"""
        mock_db_instance = AsyncMock()
        mock_db.return_value = mock_db_instance
        
        agent = MarketDataIngestorAgent()
        await agent.initialize()
        
        assert agent.data_client is not None
        assert agent.validator is not None
        assert agent.db_manager is not None
        mock_db_instance.initialize.assert_called_once()
        mock_db_instance.create_market_data_table.assert_called_once()
    
    def test_workflow_creation(self):
        """Test LangGraph workflow creation"""
        agent = MarketDataIngestorAgent()
        workflow = agent.create_workflow()
        
        assert workflow is not None
        # Verify workflow has expected nodes
        expected_nodes = ["fetch_data", "validate_data", "store_data", "handle_failures", "generate_stats"]
        # Note: In actual implementation, we'd need to check the workflow structure
        # This is a basic test to ensure workflow creation doesn't fail
    
    @patch('agents.market_data_ingestor.DatabaseManager')
    @patch('agents.market_data_ingestor.DataProviderClient')
    async def test_fetch_data_node_success(self, mock_client_class, mock_db):
        """Test successful data fetching"""
        # Mock successful data fetch
        mock_client = AsyncMock()
        mock_client.fetch_historical_data.return_value = {
            'AAPL': [
                {
                    'symbol': 'AAPL',
                    'exchange': 'NASDAQ',
                    'timestamp': datetime.now(),
                    'open': 150.00,
                    'high': 152.00,
                    'low': 149.00,
                    'close': 151.00,
                    'volume': 1000000,
                    'provider': 'alpaca'
                }
            ]
        }
        mock_client_class.return_value = mock_client
        
        agent = MarketDataIngestorAgent()
        
        # Create test state
        state = IngestionState(
            symbols=['AAPL'],
            timeframe='1Day',
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            current_provider=DataProvider.ALPACA,
            raw_data={},
            validated_data={},
            failed_symbols=[],
            ingestion_stats={},
            errors=[]
        )
        
        result_state = await agent.fetch_data_node(state)
        
        assert result_state.ingestion_stats['fetch_success'] is True
        assert result_state.ingestion_stats['records_fetched'] == 1
        assert 'AAPL' in result_state.raw_data
    
    @patch('agents.market_data_ingestor.DatabaseManager')
    @patch('agents.market_data_ingestor.DataProviderClient')
    async def test_fetch_data_node_failure(self, mock_client_class, mock_db):
        """Test data fetching failure"""
        # Mock failed data fetch
        mock_client = AsyncMock()
        mock_client.fetch_historical_data.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        agent = MarketDataIngestorAgent()
        
        # Create test state
        state = IngestionState(
            symbols=['AAPL'],
            timeframe='1Day',
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            current_provider=DataProvider.ALPACA,
            raw_data={},
            validated_data={},
            failed_symbols=[],
            ingestion_stats={},
            errors=[]
        )
        
        result_state = await agent.fetch_data_node(state)
        
        assert result_state.ingestion_stats['fetch_success'] is False
        assert len(result_state.errors) > 0
        assert "API Error" in result_state.errors[0]
    
    async def test_validate_data_node(self):
        """Test data validation node"""
        agent = MarketDataIngestorAgent()
        
        # Create test state with raw data
        state = IngestionState(
            symbols=['AAPL'],
            timeframe='1Day',
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            current_provider=DataProvider.ALPACA,
            raw_data={
                'AAPL': [
                    {
                        'symbol': 'AAPL',
                        'exchange': 'NASDAQ',
                        'timestamp': datetime.now(),
                        'open': 150.00,
                        'high': 152.00,
                        'low': 149.00,
                        'close': 151.00,
                        'volume': 1000000,
                        'provider': 'alpaca'
                    }
                ]
            },
            validated_data={},
            failed_symbols=[],
            ingestion_stats={},
            errors=[]
        )
        
        result_state = await agent.validate_data_node(state)
        
        assert 'AAPL' in result_state.validated_data
        assert len(result_state.validated_data['AAPL']) == 1
        assert result_state.ingestion_stats['validation']['valid_records'] == 1
    
    def test_should_handle_failures(self):
        """Test failure handling decision logic"""
        agent = MarketDataIngestorAgent()
        
        # Test case: fetch failed, should handle failures
        state = IngestionState(
            symbols=['AAPL'],
            timeframe='1Day',
            start_date=datetime.now(),
            end_date=datetime.now(),
            current_provider=DataProvider.ALPACA,
            raw_data={},
            validated_data={},
            failed_symbols=[],
            ingestion_stats={'fetch_success': False},
            errors=[]
        )
        
        agent.failover_attempts = 0
        result = agent.should_handle_failures(state)
        assert result == "handle_failures"
        
        # Test case: fetch succeeded, should continue
        state.ingestion_stats['fetch_success'] = True
        result = agent.should_handle_failures(state)
        assert result == "continue"
        
        # Test case: max failover attempts reached
        state.ingestion_stats['fetch_success'] = False
        agent.failover_attempts = agent.max_failover_attempts
        result = agent.should_handle_failures(state)
        assert result == "continue"
    
    async def test_handle_failures_node(self):
        """Test failure handling and provider failover"""
        agent = MarketDataIngestorAgent()
        
        # Test failover from Alpaca to Polygon
        state = IngestionState(
            symbols=['AAPL'],
            timeframe='1Day',
            start_date=datetime.now(),
            end_date=datetime.now(),
            current_provider=DataProvider.ALPACA,
            raw_data={},
            validated_data={},
            failed_symbols=[],
            ingestion_stats={},
            errors=[]
        )
        
        result_state = await agent.handle_failures_node(state)
        
        assert result_state.current_provider == DataProvider.POLYGON
        assert agent.failover_attempts == 1
        
        # Test failover from Polygon to Alpaca
        state.current_provider = DataProvider.POLYGON
        result_state = await agent.handle_failures_node(state)
        
        assert result_state.current_provider == DataProvider.ALPACA
        assert agent.failover_attempts == 2


@pytest.mark.asyncio
async def test_create_market_data_ingestor():
    """Test factory function for creating market data ingestor"""
    with patch('agents.market_data_ingestor.DatabaseManager') as mock_db:
        mock_db_instance = AsyncMock()
        mock_db.return_value = mock_db_instance
        
        agent = await create_market_data_ingestor()
        
        assert isinstance(agent, MarketDataIngestorAgent)
        mock_db_instance.initialize.assert_called_once()
        mock_db_instance.create_market_data_table.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_ingestion_workflow():
    """Integration test for full ingestion workflow"""
    # This test would require actual database and API connections
    # For now, it's a placeholder for integration testing
    
    # Mock all external dependencies
    with patch('agents.market_data_ingestor.DatabaseManager') as mock_db, \
         patch('agents.market_data_ingestor.DataProviderClient') as mock_client:
        
        # Setup mocks
        mock_db_instance = AsyncMock()
        mock_db_instance.bulk_insert_market_data.return_value = 1
        mock_db_instance.get_data_statistics.return_value = {
            'total_records': 1,
            'unique_symbols': 1,
            'by_provider': {'alpaca': 1}
        }
        mock_db.return_value = mock_db_instance
        
        mock_client_instance = AsyncMock()
        mock_client_instance.fetch_historical_data.return_value = {
            'AAPL': [
                {
                    'symbol': 'AAPL',
                    'exchange': 'NASDAQ',
                    'timestamp': datetime.now(),
                    'open': 150.00,
                    'high': 152.00,
                    'low': 149.00,
                    'close': 151.00,
                    'volume': 1000000,
                    'provider': 'alpaca'
                }
            ]
        }
        mock_client.return_value = mock_client_instance
        
        # Create and test agent
        agent = MarketDataIngestorAgent()
        await agent.initialize()
        
        result = await agent.ingest_historical_data(
            symbols=['AAPL'],
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            timeframe='1Day'
        )
        
        assert result['success'] is True
        assert 'statistics' in result
        assert len(result['failed_symbols']) == 0
        assert len(result['errors']) == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])