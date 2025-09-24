"""
Market Data Ingestor Agent - LangGraph Implementation

This agent handles autonomous ingestion and processing of global market data
with automatic failover, data validation, and PostgreSQL storage.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal
import json

# LangGraph imports
from langgraph.graph import StateGraph, END

# Data provider imports
import alpaca_trade_api as tradeapi
from polygon import RESTClient as PolygonClient
import requests
from io import StringIO

# Database imports
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Configuration
from config.settings import settings
from config.secure_config import get_api_keys

logger = logging.getLogger(__name__)

class DataProvider(Enum):
    ALPACA = "alpaca"
    POLYGON = "polygon"
    BACKUP = "backup"
    SATELLITE = "satellite"

class DataQuality(Enum):
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    MISSING = "missing"

@dataclass
class MarketData:
    """Core market data structure"""
    symbol: str
    exchange: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Optional[Decimal] = None
    provider: str = ""
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp,
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': self.volume,
            'vwap': float(self.vwap) if self.vwap else None,
            'provider': self.provider,
            'quality_score': self.quality_score
        }

@dataclass
class ValidationResult:
    """Data validation result"""
    is_valid: bool
    quality: DataQuality
    issues: List[str]
    corrected_data: Optional[MarketData] = None

@dataclass
class FailoverResult:
    """Failover operation result"""
    success: bool
    new_provider: DataProvider
    error_message: Optional[str] = None

@dataclass
class IngestionState:
    """LangGraph state for market data ingestion"""
    symbols: List[str]
    timeframe: str
    start_date: datetime
    end_date: datetime
    current_provider: DataProvider
    raw_data: Dict[str, List[Dict]]
    validated_data: Dict[str, List[MarketData]]
    failed_symbols: List[str]
    ingestion_stats: Dict[str, Any]
    errors: List[str]

class DataProviderClient:
    """Unified interface for data providers"""
    
    def __init__(self):
        self.settings = settings
        self.api_keys = get_api_keys()
        self._init_clients()
    
    def _init_clients(self):
        """Initialize API clients"""
        try:
            # Alpaca client
            self.alpaca_client = tradeapi.REST(
                key_id=self.api_keys.get('ALPACA_API_KEY'),
                secret_key=self.api_keys.get('ALPACA_SECRET_KEY'),
                base_url=self.api_keys.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            )
            
            # Polygon client
            self.polygon_client = PolygonClient(
                api_key=self.api_keys.get('POLYGON_API_KEY')
            )
            
            logger.info("Data provider clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize data provider clients: {e}")
            raise
    
    async def fetch_historical_data(self, 
                                  symbols: List[str], 
                                  start_date: datetime,
                                  end_date: datetime,
                                  timeframe: str = '1Day',
                                  provider: DataProvider = DataProvider.ALPACA) -> Dict[str, List[Dict]]:
        """Fetch historical market data from specified provider"""
        
        try:
            if provider == DataProvider.ALPACA:
                return await self._fetch_alpaca_data(symbols, start_date, end_date, timeframe)
            elif provider == DataProvider.POLYGON:
                return await self._fetch_polygon_data(symbols, start_date, end_date, timeframe)
            elif provider == DataProvider.SATELLITE:
                return await self._fetch_satellite_data(symbols, start_date, end_date, timeframe)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to fetch data from {provider.value}: {e}")
            raise
    
    async def _fetch_alpaca_data(self, symbols: List[str], start_date: datetime, 
                               end_date: datetime, timeframe: str) -> Dict[str, List[Dict]]:
        """Fetch data from Alpaca API"""
        data = {}
        
        for symbol in symbols:
            try:
                # Get historical bars
                bars = self.alpaca_client.get_bars(
                    symbol,
                    timeframe,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adjustment='raw'
                ).df
                
                if not bars.empty:
                    # Convert to our format
                    symbol_data = []
                    for idx, row in bars.iterrows():
                        bar_data = {
                            'symbol': symbol,
                            'exchange': 'NASDAQ',  # Default for Alpaca
                            'timestamp': idx,
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': int(row['volume']),
                            'vwap': float(row.get('vwap', 0)) if 'vwap' in row else None,
                            'provider': 'alpaca'
                        }
                        symbol_data.append(bar_data)
                    
                    data[symbol] = symbol_data
                    logger.info(f"Fetched {len(symbol_data)} bars for {symbol} from Alpaca")
                
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} from Alpaca: {e}")
                continue
        
        return data
    
    async def _fetch_polygon_data(self, symbols: List[str], start_date: datetime,
                                end_date: datetime, timeframe: str) -> Dict[str, List[Dict]]:
        """Fetch data from Polygon API"""
        data = {}
        
        # Convert timeframe to Polygon format
        multiplier, timespan = self._convert_timeframe_to_polygon(timeframe)
        
        for symbol in symbols:
            try:
                # Get aggregates (bars)
                aggs = self.polygon_client.get_aggs(
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d')
                )
                
                if aggs and len(aggs) > 0:
                    symbol_data = []
                    for agg in aggs:
                        bar_data = {
                            'symbol': symbol,
                            'exchange': 'NASDAQ',  # Default
                            'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                            'open': float(agg.open),
                            'high': float(agg.high),
                            'low': float(agg.low),
                            'close': float(agg.close),
                            'volume': int(agg.volume),
                            'vwap': float(agg.vwap) if hasattr(agg, 'vwap') and agg.vwap else None,
                            'provider': 'polygon'
                        }
                        symbol_data.append(bar_data)
                    
                    data[symbol] = symbol_data
                    logger.info(f"Fetched {len(symbol_data)} bars for {symbol} from Polygon")
                
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} from Polygon: {e}")
                continue
        
        return data
    
    def _convert_timeframe_to_polygon(self, timeframe: str) -> tuple:
        """Convert timeframe to Polygon API format"""
        timeframe_map = {
            '1Min': (1, 'minute'),
            '5Min': (5, 'minute'),
            '15Min': (15, 'minute'),
            '1Hour': (1, 'hour'),
            '1Day': (1, 'day')
        }
        return timeframe_map.get(timeframe, (1, 'day'))
        
    async def _fetch_satellite_data(self, symbols: List[str], start_date: datetime,
                                  end_date: datetime, timeframe: str) -> Dict[str, List[Dict]]:
        """Fetch data from satellite imagery sources
        
        This method retrieves satellite imagery data that can be used for trading signals:
        - Agricultural yield predictions
        - Oil storage monitoring
        - Shipping and logistics activity
        - Construction and development progress
        - Natural disaster impact assessment
        """
        data = {}
        
        # Get satellite API configuration from settings
        satellite_api_key = self.api_keys.get('SATELLITE_API_KEY')
        if not satellite_api_key:
            logger.error("Missing satellite API key in API keys configuration")
            return data
            
        # Base URL for satellite data API from settings
        base_url = self.settings.satellite_api_url if hasattr(self.settings, 'satellite_api_url') else 'https://api.satellite-data.com/v1'
        
        for symbol in symbols:
            try:
                # Map symbol to relevant satellite data type
                data_type = self._map_symbol_to_satellite_data(symbol)
                if not data_type:
                    logger.warning(f"No satellite data mapping for symbol: {symbol}")
                    continue
                    
                # Construct API request
                params = {
                    'api_key': satellite_api_key,
                    'data_type': data_type,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'resolution': self._convert_timeframe_to_satellite(timeframe)
                }
                
                # Make API request
                logger.info(f"Requesting satellite data from {base_url}/data for {symbol}")
                response = requests.get(f"{base_url}/data", params=params)
                response.raise_for_status()
                
                # Process response
                satellite_data = response.json()
                if satellite_data and 'data' in satellite_data and len(satellite_data['data']) > 0:
                    symbol_data = []
                    
                    for item in satellite_data['data']:
                        # Convert satellite metrics to market data format
                        # This is a simplified example - actual implementation would depend on
                        # the specific satellite data structure and how it maps to trading signals
                        timestamp = datetime.fromisoformat(item['timestamp'])
                        
                        # Calculate synthetic OHLC values based on satellite metrics
                        # In a real implementation, these would be derived from actual satellite measurements
                        base_value = float(item['value'])
                        signal_strength = float(item.get('signal_strength', 0.5))
                        confidence = float(item.get('confidence', 0.8))
                        
                        # Create synthetic price data based on satellite signals
                        # This is just an example - real implementation would use more sophisticated models
                        open_price = base_value
                        close_price = base_value * (1 + (signal_strength - 0.5) * 0.02)
                        high_price = max(open_price, close_price) * (1 + confidence * 0.01)
                        low_price = min(open_price, close_price) * (1 - confidence * 0.01)
                        volume = int(item.get('observation_count', 100) * 1000)
                        
                        bar_data = {
                            'symbol': symbol,
                            'exchange': 'SATELLITE',
                            'timestamp': timestamp,
                            'open': float(open_price),
                            'high': float(high_price),
                            'low': float(low_price),
                            'close': float(close_price),
                            'volume': volume,
                            'vwap': float((high_price + low_price + close_price) / 3),
                            'provider': 'satellite',
                            'satellite_metrics': item  # Store original satellite metrics for reference
                        }
                        symbol_data.append(bar_data)
                    
                    data[symbol] = symbol_data
                    logger.info(f"Fetched {len(symbol_data)} satellite data points for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to fetch satellite data for {symbol}: {e}")
                continue
        
        return data
    
    def _map_symbol_to_satellite_data(self, symbol: str) -> Optional[str]:
        """Map stock symbol to relevant satellite data type"""
        # This mapping would be more comprehensive in a real implementation
        # and might be stored in a database or configuration file
        satellite_mappings = {
            # Agricultural companies
            'CORN': 'crop_health_corn',
            'WEAT': 'crop_health_wheat',
            'SOYB': 'crop_health_soybean',
            
            # Oil & Gas
            'XOM': 'oil_storage',
            'CVX': 'oil_storage',
            'USO': 'oil_storage',
            
            # Shipping & Logistics
            'MAERSK': 'port_activity',
            'FDX': 'logistics_hubs',
            'UPS': 'logistics_hubs',
            
            # Retail
            'WMT': 'parking_lot_occupancy',
            'TGT': 'parking_lot_occupancy',
            
            # Mining
            'BHP': 'mining_activity',
            'RIO': 'mining_activity',
            
            # Default mapping for other symbols
            'DEFAULT': 'economic_activity'
        }
        
        return satellite_mappings.get(symbol, satellite_mappings.get('DEFAULT'))
    
    def _convert_timeframe_to_satellite(self, timeframe: str) -> str:
        """Convert timeframe to satellite API format"""
        timeframe_map = {
            '1Min': 'high',
            '5Min': 'high',
            '15Min': 'medium',
            '1Hour': 'medium',
            '1Day': 'low'
        }
        return timeframe_map.get(timeframe, 'medium')

class DataValidator:
    """Data validation and quality control"""
    
    def __init__(self):
        self.quality_thresholds = {
            'price_change_limit': 0.20,  # 20% max price change
            'volume_spike_limit': 10.0,   # 10x volume spike
            'missing_data_tolerance': 0.05  # 5% missing data tolerance
        }
    
    def validate_market_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate market data quality"""
        issues = []
        quality = DataQuality.VALID
        
        try:
            # Basic data structure validation
            required_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in data or data[field] is None:
                    issues.append(f"Missing required field: {field}")
                    quality = DataQuality.INVALID
            
            if quality == DataQuality.INVALID:
                return ValidationResult(False, quality, issues)
            
            # Price validation
            open_price = float(data['open'])
            high_price = float(data['high'])
            low_price = float(data['low'])
            close_price = float(data['close'])
            
            # Check OHLC relationships
            if not (low_price <= open_price <= high_price and low_price <= close_price <= high_price):
                issues.append("Invalid OHLC relationship")
                quality = DataQuality.SUSPICIOUS
            
            # Check for extreme price movements
            if open_price > 0:
                price_change = abs(close_price - open_price) / open_price
                if price_change > self.quality_thresholds['price_change_limit']:
                    issues.append(f"Extreme price change: {price_change:.2%}")
                    quality = DataQuality.SUSPICIOUS
            
            # Volume validation
            volume = int(data['volume'])
            if volume < 0:
                issues.append("Negative volume")
                quality = DataQuality.INVALID
            
            # Create MarketData object
            market_data = MarketData(
                symbol=data['symbol'],
                exchange=data.get('exchange', 'UNKNOWN'),
                timestamp=data['timestamp'] if isinstance(data['timestamp'], datetime) 
                         else datetime.fromisoformat(str(data['timestamp'])),
                open=Decimal(str(open_price)),
                high=Decimal(str(high_price)),
                low=Decimal(str(low_price)),
                close=Decimal(str(close_price)),
                volume=volume,
                vwap=Decimal(str(data['vwap'])) if data.get('vwap') else None,
                provider=data.get('provider', ''),
                quality_score=1.0 if quality == DataQuality.VALID else 0.5
            )
            
            return ValidationResult(
                is_valid=(quality != DataQuality.INVALID),
                quality=quality,
                issues=issues,
                corrected_data=market_data
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                issues=[f"Validation exception: {str(e)}"]
            )

class DatabaseManager:
    """PostgreSQL database operations"""
    
    def __init__(self):
        self.settings = settings
        self.engine = None
        self.async_engine = None
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Create async engine
            database_url = f"postgresql+asyncpg://{self.settings.database.username}:{self.settings.database.password}@{self.settings.database.host}:{self.settings.database.port}/{self.settings.database.database}"
            self.async_engine = create_async_engine(database_url, echo=False)
            
            # Create sync engine for some operations
            sync_url = f"postgresql://{self.settings.database.username}:{self.settings.database.password}@{self.settings.database.host}:{self.settings.database.port}/{self.settings.database.database}"
            self.engine = create_engine(sync_url)
            
            logger.info("Database connections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_market_data_table(self):
        """Create market data table with proper indexing"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS market_data (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            exchange VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DECIMAL(15,6) NOT NULL,
            high DECIMAL(15,6) NOT NULL,
            low DECIMAL(15,6) NOT NULL,
            close DECIMAL(15,6) NOT NULL,
            volume BIGINT NOT NULL,
            vwap DECIMAL(15,6),
            provider VARCHAR(20),
            quality_score DECIMAL(5,4) DEFAULT 1.0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(symbol, exchange, timestamp, provider)
        );
        
        -- Create indexes for fast queries
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
        ON market_data(symbol, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_market_data_exchange_time 
        ON market_data(exchange, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_market_data_provider 
        ON market_data(provider);
        
        CREATE INDEX IF NOT EXISTS idx_market_data_created_at 
        ON market_data(created_at DESC);
        """
        
        try:
            async with self.async_engine.begin() as conn:
                await conn.execute(text(create_table_sql))
            logger.info("Market data table and indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create market data table: {e}")
            raise
    
    async def bulk_insert_market_data(self, data_list: List[MarketData]) -> int:
        """Bulk insert market data with conflict resolution"""
        if not data_list:
            return 0
        
        insert_sql = """
        INSERT INTO market_data (
            symbol, exchange, timestamp, open, high, low, close, 
            volume, vwap, provider, quality_score
        ) VALUES (
            :symbol, :exchange, :timestamp, :open, :high, :low, :close,
            :volume, :vwap, :provider, :quality_score
        ) ON CONFLICT (symbol, exchange, timestamp, provider) 
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            vwap = EXCLUDED.vwap,
            quality_score = EXCLUDED.quality_score,
            created_at = NOW()
        """
        
        try:
            # Convert MarketData objects to dictionaries
            data_dicts = [data.to_dict() for data in data_list]
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(insert_sql), data_dicts)
                return result.rowcount
                
        except Exception as e:
            logger.error(f"Failed to bulk insert market data: {e}")
            raise
    
    async def get_data_statistics(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get ingestion statistics"""
        base_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT symbol) as unique_symbols,
            MIN(timestamp) as earliest_date,
            MAX(timestamp) as latest_date,
            AVG(quality_score) as avg_quality_score,
            provider,
            COUNT(*) as provider_count
        FROM market_data
        """
        
        if symbols:
            symbol_list = "', '".join(symbols)
            base_query += f" WHERE symbol IN ('{symbol_list}')"
        
        base_query += " GROUP BY provider"
        
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(base_query))
                rows = result.fetchall()
                
                stats = {
                    'total_records': sum(row.provider_count for row in rows),
                    'unique_symbols': max(row.unique_symbols for row in rows) if rows else 0,
                    'earliest_date': min(row.earliest_date for row in rows) if rows else None,
                    'latest_date': max(row.latest_date for row in rows) if rows else None,
                    'avg_quality_score': sum(row.avg_quality_score * row.provider_count for row in rows) / sum(row.provider_count for row in rows) if rows else 0,
                    'by_provider': {row.provider: row.provider_count for row in rows}
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
            return {}

class MarketDataIngestorAgent:
    """LangGraph-based Market Data Ingestor Agent"""
    
    def __init__(self):
        self.data_client = DataProviderClient()
        self.validator = DataValidator()
        self.db_manager = DatabaseManager()
        self.current_provider = DataProvider.ALPACA
        self.failover_attempts = 0
        self.max_failover_attempts = 4  # Increased to account for satellite provider
        
    async def initialize(self):
        """Initialize the agent"""
        await self.db_manager.initialize()
        await self.db_manager.create_market_data_table()
        logger.info("Market Data Ingestor Agent initialized")
    
    def create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for market data ingestion"""
        
        workflow = StateGraph(IngestionState)
        
        # Add nodes
        workflow.add_node("fetch_data", self.fetch_data_node)
        workflow.add_node("validate_data", self.validate_data_node)
        workflow.add_node("store_data", self.store_data_node)
        workflow.add_node("handle_failures", self.handle_failures_node)
        workflow.add_node("generate_stats", self.generate_stats_node)
        
        # Define workflow edges
        workflow.set_entry_point("fetch_data")
        workflow.add_edge("fetch_data", "validate_data")
        workflow.add_edge("validate_data", "store_data")
        workflow.add_edge("store_data", "generate_stats")
        
        # Conditional edges for error handling
        workflow.add_conditional_edges(
            "fetch_data",
            self.should_handle_failures,
            {
                "handle_failures": "handle_failures",
                "continue": "validate_data"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_failures",
            self.should_retry_or_end,
            {
                "retry": "fetch_data",
                "end": END
            }
        )
        
        workflow.add_edge("generate_stats", END)
        
        return workflow
    
    async def fetch_data_node(self, state: IngestionState) -> IngestionState:
        """Fetch market data from current provider"""
        logger.info(f"Fetching data for {len(state.symbols)} symbols from {state.current_provider.value}")
        
        try:
            raw_data = await self.data_client.fetch_historical_data(
                symbols=state.symbols,
                start_date=state.start_date,
                end_date=state.end_date,
                timeframe=state.timeframe,
                provider=state.current_provider
            )
            
            state.raw_data = raw_data
            state.ingestion_stats['fetch_success'] = True
            state.ingestion_stats['records_fetched'] = sum(len(data) for data in raw_data.values())
            
            logger.info(f"Successfully fetched {state.ingestion_stats['records_fetched']} records")
            
        except Exception as e:
            error_msg = f"Failed to fetch data from {state.current_provider.value}: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.ingestion_stats['fetch_success'] = False
        
        return state
    
    async def validate_data_node(self, state: IngestionState) -> IngestionState:
        """Validate and normalize market data"""
        logger.info("Validating market data")
        
        validated_data = {}
        validation_stats = {
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'suspicious_records': 0
        }
        
        for symbol, symbol_data in state.raw_data.items():
            validated_symbol_data = []
            
            for record in symbol_data:
                validation_result = self.validator.validate_market_data(record)
                validation_stats['total_records'] += 1
                
                if validation_result.is_valid:
                    validated_symbol_data.append(validation_result.corrected_data)
                    if validation_result.quality == DataQuality.VALID:
                        validation_stats['valid_records'] += 1
                    else:
                        validation_stats['suspicious_records'] += 1
                else:
                    validation_stats['invalid_records'] += 1
                    logger.warning(f"Invalid data for {symbol}: {validation_result.issues}")
            
            if validated_symbol_data:
                validated_data[symbol] = validated_symbol_data
            else:
                state.failed_symbols.append(symbol)
        
        state.validated_data = validated_data
        state.ingestion_stats['validation'] = validation_stats
        
        logger.info(f"Validation complete: {validation_stats['valid_records']} valid, "
                   f"{validation_stats['suspicious_records']} suspicious, "
                   f"{validation_stats['invalid_records']} invalid records")
        
        return state
    
    async def store_data_node(self, state: IngestionState) -> IngestionState:
        """Store validated data in PostgreSQL"""
        logger.info("Storing validated data in database")
        
        try:
            # Flatten all validated data
            all_data = []
            for symbol_data in state.validated_data.values():
                all_data.extend(symbol_data)
            
            if len(all_data) > 0:
                records_inserted = await self.db_manager.bulk_insert_market_data(all_data)
                state.ingestion_stats['records_stored'] = records_inserted
                state.ingestion_stats['storage_success'] = True
                
                logger.info(f"Successfully stored {records_inserted} records in database")
            else:
                state.ingestion_stats['records_stored'] = 0
                state.ingestion_stats['storage_success'] = False
                logger.warning("No valid data to store")
                
        except Exception as e:
            error_msg = f"Failed to store data in database: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.ingestion_stats['storage_success'] = False
        
        return state
    
    async def handle_failures_node(self, state: IngestionState) -> IngestionState:
        """Handle data provider failures with automatic failover"""
        logger.info(f"Handling failures, attempt {self.failover_attempts + 1}")
        
        self.failover_attempts += 1
        
        # Try failover to different provider in a circular pattern
        if state.current_provider == DataProvider.ALPACA:
            state.current_provider = DataProvider.POLYGON
            logger.info("Failing over from Alpaca to Polygon")
        elif state.current_provider == DataProvider.POLYGON:
            state.current_provider = DataProvider.SATELLITE
            logger.info("Failing over from Polygon to Satellite")
        elif state.current_provider == DataProvider.SATELLITE:
            state.current_provider = DataProvider.ALPACA
            logger.info("Failing over from Satellite to Alpaca")
        
        # Clear previous errors for retry
        state.raw_data = {}
        
        return state
    
    async def generate_stats_node(self, state: IngestionState) -> IngestionState:
        """Generate final ingestion statistics"""
        logger.info("Generating ingestion statistics")
        
        try:
            # Get database statistics
            db_stats = await self.db_manager.get_data_statistics(state.symbols)
            state.ingestion_stats['database_stats'] = db_stats
            
            # Calculate success rate
            total_symbols = len(state.symbols)
            successful_symbols = len(state.validated_data)
            state.ingestion_stats['success_rate'] = successful_symbols / total_symbols if total_symbols > 0 else 0
            
            logger.info(f"Ingestion complete: {successful_symbols}/{total_symbols} symbols successful "
                       f"({state.ingestion_stats['success_rate']:.1%} success rate)")
            
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
        
        return state
    
    def should_handle_failures(self, state: IngestionState) -> str:
        """Determine if failures should be handled"""
        if not state.ingestion_stats.get('fetch_success', True) and self.failover_attempts < self.max_failover_attempts:
            return "handle_failures"
        return "continue"
    
    def should_retry_or_end(self, state: IngestionState) -> str:
        """Determine if should retry or end"""
        if self.failover_attempts < self.max_failover_attempts:
            return "retry"
        return "end"
    
    async def ingest_historical_data(self, 
                                   symbols: List[str],
                                   start_date: datetime,
                                   end_date: datetime,
                                   timeframe: str = '1Day') -> Dict[str, Any]:
        """Main method to ingest historical market data"""
        
        # Reset failover attempts
        self.failover_attempts = 0
        
        # Create initial state
        initial_state = IngestionState(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            current_provider=self.current_provider,
            raw_data={},
            validated_data={},
            failed_symbols=[],
            ingestion_stats={
                'start_time': datetime.utcnow(),
                'symbols_requested': len(symbols)
            },
            errors=[]
        )
        
        # Create and run workflow
        workflow = self.create_workflow()
        app = workflow.compile()
        
        try:
            # Execute the workflow
            final_state = await app.ainvoke(initial_state)
            
            # Add completion time
            final_state.ingestion_stats['end_time'] = datetime.utcnow()
            final_state.ingestion_stats['duration'] = (
                final_state.ingestion_stats['end_time'] - 
                final_state.ingestion_stats['start_time']
            ).total_seconds()
            
            return {
                'success': len(final_state.errors) == 0,
                'statistics': final_state.ingestion_stats,
                'failed_symbols': final_state.failed_symbols,
                'errors': final_state.errors
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                'success': False,
                'statistics': initial_state.ingestion_stats,
                'failed_symbols': symbols,
                'errors': [str(e)]
            }

# Factory function for creating the agent
async def create_market_data_ingestor() -> MarketDataIngestorAgent:
    """Factory function to create and initialize the Market Data Ingestor Agent"""
    agent = MarketDataIngestorAgent()
    await agent.initialize()
    return agent