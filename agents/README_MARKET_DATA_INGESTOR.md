# Market Data Ingestor Agent

## Overview

The Market Data Ingestor Agent is a LangGraph-based autonomous agent that handles the ingestion and processing of global market data with automatic failover, data validation, and PostgreSQL storage. This agent is a core component of the LangGraph Adaptive Multi-Strategy AI Trading System.

## Features

### Core Capabilities
- **Multi-Provider Support**: Integrates with Alpaca and Polygon APIs for market data
- **Automatic Failover**: Seamlessly switches between data providers on failures
- **Data Validation**: Comprehensive quality control and anomaly detection
- **LangGraph Integration**: Built using LangGraph state machine for autonomous operation
- **PostgreSQL Storage**: Optimized storage with proper indexing and conflict resolution
- **Real-time Processing**: Handles both real-time and historical data ingestion

### Data Sources
- **Primary**: Alpaca Markets API
- **Secondary**: Polygon.io API
- **Supported Assets**: Equities, ETFs, Options (future), Forex (future), Crypto (future)

### Data Quality Features
- OHLCV relationship validation
- Extreme price movement detection
- Volume anomaly detection
- Missing data handling
- Quality scoring system

## Architecture

### LangGraph Workflow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Fetch Data  │───►│ Validate    │───►│ Store Data  │───►│ Generate    │
│             │    │ Data        │    │             │    │ Statistics  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                                      │
       ▼                                      │
┌─────────────┐    ┌─────────────┐           │
│ Handle      │◄───│ Should      │◄──────────┘
│ Failures    │    │ Retry?      │
└─────────────┘    └─────────────┘
```

### Components

#### 1. DataProviderClient
- Unified interface for multiple data providers
- Handles API authentication and rate limiting
- Automatic data format normalization

#### 2. DataValidator
- Real-time data quality assessment
- Configurable validation rules
- Anomaly detection and correction

#### 3. DatabaseManager
- Async PostgreSQL operations
- Bulk insert optimization
- Conflict resolution (upsert)
- Performance monitoring

#### 4. MarketDataIngestorAgent
- LangGraph state machine orchestration
- Autonomous decision making
- Error handling and recovery
- Performance optimization

## Database Schema

```sql
CREATE TABLE market_data (
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
```

### Indexes
- `idx_market_data_symbol_time`: Fast symbol-based queries
- `idx_market_data_exchange_time`: Exchange-based filtering
- `idx_market_data_provider`: Provider-based analysis
- `idx_market_data_quality`: Quality-based filtering

## Usage

### Basic Usage

```python
import asyncio
from datetime import datetime, timedelta
from agents.market_data_ingestor import create_market_data_ingestor

async def main():
    # Create the agent
    agent = await create_market_data_ingestor()
    
    # Define parameters
    symbols = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Ingest data
    result = await agent.ingest_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe="1Day"
    )
    
    if result['success']:
        print(f"Successfully ingested {result['statistics']['records_stored']} records")
    else:
        print(f"Ingestion failed: {result['errors']}")

asyncio.run(main())
```

### Advanced Configuration

```python
# Custom validation thresholds
agent.validator.quality_thresholds = {
    'price_change_limit': 0.15,  # 15% max price change
    'volume_spike_limit': 5.0,   # 5x volume spike
    'missing_data_tolerance': 0.02  # 2% missing data tolerance
}

# Custom failover settings
agent.max_failover_attempts = 5
agent.current_provider = DataProvider.POLYGON  # Start with Polygon
```

## Configuration

### Environment Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_system
DB_USER=postgres
DB_PASSWORD=your_password

# API Keys
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=your_polygon_key

# Market Data Settings
MAX_SYMBOLS_PER_REQUEST=100
DATA_RETENTION_DAYS=365
```

### Settings Configuration

The agent uses the centralized settings system:

```python
from config.settings import settings

# Access database settings
db_config = settings.database

# Access broker settings
broker_config = settings.brokers
```

## Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/test_market_data_ingestor.py -v

# Run specific test class
python -m pytest tests/test_market_data_ingestor.py::TestDataValidator -v

# Run with coverage
python -m pytest tests/test_market_data_ingestor.py --cov=agents.market_data_ingestor
```

### Integration Tests
```bash
# Run integration tests (requires database)
python -m pytest tests/test_market_data_ingestor.py -m integration -v
```

### Acceptance Test
```bash
# Run the official acceptance test
python scripts/test_market_data_acceptance.py
```

## Performance

### Benchmarks
- **Throughput**: 50+ records/second
- **Latency**: <100ms per API call
- **Memory**: <500MB for 10,000 records
- **Storage**: ~1KB per OHLCV record

### Optimization Features
- Bulk database operations
- Connection pooling
- Async I/O throughout
- Efficient data structures
- Configurable batch sizes

## Error Handling

### Automatic Recovery
- API rate limit handling
- Network timeout recovery
- Database connection retry
- Data provider failover

### Error Types
- `DataProviderError`: API-related issues
- `ValidationError`: Data quality issues
- `DatabaseError`: Storage-related issues
- `ConfigurationError`: Setup issues

### Monitoring
- Comprehensive logging
- Performance metrics
- Error tracking
- Quality statistics

## Deployment

### Dependencies
```bash
pip install langgraph langchain langchain-core
pip install alpaca-trade-api polygon-api-client
pip install asyncpg pandas numpy
```

### Docker Support
```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "agents.market_data_ingestor"]
```

### Production Considerations
- Use connection pooling
- Configure proper logging
- Set up monitoring alerts
- Implement backup strategies
- Use environment-specific configs

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Check API key configuration
   python -c "from config.secure_config import get_api_keys; print(get_api_keys())"
   ```

2. **Database Connection**
   ```bash
   # Test database connection
   python -c "from config.database import test_connection; test_connection()"
   ```

3. **Data Quality Issues**
   ```python
   # Check validation settings
   agent.validator.quality_thresholds
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable SQL logging
agent.db_manager.async_engine.echo = True
```

## Future Enhancements

### Planned Features
- Real-time streaming data
- Options chain ingestion
- Cryptocurrency support
- Alternative data sources
- Machine learning-based anomaly detection

### Scalability Improvements
- Horizontal scaling support
- Distributed processing
- Advanced caching strategies
- Load balancing

## Contributing

### Development Setup
```bash
git clone <repository>
cd trading-system
pip install -e .
pip install -r requirements-dev.txt
pre-commit install
```

### Code Standards
- Follow PEP 8
- Add type hints
- Write comprehensive tests
- Update documentation
- Use async/await patterns

## License

This project is part of the LangGraph Adaptive Multi-Strategy AI Trading System.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test cases
3. Check the logs for error details
4. Consult the LangGraph documentation