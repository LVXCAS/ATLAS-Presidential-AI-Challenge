# Satellite Data Integration for Trading Signals

## Overview

This module extends the Market Data Ingestor Agent to incorporate satellite imagery data as an additional data source for trading signals. Satellite data can provide unique insights for various asset classes, particularly those affected by physical conditions observable from space.

## Supported Satellite Data Types

### Agricultural Commodities
For agricultural commodities like corn, wheat, and soybeans, satellite data provides:
- Crop health indices
- Soil moisture levels
- Vegetation density metrics

### Oil & Gas
For energy companies and commodities, satellite data provides:
- Storage facility fill rates
- Facility activity levels
- Tanker and shipping traffic

### Retail
For retail companies, satellite data provides:
- Parking lot occupancy rates
- Delivery truck counts
- Estimated foot traffic

## Integration with Market Data Ingestor

The satellite data provider has been integrated as a first-class data source alongside Alpaca and Polygon. Key features include:

1. **Failover Support**: The system can automatically failover to satellite data if other providers fail, and vice versa.

2. **Symbol Mapping**: Automatic mapping of trading symbols to appropriate satellite data types.

3. **Timeframe Conversion**: Standard market data timeframes are converted to satellite API formats.

4. **Configuration**: API keys and endpoints are configurable through the standard settings system.

## Usage Example

```python
# Create the market data ingestor agent
agent = await create_market_data_ingestor()

# Set provider explicitly to satellite
agent.current_provider = DataProvider.SATELLITE

# Define parameters for satellite data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Last 30 days

# Ingest satellite data
result = await agent.ingest_historical_data(
    symbols=["CORN", "WEAT", "SOYB", "XOM", "WMT"],
    start_date=start_date,
    end_date=end_date,
    timeframe="1Day"
)
```

## Configuration

To use the satellite data provider, add the following environment variables:

```
SATELLITE_API_KEY=your_api_key_here
SATELLITE_API_BASE_URL=https://api.satellite-data.com/v1
```

Alternatively, you can set these values directly in the settings.py file.

## Demo

A demonstration of the satellite data integration is available in the examples directory:

```bash
python examples/market_data_ingestor_demo.py --demo satellite
```

This will run a demonstration that fetches satellite data for various symbols and displays the results.