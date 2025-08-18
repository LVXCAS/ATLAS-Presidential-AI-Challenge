"""Satellite Trading Agent Demonstration

This script demonstrates the usage of the Satellite Trading Agent
for generating trading signals based on satellite imagery data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_data_ingestor import create_market_data_ingestor, DataProvider
from agents.satellite_trading_agent import create_satellite_trading_agent
from config.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Demo configuration
SATELLITE_DEMO_SYMBOLS = [
    # Agricultural commodities
    "CORN", "WEAT", "SOYB",
    # Oil & Gas
    "XOM", "CVX", "USO",
    # Retail with parking lot monitoring
    "WMT", "TGT"
]


async def demonstrate_satellite_trading():
    """Demonstrate satellite-based trading signals"""
    logger.info("=== Satellite Trading Signals Demo ===")
    
    try:
        # Create the market data ingestor agent
        data_agent = await create_market_data_ingestor()
        
        # Set provider explicitly to satellite
        data_agent.current_provider = DataProvider.SATELLITE
        
        # Define parameters for satellite data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        # Ingest satellite data
        logger.info("Ingesting satellite data...")
        result = await data_agent.ingest_historical_data(
            symbols=SATELLITE_DEMO_SYMBOLS,
            start_date=start_date,
            end_date=end_date,
            timeframe="1Day"
        )
        
        if not result['success']:
            logger.error(f"Failed to ingest satellite data: {result['errors']}")
            return
            
        logger.info(f"✅ Successfully ingested satellite data")
        logger.info(f"   - Records stored: {result['statistics'].get('records_stored', 0)}")
        
        # Create the satellite trading agent
        trading_agent = await create_satellite_trading_agent()
        
        # Get the market data with satellite metrics
        market_data = result['statistics'].get('sample_data', [])
        
        if not market_data:
            logger.error("No market data available with satellite metrics")
            return
            
        # Generate trading signals
        logger.info("Generating trading signals from satellite data...")
        signals = await trading_agent.generate_signals(market_data)
        
        # Display the signals
        logger.info(f"Generated {len(signals)} trading signals:")
        
        for i, signal in enumerate(signals, 1):
            logger.info(f"\nSignal {i}:")
            logger.info(f"   Symbol: {signal.symbol}")
            logger.info(f"   Type: {signal.signal_type}")
            logger.info(f"   Value: {signal.value:.4f} (negative=bearish, positive=bullish)")
            logger.info(f"   Confidence: {signal.confidence:.2f}")
            logger.info(f"   Satellite Data Source: {signal.satellite_data_source}")
            logger.info(f"   Top Reasons:")
            for reason in signal.top_3_reasons:
                logger.info(f"      - {reason}")
            
            if signal.satellite_indicators:
                logger.info(f"   Key Metrics:")
                for key, value in signal.satellite_indicators.items():
                    logger.info(f"      - {key}: {value}")
        
        return signals
        
    except Exception as e:
        logger.error(f"Satellite trading demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(demonstrate_satellite_trading())