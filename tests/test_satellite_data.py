"""Tests for the satellite data integration in the Market Data Ingestor."""

import asyncio
import unittest
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_data_ingestor import create_market_data_ingestor, DataProvider
from config.logging_config import setup_logging

# Setup logging
setup_logging()


class TestSatelliteDataIntegration(unittest.TestCase):
    """Test cases for satellite data integration."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.get_event_loop()
        self.agent = self.loop.run_until_complete(create_market_data_ingestor())
        self.agent.current_provider = DataProvider.SATELLITE

    def test_satellite_data_fetch(self):
        """Test fetching satellite data for agricultural commodities."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days

        result = self.loop.run_until_complete(
            self.agent.ingest_historical_data(
                symbols=["CORN", "WEAT"],
                start_date=start_date,
                end_date=end_date,
                timeframe="1Day"
            )
        )

        self.assertTrue(result['success'], "Satellite data ingestion should succeed")
        self.assertGreater(result['statistics'].get('records_stored', 0), 0, 
                          "Should have stored some records")

    def test_satellite_data_metrics(self):
        """Test that satellite data includes the expected metrics."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)  # Just 1 day for quick test

        result = self.loop.run_until_complete(
            self.agent.ingest_historical_data(
                symbols=["CORN"],
                start_date=start_date,
                end_date=end_date,
                timeframe="1Day"
            )
        )

        self.assertTrue(result['success'], "Satellite data ingestion should succeed")
        
        # Check if sample data is available
        self.assertIn('sample_data', result['statistics'], 
                     "Result should include sample data")
        
        if result['statistics']['sample_data']:
            sample = result['statistics']['sample_data'][0]
            self.assertIn('satellite_metrics', sample, 
                         "Sample should include satellite metrics")
            
            # For agricultural commodities, check for specific metrics
            metrics = sample['satellite_metrics']
            self.assertIn('crop_health_index', metrics, 
                         "Agricultural data should include crop health index")

    def test_symbol_mapping(self):
        """Test that symbols are correctly mapped to satellite data types."""
        # Direct test of the mapping function
        self.assertEqual(self.agent.data_provider_client._map_symbol_to_satellite_data("CORN"), 
                         "agriculture", 
                         "CORN should map to agriculture data type")
        
        self.assertEqual(self.agent.data_provider_client._map_symbol_to_satellite_data("XOM"), 
                         "oil_storage", 
                         "XOM should map to oil_storage data type")
        
        self.assertEqual(self.agent.data_provider_client._map_symbol_to_satellite_data("WMT"), 
                         "retail", 
                         "WMT should map to retail data type")

    def test_timeframe_conversion(self):
        """Test that timeframes are correctly converted to satellite API format."""
        # Direct test of the conversion function
        self.assertEqual(self.agent.data_provider_client._convert_timeframe_to_satellite("1Day"), 
                         "daily", 
                         "1Day should convert to daily")
        
        self.assertEqual(self.agent.data_provider_client._convert_timeframe_to_satellite("1Hour"), 
                         "hourly", 
                         "1Hour should convert to hourly")
        
        self.assertEqual(self.agent.data_provider_client._convert_timeframe_to_satellite("5Min"), 
                         "high_frequency", 
                         "5Min should convert to high_frequency")


if __name__ == "__main__":
    unittest.main()