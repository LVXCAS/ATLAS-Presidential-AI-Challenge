"""Satellite Trading Agent for generating signals based on satellite imagery data."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .market_data_ingestor import DataProvider, MarketData
from .langgraph_workflow import Signal, SatelliteMetrics

logger = logging.getLogger(__name__)


class SatelliteTradingAgent:
    """Trading agent that generates signals based on satellite imagery data."""

    def __init__(self):
        """Initialize the satellite trading agent."""
        self.name = "satellite_trading_agent"
        self.version = "1.0.0"
        self.supported_symbols = {
            "agriculture": ["CORN", "WEAT", "SOYB"],
            "oil_storage": ["XOM", "CVX", "USO"],
            "retail": ["WMT", "TGT"]
        }
        
        # Thresholds for generating signals
        self.thresholds = {
            "agriculture": {
                "crop_health_index": 0.8,  # High crop health is bullish
                "soil_moisture": 0.6,      # Good soil moisture is bullish
                "vegetation_density": 0.7   # High vegetation density is bullish
            },
            "oil_storage": {
                "storage_fill_rate": 0.7,   # High storage fill is bearish (oversupply)
                "facility_activity": 0.8,   # High facility activity is bullish
                "tanker_traffic": 5         # High tanker traffic threshold
            },
            "retail": {
                "parking_lot_occupancy": 0.7,  # High occupancy is bullish
                "delivery_truck_count": 15,    # High delivery count is bullish
                "foot_traffic_estimate": 2000  # High foot traffic threshold
            }
        }

    async def generate_signals(self, market_data: List[Dict]) -> List[Signal]:
        """Generate trading signals based on satellite data.
        
        Args:
            market_data: List of market data points with satellite metrics
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Group data by symbol
        symbol_data = {}
        for data_point in market_data:
            symbol = data_point.get("symbol")
            if symbol not in symbol_data:
                symbol_data[symbol] = []
            symbol_data[symbol].append(data_point)
        
        # Process each symbol
        for symbol, data_points in symbol_data.items():
            # Determine satellite data type for this symbol
            satellite_data_type = self._get_satellite_data_type(symbol)
            if not satellite_data_type:
                continue
                
            # Get the most recent data point
            data_points.sort(key=lambda x: x.get("timestamp", ""))
            latest_data = data_points[-1]
            
            # Extract satellite metrics
            satellite_metrics = latest_data.get("satellite_metrics", {})
            if not satellite_metrics:
                continue
                
            # Generate signal based on satellite data type
            signal = self._generate_signal_for_type(
                symbol, 
                satellite_data_type, 
                satellite_metrics,
                latest_data
            )
            
            if signal:
                signals.append(signal)
        
        return signals

    def _get_satellite_data_type(self, symbol: str) -> Optional[str]:
        """Determine the satellite data type for a symbol."""
        for data_type, symbols in self.supported_symbols.items():
            if symbol in symbols:
                return data_type
        return None

    def _generate_signal_for_type(
        self, 
        symbol: str, 
        data_type: str, 
        metrics: Dict, 
        market_data: Dict
    ) -> Optional[Signal]:
        """Generate a signal based on the satellite data type and metrics."""
        if data_type == "agriculture":
            return self._generate_agriculture_signal(symbol, metrics, market_data)
        elif data_type == "oil_storage":
            return self._generate_oil_storage_signal(symbol, metrics, market_data)
        elif data_type == "retail":
            return self._generate_retail_signal(symbol, metrics, market_data)
        return None

    def _generate_agriculture_signal(
        self, 
        symbol: str, 
        metrics: Dict, 
        market_data: Dict
    ) -> Optional[Signal]:
        """Generate signal for agricultural commodities."""
        # Extract relevant metrics
        crop_health = metrics.get("crop_health_index", 0)
        soil_moisture = metrics.get("soil_moisture", 0)
        vegetation_density = metrics.get("vegetation_density", 0)
        
        # Calculate signal value (-1 to 1 scale)
        thresholds = self.thresholds["agriculture"]
        
        # Higher crop health and vegetation density are bullish
        # Soil moisture should be in a good range (not too dry, not too wet)
        crop_health_signal = (crop_health - thresholds["crop_health_index"]) * 2
        soil_moisture_signal = 1 - abs(soil_moisture - thresholds["soil_moisture"]) * 2
        vegetation_signal = (vegetation_density - thresholds["vegetation_density"]) * 2
        
        # Combine signals with weights
        signal_value = (
            crop_health_signal * 0.4 + 
            soil_moisture_signal * 0.3 + 
            vegetation_signal * 0.3
        )
        
        # Calculate confidence based on how strong the metrics are
        confidence = min(0.9, abs(signal_value) * 0.8 + 0.2)
        
        # Generate reasons
        reasons = []
        if crop_health > thresholds["crop_health_index"]:
            reasons.append(f"Crop health index is strong at {crop_health:.2f}")
        else:
            reasons.append(f"Crop health index is weak at {crop_health:.2f}")
            
        if abs(soil_moisture - thresholds["soil_moisture"]) < 0.1:
            reasons.append(f"Soil moisture is optimal at {soil_moisture:.2f}")
        elif soil_moisture < thresholds["soil_moisture"]:
            reasons.append(f"Soil moisture is too low at {soil_moisture:.2f}")
        else:
            reasons.append(f"Soil moisture is too high at {soil_moisture:.2f}")
            
        if vegetation_density > thresholds["vegetation_density"]:
            reasons.append(f"Vegetation density is high at {vegetation_density:.2f}")
        else:
            reasons.append(f"Vegetation density is low at {vegetation_density:.2f}")
        
        # Create signal
        return Signal(
            symbol=symbol,
            signal_type="SATELLITE_AGRICULTURE",
            value=signal_value,
            confidence=confidence,
            top_3_reasons=reasons[:3],
            timestamp=datetime.now(),
            model_version=self.version,
            agent_name=self.name,
            satellite_data_source="agriculture",
            satellite_indicators={
                "crop_health_index": crop_health,
                "soil_moisture": soil_moisture,
                "vegetation_density": vegetation_density
            }
        )

    def _generate_oil_storage_signal(
        self, 
        symbol: str, 
        metrics: Dict, 
        market_data: Dict
    ) -> Optional[Signal]:
        """Generate signal for oil & gas companies."""
        # Extract relevant metrics
        storage_fill = metrics.get("storage_fill_rate", 0)
        facility_activity = metrics.get("facility_activity", 0)
        tanker_traffic = metrics.get("tanker_traffic", 0)
        
        # Calculate signal value (-1 to 1 scale)
        thresholds = self.thresholds["oil_storage"]
        
        # Higher storage fill is bearish (oversupply)
        # Higher facility activity and tanker traffic are bullish
        storage_signal = -1 * (storage_fill - thresholds["storage_fill_rate"]) * 2
        activity_signal = (facility_activity - thresholds["facility_activity"]) * 2
        tanker_signal = (tanker_traffic - thresholds["tanker_traffic"]) / 5  # Normalize
        
        # Combine signals with weights
        signal_value = (
            storage_signal * 0.5 + 
            activity_signal * 0.3 + 
            tanker_signal * 0.2
        )
        
        # Calculate confidence
        confidence = min(0.9, abs(signal_value) * 0.7 + 0.3)
        
        # Generate reasons
        reasons = []
        if storage_fill > thresholds["storage_fill_rate"]:
            reasons.append(f"Storage fill rate is high at {storage_fill:.2f} (bearish)")
        else:
            reasons.append(f"Storage fill rate is low at {storage_fill:.2f} (bullish)")
            
        if facility_activity > thresholds["facility_activity"]:
            reasons.append(f"Facility activity is high at {facility_activity:.2f} (bullish)")
        else:
            reasons.append(f"Facility activity is low at {facility_activity:.2f} (bearish)")
            
        if tanker_traffic > thresholds["tanker_traffic"]:
            reasons.append(f"Tanker traffic is high at {tanker_traffic} (bullish)")
        else:
            reasons.append(f"Tanker traffic is low at {tanker_traffic} (bearish)")
        
        # Create signal
        return Signal(
            symbol=symbol,
            signal_type="SATELLITE_OIL_STORAGE",
            value=signal_value,
            confidence=confidence,
            top_3_reasons=reasons[:3],
            timestamp=datetime.now(),
            model_version=self.version,
            agent_name=self.name,
            satellite_data_source="oil_storage",
            satellite_indicators={
                "storage_fill_rate": storage_fill,
                "facility_activity": facility_activity,
                "tanker_traffic": float(tanker_traffic)
            }
        )

    def _generate_retail_signal(
        self, 
        symbol: str, 
        metrics: Dict, 
        market_data: Dict
    ) -> Optional[Signal]:
        """Generate signal for retail companies."""
        # Extract relevant metrics
        parking_occupancy = metrics.get("parking_lot_occupancy", 0)
        truck_count = metrics.get("delivery_truck_count", 0)
        foot_traffic = metrics.get("foot_traffic_estimate", 0)
        
        # Calculate signal value (-1 to 1 scale)
        thresholds = self.thresholds["retail"]
        
        # Higher values for all metrics are bullish for retail
        parking_signal = (parking_occupancy - thresholds["parking_lot_occupancy"]) * 2
        truck_signal = (truck_count - thresholds["delivery_truck_count"]) / 15  # Normalize
        traffic_signal = (foot_traffic - thresholds["foot_traffic_estimate"]) / 2000  # Normalize
        
        # Combine signals with weights
        signal_value = (
            parking_signal * 0.4 + 
            truck_signal * 0.3 + 
            traffic_signal * 0.3
        )
        
        # Calculate confidence
        confidence = min(0.9, abs(signal_value) * 0.8 + 0.2)
        
        # Generate reasons
        reasons = []
        if parking_occupancy > thresholds["parking_lot_occupancy"]:
            reasons.append(f"Parking lot occupancy is high at {parking_occupancy:.2f} (bullish)")
        else:
            reasons.append(f"Parking lot occupancy is low at {parking_occupancy:.2f} (bearish)")
            
        if truck_count > thresholds["delivery_truck_count"]:
            reasons.append(f"Delivery truck count is high at {truck_count} (bullish)")
        else:
            reasons.append(f"Delivery truck count is low at {truck_count} (bearish)")
            
        if foot_traffic > thresholds["foot_traffic_estimate"]:
            reasons.append(f"Foot traffic estimate is high at {foot_traffic} (bullish)")
        else:
            reasons.append(f"Foot traffic estimate is low at {foot_traffic} (bearish)")
        
        # Create signal
        return Signal(
            symbol=symbol,
            signal_type="SATELLITE_RETAIL",
            value=signal_value,
            confidence=confidence,
            top_3_reasons=reasons[:3],
            timestamp=datetime.now(),
            model_version=self.version,
            agent_name=self.name,
            satellite_data_source="retail",
            satellite_indicators={
                "parking_lot_occupancy": parking_occupancy,
                "delivery_truck_count": float(truck_count),
                "foot_traffic_estimate": float(foot_traffic)
            }
        )


async def create_satellite_trading_agent() -> SatelliteTradingAgent:
    """Factory function to create and initialize a satellite trading agent."""
    agent = SatelliteTradingAgent()
    return agent