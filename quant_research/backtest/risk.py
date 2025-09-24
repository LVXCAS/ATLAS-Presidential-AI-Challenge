"""Risk management for backtesting."""

from typing import Dict, Any
import pandas as pd


class RiskManager:
    """Simple risk manager for backtesting."""
    
    def __init__(self, config):
        """Initialize risk manager."""
        self.config = config
        self.max_position_size = getattr(config, 'max_position_size', 0.1)
        self.max_daily_loss = getattr(config, 'max_daily_loss', 0.02)
        
    def filter_signals(
        self,
        signals: Dict[str, Any],
        portfolio,
        current_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Filter signals based on risk rules.
        
        Args:
            signals: Trading signals
            portfolio: Portfolio instance
            current_data: Current market data
            
        Returns:
            Filtered signals
        """
        filtered_signals = {}
        
        for symbol, signal in signals.items():
            # Skip if no current price data
            if symbol not in current_data or current_data[symbol].empty:
                continue
                
            # Basic position size check
            if signal.get("action") == "buy":
                quantity = signal.get("quantity", 0)
                current_price = current_data[symbol]['close'].iloc[-1]
                position_value = quantity * current_price
                
                # Check position size limit
                max_position_value = portfolio.total_value * self.max_position_size
                
                if position_value <= max_position_value:
                    filtered_signals[symbol] = signal
                    
            elif signal.get("action") == "sell":
                # Allow sell signals (for now)
                filtered_signals[symbol] = signal
        
        return filtered_signals
    
    def check_daily_loss_limit(self, portfolio) -> bool:
        """Check if daily loss limit is exceeded."""
        # Simplified check
        return portfolio.total_return > -self.max_daily_loss