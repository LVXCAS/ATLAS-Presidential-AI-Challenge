"""Trade execution engine for backtesting."""

from typing import Dict, Any, Optional
from datetime import datetime


class ExecutionEngine:
    """Simple execution engine for backtesting."""
    
    def __init__(self, config):
        """Initialize execution engine."""
        self.config = config
        
    def execute_signal(
        self,
        symbol: str,
        signal: Dict[str, Any],
        current_price: float,
        portfolio,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Execute a trading signal.
        
        Args:
            symbol: Symbol to trade
            signal: Trading signal
            current_price: Current market price
            portfolio: Portfolio instance
            timestamp: Current timestamp
            
        Returns:
            Trade record or None if not executed
        """
        if not signal or "action" not in signal:
            return None
            
        action = signal["action"]
        quantity = signal.get("quantity", 0)
        
        if quantity <= 0:
            return None
            
        # Execute trade based on action
        if action.lower() == "buy":
            success = portfolio.open_position(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                timestamp=timestamp
            )
        elif action.lower() == "sell":
            success = portfolio.close_position(
                symbol=symbol,
                price=current_price,
                timestamp=timestamp,
                quantity=quantity
            )
        else:
            return None
            
        if success:
            return {
                "timestamp": timestamp,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "signal": signal
            }
        
        return None


class OrderManager:
    """Simple order manager for backtesting."""
    
    def __init__(self):
        """Initialize order manager."""
        self.pending_orders = []
        
    def submit_order(self, order):
        """Submit an order."""
        self.pending_orders.append(order)
        
    def process_orders(self, current_prices):
        """Process pending orders."""
        # Simplified - immediately execute all orders
        executed = []
        for order in self.pending_orders:
            executed.append(order)
        
        self.pending_orders.clear()
        return executed