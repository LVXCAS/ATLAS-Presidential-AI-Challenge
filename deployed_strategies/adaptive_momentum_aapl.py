#!/usr/bin/env python3
"""
Auto-generated strategy: adaptive_momentum_AAPL
Template: adaptive_momentum
Symbol: AAPL
Generated: 2025-09-13 11:57:04

Performance Metrics:
- Sharpe Ratio: 0.69
- Annual Return: 13.0%
- Max Drawdown: 14.0%
- Win Rate: 62.0%
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

class Adaptive_Momentum_AaplStrategy:
    """Auto-generated trading strategy"""
    
    def __init__(self):
        self.name = "adaptive_momentum_AAPL"
        self.symbol = "AAPL"
        self.template = "adaptive_momentum"
        self.parameters = {'fast_period': 12, 'slow_period': 50, 'volatility_lookback': 20, 'regime_threshold': 1.0}
        self.performance = {'total_return': 0.18, 'annual_return': 0.13, 'annual_volatility': 0.22, 'sharpe_ratio': 0.69, 'max_drawdown': 0.14, 'calmar_ratio': 0.93, 'win_rate': 0.62, 'total_trades': 52}
        
        # Risk management
        self.max_position_size = 0.05  # 5% max position
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.06  # 6% take profit
        
        # Strategy state
        self.current_position = 0
        self.last_signal = 0
        self.entry_price = None
    
    async def get_signal(self, market_data: Dict) -> Dict:
        """Generate trading signal based on strategy logic"""
        try:
            # Convert market data to format expected by strategy
            if 'price_data' in market_data:
                df = pd.DataFrame(market_data['price_data'])
            else:
                # Create simple DataFrame from current data
                df = pd.DataFrame({
                    'Close': [market_data.get('close', 100)],
                    'High': [market_data.get('high', 101)],
                    'Low': [market_data.get('low', 99)],
                    'Volume': [market_data.get('volume', 1000000)]
                })
            
            # Generate strategy-specific signal
            signal = self._generate_adaptive_momentum_signal(df)
            
            # Apply risk management
            signal = self._apply_risk_management(signal, market_data)
            
            return {
                'signal': signal,
                'symbol': self.symbol,
                'strategy': self.name,
                'confidence': abs(signal) if signal != 0 else 0,
                'timestamp': datetime.now().isoformat(),
                'parameters': self.parameters
            }
            
        except Exception as e:
            print(f"Error generating signal for {self.name}: {e}")
            return {
                'signal': 0,
                'symbol': self.symbol,
                'strategy': self.name,
                'confidence': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _generate_adaptive_momentum_signal(self, data: pd.DataFrame) -> int:
        """Generate signal based on adaptive_momentum template"""
        
        try:
            if len(data) < max(12, 50):
                return 0
                
            fast_sma = data['Close'].rolling(12).mean()
            slow_sma = data['Close'].rolling(50).mean()
            
            if len(fast_sma) < 2 or len(slow_sma) < 2:
                return 0
            
            # Momentum signal
            if fast_sma.iloc[-1] > slow_sma.iloc[-1] and fast_sma.iloc[-2] <= slow_sma.iloc[-2]:
                return 1  # Buy signal
            elif fast_sma.iloc[-1] < slow_sma.iloc[-1] and fast_sma.iloc[-2] >= slow_sma.iloc[-2]:
                return -1  # Sell signal
            else:
                return 0
        except:
            return 0
            
    
    def _apply_risk_management(self, signal: int, market_data: Dict) -> int:
        """Apply risk management rules"""
        current_price = market_data.get('close', market_data.get('price', 100))
        
        # Check position limits
        if abs(self.current_position) >= self.max_position_size:
            if (signal > 0 and self.current_position > 0) or (signal < 0 and self.current_position < 0):
                return 0  # Don't increase position
        
        # Stop loss check
        if self.current_position != 0 and self.entry_price:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if self.current_position > 0:  # Long position
                if pnl_pct < -self.stop_loss:
                    return -1  # Stop loss hit
                elif pnl_pct > self.take_profit:
                    return -1  # Take profit
            else:  # Short position
                if pnl_pct > self.stop_loss:
                    return 1   # Stop loss hit
                elif pnl_pct < -self.take_profit:
                    return 1   # Take profit
        
        return signal
    
    def update_position(self, new_position: float, price: float):
        """Update current position tracking"""
        if new_position != 0 and self.current_position == 0:
            self.entry_price = price  # New entry
        elif new_position == 0:
            self.entry_price = None   # Position closed
        
        self.current_position = new_position
        self.last_signal = 1 if new_position > 0 else (-1 if new_position < 0 else 0)

# Create strategy instance for import
strategy_instance = Adaptive_Momentum_AaplStrategy()

async def get_trading_signal(market_data: Dict) -> Dict:
    """Main entry point for strategy signals"""
    return await strategy_instance.get_signal(market_data)

if __name__ == "__main__":
    # Test the strategy
    import asyncio
    
    async def test_strategy():
        test_data = {
            'close': 150.0,
            'high': 152.0,
            'low': 148.0,
            'volume': 1500000
        }
        
        signal = await get_trading_signal(test_data)
        print(f"Strategy: {signal['strategy']}")
        print(f"Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']}")
        print(f"Symbol: {signal['symbol']}")
    
    asyncio.run(test_strategy())
