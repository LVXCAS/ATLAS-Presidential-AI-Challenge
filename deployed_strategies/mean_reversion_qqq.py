#!/usr/bin/env python3
"""
Auto-generated strategy: mean_reversion_QQQ
Template: mean_reversion_ml
Symbol: QQQ
Generated: 2025-09-13 11:57:04

Performance Metrics:
- Sharpe Ratio: 0.72
- Annual Return: 11.0%
- Max Drawdown: 12.0%
- Win Rate: 58.0%
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

class Mean_Reversion_QqqStrategy:
    """Auto-generated trading strategy"""
    
    def __init__(self):
        self.name = "mean_reversion_QQQ"
        self.symbol = "QQQ"
        self.template = "mean_reversion_ml"
        self.parameters = {'lookback_period': 30, 'threshold_multiplier': 2.0, 'volatility_window': 10, 'ml_confidence_threshold': 0.7}
        self.performance = {'total_return': 0.15, 'annual_return': 0.11, 'annual_volatility': 0.18, 'sharpe_ratio': 0.72, 'max_drawdown': 0.12, 'calmar_ratio': 0.92, 'win_rate': 0.58, 'total_trades': 38}
        
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
            signal = self._generate_mean_reversion_ml_signal(df)
            
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
    
    def _generate_mean_reversion_ml_signal(self, data: pd.DataFrame) -> int:
        """Generate signal based on mean_reversion_ml template"""
        
        try:
            lookback = 30
            if len(data) < lookback:
                return 0
                
            price_mean = data['Close'].rolling(lookback).mean()
            price_std = data['Close'].rolling(lookback).std()
            
            if price_std.iloc[-1] == 0:
                return 0
                
            z_score = (data['Close'].iloc[-1] - price_mean.iloc[-1]) / price_std.iloc[-1]
            threshold = 2.0
            
            if z_score < -threshold:
                return 1  # Oversold, buy
            elif z_score > threshold:
                return -1  # Overbought, sell
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
strategy_instance = Mean_Reversion_QqqStrategy()

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
