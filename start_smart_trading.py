#!/usr/bin/env python3
"""
Smart Auto-Trading System
Mean Reversion + Ensemble Strategies (Simplified)
"""

import sys
import os
import asyncio
import time
import random
import math
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env.development')
except:
    pass

# Create logs directory
os.makedirs('logs', exist_ok=True)

class SmartTradingSystem:
    """Smart trading system with mean reversion logic"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'SPY', 'QQQ']
        self.trade_count = 0
        self.position_tracker = {}
        self.price_history = {}
        self.broker = None
        
        # Initialize price history with mock data
        for symbol in self.symbols:
            self.price_history[symbol] = []
            
    def log_trade(self, message):
        """Log trade to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp}: {message}"
        
        print(log_message)
        
        try:
            with open('logs/smart_trades.log', 'a') as f:
                f.write(log_message + '\n')
        except:
            pass
    
    def simulate_price_data(self, symbol):
        """Simulate realistic price movement with mean reversion characteristics"""
        base_prices = {
            'AAPL': 190.0, 'MSFT': 380.0, 'GOOGL': 135.0, 'TSLA': 250.0,
            'NVDA': 450.0, 'AMZN': 155.0, 'SPY': 450.0, 'QQQ': 390.0
        }
        
        base_price = base_prices.get(symbol, 200.0)
        
        # Get recent prices or start fresh
        if len(self.price_history[symbol]) == 0:
            # Initialize with base price
            current_price = base_price
        else:
            # Simulate mean-reverting price movement
            last_price = self.price_history[symbol][-1]
            
            # Mean reversion force (pulls price back to base)
            mean_reversion_force = (base_price - last_price) * 0.1
            
            # Random walk component
            random_change = random.gauss(0, base_price * 0.02)
            
            # Combine forces
            price_change = mean_reversion_force + random_change
            current_price = max(last_price + price_change, base_price * 0.5)
        
        # Keep history (last 50 prices)
        self.price_history[symbol].append(current_price)
        if len(self.price_history[symbol]) > 50:
            self.price_history[symbol].pop(0)
        
        return current_price
    
    def calculate_technical_indicators(self, symbol):
        """Calculate technical indicators from price history"""
        prices = self.price_history[symbol]
        
        if len(prices) < 20:
            return None
        
        current_price = prices[-1]
        
        # Simple Moving Averages
        sma_20 = sum(prices[-20:]) / 20
        sma_10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else sma_20
        
        # Price volatility (standard deviation)
        mean_price = sum(prices[-20:]) / 20
        variance = sum((p - mean_price) ** 2 for p in prices[-20:]) / 20
        std_dev = math.sqrt(variance)
        
        # Z-Score (mean reversion indicator)
        z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
        
        # Simulated RSI-like indicator
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, change) for change in price_changes[-14:]]
        losses = [max(0, -change) for change in price_changes[-14:]]
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 1
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Band equivalent
        upper_band = mean_price + (2 * std_dev)
        lower_band = mean_price - (2 * std_dev)
        bb_position = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
        
        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_10': sma_10,
            'z_score': z_score,
            'rsi': rsi,
            'bb_position': bb_position,
            'volatility': std_dev
        }
    
    def mean_reversion_signal(self, indicators):
        """Generate mean reversion trading signals"""
        if not indicators:
            return "HOLD", 0.0, "No data"
        
        z_score = indicators['z_score']
        rsi = indicators['rsi']
        bb_position = indicators['bb_position']
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        
        signals = []
        
        # Mean Reversion Signals
        
        # Z-Score Analysis (Strong signal when price deviates significantly)
        if z_score < -1.5:  # Price well below mean
            signals.append(("BUY", 0.8, f"Z-Score oversold ({z_score:.2f})"))
        elif z_score > 1.5:  # Price well above mean
            signals.append(("SELL", 0.8, f"Z-Score overbought ({z_score:.2f})"))
        elif z_score < -1.0:
            signals.append(("BUY", 0.6, f"Z-Score moderately oversold ({z_score:.2f})"))
        elif z_score > 1.0:
            signals.append(("SELL", 0.6, f"Z-Score moderately overbought ({z_score:.2f})"))
        
        # RSI-like Analysis
        if rsi < 30:
            signals.append(("BUY", 0.7, f"RSI oversold ({rsi:.1f})"))
        elif rsi > 70:
            signals.append(("SELL", 0.7, f"RSI overbought ({rsi:.1f})"))
        
        # Bollinger Band Analysis
        if bb_position < 0.1:  # Near lower band
            signals.append(("BUY", 0.6, f"Near support level ({bb_position:.2f})"))
        elif bb_position > 0.9:  # Near upper band
            signals.append(("SELL", 0.6, f"Near resistance level ({bb_position:.2f})"))
        
        # Price vs Moving Average
        price_vs_ma = (current_price - sma_20) / sma_20
        if price_vs_ma < -0.05:  # 5% below moving average
            signals.append(("BUY", 0.5, f"Price below MA ({price_vs_ma:.1%})"))
        elif price_vs_ma > 0.05:  # 5% above moving average
            signals.append(("SELL", 0.5, f"Price above MA ({price_vs_ma:.1%})"))
        
        # Ensemble Decision
        if not signals:
            return "HOLD", 0.0, "No clear signals"
        
        buy_signals = [s for s in signals if s[0] == "BUY"]
        sell_signals = [s for s in signals if s[0] == "SELL"]
        
        if len(buy_signals) > len(sell_signals):
            total_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
            reasons = " | ".join(s[2] for s in buy_signals)
            return "BUY", total_confidence, reasons
        
        elif len(sell_signals) > len(buy_signals):
            total_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
            reasons = " | ".join(s[2] for s in sell_signals)
            return "SELL", total_confidence, reasons
        
        return "HOLD", 0.0, "Conflicting signals"
    
    async def analyze_and_trade(self, symbol):
        """Analyze symbol and make intelligent trading decision"""
        try:
            # Simulate getting new price data
            current_price = self.simulate_price_data(symbol)
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(symbol)
            
            if not indicators:
                return
            
            # Get mean reversion signal
            signal, confidence, reason = self.mean_reversion_signal(indicators)
            
            # Only trade on strong signals (>= 50% confidence)
            if signal != "HOLD" and confidence >= 0.5:
                # Position sizing based on confidence
                base_quantity = 5
                quantity = max(1, int(base_quantity * confidence))
                
                self.trade_count += 1
                
                # Execute trade
                if self.broker:
                    try:
                        await self.execute_paper_trade(symbol, signal, quantity, current_price, confidence, reason)
                    except Exception as e:
                        self.log_trade(f"SMART TRADE #{self.trade_count}: {signal} {quantity} {symbol} @ ${current_price:.2f} | Confidence: {confidence:.1%} | {reason} (API Error)")
                else:
                    self.log_trade(f"SMART TRADE #{self.trade_count}: {signal} {quantity} {symbol} @ ${current_price:.2f} | Confidence: {confidence:.1%} | {reason}")
                
                # Update position tracking
                if symbol not in self.position_tracker:
                    self.position_tracker[symbol] = 0
                
                if signal == "BUY":
                    self.position_tracker[symbol] += quantity
                elif signal == "SELL":
                    self.position_tracker[symbol] -= quantity
                
        except Exception as e:
            self.log_trade(f"{symbol}: Analysis error - {e}")
    
    async def execute_paper_trade(self, symbol, signal, quantity, price, confidence, reason):
        """Execute actual paper trade through Alpaca API"""
        try:
            from agents.broker_integration import OrderRequest, OrderSide, OrderType
            
            order_side = OrderSide.BUY if signal == "BUY" else OrderSide.SELL
            
            order_request = OrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                type=OrderType.MARKET
            )
            
            order_response = await self.broker.submit_order(order_request)
            self.log_trade(f"PAPER TRADE #{self.trade_count}: {signal} {quantity} {symbol} @ ${price:.2f} | ID: {order_response.id} | Confidence: {confidence:.1%} | {reason}")
            
        except Exception as e:
            self.log_trade(f"PAPER TRADE #{self.trade_count}: {signal} {quantity} {symbol} @ ${price:.2f} | Confidence: {confidence:.1%} | {reason} (API Error)")
    
    async def start_smart_trading(self):
        """Start the smart trading system"""
        print("HIVE TRADE - SMART AUTOMATED TRADING")
        print("=" * 45)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Strategy: Mean Reversion + Technical Analysis")
        print("Mode: Paper Trading (Safe)")
        print("Symbols:", ", ".join(self.symbols))
        print("-" * 45)
        
        # Initialize broker
        try:
            from agents.broker_integration import AlpacaBrokerIntegration
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            account_info = await self.broker.get_account_info()
            if account_info:
                self.log_trade(f"Connected to Alpaca: {account_info.get('account_number', 'N/A')}")
                self.log_trade(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            else:
                self.log_trade("Alpaca connection failed - using smart mock mode")
                self.broker = None
        except Exception as e:
            self.log_trade(f"Broker error: {e}")
            self.log_trade("Running in smart simulation mode")
            self.broker = None
        
        self.log_trade("Starting intelligent mean reversion analysis...")
        self.log_trade("Building price history for technical analysis...")
        
        # Build initial price history
        for symbol in self.symbols:
            for _ in range(25):  # Build 25 days of history
                self.simulate_price_data(symbol)
        
        self.log_trade("Price history initialized")
        
        try:
            analysis_count = 0
            while True:
                current_time = datetime.now()
                current_hour = current_time.hour
                
                # Trade during market hours (9 AM - 4 PM)
                if 9 <= current_hour <= 16:
                    analysis_count += 1
                    self.log_trade(f"Analysis Cycle #{analysis_count}")
                    
                    # Analyze each symbol
                    for symbol in self.symbols:
                        await self.analyze_and_trade(symbol)
                        await asyncio.sleep(2)  # Small delay between symbols
                    
                    # Log current positions every 5 cycles
                    if analysis_count % 5 == 0 and self.position_tracker:
                        positions_str = ", ".join([f"{sym}: {pos}" for sym, pos in self.position_tracker.items() if pos != 0])
                        if positions_str:
                            self.log_trade(f"Current Positions: {positions_str}")
                
                # Wait 2 minutes before next analysis cycle
                await asyncio.sleep(120)
                
        except KeyboardInterrupt:
            self.log_trade(f"Smart trading stopped by user. Total trades: {self.trade_count}")
        except Exception as e:
            self.log_trade(f"Trading error: {e}")
        
        self.log_trade("Smart trading session ended")
        return self.trade_count

async def main():
    """Main function"""
    print("Starting HiveTrading Smart Auto-Trading System...")
    print("Mean Reversion + Technical Analysis")
    print("Paper Trading Mode (No Real Money)")
    print()
    
    trading_system = SmartTradingSystem()
    result = await trading_system.start_smart_trading()
    
    print(f"\nSmart trading system completed!")
    print(f"Total intelligent trades: {result}")

if __name__ == "__main__":
    asyncio.run(main())