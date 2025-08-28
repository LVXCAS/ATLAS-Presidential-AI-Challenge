#!/usr/bin/env python3
"""
Intelligent Auto-Trading System
Combines Mean Reversion + Ensemble Strategies for Paper Trading
"""

import sys
import os
import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

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

class IntelligentTradingSystem:
    """Intelligent trading system with mean reversion and ensemble strategies"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'SPY', 'QQQ']
        self.trade_count = 0
        self.position_tracker = {}
        self.broker = None
        
    def log_trade(self, message):
        """Log trade to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp}: {message}"
        
        print(log_message)
        
        try:
            with open('logs/intelligent_trades.log', 'a') as f:
                f.write(log_message + '\n')
        except:
            pass
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for trading decisions"""
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Z-Score for Mean Reversion
        df['Price_Mean'] = df['Close'].rolling(30).mean()
        df['Price_Std'] = df['Close'].rolling(30).std()
        df['Z_Score'] = (df['Close'] - df['Price_Mean']) / df['Price_Std']
        
        return df
    
    def mean_reversion_signal(self, df):
        """Generate mean reversion trading signals"""
        if len(df) < 50:
            return "HOLD", 0.0, "Insufficient data"
        
        latest = df.iloc[-1]
        z_score = latest['Z_Score']
        rsi = latest['RSI']
        bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
        
        # Mean Reversion Logic
        signals = []
        confidence = 0.0
        
        # Z-Score based signals
        if z_score < -2:  # Price significantly below mean
            signals.append(("BUY", 0.8, "Z-Score oversold"))
        elif z_score > 2:  # Price significantly above mean
            signals.append(("SELL", 0.8, "Z-Score overbought"))
        
        # RSI based signals
        if rsi < 30:  # Oversold
            signals.append(("BUY", 0.7, "RSI oversold"))
        elif rsi > 70:  # Overbought
            signals.append(("SELL", 0.7, "RSI overbought"))
        
        # Bollinger Bands signals
        if bb_position < 0.1:  # Near lower band
            signals.append(("BUY", 0.6, "Bollinger Band support"))
        elif bb_position > 0.9:  # Near upper band
            signals.append(("SELL", 0.6, "Bollinger Band resistance"))
        
        # Combine signals
        if signals:
            buy_signals = [s for s in signals if s[0] == "BUY"]
            sell_signals = [s for s in signals if s[0] == "SELL"]
            
            if len(buy_signals) > len(sell_signals):
                avg_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
                reasons = ", ".join(s[2] for s in buy_signals)
                return "BUY", avg_confidence, reasons
            elif len(sell_signals) > len(buy_signals):
                avg_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
                reasons = ", ".join(s[2] for s in sell_signals)
                return "SELL", avg_confidence, reasons
        
        return "HOLD", 0.0, "No clear signal"
    
    def momentum_signal(self, df):
        """Generate momentum trading signals"""
        if len(df) < 50:
            return "HOLD", 0.0, "Insufficient data"
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Momentum indicators
        price_change = (latest['Close'] - prev['Close']) / prev['Close']
        macd_signal = latest['MACD'] > latest['MACD_Signal']
        sma_trend = latest['Close'] > latest['SMA_20'] > latest['SMA_50']
        
        signals = []
        
        if price_change > 0.02 and macd_signal and sma_trend:  # Strong upward momentum
            signals.append(("BUY", 0.7, "Strong upward momentum"))
        elif price_change < -0.02 and not macd_signal and not sma_trend:  # Strong downward momentum
            signals.append(("SELL", 0.7, "Strong downward momentum"))
        
        if signals:
            return signals[0]
        
        return "HOLD", 0.0, "No momentum signal"
    
    def ensemble_decision(self, symbol, df):
        """Make ensemble trading decision combining multiple strategies"""
        # Get signals from different strategies
        mean_rev_signal, mean_rev_conf, mean_rev_reason = self.mean_reversion_signal(df)
        momentum_sig, momentum_conf, momentum_reason = self.momentum_signal(df)
        
        # Ensemble decision logic
        signals = []
        
        if mean_rev_signal in ["BUY", "SELL"] and mean_rev_conf > 0.6:
            signals.append((mean_rev_signal, mean_rev_conf * 0.7, f"Mean Reversion: {mean_rev_reason}"))
        
        if momentum_sig in ["BUY", "SELL"] and momentum_conf > 0.5:
            signals.append((momentum_sig, momentum_conf * 0.5, f"Momentum: {momentum_reason}"))
        
        # Combine signals
        if not signals:
            return "HOLD", 0.0, "No trading signals", {}
        
        buy_signals = [s for s in signals if s[0] == "BUY"]
        sell_signals = [s for s in signals if s[0] == "SELL"]
        
        if buy_signals and len(buy_signals) >= len(sell_signals):
            total_confidence = sum(s[1] for s in buy_signals)
            reasons = " | ".join(s[2] for s in buy_signals)
            
            # Position sizing based on confidence
            quantity = max(1, int(total_confidence * 10))
            
            return "BUY", total_confidence, reasons, {"quantity": quantity}
            
        elif sell_signals and len(sell_signals) > len(buy_signals):
            total_confidence = sum(s[1] for s in sell_signals)
            reasons = " | ".join(s[2] for s in sell_signals)
            
            # Position sizing based on confidence
            quantity = max(1, int(total_confidence * 10))
            
            return "SELL", total_confidence, reasons, {"quantity": quantity}
        
        return "HOLD", 0.0, "Conflicting signals", {}
    
    async def analyze_and_trade(self, symbol):
        """Analyze symbol and make trading decision"""
        try:
            # Fetch recent market data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='3mo', interval='1d')
            
            if df.empty or len(df) < 50:
                self.log_trade(f"{symbol}: Insufficient data for analysis")
                return
            
            # Calculate indicators
            df = self.calculate_technical_indicators(df)
            
            # Get ensemble decision
            signal, confidence, reason, params = self.ensemble_decision(symbol, df)
            
            if signal != "HOLD" and confidence > 0.5:
                quantity = params.get("quantity", 1)
                current_price = df['Close'].iloc[-1]
                
                self.trade_count += 1
                
                # Execute trade
                if self.broker:
                    try:
                        await self.execute_paper_trade(symbol, signal, quantity, current_price, reason)
                    except Exception as e:
                        self.log_trade(f"INTELLIGENT TRADE #{self.trade_count}: {signal} {quantity} {symbol} @ ${current_price:.2f} | Confidence: {confidence:.2%} | Reason: {reason} (API Error)")
                else:
                    self.log_trade(f"INTELLIGENT TRADE #{self.trade_count}: {signal} {quantity} {symbol} @ ${current_price:.2f} | Confidence: {confidence:.2%} | Reason: {reason}")
                
                # Update position tracking
                if symbol not in self.position_tracker:
                    self.position_tracker[symbol] = 0
                
                if signal == "BUY":
                    self.position_tracker[symbol] += quantity
                else:
                    self.position_tracker[symbol] -= quantity
                
        except Exception as e:
            self.log_trade(f"{symbol}: Analysis error - {e}")
    
    async def execute_paper_trade(self, symbol, signal, quantity, price, reason):
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
            self.log_trade(f"PAPER TRADE #{self.trade_count}: {signal} {quantity} {symbol} @ ${price:.2f} | Order ID: {order_response.id} | Reason: {reason}")
            
        except Exception as e:
            self.log_trade(f"PAPER TRADE #{self.trade_count}: {signal} {quantity} {symbol} @ ${price:.2f} | Reason: {reason} (API Error: {e})")
    
    async def start_intelligent_trading(self):
        """Start the intelligent trading system"""
        print("HIVE TRADE - INTELLIGENT AUTOMATED TRADING")
        print("=" * 50)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Strategies: Mean Reversion + Momentum + Ensemble")
        print("Mode: Paper Trading (Safe)")
        print("Symbols:", ", ".join(self.symbols))
        print("-" * 50)
        
        # Try to initialize broker
        try:
            from agents.broker_integration import AlpacaBrokerIntegration
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            account_info = await self.broker.get_account_info()
            if account_info:
                self.log_trade(f"Connected to Alpaca: {account_info.get('account_number', 'N/A')}")
                self.log_trade(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            else:
                self.log_trade("Alpaca connection failed - using mock mode")
                self.broker = None
        except Exception as e:
            self.log_trade(f"Broker initialization error: {e}")
            self.log_trade("Running in intelligent mock mode")
            self.broker = None
        
        self.log_trade("Starting intelligent trading analysis...")
        self.log_trade("Using Mean Reversion + Ensemble Strategies")
        
        try:
            analysis_count = 0
            while True:
                current_time = datetime.now()
                current_hour = current_time.hour
                
                # Trade during market hours (9 AM - 4 PM)
                if 9 <= current_hour <= 16:
                    analysis_count += 1
                    self.log_trade(f"=== Analysis Cycle #{analysis_count} ===")
                    
                    # Analyze each symbol
                    for symbol in self.symbols:
                        await self.analyze_and_trade(symbol)
                        await asyncio.sleep(5)  # Small delay between symbols
                    
                    # Log current positions
                    if self.position_tracker:
                        positions_str = ", ".join([f"{sym}: {pos}" for sym, pos in self.position_tracker.items() if pos != 0])
                        if positions_str:
                            self.log_trade(f"Current Positions: {positions_str}")
                
                # Wait 5 minutes before next analysis cycle
                await asyncio.sleep(300)
                
        except KeyboardInterrupt:
            self.log_trade(f"Intelligent trading stopped. Total trades: {self.trade_count}")
        except Exception as e:
            self.log_trade(f"Trading error: {e}")
        
        self.log_trade("Intelligent trading session ended")
        return self.trade_count

async def main():
    """Main function to run intelligent trading"""
    print("Starting HiveTrading Intelligent Auto-Trading System...")
    print("This system uses:")
    print("- Mean Reversion Strategy (RSI, Bollinger Bands, Z-Score)")
    print("- Ensemble Approach (Multiple strategies combined)")
    print("- Confidence-based position sizing")
    print("- All trades in PAPER TRADING mode (no real money)")
    print()
    
    trading_system = IntelligentTradingSystem()
    result = await trading_system.start_intelligent_trading()
    
    print(f"\nIntelligent trading system stopped successfully!")
    print(f"Total intelligent trades generated: {result}")

if __name__ == "__main__":
    asyncio.run(main())