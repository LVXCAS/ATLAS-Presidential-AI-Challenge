#!/usr/bin/env python3
"""
Hive Trade - Live Trading Engine
Connects Bloomberg Terminal to Alpaca Markets for real trading execution
"""

import alpaca_trade_api as tradeapi
import asyncio
import os
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np

class HiveTradingEngine:
    def __init__(self):
        load_dotenv()
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            api_version='v2'
        )
        
        # Trading parameters
        self.max_position_size = 0.1  # Max 10% of portfolio per position
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.05  # 5% take profit
        
        # AI Agent signals (simulated for now)
        self.agent_signals = {
            'MOMENTUM_01': 'NEUTRAL',
            'SENTIMENT_02': 'NEUTRAL', 
            'MEAN_REV_03': 'NEUTRAL',
            'NEWS_NLP_04': 'NEUTRAL',
            'RISK_MGR_05': 'ACTIVE',
            'ARBIT_06': 'NEUTRAL'
        }
        
        # Watchlist
        self.watchlist = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN']
        
        print("=" * 60)
        print("HIVE TRADE - LIVE TRADING ENGINE INITIALIZED")
        print("=" * 60)
        print(f"Account: {self.api.get_account().status}")
        print(f"Buying Power: ${float(self.api.get_account().buying_power):,.2f}")
        print(f"Market Open: {self.api.get_clock().is_open}")
        print("=" * 60)

    async def get_market_data(self, symbol):
        """Get real-time market data for a symbol"""
        try:
            # Get latest quote
            latest_trade = self.api.get_latest_trade(symbol)
            latest_quote = self.api.get_latest_quote(symbol)
            
            return {
                'symbol': symbol,
                'price': float(latest_trade.price),
                'bid': float(latest_quote.bid_price),
                'ask': float(latest_quote.ask_price),
                'volume': int(latest_trade.size),
                'timestamp': latest_trade.timestamp
            }
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            return None

    def calculate_position_size(self, symbol, signal_strength=0.5):
        """Calculate optimal position size based on portfolio and risk management"""
        account = self.api.get_account()
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value)
        
        # Base position size as percentage of portfolio
        base_size = portfolio_value * self.max_position_size * signal_strength
        
        # Ensure we don't exceed buying power
        max_affordable = min(base_size, buying_power * 0.8)  # Use 80% of buying power max
        
        return max_affordable

    async def execute_trade(self, symbol, side, qty, order_type='market'):
        """Execute a trade through Alpaca"""
        try:
            print(f"\n[TRADE EXECUTION] {side} {qty} shares of {symbol}")
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day'
            )
            
            print(f"[ORDER SUBMITTED] ID: {order.id} | Status: {order.status}")
            return order
            
        except Exception as e:
            print(f"[TRADE ERROR] Failed to execute {side} {symbol}: {e}")
            return None

    async def analyze_signals(self, symbol):
        """Analyze various signals and generate trading decision"""
        try:
            # Get market data
            data = await self.get_market_data(symbol)
            if not data:
                return None
                
            # Simple momentum strategy (demo)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d', interval='1h')
            
            if len(hist) < 10:
                return None
                
            # Calculate simple indicators
            current_price = data['price']
            sma_short = hist['Close'].rolling(window=5).mean().iloc[-1]
            sma_long = hist['Close'].rolling(window=10).mean().iloc[-1]
            rsi = self.calculate_rsi(hist['Close'])
            
            # Generate signal
            signal = 'HOLD'
            confidence = 0.5
            
            if sma_short > sma_long and rsi < 70:
                signal = 'BUY'
                confidence = min(0.8, (sma_short - sma_long) / sma_long + 0.5)
            elif sma_short < sma_long and rsi > 30:
                signal = 'SELL'  
                confidence = min(0.8, (sma_long - sma_short) / sma_long + 0.5)
                
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'price': current_price,
                'sma_short': sma_short,
                'sma_long': sma_long,
                'rsi': rsi,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error analyzing signals for {symbol}: {e}")
            return None

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    async def risk_check(self, symbol, side, qty):
        """Perform risk checks before executing trade"""
        account = self.api.get_account()
        positions = self.api.list_positions()
        
        # Check if we already have a position in this symbol
        current_position = None
        for pos in positions:
            if pos.symbol == symbol:
                current_position = pos
                break
        
        # Portfolio concentration check
        portfolio_value = float(account.portfolio_value)
        market_data = await self.get_market_data(symbol)
        if market_data:
            position_value = qty * market_data['price']
            concentration = position_value / portfolio_value
            
            if concentration > self.max_position_size:
                print(f"[RISK CHECK] Position too large: {concentration:.2%} > {self.max_position_size:.2%}")
                return False
        
        # Day trade count check
        if account.pattern_day_trader and int(account.daytrade_count) >= 3:
            print("[RISK CHECK] Day trade limit reached")
            return False
        
        print("[RISK CHECK] Passed all risk checks")
        return True

    async def monitor_positions(self):
        """Monitor open positions for stop loss/take profit"""
        positions = self.api.list_positions()
        
        for position in positions:
            try:
                current_price_data = await self.get_market_data(position.symbol)
                if not current_price_data:
                    continue
                    
                current_price = current_price_data['price']
                entry_price = float(position.avg_cost)
                qty = int(position.qty)
                side = 'long' if qty > 0 else 'short'
                
                # Calculate P&L percentage
                if side == 'long':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                print(f"[POSITION MONITOR] {position.symbol}: {pnl_pct:.2%} P&L")
                
                # Check stop loss
                if pnl_pct <= -self.stop_loss:
                    print(f"[STOP LOSS TRIGGERED] {position.symbol} at {pnl_pct:.2%}")
                    await self.execute_trade(
                        position.symbol, 
                        'sell' if side == 'long' else 'buy', 
                        abs(qty)
                    )
                
                # Check take profit  
                elif pnl_pct >= self.take_profit:
                    print(f"[TAKE PROFIT TRIGGERED] {position.symbol} at {pnl_pct:.2%}")
                    await self.execute_trade(
                        position.symbol,
                        'sell' if side == 'long' else 'buy',
                        abs(qty)
                    )
                    
            except Exception as e:
                print(f"Error monitoring position {position.symbol}: {e}")

    async def trading_loop(self):
        """Main trading loop"""
        print("\n[TRADING LOOP STARTED]")
        
        while True:
            try:
                # Check if market is open
                clock = self.api.get_clock()
                if not clock.is_open:
                    print(f"[MARKET CLOSED] Next open: {clock.next_open}")
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                print(f"\n[SCAN] {datetime.now().strftime('%H:%M:%S')} - Scanning watchlist...")
                
                # Analyze each symbol in watchlist
                for symbol in self.watchlist:
                    signal_data = await self.analyze_signals(symbol)
                    
                    if signal_data and signal_data['signal'] != 'HOLD':
                        print(f"[SIGNAL] {symbol}: {signal_data['signal']} (Confidence: {signal_data['confidence']:.2f})")
                        
                        # Calculate position size
                        position_value = self.calculate_position_size(symbol, signal_data['confidence'])
                        qty = int(position_value / signal_data['price'])
                        
                        if qty > 0:
                            # Perform risk check
                            side = 'buy' if signal_data['signal'] == 'BUY' else 'sell'
                            
                            if await self.risk_check(symbol, side, qty):
                                # Execute trade
                                await self.execute_trade(symbol, side, qty)
                            else:
                                print(f"[RISK BLOCK] Trade blocked for {symbol}")
                
                # Monitor existing positions
                await self.monitor_positions()
                
                # Wait before next scan
                await asyncio.sleep(60)  # Scan every minute
                
            except Exception as e:
                print(f"[TRADING LOOP ERROR] {e}")
                await asyncio.sleep(30)

    async def start_trading(self):
        """Start the live trading engine"""
        print("\n*** STARTING LIVE TRADING ENGINE ***")
        print("Press Ctrl+C to stop trading")
        
        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Trading engine stopped by user")
        except Exception as e:
            print(f"[FATAL ERROR] Trading engine crashed: {e}")

async def main():
    engine = HiveTradingEngine()
    await engine.start_trading()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTrading engine shutdown complete.")