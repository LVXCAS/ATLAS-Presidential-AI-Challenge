#!/usr/bin/env python3
"""
Simple Auto-Trading System for Paper Trading
"""

import sys
import os
import asyncio
import time
import random
from datetime import datetime

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

def log_trade(message):
    """Log trade to file and console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp}: {message}"
    
    print(log_message)
    
    try:
        with open('logs/paper_trades.log', 'a') as f:
            f.write(log_message + '\n')
    except:
        pass

async def start_mock_trading():
    """Start mock trading system"""
    
    print("HIVE TRADE - AUTOMATED PAPER TRADING")
    print("=" * 40)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mode: Paper Trading (Mock)")
    print("Status: ACTIVE")
    print("-" * 40)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'SPY', 'QQQ']
    trade_count = 0
    
    try:
        # Try to import and test Alpaca connection
        from agents.broker_integration import AlpacaBrokerIntegration
        
        broker = AlpacaBrokerIntegration(paper_trading=True)
        log_trade("Alpaca broker initialized successfully")
        
        # Test connection
        account_info = await broker.get_account_info()
        if account_info:
            log_trade(f"Connected to Alpaca account: {account_info.get('account_number', 'N/A')}")
            log_trade(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        else:
            log_trade("Alpaca connection failed - using mock mode")
            broker = None
            
    except Exception as e:
        log_trade(f"Alpaca connection error: {e}")
        log_trade("Running in mock mode")
        broker = None
    
    log_trade("Starting automated trading loop...")
    log_trade("Press Ctrl+C to stop")
    
    try:
        while True:
            current_time = datetime.now()
            current_hour = current_time.hour
            
            # Trade during market hours (9 AM - 4 PM)
            if 9 <= current_hour <= 16:
                
                # Random chance to generate trade signal (10%)
                if random.random() < 0.1:
                    symbol = random.choice(symbols)
                    side = random.choice(['BUY', 'SELL'])
                    quantity = random.randint(1, 10)
                    price = round(random.uniform(100, 400), 2)
                    
                    trade_count += 1
                    
                    if broker:
                        try:
                            # Create actual paper trade request
                            from agents.broker_integration import OrderRequest, OrderSide, OrderType
                            
                            order_side = OrderSide.BUY if side == 'BUY' else OrderSide.SELL
                            
                            order_request = OrderRequest(
                                symbol=symbol,
                                qty=quantity,
                                side=order_side,
                                type=OrderType.MARKET
                            )
                            
                            order_response = await broker.submit_order(order_request)
                            log_trade(f"PAPER TRADE #{trade_count}: {side} {quantity} {symbol} - Order ID: {order_response.id}")
                            
                        except Exception as e:
                            log_trade(f"MOCK TRADE #{trade_count}: {side} {quantity} {symbol} @ ${price} (API error: {e})")
                    else:
                        log_trade(f"MOCK TRADE #{trade_count}: {side} {quantity} {symbol} @ ${price}")
            
            # Wait 30 seconds before next check
            await asyncio.sleep(30)
            
    except KeyboardInterrupt:
        log_trade(f"Trading stopped by user. Total trades: {trade_count}")
    except Exception as e:
        log_trade(f"Trading error: {e}")
    
    log_trade("Auto-trading session ended")
    print("\nTrading system stopped successfully!")
    return trade_count

if __name__ == "__main__":
    print("Starting HiveTrading Auto-Trading System...")
    print("This will run continuously and generate trades automatically")
    print("All trades are in PAPER TRADING mode (no real money)")
    print()
    
    result = asyncio.run(start_mock_trading())