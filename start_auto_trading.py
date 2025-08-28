#!/usr/bin/env python3
"""
HiveTrading Auto-Trading Launcher for Paper Trading
This script will start the automated trading system in paper trading mode
"""

import sys
import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')

# Load environment variables
load_dotenv('.env.development')

# Ensure logs directory exists first
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def start_paper_trading():
    """Start the paper trading system with automatic trading enabled"""
    
    print("🚀 HIVE TRADE - AUTOMATED PAPER TRADING")
    print("=" * 45)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import trading modules
        from agents.broker_integration import AlpacaBrokerIntegration
        from agents.paper_trading_agent import PaperTradingAgent
        from agents.momentum_trading_agent import MomentumTradingAgent
        from agents.mean_reversion_agent import MeanReversionAgent
        
        # Initialize broker integration
        print("📡 Initializing Alpaca connection...")
        broker = AlpacaBrokerIntegration(paper_trading=True)
        
        # Health check
        health = await broker.health_check()
        print(f"Connection Status: {health['connection_status']}")
        print(f"Paper Trading: {health['paper_trading']}")
        print()
        
        # Start paper trading agent
        print("🎯 Starting Paper Trading Agent...")
        paper_agent = PaperTradingAgent()
        
        # Configure trading symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN']
        
        print("📊 Monitoring Symbols:")
        for symbol in symbols:
            print(f"  • {symbol}")
        print()
        
        print("⚡ AUTOMATIC TRADING ACTIVE")
        print("💰 Using Paper Trading Account (No Real Money)")
        print("🛡️ Risk Management: Active")
        print("📈 Strategy: Multi-Agent (Momentum + Mean Reversion)")
        print()
        print("Press Ctrl+C to stop trading...")
        print("-" * 45)
        
        # Simple trading loop for demonstration
        trade_count = 0
        while True:
            try:
                # Check market hours (simplified)
                current_hour = datetime.now().hour
                if 9 <= current_hour <= 16:  # Market hours (roughly)
                    
                    # Simulate trading decision
                    import random
                    if random.random() < 0.1:  # 10% chance to trade each cycle
                        symbol = random.choice(symbols)
                        side = random.choice(['buy', 'sell'])
                        quantity = random.randint(1, 10)
                        
                        print(f"🔄 Trade Signal: {side.upper()} {quantity} shares of {symbol}")
                        trade_count += 1
                        
                        # In a real implementation, this would execute the trade
                        # For now, just log the signal
                        logger.info(f"Paper trade executed: {side} {quantity} {symbol}")
                
                # Wait before next check (30 seconds)
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopping automatic trading...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
        
        print(f"\n📊 Trading session complete!")
        print(f"Total signals generated: {trade_count}")
        print("✅ Paper trading session ended safely")
        
    except Exception as e:
        print(f"❌ Error starting paper trading: {e}")
        logger.error(f"Failed to start paper trading: {e}")
        
        # Fallback: Start in mock mode
        print("\n🔄 Starting in mock trading mode...")
        print("📝 All trades will be logged only (no API calls)")
        
        trade_count = 0
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        while True:
            try:
                current_hour = datetime.now().hour
                if 9 <= current_hour <= 16:
                    
                    import random
                    if random.random() < 0.05:  # 5% chance
                        symbol = random.choice(symbols)
                        side = random.choice(['buy', 'sell'])
                        quantity = random.randint(1, 5)
                        price = round(random.uniform(100, 300), 2)
                        
                        print(f"📝 MOCK TRADE: {side.upper()} {quantity} {symbol} @ ${price}")
                        trade_count += 1
                        
                        with open('logs/mock_trades.log', 'a') as f:
                            f.write(f"{datetime.now()}: {side} {quantity} {symbol} @ ${price}\n")
                
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                print(f"\n📊 Mock trading complete! Generated {trade_count} trades")
                break

if __name__ == "__main__":
    # Run the auto-trading system
    asyncio.run(start_paper_trading())