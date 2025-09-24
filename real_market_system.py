#!/usr/bin/env python3
"""
REAL MARKET AUTONOMOUS TRADING SYSTEM
- Real market data from Yahoo Finance
- Actual market hours detection
- Real-time price feeds
- Continuous operation during market hours
- Paper trading with REAL market responses
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timezone
import pytz
import asyncio
import alpaca_trade_api as tradeapi
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_trading.log'),
        logging.StreamHandler()
    ]
)

class RealMarketSystem:
    def __init__(self):
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BTC-USD', 'ETH-USD']
        self.positions = {}
        self.capital = 500000.0
        self.is_market_open = False
        self.market_timezone = pytz.timezone('America/New_York')

        # Initialize Alpaca for paper trading
        self.alpaca = None
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if api_key and secret_key:
                self.alpaca = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
                logging.info(f"Connected to Alpaca paper trading: {base_url}")
            else:
                logging.warning("Alpaca API keys not found - using simulation mode")
        except Exception as e:
            logging.error(f"Failed to connect to Alpaca: {e}")

        logging.info("REAL MARKET SYSTEM INITIALIZED")
        logging.info(f"Tracking {len(self.symbols)} symbols")
        logging.info(f"Starting capital: ${self.capital:,.2f}")

    def check_market_hours(self):
        """Check if market is currently open"""
        try:
            now = datetime.now(self.market_timezone)

            # Check if weekday (0=Monday, 6=Sunday)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False

            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

            # Extended hours: 4:00 AM - 8:00 PM ET
            extended_open = now.replace(hour=4, minute=0, second=0, microsecond=0)
            extended_close = now.replace(hour=20, minute=0, second=0, microsecond=0)

            is_regular_hours = market_open <= now <= market_close
            is_extended_hours = extended_open <= now <= extended_close

            if is_regular_hours:
                logging.info(f"MARKET IS OPEN - Regular hours - {now.strftime('%H:%M:%S ET')}")
                return True
            elif is_extended_hours:
                logging.info(f"MARKET IS OPEN - Extended hours - {now.strftime('%H:%M:%S ET')}")
                return True
            else:
                logging.info(f"MARKET IS CLOSED - {now.strftime('%H:%M:%S ET')}")
                return False

        except Exception as e:
            logging.error(f"Error checking market hours: {e}")
            return False

    def get_real_market_data(self):
        """Get real-time market data from Yahoo Finance"""
        try:
            market_data = {}

            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)

                    # Get current price
                    info = ticker.info
                    current_price = info.get('regularMarketPrice') or info.get('previousClose', 0)

                    # Get recent data for analysis
                    hist = ticker.history(period="1d", interval="1m")

                    if not hist.empty:
                        latest = hist.iloc[-1]

                        market_data[symbol] = {
                            'price': current_price,
                            'volume': latest.get('Volume', 0),
                            'open': latest.get('Open', current_price),
                            'high': latest.get('High', current_price),
                            'low': latest.get('Low', current_price),
                            'change': ((current_price - latest.get('Open', current_price)) / latest.get('Open', current_price)) * 100 if latest.get('Open') else 0,
                            'timestamp': datetime.now().isoformat()
                        }

                        logging.info(f"{symbol}: ${current_price:.2f} ({market_data[symbol]['change']:+.2f}%)")

                except Exception as e:
                    logging.error(f"Error getting data for {symbol}: {e}")
                    continue

            return market_data

        except Exception as e:
            logging.error(f"Error getting market data: {e}")
            return {}

    def analyze_signal(self, symbol, data):
        """Analyze real market data for trading signals"""
        try:
            price = data['price']
            change = data['change']
            volume = data['volume']

            # Simple momentum strategy with real data
            signal_strength = 0.0
            action = 'HOLD'

            # Strong upward momentum
            if change > 0.8 and volume > 500000:
                signal_strength = min(abs(change) / 3.0, 0.9)
                action = 'BUY'

            # Strong downward momentum
            elif change < -0.8 and volume > 500000:
                signal_strength = min(abs(change) / 3.0, 0.9)
                action = 'SELL'

            # Moderate signals
            elif abs(change) > 0.3:
                signal_strength = min(abs(change) / 5.0, 0.7)
                action = 'BUY' if change > 0 else 'SELL'

            if signal_strength > 0.15:  # More sensitive to smaller moves
                logging.info(f"SIGNAL: {symbol} {action} - Strength: {signal_strength:.2f} - Change: {change:+.2f}% - Price: ${price:.2f}")
                return {
                    'symbol': symbol,
                    'action': action,
                    'strength': signal_strength,
                    'price': price,
                    'change': change,
                    'confidence': signal_strength
                }

            return None

        except Exception as e:
            logging.error(f"Error analyzing signal for {symbol}: {e}")
            return None

    def execute_trade(self, signal):
        """Execute trade with real broker or simulation"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
            confidence = signal['confidence']

            # Calculate position size based on confidence
            position_value = self.capital * confidence * 0.1  # Max 10% per position
            quantity = int(position_value / price)

            if quantity <= 0:
                return False

            if self.alpaca:
                # Execute with Alpaca paper trading
                try:
                    side = 'buy' if action == 'BUY' else 'sell'

                    order = self.alpaca.submit_order(
                        symbol=symbol.replace('-USD', ''),  # Remove crypto suffix for Alpaca
                        qty=quantity,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )

                    logging.info(f"REAL EXECUTION: {symbol} {action} {quantity} @ ${price:.2f} - Order ID: {order.id}")
                    return True

                except Exception as e:
                    logging.error(f"Alpaca execution failed: {e}")
                    # Fall back to simulation

            # Simulation execution
            execution_price = price * (1 + np.random.uniform(-0.001, 0.001))  # Small slippage

            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0, 'avg_price': 0}

            if action == 'BUY':
                old_quantity = self.positions[symbol]['quantity']
                old_value = old_quantity * self.positions[symbol]['avg_price']
                new_value = quantity * execution_price
                total_quantity = old_quantity + quantity

                if total_quantity > 0:
                    self.positions[symbol]['avg_price'] = (old_value + new_value) / total_quantity
                    self.positions[symbol]['quantity'] = total_quantity

            elif action == 'SELL':
                self.positions[symbol]['quantity'] = max(0, self.positions[symbol]['quantity'] - quantity)

            logging.info(f"SIMULATED EXECUTION: {symbol} {action} {quantity} @ ${execution_price:.2f}")
            logging.info(f"Position: {self.positions[symbol]['quantity']} shares @ ${self.positions[symbol]['avg_price']:.2f}")

            return True

        except Exception as e:
            logging.error(f"Error executing trade: {e}")
            return False

    async def run_continuous_trading(self):
        """Run continuous autonomous trading during market hours"""
        logging.info("STARTING CONTINUOUS AUTONOMOUS TRADING")
        logging.info("Monitoring real market data and executing trades...")

        while True:
            try:
                # Check market hours
                self.is_market_open = self.check_market_hours()

                if self.is_market_open:
                    # Get real market data
                    market_data = self.get_real_market_data()

                    if market_data:
                        signals_generated = 0
                        trades_executed = 0

                        # Analyze each symbol
                        for symbol, data in market_data.items():
                            signal = self.analyze_signal(symbol, data)

                            if signal:
                                signals_generated += 1

                                # Execute trade
                                if self.execute_trade(signal):
                                    trades_executed += 1

                        if signals_generated > 0:
                            logging.info(f"Cycle complete: {signals_generated} signals, {trades_executed} executions")

                    # Wait 30 seconds between cycles during market hours
                    await asyncio.sleep(30)

                else:
                    # Market closed - wait 5 minutes before checking again
                    logging.info("Market closed - waiting 5 minutes...")
                    await asyncio.sleep(300)

            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)

async def main():
    """Main function to run the real market system"""
    logging.info("=" * 80)
    logging.info("REAL MARKET AUTONOMOUS TRADING SYSTEM")
    logging.info("- Real Yahoo Finance market data")
    logging.info("- Actual market hours detection")
    logging.info("- Continuous operation")
    logging.info("- Paper trading with real market responses")
    logging.info("=" * 80)

    system = RealMarketSystem()
    await system.run_continuous_trading()

if __name__ == "__main__":
    asyncio.run(main())