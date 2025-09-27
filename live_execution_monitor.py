#!/usr/bin/env python3
"""
LIVE EXECUTION MONITOR
Real-time monitoring for the 4-trade execution plan
Tracks META, SPY, AAPL calls, GOOGL calls
"""

import yfinance as yf
import time
from datetime import datetime, timedelta
import logging

class LiveExecutionMonitor:
    """Real-time execution monitoring for Intel-puts-style trades"""

    def __init__(self):
        # The 4 execution targets
        self.targets = {
            'META': {'type': 'STOCK', 'allocation': 0.35, 'shares': 432, 'amount': 311040},
            'SPY': {'type': 'STOCK', 'allocation': 0.28, 'shares': 376, 'amount': 248634},
            'AAPL': {'type': 'CALL', 'strike': 265, 'contracts': 235, 'amount': 177801},
            'GOOGL': {'type': 'CALL', 'strike': 252, 'contracts': 123, 'amount': 88560}
        }

        self.total_deployment = 826035

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - EXECUTION - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_market_open_countdown(self):
        """Calculate time until market open"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)  # 9:30 AM EST
        if now.hour >= 16:  # After market close, next day
            market_open += timedelta(days=1)

        time_to_open = market_open - now
        return time_to_open

    def get_real_time_prices(self):
        """Get real-time prices for execution targets"""
        prices = {}

        for symbol in ['META', 'SPY', 'AAPL', 'GOOGL']:
            try:
                ticker = yf.Ticker(symbol)
                # Get latest price
                hist = ticker.history(period='1d', interval='1m')
                if not hist.empty:
                    prices[symbol] = {
                        'price': float(hist['Close'].iloc[-1]),
                        'volume': int(hist['Volume'].iloc[-1]),
                        'time': hist.index[-1].strftime('%H:%M:%S')
                    }
                else:
                    prices[symbol] = {'price': 'N/A', 'volume': 'N/A', 'time': 'N/A'}

            except Exception as e:
                prices[symbol] = {'price': 'ERROR', 'volume': 'ERROR', 'time': str(e)[:20]}

        return prices

    def analyze_execution_readiness(self, prices):
        """Analyze if conditions are optimal for execution"""
        readiness = {}

        for symbol, data in prices.items():
            if data['price'] != 'N/A' and data['price'] != 'ERROR':
                target_info = self.targets[symbol]

                # Calculate position value at current price
                if target_info['type'] == 'STOCK':
                    current_value = data['price'] * target_info['shares']
                    price_deviation = abs(current_value - target_info['amount']) / target_info['amount']
                else:  # CALL options - simplified analysis
                    current_value = target_info['amount']  # Use planned amount
                    price_deviation = 0.02  # Assume 2% options deviation

                readiness[symbol] = {
                    'current_price': data['price'],
                    'target_shares/contracts': target_info.get('shares', target_info.get('contracts')),
                    'current_value': current_value,
                    'planned_value': target_info['amount'],
                    'deviation': price_deviation,
                    'volume': data['volume'],
                    'ready': price_deviation < 0.05,  # Ready if less than 5% deviation
                    'type': target_info['type']
                }
            else:
                readiness[symbol] = {'ready': False, 'error': data['price']}

        return readiness

    def run_live_monitor(self):
        """Run continuous monitoring until market open"""
        print("LIVE EXECUTION MONITOR - INTEL-PUTS-STYLE TRADES")
        print("=" * 80)
        print("Monitoring 4-trade execution plan:")
        print("META (35%), SPY (28%), AAPL CALLS (20%), GOOGL CALLS (10%)")
        print("=" * 80)

        while True:
            try:
                # Get countdown
                countdown = self.get_market_open_countdown()

                if countdown.total_seconds() <= 0:
                    print("\n*** MARKET IS OPEN - EXECUTE TRADES NOW! ***")
                    break

                # Get prices
                prices = self.get_real_time_prices()

                # Analyze readiness
                readiness = self.analyze_execution_readiness(prices)

                # Display status
                print(f"\n=== EXECUTION STATUS - {datetime.now().strftime('%H:%M:%S')} ===")
                print(f"Market opens in: {str(countdown).split('.')[0]}")
                print(f"Total deployment ready: ${self.total_deployment:,.0f}")

                print("\nTARGET READINESS:")
                print("Symbol | Type  | Price    | Volume   | Value     | Status")
                print("-" * 65)

                for symbol, status in readiness.items():
                    if 'ready' in status and status['ready']:
                        ready_status = "[OK] READY"
                    elif 'error' in status:
                        ready_status = f"[ERR] {status['error']}"
                    else:
                        ready_status = "[WAIT] WAIT"

                    if 'current_price' in status:
                        print(f"{symbol:6} | {status['type']:5} | ${status['current_price']:7.2f} | {status['volume']:8,} | ${status['current_value']:8,.0f} | {ready_status}")
                    else:
                        print(f"{symbol:6} | ERROR | ERROR    | ERROR    | ERROR     | {ready_status}")

                # Overall readiness
                ready_count = sum(1 for s in readiness.values() if s.get('ready', False))
                print(f"\nOVERALL READINESS: {ready_count}/4 targets ready")

                if ready_count == 4:
                    print("[READY] ALL TARGETS READY FOR EXECUTION!")
                elif ready_count >= 2:
                    print("[PARTIAL] PARTIAL READINESS - Monitor closely")
                else:
                    print("[WAIT] WAITING FOR OPTIMAL CONDITIONS")

                time.sleep(30)  # Update every 30 seconds

            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(10)

def main():
    """Run live execution monitoring"""
    monitor = LiveExecutionMonitor()
    monitor.run_live_monitor()

if __name__ == "__main__":
    main()