#!/usr/bin/env python3
"""
AUTONOMOUS TRADE READINESS MONITOR
Continuously monitor account status and execute Intel-puts-style trades the moment account unlocks
Works 24/7 until trades are executed
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import json
import logging

class AutonomousTradeReadinessMonitor:
    """Monitor account and execute trades when ready"""

    def __init__(self):
        load_dotenv()

        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL')
        )

        # The queued Intel-puts-style trades (from hybrid system)
        self.queued_trades = [
            {
                'symbol': 'META',
                'type': 'STOCK',
                'quantity': 432,
                'allocation_pct': 35.0,
                'target_amount': 311040,
                'reasoning': 'Highest genetic score (7.20) - AI-optimized quality stock'
            },
            {
                'symbol': 'SPY',
                'type': 'STOCK',
                'quantity': 376,
                'allocation_pct': 28.0,
                'target_amount': 248634,
                'reasoning': 'Fed meeting catalyst - 17 days to rate cut decision'
            }
        ]

        self.total_queued_value = 559674  # Focus on stocks first, then options

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - MONITOR - %(message)s')
        self.logger = logging.getLogger(__name__)

    def check_account_trading_readiness(self):
        """Check if account can execute new trades"""
        try:
            account = self.api.get_account()

            equity = float(account.equity)
            buying_power = float(account.buying_power)
            cash = float(account.cash)

            # Test trade readiness by checking restrictions
            trading_blocked = equity < 0
            has_buying_power = buying_power >= self.total_queued_value

            status = {
                'equity': equity,
                'buying_power': buying_power,
                'cash': cash,
                'trading_blocked': trading_blocked,
                'has_sufficient_capital': has_buying_power,
                'ready_to_trade': not trading_blocked and has_buying_power,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }

            return status

        except Exception as e:
            self.logger.error(f"Error checking readiness: {e}")
            return None

    def execute_queued_trade(self, trade):
        """Execute individual queued trade"""
        symbol = trade['symbol']
        quantity = trade['quantity']

        try:
            print(f"[EXECUTING] {symbol} - {quantity} shares")

            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='day'
            )

            print(f"[SUCCESS] Order submitted: {order.id}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to execute {symbol}: {e}")
            return False

    def execute_all_queued_trades(self):
        """Execute all queued trades when account is ready"""
        print(f"\n[EXECUTION] DEPLOYING INTEL-PUTS-STYLE TRADES")
        print(f"Executing {len(self.queued_trades)} trades worth ${self.total_queued_value:,.0f}")
        print("=" * 60)

        successful_trades = 0

        for trade in self.queued_trades:
            success = self.execute_queued_trade(trade)
            if success:
                successful_trades += 1
            time.sleep(3)  # Brief pause between trades

        print(f"\n[RESULTS] Executed {successful_trades}/{len(self.queued_trades)} trades")

        if successful_trades > 0:
            print("[SUCCESS] Intel-puts-style deployment initiated!")
            print("Monitoring positions for 25-50% monthly returns...")
            return True
        else:
            print("[FAILED] No trades executed - monitoring continues...")
            return False

    def run_continuous_monitoring(self):
        """Run 24/7 monitoring until trades execute"""
        print("AUTONOMOUS TRADE READINESS MONITOR")
        print("=" * 80)
        print("Monitoring account status 24/7 until Intel-puts-style trades execute")
        print(f"Queued trades: ${self.total_queued_value:,.0f} (META + SPY)")
        print("Will execute immediately when account restrictions lift")
        print("=" * 80)

        check_count = 0

        while True:
            try:
                check_count += 1

                # Check trading readiness
                status = self.check_account_trading_readiness()

                if not status:
                    print(f"[{check_count:04d}] Error checking account - retrying in 1 minute")
                    time.sleep(60)
                    continue

                # Display status
                equity = status['equity']
                buying_power = status['buying_power']
                timestamp = status['timestamp']
                ready = status['ready_to_trade']

                print(f"[{check_count:04d}] {timestamp} | Equity: ${equity:>10,.0f} | "
                      f"Buying Power: ${buying_power:>12,.0f} | "
                      f"Ready: {'YES' if ready else 'NO'}")

                # Execute trades if ready
                if ready:
                    print(f"\n[READY] ACCOUNT UNLOCKED FOR TRADING!")
                    success = self.execute_all_queued_trades()

                    if success:
                        print(f"\n[COMPLETE] Mission accomplished - trades deployed!")
                        print("Intel-puts-style positions are now active")
                        print("Continue monitoring with other autonomous systems...")
                        break
                    else:
                        print(f"\n[RETRY] Execution failed - continuing monitoring...")

                # Wait before next check
                if check_count % 20 == 0:  # Every 20 checks (~10 minutes)
                    print(f"[STATUS] Completed {check_count} readiness checks")
                    print(f"         Waiting for positive equity (currently ${equity:,.0f})")

                time.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                print(f"\n[STOPPED] Monitoring stopped by user after {check_count} checks")
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def save_trade_queue(self):
        """Save current trade queue to file for persistence"""
        try:
            queue_data = {
                'trades': self.queued_trades,
                'total_value': self.total_queued_value,
                'created': datetime.now().isoformat()
            }

            with open('intel_puts_trade_queue.json', 'w') as f:
                json.dump(queue_data, f, indent=2)

            print("[SAVED] Trade queue saved to intel_puts_trade_queue.json")

        except Exception as e:
            print(f"[ERROR] Failed to save trade queue: {e}")

def main():
    """Run autonomous trade readiness monitor"""
    monitor = AutonomousTradeReadinessMonitor()

    # Save trade queue for persistence
    monitor.save_trade_queue()

    # Start continuous monitoring
    monitor.run_continuous_monitoring()

if __name__ == "__main__":
    main()