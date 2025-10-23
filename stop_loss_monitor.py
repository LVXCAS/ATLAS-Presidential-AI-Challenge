#!/usr/bin/env python3
"""
STOP LOSS MONITOR - Week 2
Monitors all open positions and closes losing positions that hit stop loss thresholds
"""

import os
import time
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime

load_dotenv('.env.paper')

class StopLossMonitor:
    """Monitor positions and execute stop losses"""

    def __init__(self, stop_loss_pct=0.20, check_interval=60):
        """
        Args:
            stop_loss_pct: Stop loss threshold (default 20% loss)
            check_interval: How often to check positions in seconds (default 60s)
        """
        self.api = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )

        self.stop_loss_pct = stop_loss_pct
        self.check_interval = check_interval
        self.closed_positions = []

        print("=" * 70)
        print("STOP LOSS MONITOR - ACTIVE")
        print("=" * 70)
        print(f"Stop Loss Threshold: {self.stop_loss_pct * 100}%")
        print(f"Check Interval: {self.check_interval}s")
        print("=" * 70)

    def check_positions(self):
        """Check all positions and execute stop losses if needed"""

        try:
            positions = self.api.get_all_positions()

            if not positions:
                return

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking {len(positions)} positions...")

            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc)  # Percentage P&L

                # Skip if position is profitable
                if unrealized_plpc >= 0:
                    continue

                # Check if loss exceeds stop loss threshold
                loss_pct = abs(unrealized_plpc)

                if loss_pct >= self.stop_loss_pct:
                    print(f"\n[STOP LOSS TRIGGERED] {symbol}")
                    print(f"  Position: {qty} @ ${entry_price:.2f}")
                    print(f"  Current Price: ${current_price:.2f}")
                    print(f"  Loss: ${unrealized_pl:.2f} ({unrealized_plpc * 100:.1f}%)")
                    print(f"  Threshold: {self.stop_loss_pct * 100}%")

                    # Execute stop loss
                    self.close_position(symbol, qty, unrealized_pl)

        except Exception as e:
            print(f"[ERROR] Position check failed: {e}")

    def close_position(self, symbol, qty, loss):
        """Close a position that hit stop loss"""

        try:
            # Determine side (if qty negative, buy to close; if positive, sell to close)
            abs_qty = abs(qty)

            if qty > 0:
                side = OrderSide.SELL
                action = "SELL"
            else:
                side = OrderSide.BUY
                action = "BUY"

            print(f"  [CLOSING] {action} {abs_qty} {symbol}...")

            # Submit market order to close
            order = self.api.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            )

            print(f"  [OK] STOP LOSS EXECUTED: Order ID {order.id}")
            print(f"  Realized Loss: ${loss:.2f}")

            # Track closed position
            self.closed_positions.append({
                'symbol': symbol,
                'qty': qty,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'order_id': order.id
            })

        except Exception as e:
            print(f"  [X] STOP LOSS FAILED: {e}")

    def run_continuous(self):
        """Run continuous stop loss monitoring"""

        print(f"\n[MONITORING] Stop loss monitoring active...")
        print(f"Press Ctrl+C to stop\n")

        try:
            while True:
                self.check_positions()
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n\n[STOPPED] Stop loss monitor stopped by user")
            self.print_summary()

    def print_summary(self):
        """Print summary of stop losses executed"""

        if self.closed_positions:
            print(f"\n{'='*70}")
            print("STOP LOSS SUMMARY")
            print(f"{'='*70}")
            print(f"Total positions closed: {len(self.closed_positions)}")

            total_loss = sum(pos['loss'] for pos in self.closed_positions)
            print(f"Total realized loss: ${total_loss:.2f}")

            print(f"\nClosed Positions:")
            for pos in self.closed_positions:
                print(f"  {pos['symbol']}: {pos['qty']} | Loss: ${pos['loss']:.2f}")

            print(f"{'='*70}")

def main():
    """Run stop loss monitor"""

    # Stop loss settings
    stop_loss_pct = 0.20  # 20% loss threshold
    check_interval = 60   # Check every 60 seconds

    monitor = StopLossMonitor(
        stop_loss_pct=stop_loss_pct,
        check_interval=check_interval
    )

    monitor.run_continuous()

if __name__ == "__main__":
    main()
