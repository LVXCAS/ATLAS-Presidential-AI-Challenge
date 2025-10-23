#!/usr/bin/env python3
"""
ENHANCED STOP LOSS MONITOR
Automatically close positions that hit stop loss thresholds
Supports stocks AND options spreads
"""
import os
import time
import sys
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from datetime import datetime

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.telegram_notifier import get_notifier
    from utils.trade_database import get_database
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False
    print("[WARNING] Integrations not available - running standalone")

load_dotenv()


class EnhancedStopLossMonitor:
    """Monitor all positions and execute stop losses"""

    def __init__(self, stop_loss_pct: float = 0.20, check_interval: int = 60):
        """
        Args:
            stop_loss_pct: Stop loss threshold (default 20% loss)
            check_interval: Check frequency in seconds (default 60s)
        """
        self.api = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )

        self.stop_loss_pct = stop_loss_pct
        self.check_interval = check_interval
        self.closed_positions = []

        # Get integrations
        self.notifier = get_notifier() if INTEGRATIONS_AVAILABLE else None
        self.database = get_database() if INTEGRATIONS_AVAILABLE else None

        print("=" * 70)
        print("ENHANCED STOP LOSS MONITOR - ACTIVE")
        print("=" * 70)
        print(f"Stop Loss Threshold: {self.stop_loss_pct * 100}%")
        print(f"Check Interval: {self.check_interval}s")
        print(f"Telegram Alerts: {'Enabled' if self.notifier and self.notifier.enabled else 'Disabled'}")
        print(f"Database Logging: {'Enabled' if self.database else 'Disabled'}")
        print("=" * 70)
        print()

    def check_positions(self):
        """Check all positions and execute stop losses if needed"""
        try:
            positions = self.api.get_all_positions()

            if not positions:
                return

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking {len(positions)} positions...")

            for position in positions:
                symbol = position.symbol
                qty = float(position.qty)
                current_price = float(position.current_price)
                avg_entry = float(position.avg_entry_price)
                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc)

                # Calculate loss percentage
                loss_pct = abs(unrealized_plpc) if unrealized_plpc < 0 else 0

                # Check if stop loss hit
                if loss_pct >= self.stop_loss_pct:
                    print(f"\n🛑 STOP LOSS HIT: {symbol}")
                    print(f"   Entry: ${avg_entry:.2f}")
                    print(f"   Current: ${current_price:.2f}")
                    print(f"   Loss: ${abs(unrealized_pl):.2f} ({loss_pct * 100:.1f}%)")

                    # Close position
                    success = self._close_position(symbol, unrealized_pl, loss_pct)

                    if success:
                        # Send alert
                        if self.notifier and self.notifier.enabled:
                            self.notifier.stop_loss_hit(
                                symbol=symbol,
                                loss=unrealized_pl,
                                reason=f"{loss_pct * 100:.1f}% loss threshold"
                            )

                        # Log to database
                        if self.database:
                            self.database.log_system_event(
                                event_type='STOP_LOSS',
                                component='STOP_LOSS_MONITOR',
                                message=f"Closed {symbol} at {loss_pct * 100:.1f}% loss",
                                severity='WARNING'
                            )

                        self.closed_positions.append({
                            'symbol': symbol,
                            'loss': unrealized_pl,
                            'loss_pct': loss_pct,
                            'timestamp': datetime.now()
                        })

                # Warn on large losses approaching threshold
                elif 0.15 <= loss_pct < self.stop_loss_pct:
                    print(f"⚠️  WARNING: {symbol} at -{loss_pct * 100:.1f}% (approaching stop loss)")

                    # Send warning once per position
                    if self.notifier and self.notifier.enabled:
                        # Check if we haven't warned recently
                        self.notifier.large_loss_warning(
                            symbol=symbol,
                            current_loss=unrealized_pl,
                            threshold=self.stop_loss_pct * 100
                        )

        except Exception as e:
            print(f"[ERROR] Check positions failed: {e}")
            if self.notifier and self.notifier.enabled:
                self.notifier.system_error(
                    component='STOP_LOSS_MONITOR',
                    error=str(e)
                )

    def _close_position(self, symbol: str, loss: float, loss_pct: float) -> bool:
        """Close a position"""
        try:
            # Attempt to close via API
            close_request = ClosePositionRequest()
            self.api.close_position(symbol)

            print(f"   ✓ Position closed: {symbol}")
            return True

        except Exception as e:
            print(f"   ✗ Failed to close {symbol}: {e}")

            # Try market order as fallback
            try:
                position = self.api.get_open_position(symbol)
                qty = abs(float(position.qty))
                side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY

                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )

                self.api.submit_order(order_data)
                print(f"   ✓ Market order submitted: {symbol}")
                return True

            except Exception as e2:
                print(f"   ✗ Market order also failed: {e2}")
                return False

    def run(self):
        """Run continuous monitoring loop"""
        print(f"\n[START] Stop loss monitoring active")
        print(f"Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while True:
                iteration += 1
                print(f"\n{'='*70}")
                print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")

                self.check_positions()

                # Show summary
                if self.closed_positions:
                    print(f"\n[SUMMARY] Closed {len(self.closed_positions)} positions today:")
                    for pos in self.closed_positions:
                        print(f"  - {pos['symbol']}: ${pos['loss']:.2f} ({pos['loss_pct']*100:.1f}%)")

                print(f"\n[WAITING] Next check in {self.check_interval}s...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] Stop loss monitor stopped by user")
            print(f"Total positions closed: {len(self.closed_positions)}")


if __name__ == "__main__":
    # Allow custom thresholds
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Stop Loss Monitor')
    parser.add_argument('--threshold', type=float, default=0.20,
                       help='Stop loss threshold (default: 0.20 = 20%%)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds (default: 60)')

    args = parser.parse_args()

    monitor = EnhancedStopLossMonitor(
        stop_loss_pct=args.threshold,
        check_interval=args.interval
    )

    monitor.run()
