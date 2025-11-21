"""
DAILY DRAWDOWN TRACKER

Add this to E8_FOREX_BOT.py to prevent hitting daily DD limit.
This is the critical safety feature that was missing.
"""

import json
from pathlib import Path
from datetime import datetime

class DailyDDTracker:
    """
    Track daily profit/loss and block trading if daily DD limit exceeded.

    E8 has TWO drawdown rules:
    1. Trailing DD: 6% from peak balance (we had this)
    2. Daily DD: 2-4% max loss per day (WE DIDN'T HAVE THIS - cost us $600)

    This tracker prevents daily DD violations.
    """

    def __init__(self, daily_dd_limit=4000, warning_threshold=0.75):
        """
        Initialize daily DD tracker.

        Args:
            daily_dd_limit: Max loss allowed per day (default $4,000 = 2% of $200k)
            warning_threshold: Warn when reaching this % of limit (default 75%)
        """
        self.daily_dd_limit = daily_dd_limit
        self.warning_threshold = warning_threshold
        self.tracker_file = Path('BOTS/daily_pnl_tracker.json')

    def _load_tracker(self):
        """Load daily P/L tracker from file"""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load daily tracker: {e}")
                return {}
        return {}

    def _save_tracker(self, tracker):
        """Save daily P/L tracker to file"""
        try:
            # Ensure directory exists
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.tracker_file, 'w') as f:
                json.dump(tracker, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save daily tracker: {e}")

    def initialize_day(self, current_equity):
        """
        Initialize tracking for today if not already done.

        Call this at the start of each scan.
        """
        today = datetime.now().strftime('%Y-%m-%d')
        tracker = self._load_tracker()

        if today not in tracker:
            print(f"\n[DAILY DD] New trading day: {today}")
            print(f"[DAILY DD] Starting equity: ${current_equity:,.2f}")
            print(f"[DAILY DD] Daily loss limit: ${self.daily_dd_limit:,.2f}")

            tracker[today] = {
                'start_equity': current_equity,
                'current_loss': 0,
                'max_loss_seen': 0,
                'trades_today': 0,
                'violations': 0
            }

            self._save_tracker(tracker)

        return tracker[today]

    def check_daily_dd(self, current_equity):
        """
        Check if daily DD limit has been exceeded.

        Returns:
            (can_trade, status_message)
            can_trade: True if safe to trade, False if daily DD limit hit
            status_message: Human-readable status
        """
        today = datetime.now().strftime('%Y-%m-%d')
        tracker = self._load_tracker()

        # Initialize if needed
        if today not in tracker:
            self.initialize_day(current_equity)
            tracker = self._load_tracker()

        day_data = tracker[today]
        start_equity = day_data['start_equity']

        # Calculate today's P/L
        daily_pnl = current_equity - start_equity
        daily_loss = max(0, -daily_pnl)  # Only count losses

        # Update tracker
        day_data['current_loss'] = daily_loss
        day_data['max_loss_seen'] = max(day_data['max_loss_seen'], daily_loss)

        # Check if exceeded limit
        if daily_loss >= self.daily_dd_limit:
            day_data['violations'] += 1
            self._save_tracker(tracker)

            msg = f"[DAILY DD VIOLATION] Lost ${daily_loss:,.2f} today (limit: ${self.daily_dd_limit:,.2f})"
            return False, msg

        # Check if approaching limit (warning)
        warning_level = self.daily_dd_limit * self.warning_threshold
        if daily_loss >= warning_level:
            self._save_tracker(tracker)

            remaining = self.daily_dd_limit - daily_loss
            pct = (daily_loss / self.daily_dd_limit) * 100

            msg = f"[DAILY DD WARNING] Lost ${daily_loss:,.2f} today ({pct:.0f}% of limit, ${remaining:,.2f} remaining)"
            return True, msg

        # Safe zone
        self._save_tracker(tracker)

        remaining = self.daily_dd_limit - daily_loss
        pct = (daily_loss / self.daily_dd_limit) * 100 if daily_loss > 0 else 0

        if daily_pnl >= 0:
            msg = f"[DAILY DD] Profit today: ${daily_pnl:,.2f} (no DD risk)"
        else:
            msg = f"[DAILY DD] Loss today: ${daily_loss:,.2f} ({pct:.0f}% of limit, ${remaining:,.2f} remaining)"

        return True, msg

    def record_trade(self):
        """Record that a trade was placed today"""
        today = datetime.now().strftime('%Y-%m-%d')
        tracker = self._load_tracker()

        if today in tracker:
            tracker[today]['trades_today'] = tracker[today].get('trades_today', 0) + 1
            self._save_tracker(tracker)

    def get_today_summary(self):
        """Get summary of today's trading activity"""
        today = datetime.now().strftime('%Y-%m-%d')
        tracker = self._load_tracker()

        if today not in tracker:
            return None

        return tracker[today]

    def get_weekly_summary(self):
        """Get summary of last 7 days"""
        tracker = self._load_tracker()

        from datetime import timedelta
        today_date = datetime.now()

        weekly_data = []
        for i in range(7):
            date = (today_date - timedelta(days=i)).strftime('%Y-%m-%d')
            if date in tracker:
                weekly_data.append({
                    'date': date,
                    **tracker[date]
                })

        return weekly_data


# ==============================================================================
# HOW TO INTEGRATE INTO E8_FOREX_BOT.py
# ==============================================================================

"""
1. In __init__:

    # Daily DD tracker (CRITICAL SAFETY FEATURE)
    from daily_dd_tracker import DailyDDTracker
    self.daily_dd_tracker = DailyDDTracker(
        daily_dd_limit=4000,  # $4,000 = 2% of $200k (adjust based on E8's actual limit)
        warning_threshold=0.75  # Warn at 75% of limit
    )


2. In scan_forex(), BEFORE scanning for setups:

    def scan_forex(self):
        print(f"\n{'='*70}")
        print(f"SCANNING FOREX PAIRS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # Get account balance
        balance, challenge_status = self.get_account_balance()

        # CHECK DAILY DD FIRST (CRITICAL!)
        can_trade, dd_message = self.daily_dd_tracker.check_daily_dd(balance)
        print(f"\n{dd_message}")

        if not can_trade:
            print("\n[STOP] Daily DD limit exceeded - no trading until tomorrow")
            return  # <-- BLOCKS ALL TRADING FOR REST OF DAY

        # Rest of scanning logic...


3. After placing a trade:

    if order_id:
        print(f"[SUCCESS] Order placed: {order_id}")
        # Record trade in daily tracker
        self.daily_dd_tracker.record_trade()


4. At end of day (optional), add reporting:

    def print_daily_summary(self):
        summary = self.daily_dd_tracker.get_today_summary()
        if summary:
            print(f"\n{'='*70}")
            print(f"DAILY SUMMARY")
            print(f"{'='*70}")
            print(f"  Start Equity: ${summary['start_equity']:,.2f}")
            print(f"  Current Loss: ${summary['current_loss']:,.2f}")
            print(f"  Max Loss Today: ${summary['max_loss_seen']:,.2f}")
            print(f"  Trades Placed: {summary['trades_today']}")
            print(f"  DD Violations: {summary['violations']}")
            print(f"{'='*70}")
"""


# ==============================================================================
# STANDALONE TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAILY DD TRACKER - TESTING")
    print("=" * 70)

    # Create tracker
    tracker = DailyDDTracker(daily_dd_limit=4000)

    # Simulate trading day
    print("\n--- Morning: Start of Day ---")
    start_equity = 200000
    tracker.initialize_day(start_equity)

    print("\n--- 10 AM: First trade loses $1,000 ---")
    equity = 199000
    can_trade, msg = tracker.check_daily_dd(equity)
    print(msg)
    print(f"Can trade: {can_trade}")

    print("\n--- 11 AM: Second trade loses $2,000 more (total -$3,000) ---")
    equity = 197000
    can_trade, msg = tracker.check_daily_dd(equity)
    print(msg)
    print(f"Can trade: {can_trade}")

    print("\n--- 12 PM: Third trade loses $1,500 more (total -$4,500) ---")
    equity = 195500
    can_trade, msg = tracker.check_daily_dd(equity)
    print(msg)
    print(f"Can trade: {can_trade}")  # Should be FALSE - exceeded limit!

    print("\n--- Summary ---")
    summary = tracker.get_today_summary()
    print(f"Start: ${summary['start_equity']:,.2f}")
    print(f"Current Loss: ${summary['current_loss']:,.2f}")
    print(f"Max Loss: ${summary['max_loss_seen']:,.2f}")
    print(f"Violations: {summary['violations']}")

    print("\n" + "=" * 70)
    print("If this tracker had been in the bot, it would have saved $600")
    print("=" * 70)
