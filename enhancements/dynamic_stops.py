#!/usr/bin/env python3
"""
Dynamic Stop Loss Management
Time-based and profit-based trailing stops
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class DynamicStopManager:
    """Manage dynamic stop losses"""

    def __init__(self):
        # Time-based stop tightening
        self.time_stops = {
            'DAY_1_3': -0.60,    # Days 1-3: -60% stop
            'DAY_4_7': -0.50,    # Days 4-7: -50% stop
            'DAY_8_14': -0.40,   # Days 8-14: -40% stop
            'DAY_15_PLUS': -0.35  # Days 15+: -35% stop
        }

        # Profit-based stop adjustments
        self.profit_stops = {
            'BREAKEVEN': 0.30,    # Move to breakeven at +30%
            'LOCK_PROFIT': 0.50,  # Lock in profit at +50%
            'TRAIL_START': 0.60   # Start trailing at +60%
        }

    def calculate_stop_loss(self, entry_price: float, entry_date: datetime,
                           current_price: float, current_pnl_pct: float) -> Dict:
        """
        Calculate dynamic stop loss based on time and profit

        Args:
            entry_price: Entry price of option
            entry_date: When position was entered
            current_price: Current option price
            current_pnl_pct: Current P&L as percentage (e.g., 0.35 for +35%)

        Returns:
            {
                'stop_price': float,
                'stop_pct': float,
                'stop_type': str,
                'reasoning': str
            }
        """
        days_held = (datetime.now() - entry_date).days

        # Determine time-based stop
        if days_held <= 3:
            time_stop_pct = self.time_stops['DAY_1_3']
        elif days_held <= 7:
            time_stop_pct = self.time_stops['DAY_4_7']
        elif days_held <= 14:
            time_stop_pct = self.time_stops['DAY_8_14']
        else:
            time_stop_pct = self.time_stops['DAY_15_PLUS']

        # Check profit-based stops
        if current_pnl_pct >= self.profit_stops['TRAIL_START']:
            # Trailing stop: 30% below current price
            stop_price = current_price * 0.70
            stop_pct = (stop_price - entry_price) / entry_price
            stop_type = 'TRAILING'
            reasoning = f"Trailing stop (30% below peak) after {current_pnl_pct:.0%} gain"

        elif current_pnl_pct >= self.profit_stops['LOCK_PROFIT']:
            # Lock in profit: stop at +30% from entry
            stop_price = entry_price * 1.30
            stop_pct = 0.30
            stop_type = 'PROFIT_LOCK'
            reasoning = f"Locking {stop_pct:.0%} profit after {current_pnl_pct:.0%} gain"

        elif current_pnl_pct >= self.profit_stops['BREAKEVEN']:
            # Move to breakeven
            stop_price = entry_price
            stop_pct = 0.0
            stop_type = 'BREAKEVEN'
            reasoning = f"Breakeven stop after {current_pnl_pct:.0%} gain"

        else:
            # Time-based stop
            stop_price = entry_price * (1 + time_stop_pct)
            stop_pct = time_stop_pct
            stop_type = 'TIME_BASED'
            reasoning = f"Day {days_held} time-based stop ({time_stop_pct:.0%})"

        return {
            'stop_price': float(stop_price),
            'stop_pct': float(stop_pct),
            'stop_type': stop_type,
            'reasoning': reasoning,
            'days_held': days_held
        }

    def should_exit(self, entry_price: float, entry_date: datetime,
                   current_price: float, peak_price: float = None) -> Dict:
        """
        Determine if position should be exited based on stop

        Args:
            entry_price: Entry price
            entry_date: Entry date
            current_price: Current price
            peak_price: Highest price reached (for trailing)

        Returns:
            {
                'exit': bool,
                'reason': str,
                'stop_hit': str
            }
        """
        # Calculate current P&L
        current_pnl_pct = (current_price - entry_price) / entry_price

        # Use peak price for trailing stop calculation
        if peak_price is None or peak_price < current_price:
            peak_price = current_price

        peak_pnl_pct = (peak_price - entry_price) / entry_price

        # Get stop level
        stop_info = self.calculate_stop_loss(
            entry_price=entry_price,
            entry_date=entry_date,
            current_price=peak_price,  # Use peak for stop calculation
            current_pnl_pct=peak_pnl_pct
        )

        # Check if stop hit
        if current_price <= stop_info['stop_price']:
            return {
                'exit': True,
                'reason': f"Stop hit: {stop_info['reasoning']}",
                'stop_hit': stop_info['stop_type'],
                'stop_price': stop_info['stop_price'],
                'current_pnl_pct': current_pnl_pct
            }

        return {
            'exit': False,
            'reason': f"Position OK - stop at ${stop_info['stop_price']:.2f} ({stop_info['stop_type']})",
            'stop_hit': None,
            'stop_price': stop_info['stop_price'],
            'current_pnl_pct': current_pnl_pct
        }

    def update_peak_price(self, current_price: float, peak_price: float = None) -> float:
        """Update peak price for trailing stop"""
        if peak_price is None:
            return current_price
        return max(peak_price, current_price)


# Global instance
_stop_manager = None

def get_dynamic_stop_manager() -> DynamicStopManager:
    """Get singleton stop manager"""
    global _stop_manager
    if _stop_manager is None:
        _stop_manager = DynamicStopManager()
    return _stop_manager


if __name__ == "__main__":
    # Test
    manager = DynamicStopManager()

    print("="*70)
    print("DYNAMIC STOP LOSS TEST")
    print("="*70)

    # Simulate a trade
    entry_price = 2.50
    entry_date = datetime.now() - timedelta(days=5)

    # Test scenarios
    scenarios = [
        (1.00, "Early loss"),
        (2.50, "Breakeven"),
        (3.25, "+30% gain"),
        (3.75, "+50% gain"),
        (4.50, "+80% gain - trailing"),
        (3.50, "After peak - trailing test")
    ]

    print(f"\nEntry: ${entry_price:.2f}, 5 days ago\n")

    peak_price = entry_price
    for current_price, label in scenarios:
        pnl_pct = (current_price - entry_price) / entry_price

        # Update peak
        peak_price = manager.update_peak_price(current_price, peak_price)

        # Calculate stop
        stop_info = manager.calculate_stop_loss(
            entry_price=entry_price,
            entry_date=entry_date,
            current_price=peak_price,
            current_pnl_pct=(peak_price - entry_price) / entry_price
        )

        # Check exit
        exit_check = manager.should_exit(
            entry_price=entry_price,
            entry_date=entry_date,
            current_price=current_price,
            peak_price=peak_price
        )

        print(f"{label}: ${current_price:.2f} ({pnl_pct:+.0%})")
        print(f"  Stop: ${stop_info['stop_price']:.2f} ({stop_info['stop_type']})")
        print(f"  Exit: {exit_check['exit']} - {exit_check['reason']}")
        print(f"  Peak: ${peak_price:.2f}\n")
