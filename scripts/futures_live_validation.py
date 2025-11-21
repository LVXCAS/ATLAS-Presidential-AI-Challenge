#!/usr/bin/env python3
"""
FUTURES LIVE VALIDATION
48-Hour Observation Mode - No Real Trading

Tests strategy on LIVE market data without executing trades.
Tracks signals for 48 hours to validate win rate before enabling real execution.

Usage:
    python futures_live_validation.py --duration 48  # Run 48-hour validation
    python futures_live_validation.py --duration 24  # Quick 24-hour validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanners.futures_scanner import AIEnhancedFuturesScanner
from data.futures_data_fetcher import FuturesDataFetcher, MICRO_FUTURES
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict
import argparse


class FuturesLiveValidator:
    """
    Validates futures strategy in LIVE observation mode

    Features:
    - Generates signals but does NOT execute
    - Tracks if signals would have won/lost
    - Calculates real-time win rate
    - After 48 hours + 60% WR → Recommends enabling execution
    """

    def __init__(self, observation_hours: int = 48, target_win_rate: float = 0.60):
        """
        Initialize live validator

        Args:
            observation_hours: Hours to observe (default 48)
            target_win_rate: Target win rate to validate (default 60%)
        """
        print("\n" + "="*70)
        print("FUTURES LIVE VALIDATION - OBSERVATION MODE")
        print("="*70)
        print(f"Duration: {observation_hours} hours")
        print(f"Target Win Rate: {target_win_rate:.0%}")
        print(f"Mode: SIGNAL TRACKING ONLY (No Real Trades)")
        print("="*70 + "\n")

        self.observation_hours = observation_hours
        self.target_win_rate = target_win_rate
        self.scanner = AIEnhancedFuturesScanner(paper_trading=True)
        self.data_fetcher = FuturesDataFetcher(paper_trading=True)

        self.signals_tracked = []
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=observation_hours)

        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        try:
            price = self.data_fetcher.get_current_price(symbol)
            return price if price else 0.0
        except:
            return 0.0

    def track_signal(self, opportunity: Dict):
        """Track a new signal"""

        signal = {
            'id': len(self.signals_tracked) + 1,
            'symbol': opportunity['symbol'],
            'direction': opportunity['direction'],
            'entry_price': opportunity['entry_price'],
            'entry_time': datetime.now(),
            'stop_loss': opportunity['stop_loss'],
            'take_profit': opportunity['take_profit'],
            'status': 'TRACKING',
            'score': opportunity['final_score'],
            'confidence': opportunity['confidence'],
            'risk_per_contract': opportunity['risk_per_contract'],
            'exit_price': None,
            'exit_time': None,
            'exit_reason': None
        }

        self.signals_tracked.append(signal)

        print(f"\n[SIGNAL #{signal['id']} TRACKED] {signal['symbol']} {signal['direction']}")
        print(f"  Entry: ${signal['entry_price']:.2f}")
        print(f"  Stop: ${signal['stop_loss']:.2f}")
        print(f"  Target: ${signal['take_profit']:.2f}")
        print(f"  Score: {signal['score']:.2f} | Confidence: {signal['confidence']:.0%}")
        print(f"  Risk/Contract: ${signal['risk_per_contract']:.2f}")

    def update_tracked_signals(self):
        """Update status of all tracked signals"""

        for signal in self.signals_tracked:
            if signal['status'] == 'TRACKING':
                current_price = self.get_current_price(signal['symbol'])

                if current_price <= 0:
                    continue

                # Check LONG positions
                if signal['direction'] == 'LONG':
                    # Hit stop loss
                    if current_price <= signal['stop_loss']:
                        signal['status'] = 'LOSS'
                        signal['exit_price'] = signal['stop_loss']
                        signal['exit_time'] = datetime.now()
                        signal['exit_reason'] = 'Stop Loss Hit'
                        self._log_signal_exit(signal)

                    # Hit take profit
                    elif current_price >= signal['take_profit']:
                        signal['status'] = 'WIN'
                        signal['exit_price'] = signal['take_profit']
                        signal['exit_time'] = datetime.now()
                        signal['exit_reason'] = 'Take Profit Hit'
                        self._log_signal_exit(signal)

                # Check SHORT positions
                elif signal['direction'] == 'SHORT':
                    # Hit stop loss
                    if current_price >= signal['stop_loss']:
                        signal['status'] = 'LOSS'
                        signal['exit_price'] = signal['stop_loss']
                        signal['exit_time'] = datetime.now()
                        signal['exit_reason'] = 'Stop Loss Hit'
                        self._log_signal_exit(signal)

                    # Hit take profit
                    elif current_price <= signal['take_profit']:
                        signal['status'] = 'WIN'
                        signal['exit_price'] = signal['take_profit']
                        signal['exit_time'] = datetime.now()
                        signal['exit_reason'] = 'Take Profit Hit'
                        self._log_signal_exit(signal)

    def _log_signal_exit(self, signal: Dict):
        """Log when a signal exits"""

        duration = (signal['exit_time'] - signal['entry_time']).total_seconds() / 3600
        pnl = (signal['exit_price'] - signal['entry_price']) if signal['direction'] == 'LONG' else (signal['entry_price'] - signal['exit_price'])
        pnl_pct = (pnl / signal['entry_price']) * 100

        status_emoji = "✓" if signal['status'] == 'WIN' else "✗"

        print(f"\n{status_emoji} [SIGNAL #{signal['id']} {signal['status']}] {signal['symbol']}")
        print(f"  Entry: ${signal['entry_price']:.2f} → Exit: ${signal['exit_price']:.2f}")
        print(f"  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        print(f"  Duration: {duration:.1f} hours")
        print(f"  Reason: {signal['exit_reason']}")

    def get_statistics(self) -> Dict:
        """Calculate current validation statistics"""

        completed = [s for s in self.signals_tracked if s['status'] in ['WIN', 'LOSS']]
        wins = [s for s in completed if s['status'] == 'WIN']
        losses = [s for s in completed if s['status'] == 'LOSS']
        tracking = [s for s in self.signals_tracked if s['status'] == 'TRACKING']

        win_rate = len(wins) / len(completed) if completed else 0.0

        # Calculate average P&L
        if completed:
            avg_win = sum([
                (s['exit_price'] - s['entry_price']) if s['direction'] == 'LONG'
                else (s['entry_price'] - s['exit_price'])
                for s in wins
            ]) / len(wins) if wins else 0.0

            avg_loss = sum([
                abs((s['exit_price'] - s['entry_price']) if s['direction'] == 'LONG'
                    else (s['entry_price'] - s['exit_price']))
                for s in losses
            ]) / len(losses) if losses else 0.0
        else:
            avg_win = 0.0
            avg_loss = 0.0

        time_elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        time_remaining = (self.end_time - datetime.now()).total_seconds() / 3600
        progress = min((time_elapsed / self.observation_hours) * 100, 100)

        return {
            'total_signals': len(self.signals_tracked),
            'completed': len(completed),
            'wins': len(wins),
            'losses': len(losses),
            'tracking': len(tracking),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'time_elapsed_hours': time_elapsed,
            'time_remaining_hours': max(time_remaining, 0),
            'progress_pct': progress
        }

    def display_status(self):
        """Display current validation status"""

        stats = self.get_statistics()

        print("\n" + "="*70)
        print("VALIDATION STATUS")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Progress: {stats['progress_pct']:.1f}% ({stats['time_elapsed_hours']:.1f}h / {self.observation_hours}h)")
        print(f"Remaining: {stats['time_remaining_hours']:.1f} hours")
        print()
        print(f"Signals Tracked: {stats['total_signals']}")
        print(f"  - Completed: {stats['completed']} (Wins: {stats['wins']}, Losses: {stats['losses']})")
        print(f"  - Still Tracking: {stats['tracking']}")
        print()

        if stats['completed'] > 0:
            print(f"Win Rate: {stats['win_rate']:.1%} (Target: {self.target_win_rate:.0%})")
            print(f"Average Win: ${stats['avg_win']:.2f}")
            print(f"Average Loss: ${stats['avg_loss']:.2f}")

            if stats['avg_loss'] > 0:
                print(f"Win/Loss Ratio: {stats['avg_win']/stats['avg_loss']:.2f}")
        else:
            print("Win Rate: N/A (No completed signals yet)")

        print("="*70)

    def run_validation(self):
        """Run complete validation cycle"""

        print("\nStarting 48-hour futures strategy observation...")
        print("Generating signals but NOT executing trades\n")

        scan_interval = 900  # 15 minutes
        status_interval = 3600  # 1 hour
        last_status_time = time.time()

        try:
            while datetime.now() < self.end_time:
                # Scan for new opportunities
                try:
                    opportunities = self.scanner.scan_all_futures()

                    # Track new signals
                    for opp in opportunities:
                        # Only track if we don't already have an active signal for this symbol
                        active_symbols = [s['symbol'] for s in self.signals_tracked if s['status'] == 'TRACKING']
                        if opp['symbol'] not in active_symbols:
                            self.track_signal(opp)

                except Exception as e:
                    print(f"[ERROR] Scan failed: {e}")

                # Update tracked signals
                self.update_tracked_signals()

                # Display status every hour
                if time.time() - last_status_time >= status_interval:
                    self.display_status()
                    self.save_checkpoint()
                    last_status_time = time.time()

                # Wait before next scan
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Validation stopped by user")

        # Final results
        self.display_final_results()

    def save_checkpoint(self):
        """Save current validation state"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'futures_validation_checkpoint_{timestamp}.json'

        data = {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'current_time': datetime.now().isoformat(),
            'observation_hours': self.observation_hours,
            'target_win_rate': self.target_win_rate,
            'signals_tracked': self.signals_tracked,
            'statistics': self.get_statistics()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\n[CHECKPOINT SAVED] {filename}")

    def display_final_results(self):
        """Display final validation results"""

        stats = self.get_statistics()

        print("\n\n" + "="*70)
        print("48-HOUR VALIDATION COMPLETE")
        print("="*70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {stats['time_elapsed_hours']:.1f} hours")
        print()
        print("FINAL RESULTS:")
        print(f"  Signals Tracked: {stats['total_signals']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Still Tracking: {stats['tracking']}")
        print()

        if stats['completed'] >= 10:
            print(f"WIN RATE: {stats['win_rate']:.1%}")
            print(f"Average Win: ${stats['avg_win']:.2f}")
            print(f"Average Loss: ${stats['avg_loss']:.2f}")

            if stats['avg_loss'] > 0:
                print(f"Win/Loss Ratio: {stats['avg_win']/stats['avg_loss']:.2f}")

            print()

            # Recommendation
            if stats['win_rate'] >= self.target_win_rate:
                print("✓ STRATEGY VALIDATED - Ready for live execution")
                print(f"  Win rate {stats['win_rate']:.1%} exceeds target {self.target_win_rate:.0%}")
                print("\nNEXT STEPS:")
                print("  1. Run: python start_futures_paper_trading.py")
                print("  2. Start with VERY small size (1 contract)")
                print("  3. Monitor closely for first week")
            else:
                print("✗ STRATEGY NEEDS IMPROVEMENT - Do not enable")
                print(f"  Win rate {stats['win_rate']:.1%} below target {self.target_win_rate:.0%}")
                print("\nNEXT STEPS:")
                print("  1. Review losing trades")
                print("  2. Adjust strategy parameters")
                print("  3. Run another 48-hour validation")
        else:
            print("⚠ INSUFFICIENT DATA - Need at least 10 completed signals")
            print(f"  Only {stats['completed']} signals completed")
            print("\nRECOMMENDATION: Extend validation period")

        print("="*70)

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'futures_validation_final_{timestamp}.json'

        data = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'observation_hours': self.observation_hours,
            'target_win_rate': self.target_win_rate,
            'signals_tracked': self.signals_tracked,
            'final_statistics': stats,
            'recommendation': 'APPROVED' if (stats['completed'] >= 10 and stats['win_rate'] >= self.target_win_rate) else 'NOT_APPROVED'
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\n[RESULTS SAVED] {filename}\n")


def main():
    """Run futures live validation"""

    parser = argparse.ArgumentParser(description='Futures Live Validation - 48-Hour Observation Mode')
    parser.add_argument('--duration', type=int, default=48, help='Observation duration in hours (default: 48)')
    parser.add_argument('--target-wr', type=float, default=0.60, help='Target win rate (default: 0.60)')
    parser.add_argument('--quick-test', action='store_true', help='Quick 1-hour test mode')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.duration = 1
        print("\n[QUICK TEST MODE] Running 1-hour validation for testing purposes\n")

    # Initialize validator
    validator = FuturesLiveValidator(
        observation_hours=args.duration,
        target_win_rate=args.target_wr
    )

    # Run validation
    validator.run_validation()


if __name__ == "__main__":
    main()
