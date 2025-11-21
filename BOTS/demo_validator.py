"""
DEMO VALIDATION TRACKER

Track 60-day Match Trader demo performance to decide if strategy is worth $600.

Success Criteria:
1. ZERO daily DD violations (absolute requirement)
2. Positive ROI (any amount, even 5%)
3. Max trailing DD < 4%
4. Win rate > 55%

Decision Tree:
- ALL criteria pass → Pay $600 for E8 evaluation
- 1-2 violations → Adjust parameters, run another 30 days
- Multiple violations → DON'T pay $600, pivot to options
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List


class DemoValidator:
    """Track and analyze demo performance against success criteria"""

    def __init__(self, validation_file='BOTS/demo_validation_results.json'):
        self.validation_file = Path(validation_file)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create validation file if doesn't exist"""
        if not self.validation_file.exists():
            self.validation_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'start_balance': 200000,
                'target_days': 60,
                'current_balance': 200000,
                'peak_balance': 200000,
                'trades': [],
                'daily_results': {},
                'daily_dd_violations': 0,
                'max_trailing_dd_seen': 0,
                'success_criteria': {
                    'zero_daily_dd_violations': None,
                    'positive_roi': None,
                    'max_trailing_dd_under_4pct': None,
                    'win_rate_above_55pct': None
                },
                'final_verdict': None
            }

            with open(self.validation_file, 'w') as f:
                json.dump(data, f, indent=2)

    def load_data(self) -> Dict:
        """Load validation data"""
        with open(self.validation_file, 'r') as f:
            return json.load(f)

    def save_data(self, data: Dict):
        """Save validation data"""
        with open(self.validation_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_daily_result(self, date: str, equity: float, daily_pnl: float,
                           daily_dd_violation: bool, trades_count: int):
        """Record end-of-day results"""
        data = self.load_data()

        # Update daily results
        data['daily_results'][date] = {
            'equity': equity,
            'daily_pnl': daily_pnl,
            'daily_dd_violation': daily_dd_violation,
            'trades_count': trades_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Update running totals
        data['current_balance'] = equity

        if equity > data['peak_balance']:
            data['peak_balance'] = equity

        if daily_dd_violation:
            data['daily_dd_violations'] += 1

        # Calculate trailing DD
        trailing_dd = (data['peak_balance'] - equity) / data['peak_balance'] * 100
        data['max_trailing_dd_seen'] = max(data['max_trailing_dd_seen'], trailing_dd)

        self.save_data(data)

    def record_trade(self, pair: str, signal: str, entry: float, exit_price: float,
                    units: int, pnl: float, outcome: str):
        """Record trade result"""
        data = self.load_data()

        trade = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pair': pair,
            'signal': signal,
            'entry': entry,
            'exit': exit_price,
            'units': units,
            'pnl': pnl,
            'outcome': outcome  # 'win' or 'loss'
        }

        data['trades'].append(trade)
        self.save_data(data)

    def calculate_statistics(self) -> Dict:
        """Calculate performance statistics"""
        data = self.load_data()

        # Calculate days elapsed
        start = datetime.strptime(data['start_date'], '%Y-%m-%d')
        days_elapsed = (datetime.now() - start).days

        # Calculate ROI
        roi = ((data['current_balance'] - data['start_balance']) / data['start_balance']) * 100

        # Calculate win rate
        trades = data['trades']
        wins = len([t for t in trades if t['outcome'] == 'win'])
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # Get trading days
        trading_days = len(data['daily_results'])

        # Calculate average trades per week
        weeks_elapsed = days_elapsed / 7 if days_elapsed > 0 else 1
        trades_per_week = total_trades / weeks_elapsed

        return {
            'days_elapsed': days_elapsed,
            'trading_days': trading_days,
            'roi': roi,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'trades_per_week': trades_per_week,
            'daily_dd_violations': data['daily_dd_violations'],
            'max_trailing_dd': data['max_trailing_dd_seen'],
            'current_balance': data['current_balance'],
            'peak_balance': data['peak_balance']
        }

    def evaluate_success_criteria(self) -> Dict:
        """Evaluate against success criteria"""
        stats = self.calculate_statistics()
        data = self.load_data()

        # Check each criterion
        criteria = {
            'zero_daily_dd_violations': stats['daily_dd_violations'] == 0,
            'positive_roi': stats['roi'] > 0,
            'max_trailing_dd_under_4pct': stats['max_trailing_dd'] < 4.0,
            'win_rate_above_55pct': stats['win_rate'] > 55.0
        }

        # Update data
        data['success_criteria'] = criteria
        self.save_data(data)

        return criteria

    def get_verdict(self) -> str:
        """Get final verdict on whether to proceed with E8 evaluation"""
        stats = self.calculate_statistics()
        criteria = self.evaluate_success_criteria()

        # Check if 60 days completed
        if stats['days_elapsed'] < 60:
            return f"IN_PROGRESS (Day {stats['days_elapsed']}/60)"

        # All criteria must pass
        all_pass = all(criteria.values())

        if all_pass:
            return "PASS - Pay $600 for E8 evaluation with exact same settings"

        # Check how many violations
        violations = stats['daily_dd_violations']

        if violations <= 2:
            return "MARGINAL - Reduce to 1.5 lots, tighten filters (ADX 35), run another 30 days"

        return "FAIL - DON'T pay $600, pivot to options with $4k"

    def print_report(self):
        """Print comprehensive demo validation report"""
        stats = self.calculate_statistics()
        criteria = self.evaluate_success_criteria()
        verdict = self.get_verdict()

        print("=" * 70)
        print("DEMO VALIDATION REPORT")
        print("=" * 70)
        print(f"\n[PROGRESS]")
        print(f"  Days Elapsed: {stats['days_elapsed']}/60")
        print(f"  Trading Days: {stats['trading_days']}")
        print(f"  Target: {'REACHED' if stats['days_elapsed'] >= 60 else f'{60 - stats['days_elapsed']} days remaining'}")

        print(f"\n[PERFORMANCE]")
        print(f"  Starting Balance: $200,000")
        print(f"  Current Balance: ${stats['current_balance']:,.2f}")
        print(f"  Peak Balance: ${stats['peak_balance']:,.2f}")
        print(f"  ROI: {stats['roi']:+.2f}%")

        print(f"\n[TRADING ACTIVITY]")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Trades/Week: {stats['trades_per_week']:.1f}")

        print(f"\n[RISK METRICS]")
        print(f"  Daily DD Violations: {stats['daily_dd_violations']}")
        print(f"  Max Trailing DD: {stats['max_trailing_dd']:.2f}%")

        print(f"\n[SUCCESS CRITERIA]")
        for criterion, passed in criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {criterion.replace('_', ' ').title()}")

        print(f"\n[VERDICT]")
        print(f"  {verdict}")

        if verdict.startswith("PASS"):
            print(f"\n  → Next Step: Pay $600 for E8 evaluation")
            print(f"  → Estimated Pass Probability: 30-40%")
            print(f"  → Timeline: 90 days to profit target")
            print(f"  → Monthly Income: $600-1,200 (if pass)")
        elif verdict.startswith("MARGINAL"):
            print(f"\n  → Next Step: Adjust parameters and run 30 more days")
            print(f"  → Reduce max lots to 1.5")
            print(f"  → Increase ADX filter to 35")
            print(f"  → DO NOT pay $600 yet")
        else:
            print(f"\n  → Next Step: PIVOT TO OPTIONS")
            print(f"  → Your $4k can generate $1-2k/month faster")
            print(f"  → You just SAVED $600 by validating first")
            print(f"  → Options profit in 2 weeks vs 5 months for forex")

        print("=" * 70)

    def export_trade_log(self, output_file='demo_trade_log.csv'):
        """Export trade history to CSV"""
        data = self.load_data()
        trades = data['trades']

        if not trades:
            print("No trades to export")
            return

        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)

        print(f"Exported {len(trades)} trades to {output_file}")

    def get_weekly_summary(self) -> List[Dict]:
        """Get week-by-week performance summary"""
        data = self.load_data()
        daily_results = data['daily_results']

        if not daily_results:
            return []

        # Group by week
        weeks = {}
        for date_str, day_data in daily_results.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            week_num = date.isocalendar()[1]  # ISO week number
            year = date.year

            key = f"{year}-W{week_num:02d}"

            if key not in weeks:
                weeks[key] = {
                    'week': key,
                    'trades': 0,
                    'pnl': 0,
                    'violations': 0,
                    'days': []
                }

            weeks[key]['trades'] += day_data['trades_count']
            weeks[key]['pnl'] += day_data['daily_pnl']
            if day_data['daily_dd_violation']:
                weeks[key]['violations'] += 1
            weeks[key]['days'].append(date_str)

        return list(weeks.values())

    def print_weekly_summary(self):
        """Print week-by-week breakdown"""
        weekly = self.get_weekly_summary()

        if not weekly:
            print("No weekly data available")
            return

        print("\n" + "=" * 70)
        print("WEEKLY PERFORMANCE BREAKDOWN")
        print("=" * 70)
        print(f"{'Week':<15} {'Trades':<8} {'P/L':<12} {'DD Violations':<15}")
        print("-" * 70)

        for week in weekly:
            print(f"{week['week']:<15} {week['trades']:<8} ${week['pnl']:>10,.2f} {week['violations']:<15}")

        print("=" * 70)


# ==============================================================================
# STANDALONE USAGE
# ==============================================================================

if __name__ == "__main__":
    import sys

    validator = DemoValidator()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "report":
            validator.print_report()

        elif command == "weekly":
            validator.print_weekly_summary()

        elif command == "export":
            validator.export_trade_log()

        elif command == "stats":
            stats = validator.calculate_statistics()
            print(json.dumps(stats, indent=2))

        else:
            print(f"Unknown command: {command}")
            print("Available commands: report, weekly, export, stats")

    else:
        # Default: print full report
        validator.print_report()
        print("\n")
        validator.print_weekly_summary()
