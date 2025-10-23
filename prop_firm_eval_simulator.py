#!/usr/bin/env python3
"""
PROP FIRM EVALUATION SIMULATOR
================================
Simulates prop firm evaluation rules for your scanner

Typical prop firm requirements:
- Phase 1: 8% profit target, 5% max daily loss, 10% max total drawdown
- Phase 2: 5% profit target, same risk limits
- No overnight holds (some firms)
"""

import json
from datetime import datetime, timedelta

class PropFirmEvalSimulator:
    """Simulate prop firm evaluation rules"""

    def __init__(self,
                 starting_balance=50000,
                 phase=1,
                 firm='Apex'):

        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.peak_balance = starting_balance
        self.phase = phase
        self.firm = firm

        # Prop firm rules
        self.rules = {
            'Apex': {
                'phase1_target': 0.08,  # 8%
                'phase2_target': 0.05,  # 5%
                'max_daily_loss': 0.05,  # 5%
                'max_total_drawdown': 0.10,  # 10%
                'allows_overnight': True,
                'allows_automation': True
            },
            'TopstepTrader': {
                'phase1_target': 0.06,  # 6%
                'phase2_target': 0.04,  # 4%
                'max_daily_loss': 0.05,
                'max_total_drawdown': 0.06,  # Only 6%!
                'allows_overnight': False,
                'allows_automation': True
            },
            'FTMO': {
                'phase1_target': 0.08,
                'phase2_target': 0.05,
                'max_daily_loss': 0.05,
                'max_total_drawdown': 0.10,
                'allows_overnight': True,
                'allows_automation': False  # Case by case
            }
        }

        # Tracking
        self.daily_start_balance = starting_balance
        self.days_traded = 0
        self.passed = False
        self.failed = False
        self.failure_reason = None

        self.trade_log = []

    def new_trading_day(self):
        """Reset daily tracking"""
        self.daily_start_balance = self.current_balance
        self.days_traded += 1
        print(f"\n{'='*70}")
        print(f"DAY {self.days_traded} - Starting Balance: ${self.current_balance:,.2f}")
        print(f"{'='*70}")

    def record_trade(self, pnl, description=""):
        """Record a trade P&L"""
        self.current_balance += pnl

        # Update peak
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Log trade
        self.trade_log.append({
            'day': self.days_traded,
            'pnl': pnl,
            'balance': self.current_balance,
            'description': description,
            'timestamp': datetime.now().isoformat()
        })

        print(f"  Trade: {description}")
        print(f"  P&L: ${pnl:+,.2f}")
        print(f"  New Balance: ${self.current_balance:,.2f}")

        # Check rules
        self._check_rules()

    def _check_rules(self):
        """Check if any prop firm rules violated"""
        rules = self.rules[self.firm]

        # 1. Check daily loss limit
        daily_pnl = self.current_balance - self.daily_start_balance
        daily_loss_pct = daily_pnl / self.daily_start_balance

        if daily_loss_pct <= -rules['max_daily_loss']:
            self.failed = True
            self.failure_reason = f"Daily loss limit hit: {daily_loss_pct:.1%} (max {-rules['max_daily_loss']:.1%})"
            print(f"\n  ❌ [FAILED] {self.failure_reason}")
            return

        # 2. Check total drawdown
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance

        if drawdown >= rules['max_total_drawdown']:
            self.failed = True
            self.failure_reason = f"Total drawdown limit hit: {drawdown:.1%} (max {rules['max_total_drawdown']:.1%})"
            print(f"\n  ❌ [FAILED] {self.failure_reason}")
            return

        # 3. Check profit target
        total_profit = (self.current_balance - self.starting_balance) / self.starting_balance
        target = rules['phase1_target'] if self.phase == 1 else rules['phase2_target']

        if total_profit >= target:
            self.passed = True
            print(f"\n  ✅ [PASSED] Profit target hit: {total_profit:.1%} (target: {target:.1%})")
            return

        # 4. Show status
        print(f"  Daily P&L: {daily_loss_pct:+.1%} (limit: {-rules['max_daily_loss']:.1%})")
        print(f"  Total Profit: {total_profit:+.1%} (target: {target:.1%})")
        print(f"  Drawdown: {drawdown:.1%} (max: {rules['max_total_drawdown']:.1%})")

    def end_of_day_summary(self):
        """Print end of day summary"""
        rules = self.rules[self.firm]

        daily_pnl = self.current_balance - self.daily_start_balance
        total_profit = (self.current_balance - self.starting_balance) / self.starting_balance
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        target = rules['phase1_target'] if self.phase == 1 else rules['phase2_target']

        print(f"\n{'='*70}")
        print(f"END OF DAY {self.days_traded} SUMMARY")
        print(f"{'='*70}")
        print(f"Starting Balance: ${self.daily_start_balance:,.2f}")
        print(f"Ending Balance:   ${self.current_balance:,.2f}")
        print(f"Daily P&L:        ${daily_pnl:+,.2f} ({daily_pnl/self.daily_start_balance:+.1%})")
        print(f"\nProgress to Target:")
        print(f"  Current Profit: {total_profit:.1%}")
        print(f"  Target:         {target:.1%}")
        print(f"  Remaining:      {(target - total_profit):.1%}")
        print(f"\nRisk Metrics:")
        print(f"  Max Daily Loss: {daily_pnl/self.daily_start_balance:.1%} / {-rules['max_daily_loss']:.1%} limit")
        print(f"  Total Drawdown: {drawdown:.1%} / {rules['max_total_drawdown']:.1%} limit")
        print(f"{'='*70}\n")

    def get_final_report(self):
        """Generate final evaluation report"""
        rules = self.rules[self.firm]
        total_profit = (self.current_balance - self.starting_balance) / self.starting_balance
        target = rules['phase1_target'] if self.phase == 1 else rules['phase2_target']

        report = {
            'firm': self.firm,
            'phase': self.phase,
            'starting_balance': self.starting_balance,
            'ending_balance': self.current_balance,
            'total_profit': total_profit,
            'profit_target': target,
            'days_traded': self.days_traded,
            'passed': self.passed,
            'failed': self.failed,
            'failure_reason': self.failure_reason,
            'trades': len(self.trade_log)
        }

        return report


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PROP FIRM EVALUATION SIMULATOR")
    print("Testing your strategy against Apex Trader Funding Phase 1 rules")
    print("="*70)

    # Simulate Apex Phase 1 evaluation
    eval_sim = PropFirmEvalSimulator(
        starting_balance=50000,
        phase=1,
        firm='Apex'
    )

    # Simulate 10 days of trading with your expected results
    # (Replace this with actual trades from your scanner)

    # Day 1: Good day
    eval_sim.new_trading_day()
    eval_sim.record_trade(1250, "INTC Iron Condor - Win")
    eval_sim.record_trade(800, "TSLA Dual Options - Win")
    eval_sim.record_trade(600, "AAPL Iron Condor - Win")
    eval_sim.end_of_day_summary()

    # Day 2: Mixed
    eval_sim.new_trading_day()
    eval_sim.record_trade(-400, "NVDA Iron Condor - Loss")
    eval_sim.record_trade(900, "MSFT Dual Options - Win")
    eval_sim.record_trade(700, "AMD Iron Condor - Win")
    eval_sim.end_of_day_summary()

    # Day 3-5: Continue with wins (70% win rate)
    for day in range(3):
        eval_sim.new_trading_day()
        eval_sim.record_trade(1100, "Iron Condor - Win")
        eval_sim.record_trade(850, "Iron Condor - Win")
        eval_sim.record_trade(-300, "Iron Condor - Loss")
        eval_sim.end_of_day_summary()

    # Final report
    print("\n" + "="*70)
    print("FINAL EVALUATION REPORT")
    print("="*70)

    report = eval_sim.get_final_report()

    for key, value in report.items():
        if key == 'trades':
            continue
        if isinstance(value, float):
            if 'profit' in key or 'target' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value}")

    print("="*70)

    if report['passed']:
        print("\n✅ EVALUATION PASSED! Ready for funded account.")
        print(f"   Profit: {report['total_profit']:.1%} in {report['days_traded']} days")
        print(f"   Next: Phase 2 evaluation (5% target)")
    elif report['failed']:
        print(f"\n❌ EVALUATION FAILED: {report['failure_reason']}")
        print("   Need to retry evaluation")
    else:
        print(f"\n⏳ IN PROGRESS: {report['total_profit']:.1%} / {report['profit_target']:.1%}")
        print(f"   Need {(report['profit_target'] - report['total_profit']):.1%} more to pass")
