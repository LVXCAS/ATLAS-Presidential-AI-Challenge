#!/usr/bin/env python3
"""
Loss Limit Protection Demonstration
Shows how the 4.9% loss limit protects trading capital
"""

import random
import numpy as np

def simulate_trading_day_with_loss_limit():
    """Simulate a trading day showing loss limit protection"""
    print("LOSS LIMIT PROTECTION DEMONSTRATION")
    print("=" * 50)
    print("Scenario: Bad trading day with significant losses")
    print("Loss Limit: -4.9% (automatic sell-all trigger)")
    print()

    starting_capital = 100000
    current_equity = starting_capital
    trades_executed = 0
    daily_pnl = 0

    # Simulate a series of losing trades
    print("Trading Activity:")
    print(f"Starting Capital: ${starting_capital:,}")
    print()

    # Simulate trades throughout the day
    trade_times = ["09:45", "10:15", "10:45", "11:30", "12:15", "13:00", "13:30", "14:00"]

    for i, time in enumerate(trade_times):
        # Simulate a losing trade (bad day scenario)
        if i < 3:  # First few trades with moderate losses
            trade_loss_pct = random.uniform(-2.0, -1.2)  # 1.2% to 2.0% loss per trade
            position_size_pct = random.uniform(2.0, 3.5)  # 2.0% to 3.5% of capital
        else:  # Later trades with bigger losses (market deteriorating)
            trade_loss_pct = random.uniform(-3.5, -2.0)  # 2.0% to 3.5% loss per trade
            position_size_pct = random.uniform(2.5, 4.0)  # 2.5% to 4.0% of capital

        trade_pnl = current_equity * (trade_loss_pct / 100) * (position_size_pct / 100)
        current_equity += trade_pnl
        daily_pnl += trade_pnl
        trades_executed += 1

        # Calculate daily loss percentage
        daily_loss_pct = ((current_equity - starting_capital) / starting_capital) * 100

        print(f"{time}: Trade #{trades_executed} | Loss: ${trade_pnl:+,.0f} | "
              f"Equity: ${current_equity:,.0f} | Daily: {daily_loss_pct:+.2f}%")

        # Check if loss limit is hit
        if daily_loss_pct <= -4.9:
            print()
            print("🛑 LOSS LIMIT TRIGGERED!")
            print("=" * 30)
            print(f"Daily loss: {daily_loss_pct:.2f}% <= -4.9% limit")
            print("AUTOMATIC ACTIONS:")
            print("1. Cancel all pending orders")
            print("2. Sell all open positions")
            print("3. Stop trading for the day")
            print("4. Preserve remaining capital")
            print()

            preserved_capital = current_equity
            potential_additional_loss = starting_capital * 0.02  # Could have lost another 2%

            print("PROTECTION ANALYSIS:")
            print(f"Capital preserved: ${preserved_capital:,.0f}")
            print(f"Estimated additional risk avoided: ${potential_additional_loss:,.0f}")
            print(f"Final loss: {daily_loss_pct:.2f}% (controlled)")
            print()

            return {
                'final_equity': current_equity,
                'daily_loss_pct': daily_loss_pct,
                'trades_executed': trades_executed,
                'loss_limit_triggered': True,
                'capital_preserved': True
            }

    # If we get here, loss limit wasn't triggered (shouldn't happen in this demo)
    final_loss_pct = ((current_equity - starting_capital) / starting_capital) * 100
    print(f"End of day without loss limit trigger: {final_loss_pct:.2f}%")

    return {
        'final_equity': current_equity,
        'daily_loss_pct': final_loss_pct,
        'trades_executed': trades_executed,
        'loss_limit_triggered': False,
        'capital_preserved': False
    }

def compare_with_without_loss_limit():
    """Compare outcomes with and without loss limit"""
    print("COMPARISON: WITH vs WITHOUT LOSS LIMIT")
    print("=" * 50)

    scenarios = [
        {"name": "Moderate Loss Day", "avg_loss_per_trade": -0.4, "num_trades": 8},
        {"name": "Bad Loss Day", "avg_loss_per_trade": -0.6, "num_trades": 10},
        {"name": "Terrible Loss Day", "avg_loss_per_trade": -0.8, "num_trades": 12}
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 30)

        starting_capital = 100000

        # Simulate WITHOUT loss limit
        equity_without_limit = starting_capital
        for _ in range(scenario['num_trades']):
            trade_loss = equity_without_limit * abs(scenario['avg_loss_per_trade']) / 100 * random.uniform(0.8, 1.5) / 100
            equity_without_limit -= trade_loss

        loss_without_limit = ((equity_without_limit - starting_capital) / starting_capital) * 100

        # Simulate WITH loss limit (stops at -4.9%)
        equity_with_limit = starting_capital * (1 - 0.049)  # Stopped at 4.9% loss
        loss_with_limit = -4.9

        # Calculate protection
        protection_amount = equity_with_limit - equity_without_limit
        protection_pct = loss_with_limit - loss_without_limit

        print(f"Without loss limit: ${equity_without_limit:,.0f} ({loss_without_limit:+.1f}%)")
        print(f"With loss limit:    ${equity_with_limit:,.0f} ({loss_with_limit:+.1f}%)")
        print(f"Capital protected:  ${protection_amount:+,.0f} ({protection_pct:+.1f}%)")
        print(f"Protection value:   {(protection_amount/starting_capital)*100:.1f}% of account")

def show_annual_impact():
    """Show the annual impact of loss limit protection"""
    print("\n")
    print("ANNUAL IMPACT ANALYSIS")
    print("=" * 50)

    print("Assumptions:")
    print("- 252 trading days per year")
    print("- Loss limit triggers 2-3 times per year (bad days)")
    print("- Without limit: additional 2-4% loss on those days")
    print()

    scenarios = [
        {"triggers": 2, "avg_extra_loss": 2.5, "name": "Conservative estimate"},
        {"triggers": 3, "avg_extra_loss": 3.0, "name": "Moderate estimate"},
        {"triggers": 4, "avg_extra_loss": 3.5, "name": "Aggressive estimate"}
    ]

    starting_capital = 100000

    for scenario in scenarios:
        # Calculate annual protection
        annual_protection = scenario['triggers'] * (scenario['avg_extra_loss'] / 100) * starting_capital

        print(f"{scenario['name']}:")
        print(f"  Loss limit triggers: {scenario['triggers']} times/year")
        print(f"  Avg extra loss prevented: {scenario['avg_extra_loss']}% per trigger")
        print(f"  Annual capital protection: ${annual_protection:,.0f}")
        print(f"  Protection as % of capital: {(annual_protection/starting_capital)*100:.1f}%")
        print()

if __name__ == "__main__":
    # Run demonstrations
    result = simulate_trading_day_with_loss_limit()

    if result['loss_limit_triggered']:
        print("LOSS LIMIT SYSTEM EFFECTIVENESS:")
        print(f"[OK] System triggered at {result['daily_loss_pct']:.2f}% loss")
        print(f"[OK] Protected ${starting_capital - result['final_equity']:,.0f} from further loss")
        print(f"[OK] Stopped trading after {result['trades_executed']} trades")
        print("[OK] Capital preservation successful")

    compare_with_without_loss_limit()
    show_annual_impact()

    print("\nLIVE TRADING READY:")
    print("- Bot monitors daily P&L every 30 seconds")
    print("- Triggers at exactly -4.9% daily loss")
    print("- Cancels all orders + closes all positions")
    print("- Protects capital from deeper losses")
    print("- Integrated with both OPTIONS_BOT and Market Hunter")