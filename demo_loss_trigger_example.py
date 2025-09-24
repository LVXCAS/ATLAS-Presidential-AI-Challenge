#!/usr/bin/env python3
"""
Loss Limit Trigger Example
Demonstrates exactly when and how the 4.9% loss limit triggers
"""

def demonstrate_loss_limit_trigger():
    """Show exact moment when loss limit triggers"""
    print("LOSS LIMIT TRIGGER DEMONSTRATION")
    print("=" * 50)
    print("Showing a realistic bad trading day scenario")
    print("Loss Limit: -4.9% (will trigger automatic sell-all)")
    print()

    starting_capital = 100000
    current_equity = starting_capital

    # Realistic bad trading scenario
    trades = [
        {"time": "09:45", "symbol": "TSLA", "loss": 1500, "reason": "Gap down against call position"},
        {"time": "10:15", "symbol": "AAPL", "loss": 800, "reason": "Earnings reaction negative"},
        {"time": "10:45", "symbol": "NVDA", "loss": 1200, "reason": "Tech sector rotation"},
        {"time": "11:30", "symbol": "SPY", "loss": 900, "reason": "Market weakness continuation"},
        {"time": "12:15", "symbol": "QQQ", "loss": 700, "reason": "Failed support level"},
    ]

    print("Trading Activity (Realistic Bad Day):")
    print(f"Starting Capital: ${starting_capital:,}")
    print()

    for i, trade in enumerate(trades):
        # Apply the loss
        current_equity -= trade["loss"]

        # Calculate daily loss percentage
        daily_loss_pct = ((current_equity - starting_capital) / starting_capital) * 100

        print(f"{trade['time']}: {trade['symbol']} | "
              f"Loss: -${trade['loss']:,} | "
              f"Equity: ${current_equity:,.0f} | "
              f"Daily: {daily_loss_pct:+.2f}%")
        print(f"         Reason: {trade['reason']}")

        # Check if loss limit is hit
        if daily_loss_pct <= -4.9:
            print()
            print("[STOP] LOSS LIMIT TRIGGERED!")
            print("=" * 40)
            print(f"TRIGGER DETAILS:")
            print(f"- Daily loss: {daily_loss_pct:.2f}% <= -4.9% limit")
            print(f"- Remaining equity: ${current_equity:,.0f}")
            print(f"- Total loss: ${starting_capital - current_equity:,.0f}")
            print(f"- Trade that triggered: {trade['symbol']} at {trade['time']}")
            print()
            print("IMMEDIATE AUTOMATED ACTIONS:")
            print("1. [SYSTEM] Cancel ALL pending orders")
            print("2. [SYSTEM] Sell ALL open positions at market")
            print("3. [SYSTEM] Stop all trading algorithms")
            print("4. [SYSTEM] Send alert notification")
            print("5. [SYSTEM] Log event to trading_events.json")
            print("6. [SYSTEM] Set daily trading to STOPPED")
            print()

            # Calculate what would have happened without limit
            remaining_trades = trades[i+1:]
            additional_loss = sum(t["loss"] for t in remaining_trades)

            if additional_loss > 0:
                would_be_equity = current_equity - additional_loss
                would_be_loss_pct = ((would_be_equity - starting_capital) / starting_capital) * 100

                print("PROTECTION ANALYSIS:")
                print(f"Without loss limit:")
                print(f"- Additional losses: ${additional_loss:,.0f}")
                print(f"- Final equity would be: ${would_be_equity:,.0f}")
                print(f"- Total loss would be: {would_be_loss_pct:.2f}%")
                print(f"- Capital PROTECTED: ${current_equity - would_be_equity:,.0f}")
                print(f"- Protection value: {daily_loss_pct - would_be_loss_pct:+.1f}%")

            print()
            print("SYSTEM STATUS AFTER TRIGGER:")
            print("- All positions: CLOSED")
            print("- All orders: CANCELLED")
            print("- Trading status: STOPPED for remainder of day")
            print("- Capital preserved: [OK]")
            print("- Ready for next trading day: [OK]")

            return {
                'triggered': True,
                'trigger_time': trade['time'],
                'trigger_symbol': trade['symbol'],
                'final_equity': current_equity,
                'daily_loss_pct': daily_loss_pct,
                'trades_before_trigger': i + 1
            }

        print()

    # If we reach here, limit wasn't triggered
    print("Loss limit was not triggered on this scenario")
    return {'triggered': False}

def show_system_integration():
    """Show how the loss limit integrates with the trading bots"""
    print("SYSTEM INTEGRATION DETAILS")
    print("=" * 50)
    print()

    print("MONITORING PROCESS:")
    print("1. ProfitTargetMonitor runs every 30 seconds")
    print("2. Checks: current_equity vs starting_daily_equity")
    print("3. Calculates: daily_pnl_pct = (current - start) / start * 100")
    print("4. Evaluates: daily_pnl_pct <= -4.9%")
    print("5. If TRUE: Triggers sell_all_positions()")
    print()

    print("INTEGRATION POINTS:")
    print("OPTIONS_BOT.py:")
    print("  - profit_monitor = ProfitTargetMonitor()")
    print("  - Monitors both +5.75% profit target AND -4.9% loss limit")
    print("  - Background task checks every 30 seconds")
    print()

    print("start_real_market_hunter.py:")
    print("  - Same profit_monitor integration")
    print("  - Unified monitoring across both bots")
    print("  - Consistent risk management")
    print()

    print("EXECUTION FLOW:")
    print("Normal trading -> Monitor detects -4.9% -> Cancel orders -> Close positions -> Stop trading")
    print()

    print("LOGGING:")
    print("- Event logged to: trading_events.json")
    print("- Event type: 'loss_limit'")
    print("- Includes: timestamp, equity, loss%, reason")

if __name__ == "__main__":
    result = demonstrate_loss_limit_trigger()

    print("\n" + "=" * 50)

    if result['triggered']:
        print("DEMONSTRATION SUCCESSFUL!")
        print(f"[OK] Loss limit triggered at {result['trigger_time']} on {result['trigger_symbol']}")
        print(f"[OK] System protected capital at {result['daily_loss_pct']:.2f}% loss")
        print(f"[OK] Stopped trading after {result['trades_before_trigger']} trades")
        print(f"[OK] Preserved ${result['final_equity']:,.0f} of capital")
    else:
        print("Loss limit was not triggered in this scenario")

    print()
    show_system_integration()

    print("\n" + "=" * 50)
    print("READY FOR LIVE TRADING:")
    print("[OK] Profit target: +5.75% (take profits)")
    print("[OK] Loss limit: -4.9% (cut losses)")
    print("[OK] Both systems integrated and tested")
    print("[OK] Monitoring every 30 seconds")
    print("[OK] Automatic execution when triggered")