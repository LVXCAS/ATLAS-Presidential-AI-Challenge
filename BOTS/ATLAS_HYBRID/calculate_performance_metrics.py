#!/usr/bin/env python3
"""
Calculate comprehensive performance metrics for ATLAS trading system
- Sharpe Ratio
- Win Rate
- Profit Factor
- Max Drawdown
- Projected ROI (weekly, monthly, annual)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

def load_all_trades():
    """Load all trade logs from session files"""
    logs_dir = Path("logs/trades")
    all_trades = []

    if not logs_dir.exists():
        print("No trade logs found")
        return []

    for session_file in logs_dir.glob("session_*.json"):
        try:
            with open(session_file) as f:
                session = json.load(f)
                trades = session.get("trades", [])
                all_trades.extend(trades)
        except Exception as e:
            print(f"Error loading {session_file.name}: {e}")

    return all_trades

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe Ratio

    Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns

    > 1.0 = Good
    > 2.0 = Very Good
    > 3.0 = Excellent
    """
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return

    # Annualize (assuming 252 trading days)
    sharpe_annual = sharpe * np.sqrt(252)

    return sharpe_annual

def calculate_metrics(trades):
    """Calculate all performance metrics"""

    if not trades:
        return None

    # Separate winning and losing trades
    closed_trades = [t for t in trades if t.get("status") == "closed"]

    if not closed_trades:
        print("No closed trades yet")
        return None

    # Calculate PnL for each trade
    pnls = []
    wins = []
    losses = []

    for trade in closed_trades:
        pnl = trade.get("realized_pnl", 0)
        pnls.append(pnl)

        if pnl > 0:
            wins.append(pnl)
        else:
            losses.append(pnl)

    # Basic metrics
    total_trades = len(closed_trades)
    winning_trades = len(wins)
    losing_trades = len(losses)

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_profit = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0
    net_profit = sum(pnls)

    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0

    # Profit factor
    profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')

    # Sharpe ratio
    returns = np.array(pnls)
    sharpe = calculate_sharpe_ratio(returns)

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Calculate time span
    timestamps = [datetime.fromisoformat(t.get("entry_time", "")) for t in closed_trades if t.get("entry_time")]

    if len(timestamps) >= 2:
        time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
        days_trading = time_span / 24
    else:
        days_trading = 1

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "total_loss": total_loss,
        "net_profit": net_profit,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "days_trading": days_trading,
        "pnls": pnls
    }

def project_roi(metrics, starting_balance=183000):
    """Project ROI for different timeframes"""

    if not metrics or metrics["days_trading"] == 0:
        return None

    # Daily average return
    daily_return = metrics["net_profit"] / metrics["days_trading"]
    daily_roi = (daily_return / starting_balance) * 100

    # Projections (assuming consistent performance)
    weekly_profit = daily_return * 7
    weekly_roi = (weekly_profit / starting_balance) * 100

    monthly_profit = daily_return * 30
    monthly_roi = (monthly_profit / starting_balance) * 100

    annual_profit = daily_return * 365
    annual_roi = (annual_profit / starting_balance) * 100

    # Conservative estimate (70% of observed performance)
    conservative_multiplier = 0.7

    return {
        "daily": {
            "profit": daily_return,
            "roi": daily_roi,
            "conservative_roi": daily_roi * conservative_multiplier
        },
        "weekly": {
            "profit": weekly_profit,
            "roi": weekly_roi,
            "conservative_roi": weekly_roi * conservative_multiplier
        },
        "monthly": {
            "profit": monthly_profit,
            "roi": monthly_roi,
            "conservative_roi": monthly_roi * conservative_multiplier
        },
        "annual": {
            "profit": annual_profit,
            "roi": annual_roi,
            "conservative_roi": annual_roi * conservative_multiplier
        }
    }

def main():
    print("=" * 70)
    print("ATLAS PERFORMANCE ANALYSIS")
    print("=" * 70)

    trades = load_all_trades()

    if not trades:
        print("\nNo trades found. System may still be in observation mode.")
        return

    print(f"\nTotal trades in logs: {len(trades)}")

    metrics = calculate_metrics(trades)

    if not metrics:
        print("Insufficient data for analysis")
        return

    print("\n" + "=" * 70)
    print("TRADE STATISTICS")
    print("=" * 70)

    print(f"\nTotal Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")

    print(f"\nTotal Profit: ${metrics['total_profit']:,.2f}")
    print(f"Total Loss: ${metrics['total_loss']:,.2f}")
    print(f"Net Profit: ${metrics['net_profit']:,.2f}")

    print(f"\nAverage Win: ${metrics['avg_win']:,.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:,.2f}")
    print(f"Risk/Reward: {metrics['avg_win'] / metrics['avg_loss']:.2f}x" if metrics['avg_loss'] > 0 else "N/A")

    print("\n" + "=" * 70)
    print("RISK METRICS")
    print("=" * 70)

    print(f"\nProfit Factor: {metrics['profit_factor']:.2f}")
    if metrics['profit_factor'] > 2.0:
        print("  [EXCELLENT] > 2.0")
    elif metrics['profit_factor'] > 1.5:
        print("  [GOOD] > 1.5")
    elif metrics['profit_factor'] > 1.0:
        print("  [PROFITABLE] > 1.0")
    else:
        print("  [NEEDS IMPROVEMENT] < 1.0")

    print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    if metrics['sharpe_ratio'] > 3.0:
        print("  [EXCELLENT] > 3.0 (institutional grade)")
    elif metrics['sharpe_ratio'] > 2.0:
        print("  [VERY GOOD] > 2.0")
    elif metrics['sharpe_ratio'] > 1.0:
        print("  [GOOD] > 1.0")
    else:
        print("  [NEEDS IMPROVEMENT] < 1.0")

    print(f"\nMax Drawdown: ${metrics['max_drawdown']:,.2f}")
    print(f"Days Trading: {metrics['days_trading']:.1f}")

    print("\n" + "=" * 70)
    print("PROJECTED ROI")
    print("=" * 70)

    projections = project_roi(metrics)

    if projections:
        print("\n[OPTIMISTIC - Based on Observed Performance]")
        print(f"Daily:   ${projections['daily']['profit']:,.2f} ({projections['daily']['roi']:,.2f}%)")
        print(f"Weekly:  ${projections['weekly']['profit']:,.2f} ({projections['weekly']['roi']:,.2f}%)")
        print(f"Monthly: ${projections['monthly']['profit']:,.2f} ({projections['monthly']['roi']:,.2f}%)")
        print(f"Annual:  ${projections['annual']['profit']:,.2f} ({projections['annual']['roi']:,.2f}%)")

        print("\n[CONSERVATIVE - 70% of Observed Performance]")
        print(f"Daily:   ({projections['daily']['conservative_roi']:,.2f}%)")
        print(f"Weekly:  ({projections['weekly']['conservative_roi']:,.2f}%)")
        print(f"Monthly: ({projections['monthly']['conservative_roi']:,.2f}%)")
        print(f"Annual:  ({projections['annual']['conservative_roi']:,.2f}%)")

        # Compound growth projection
        print("\n" + "=" * 70)
        print("COMPOUND GROWTH PROJECTION (Monthly Reinvestment)")
        print("=" * 70)

        starting_capital = 183000
        monthly_roi = projections['monthly']['conservative_roi'] / 100

        print(f"\nStarting Capital: ${starting_capital:,.2f}")
        print(f"Monthly ROI (Conservative): {monthly_roi * 100:.2f}%")

        balance = starting_capital
        print("\nMonth | Balance")
        print("-" * 40)
        for month in [1, 3, 6, 12]:
            balance_projected = starting_capital * ((1 + monthly_roi) ** month)
            print(f"{month:5d} | ${balance_projected:,.2f}")

    print("\n" + "=" * 70)
    print("RECENT TRADES (Last 5)")
    print("=" * 70)

    closed_trades = [t for t in trades if t.get("status") == "closed"]
    for trade in closed_trades[-5:]:
        pnl = trade.get("realized_pnl", 0)
        result = "WIN" if pnl > 0 else "LOSS"
        print(f"\n{trade.get('instrument')} {trade.get('type').upper()} - {result}")
        print(f"  Entry: {trade.get('entry_price')} @ {trade.get('entry_time', 'N/A')[:16]}")
        print(f"  Exit:  {trade.get('exit_price')} @ {trade.get('exit_time', 'N/A')[:16]}")
        print(f"  P/L: ${pnl:,.2f}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
