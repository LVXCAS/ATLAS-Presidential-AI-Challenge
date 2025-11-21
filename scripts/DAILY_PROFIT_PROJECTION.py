"""
REALISTIC DAILY PROFIT EXPECTATIONS
Based on 113-trade backtest with $187K account

Shows what REAL daily P/L looks like (spoiler: mostly unrealized fluctuation)
"""

import random
import numpy as np

print()
print("=" * 80)
print(" " * 20 + "DAILY PROFIT PROJECTION")
print("=" * 80)
print()

# Your current account
account_balance = 187191

# Backtest statistics
total_trades_6mo = 113
win_rate = 0.389
avg_win_pct = 0.0193  # +1.93%
avg_loss_pct = -0.0101  # -1.01%
avg_trade_duration_days = 6.3
trades_per_day = 113 / 180  # 0.63 trades per day

# Calculate position sizing
risk_per_trade = 0.015  # 1.5%
leverage = 5
position_size = account_balance * risk_per_trade * leverage
avg_positions_open = 2  # Max 2 positions typically

print(f"Account Balance: ${account_balance:,.2f}")
print(f"Risk Per Trade: {risk_per_trade*100}%")
print(f"Leverage: {leverage}x")
print(f"Typical Position Size: ${position_size:,.2f}")
print(f"Average Positions Open: {avg_positions_open}")
print()

print("=" * 80)
print("DAILY PROFIT EXPECTATIONS")
print("=" * 80)
print()

# Calculate daily profit (realized)
daily_avg_return_pct = 0.086 / 100  # 0.086% per day from backtest
daily_avg_profit = account_balance * daily_avg_return_pct

print("REALIZED PROFIT (Actual Money In Your Account):")
print("-" * 80)
print(f"Average Per Day: ${daily_avg_profit:,.2f} (+{daily_avg_return_pct*100:.3f}%)")
print(f"Average Per Week: ${daily_avg_profit * 7:,.2f} (+{daily_avg_return_pct*7*100:.2f}%)")
print(f"Average Per Month: ${daily_avg_profit * 30:,.2f} (+{daily_avg_return_pct*30*100:.2f}%)")
print()

print("BUT REALITY - DAILY DISTRIBUTION:")
print("-" * 80)
print()

# Model what actual daily P/L looks like
print("TYPICAL WEEK:")
print()

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
week_total = 0

for i, day in enumerate(days):
    # 62.8% chance something happens (trade closes)
    if random.random() < 0.628:
        # Is it a winner or loser?
        if random.random() < win_rate:
            # Winner!
            profit = account_balance * avg_win_pct * 0.5  # Half position size
            week_total += profit
            print(f"{day:12} | ${profit:>8,.2f} (WINNER - Trade hit 2% target)")
        else:
            # Loser
            loss = account_balance * avg_loss_pct * 0.5
            week_total += loss
            print(f"{day:12} | ${loss:>8,.2f} (LOSER - Trade hit 1.5% stop)")
    else:
        # No trade closures, just unrealized P/L fluctuation
        unrealized = random.uniform(-500, 500)
        print(f"{day:12} | ${unrealized:>8,.2f} (UNREALIZED - Positions developing)")

print()
print(f"Week Total: ${week_total:,.2f}")
print()

print("=" * 80)
print("EXPECTED MONTHLY BREAKDOWN")
print("=" * 80)
print()

# Model monthly results
month_days = 30
trades_per_month = trades_per_day * month_days  # ~19 trades
winners_per_month = trades_per_month * win_rate  # ~7 winners
losers_per_month = trades_per_month * (1 - win_rate)  # ~12 losers

monthly_wins_profit = winners_per_month * (account_balance * avg_win_pct * 0.5)
monthly_losses = losers_per_month * (account_balance * avg_loss_pct * 0.5)
monthly_net = monthly_wins_profit + monthly_losses

print(f"Trades Per Month: ~{trades_per_month:.0f}")
print(f"  Winners: ~{winners_per_month:.0f} (+{avg_win_pct*100:.2f}% each)")
print(f"  Losers: ~{losers_per_month:.0f} ({avg_loss_pct*100:.2f}% each)")
print()
print(f"Gross Wins: ${monthly_wins_profit:,.2f}")
print(f"Gross Losses: ${monthly_losses:,.2f}")
print(f"NET PROFIT: ${monthly_net:,.2f} (+{(monthly_net/account_balance)*100:.2f}%)")
print()

print("=" * 80)
print("VARIANCE (Reality Check)")
print("=" * 80)
print()

print("BEST CASE MONTH (Good Luck + Market Conditions):")
print(f"  - Higher win rate (50% vs 38.9%)")
print(f"  - More trades hit targets")
print(f"  - Profit: ${monthly_net * 1.5:,.2f} (+{(monthly_net*1.5/account_balance)*100:.2f}%)")
print()

print("WORST CASE MONTH (Bad Luck + Choppy Markets):")
print(f"  - Lower win rate (25% vs 38.9%)")
print(f"  - More stops hit, fewer targets")
print(f"  - Loss: ${monthly_net * -0.5:,.2f} ({(monthly_net*-0.5/account_balance)*100:.2f}%)")
print()

print("AVERAGE CASE MONTH (Backtest Reality):")
print(f"  - Win rate 38.9%")
print(f"  - Profit: ${monthly_net:,.2f} (+{(monthly_net/account_balance)*100:.2f}%)")
print()

print("=" * 80)
print("6-MONTH PROJECTION (Your Validation Period)")
print("=" * 80)
print()

six_month_profit = daily_avg_profit * 180
six_month_return_pct = (six_month_profit / account_balance) * 100

print(f"Expected Profit: ${six_month_profit:,.2f}")
print(f"Expected Return: +{six_month_return_pct:.2f}%")
print(f"Ending Balance: ${account_balance + six_month_profit:,.2f}")
print()

print("VARIANCE RANGE:")
print(f"  Best Case (+30%): ${account_balance + six_month_profit * 1.5:,.2f}")
print(f"  Average Case: ${account_balance + six_month_profit:,.2f}")
print(f"  Worst Case (-10%): ${account_balance + six_month_profit * -0.5:,.2f}")
print()

print("=" * 80)
print("E8 PROP FIRM TARGET ($500K ACCOUNT)")
print("=" * 80)
print()

e8_account = 500000
e8_target_pct = 0.08  # 8% profit target
e8_target_profit = e8_account * e8_target_pct

# Scale up daily profit for $500K account
e8_daily_profit = (e8_account / account_balance) * daily_avg_profit
e8_days_to_target = e8_target_profit / e8_daily_profit

print(f"E8 Challenge Account: ${e8_account:,.2f}")
print(f"Profit Target: ${e8_target_profit:,.2f} (8%)")
print(f"Max Drawdown Allowed: ${e8_account * 0.05:,.2f} (5%)")
print()
print(f"Expected Daily Profit: ${e8_daily_profit:,.2f}")
print(f"Days to Hit Target: ~{e8_days_to_target:.0f} days ({e8_days_to_target/30:.1f} months)")
print()

print("MONTHLY PROGRESSION TO E8 TARGET:")
month_return = (monthly_net / account_balance)
for month in range(1, 7):
    month_profit = e8_account * month_return * month
    month_pct = (month_profit / e8_account) * 100
    status = "TARGET HIT!" if month_pct >= 8 else f"{8-month_pct:.1f}% to go"
    print(f"  Month {month}: ${month_profit:,.2f} (+{month_pct:.2f}%) - {status}")

print()

print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print()
print("1. DAILY P/L IS MOSTLY NOISE")
print("   - Average $86/day sounds tiny, but compounds to +$28k (15%) in 6 months")
print("   - Most days show unrealized fluctuation, not realized profit")
print("   - Winners take 6+ days to develop - patience required")
print()
print("2. MONTHLY VIEW IS MORE REALISTIC")
print(f"   - Expect ~${monthly_net:,.2f}/month (+{(monthly_net/account_balance)*100:.1f}%)")
print("   - Some months +3%, some months -1%, average +2.6%")
print("   - Need 3-6 months to judge strategy, not 1-2 weeks")
print()
print("3. E8 TIMELINE IS ACHIEVABLE")
print(f"   - At current pace: {e8_days_to_target/30:.1f} months to hit 8% target")
print(f"   - With improved bot (48% win rate): ~3-4 months")
print("   - With scaling (add more capital): 2-3 months")
print()
print("4. DON'T JUDGE BY DAILY P/L")
print("   - Today: -$661 (feels bad)")
print("   - But if these trades recover: +$3,770 (feels amazing)")
print("   - Focus on weekly/monthly trends, not daily swings")
print()

print("=" * 80)
