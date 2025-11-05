"""
Monte Carlo Simulation for Forex Trading Strategy
Runs 10,000 simulations to understand probability of outcomes
"""

import json
import numpy as np
import random
from collections import defaultdict

print("\n" + "="*90)
print(" " * 25 + "MONTE CARLO SIMULATION - FOREX STRATEGY")
print("="*90)

# Load backtest data
with open('enhanced_backtest_results.json', 'r') as f:
    backtest = json.load(f)

# Extract trade outcomes
trades = backtest['trades']
win_pnls = [t['pnl_pct'] for t in trades if t['exit_reason'] == 'TAKE_PROFIT']
loss_pnls = [t['pnl_pct'] for t in trades if t['exit_reason'] == 'STOP_LOSS']

win_rate = len(win_pnls) / (len(win_pnls) + len(loss_pnls))
avg_win = np.mean(win_pnls) if win_pnls else 0
avg_loss = np.mean(loss_pnls) if loss_pnls else 0

print(f"\nBASELINE STATISTICS (from backtest):")
print(f"  Win Rate: {win_rate*100:.2f}%")
print(f"  Average Win: {avg_win*100:.3f}%")
print(f"  Average Loss: {avg_loss*100:.3f}%")
print(f"  Profit Factor: {abs(avg_win / avg_loss):.3f}")
print(f"  Sample Size: {len(win_pnls)} wins, {len(loss_pnls)} losses")

# Monte Carlo parameters
num_simulations = 10000
trades_per_sim = 100  # Simulate 100 trades (about 4-6 months)
starting_balance = 191640

print(f"\n" + "="*90)
print(f"RUNNING {num_simulations:,} MONTE CARLO SIMULATIONS")
print(f"  Simulating: {trades_per_sim} trades per simulation")
print(f"  Starting Balance: ${starting_balance:,.2f}")
print("="*90)

# Run simulations
results = []
max_drawdowns = []
bankruptcy_count = 0

for sim in range(num_simulations):
    balance = starting_balance
    peak_balance = starting_balance
    max_dd = 0

    for trade_num in range(trades_per_sim):
        # Random trade outcome based on win rate
        if random.random() < win_rate:
            # Win - sample from actual winning trades
            pnl_pct = random.choice(win_pnls)
        else:
            # Loss - sample from actual losing trades
            pnl_pct = random.choice(loss_pnls)

        # Apply P/L to balance
        pnl_dollars = balance * pnl_pct
        balance += pnl_dollars

        # Track drawdown
        if balance > peak_balance:
            peak_balance = balance

        current_dd = ((peak_balance - balance) / peak_balance) * 100
        if current_dd > max_dd:
            max_dd = current_dd

        # Check for bankruptcy
        if balance <= 0:
            bankruptcy_count += 1
            break

    results.append(balance)
    max_drawdowns.append(max_dd)

# Analyze results
results = np.array(results)
max_drawdowns = np.array(max_drawdowns)

total_returns = ((results - starting_balance) / starting_balance) * 100

# Calculate percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
result_percentiles = np.percentile(total_returns, percentiles)

print("\n" + "="*90)
print("SIMULATION RESULTS - ACCOUNT BALANCE AFTER 100 TRADES")
print("="*90)

print(f"\nPercentile    Return        Final Balance")
print("-" * 50)
for i, p in enumerate(percentiles):
    final_balance = starting_balance * (1 + result_percentiles[i]/100)
    print(f"{p:>3}th         {result_percentiles[i]:>+7.2f}%      ${final_balance:>13,.2f}")

print("\n" + "-" * 50)
print(f"Best Case:    {np.max(total_returns):>+7.2f}%      ${np.max(results):>13,.2f}")
print(f"Worst Case:   {np.min(total_returns):>+7.2f}%      ${np.min(results):>13,.2f}")
print(f"Average:      {np.mean(total_returns):>+7.2f}%      ${np.mean(results):>13,.2f}")

# Probability analysis
prob_profit = (np.sum(results > starting_balance) / num_simulations) * 100
prob_10pct = (np.sum(total_returns > 10) / num_simulations) * 100
prob_20pct = (np.sum(total_returns > 20) / num_simulations) * 100
prob_loss = (np.sum(results < starting_balance) / num_simulations) * 100
prob_10pct_loss = (np.sum(total_returns < -10) / num_simulations) * 100

print("\n" + "="*90)
print("PROBABILITY ANALYSIS")
print("="*90)

print(f"\nProfitability after 100 trades:")
print(f"  Probability of ANY profit:       {prob_profit:.1f}%")
print(f"  Probability of >10% return:      {prob_10pct:.1f}%")
print(f"  Probability of >20% return:      {prob_20pct:.1f}%")
print(f"\nDownside risk:")
print(f"  Probability of ANY loss:         {prob_loss:.1f}%")
print(f"  Probability of >10% loss:        {prob_10pct_loss:.1f}%")
print(f"  Probability of bankruptcy:       {bankruptcy_count/num_simulations*100:.2f}%")

# Drawdown analysis
dd_percentiles = np.percentile(max_drawdowns, percentiles)

print("\n" + "="*90)
print("MAXIMUM DRAWDOWN ANALYSIS")
print("="*90)

print(f"\nPercentile    Max Drawdown")
print("-" * 30)
for i, p in enumerate(percentiles):
    print(f"{p:>3}th         {dd_percentiles[i]:>6.2f}%")

print(f"\nAverage Max Drawdown:  {np.mean(max_drawdowns):.2f}%")
print(f"Worst Drawdown:        {np.max(max_drawdowns):.2f}%")

# E8 Challenge Analysis
print("\n" + "="*90)
print("E8 PROP FIRM CHALLENGE SIMULATION ($250K account, need 8% profit)")
print("="*90)

e8_starting = 250000
e8_target = e8_starting * 0.08  # $20,000 profit needed
e8_max_loss = e8_starting * 0.08  # 8% max drawdown (typical prop firm rule)

e8_results = []
e8_pass_count = 0
e8_fail_dd_count = 0
e8_fail_target_count = 0

for sim in range(num_simulations):
    balance = e8_starting
    peak_balance = e8_starting
    passed = False
    failed_dd = False

    for trade_num in range(200):  # Max 200 trades
        # Random trade outcome
        if random.random() < win_rate:
            pnl_pct = random.choice(win_pnls)
        else:
            pnl_pct = random.choice(loss_pnls)

        pnl_dollars = balance * pnl_pct
        balance += pnl_dollars

        # Update peak
        if balance > peak_balance:
            peak_balance = balance

        # Check drawdown failure
        current_dd = peak_balance - balance
        if current_dd >= e8_max_loss:
            failed_dd = True
            e8_fail_dd_count += 1
            break

        # Check if target hit
        profit = balance - e8_starting
        if profit >= e8_target:
            passed = True
            e8_pass_count += 1
            e8_results.append(trade_num + 1)  # Number of trades to pass
            break

    if not passed and not failed_dd:
        e8_fail_target_count += 1

e8_pass_rate = (e8_pass_count / num_simulations) * 100
e8_avg_trades = np.mean(e8_results) if e8_results else 0

print(f"\nChallenge Rules:")
print(f"  Starting Capital: ${e8_starting:,}")
print(f"  Profit Target: ${e8_target:,} (8%)")
print(f"  Max Drawdown: ${e8_max_loss:,} (8%)")
print(f"  Max Trades Simulated: 200")

print(f"\nSimulation Results ({num_simulations:,} trials):")
print(f"  Pass Rate: {e8_pass_rate:.1f}%")
print(f"  Fail (hit drawdown): {e8_fail_dd_count/num_simulations*100:.1f}%")
print(f"  Fail (didn't reach target): {e8_fail_target_count/num_simulations*100:.1f}%")

if e8_results:
    print(f"\nFor successful challenges:")
    print(f"  Average trades to pass: {e8_avg_trades:.0f} trades")
    print(f"  Fastest pass: {np.min(e8_results):.0f} trades")
    print(f"  Slowest pass: {np.max(e8_results):.0f} trades")

    # Estimate time to pass
    trades_per_day = 1  # Conservative estimate (1 trade per day)
    avg_days = e8_avg_trades / trades_per_day
    print(f"\nEstimated time to pass: {avg_days:.0f} days (~{avg_days/30:.1f} months)")

# Investment Analysis
print("\n" + "="*90)
print("INVESTMENT SCENARIO ANALYSIS")
print("="*90)

challenge_cost = 1227
payout_pct = 0.80  # 80% profit split

print(f"\nScenario: Buy 1x $250K E8 Challenge (${challenge_cost})")
print(f"  Pass Rate: {e8_pass_rate:.1f}%")
print(f"  Target Payout: ${e8_target * payout_pct:,.0f} (80% of ${e8_target:,})")
print(f"  Expected Value: ${e8_pass_rate/100 * e8_target * payout_pct - challenge_cost:,.0f}")

if e8_pass_rate > 0:
    expected_attempts = 100 / e8_pass_rate
    total_cost = challenge_cost * expected_attempts
    net_profit = (e8_target * payout_pct) - total_cost

    print(f"\nExpected attempts to pass: {expected_attempts:.1f}")
    print(f"Total cost (if buying multiple attempts): ${total_cost:,.0f}")
    print(f"Net profit after passing: ${net_profit:,.0f}")

    if net_profit > 0:
        roi = (net_profit / total_cost) * 100
        print(f"ROI: {roi:.1f}%")

# Recommendations
print("\n" + "="*90)
print("RECOMMENDATIONS BASED ON MONTE CARLO ANALYSIS")
print("="*90)

print("\n1. STRATEGY VIABILITY:")
if prob_profit >= 75:
    print(f"   [EXCELLENT] {prob_profit:.0f}% probability of profit after 100 trades")
elif prob_profit >= 60:
    print(f"   [GOOD] {prob_profit:.0f}% probability of profit after 100 trades")
else:
    print(f"   [RISKY] Only {prob_profit:.0f}% probability of profit after 100 trades")

print("\n2. E8 CHALLENGE READINESS:")
if e8_pass_rate >= 60:
    print(f"   [READY] {e8_pass_rate:.0f}% pass rate is excellent")
    print(f"   ACTION: Buy challenge immediately, expect to pass in ~{avg_days:.0f} days")
elif e8_pass_rate >= 40:
    print(f"   [VIABLE] {e8_pass_rate:.0f}% pass rate is acceptable")
    print(f"   ACTION: Consider buying 2-3 challenge attempts")
else:
    print(f"   [NOT READY] Only {e8_pass_rate:.0f}% pass rate")
    print(f"   ACTION: Deploy IMPROVED_FOREX_BOT.py first to improve win rate")

print("\n3. RISK MANAGEMENT:")
print(f"   Average Max Drawdown: {np.mean(max_drawdowns):.1f}%")
print(f"   95th Percentile DD: {dd_percentiles[7]:.1f}%")
if np.mean(max_drawdowns) < 10:
    print("   [GOOD] Drawdowns are manageable")
else:
    print("   [WARNING] Large drawdowns possible - prepare mentally")

print("\n4. EXPECTED OUTCOMES (100 trades):")
print(f"   Best realistic (95th %ile): +{result_percentiles[7]:.1f}%")
print(f"   Most likely (50th %ile): +{result_percentiles[4]:.1f}%")
print(f"   Worst realistic (5th %ile): {result_percentiles[1]:+.1f}%")

print("\n" + "="*90 + "\n")
