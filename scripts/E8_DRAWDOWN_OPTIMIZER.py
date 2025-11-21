"""
E8 Drawdown Limit Optimizer
Determines optimal drawdown limit for maximum pass rate and ROI
"""

import numpy as np
import json
from collections import defaultdict

print("\n" + "="*90)
print(" " * 25 + "E8 DRAWDOWN LIMIT OPTIMIZATION")
print("="*90)

# Load backtest data
with open('enhanced_backtest_results.json', 'r') as f:
    backtest = json.load(f)

trades = backtest['trades']
win_pnls = [t['pnl_pct'] for t in trades if t['exit_reason'] == 'TAKE_PROFIT']
loss_pnls = [t['pnl_pct'] for t in trades if t['exit_reason'] == 'STOP_LOSS']
win_rate = len(win_pnls) / (len(win_pnls) + len(loss_pnls))
avg_win = np.mean(win_pnls)
avg_loss = np.mean(loss_pnls)

print("\nBASELINE STRATEGY:")
print(f"  Win Rate:       {win_rate*100:.2f}%")
print(f"  Average Win:    {avg_win*100:.3f}%")
print(f"  Average Loss:   {avg_loss*100:.3f}%")
print(f"  Profit Factor:  {abs(avg_win / avg_loss):.3f}")

# E8 Challenge options
account_size = 500000
profit_target_pct = 0.08  # 8% = $40,000 (fixed)
profit_target_dollars = account_size * profit_target_pct
challenge_cost = 1627
profit_split = 0.80

# Available drawdown limits
drawdown_options = [0.05, 0.06, 0.08, 0.10, 0.12]  # 5%, 6%, 8%, 10%, 12%

print("\n" + "="*90)
print("E8 CHALLENGE OPTIONS")
print("="*90)

print(f"\nFixed Parameters:")
print(f"  Account Size:       ${account_size:,}")
print(f"  Profit Target:      8% (${profit_target_dollars:,})")
print(f"  Challenge Cost:     ${challenge_cost}")
print(f"  Profit Split:       {profit_split*100:.0f}% to you")
print(f"  Target Payout:      ${profit_target_dollars * profit_split:,.0f}")

print(f"\n" + "-"*90)
print("DRAWDOWN LIMIT OPTIONS:")
print("-"*90)
print(f"{'Limit':<10} {'Dollar Amount':<20} {'Common Name':<30}")
print("-"*90)
for dd_pct in drawdown_options:
    dd_dollars = account_size * dd_pct
    if dd_pct == 0.05:
        name = "Ultra Conservative"
    elif dd_pct == 0.06:
        name = "Conservative"
    elif dd_pct == 0.08:
        name = "Standard (Most Common)"
    elif dd_pct == 0.10:
        name = "Aggressive"
    else:
        name = "Very Aggressive"
    print(f"{dd_pct*100:<9.0f}%  ${dd_dollars:<18,.0f}  {name:<30}")

# Monte Carlo simulation for each drawdown limit
print("\n" + "="*90)
print("MONTE CARLO SIMULATION (5,000 runs per drawdown limit)")
print("="*90)

def simulate_challenge_with_dd(max_dd_pct, risk_pct=0.015, num_sims=5000):
    """Run Monte Carlo for E8 challenge with specific DD limit"""
    max_dd_dollars = account_size * max_dd_pct
    pass_count = 0
    fail_dd_count = 0
    trades_to_pass = []
    max_dds_reached = []

    for sim in range(num_sims):
        balance = account_size
        peak_balance = account_size
        max_dd_reached = 0

        for trade_num in range(300):  # Increased to 300 max trades
            # Win or loss
            if np.random.random() < win_rate:
                pnl_pct = np.random.choice(win_pnls)
            else:
                pnl_pct = np.random.choice(loss_pnls)

            # Scale P/L by risk level
            scaled_pnl = pnl_pct * (risk_pct / 0.01)
            pnl_dollars = balance * scaled_pnl
            balance += pnl_dollars

            # Update peak
            if balance > peak_balance:
                peak_balance = balance

            # Track max DD reached
            current_dd_dollars = peak_balance - balance
            current_dd_pct = (current_dd_dollars / account_size) * 100
            if current_dd_pct > max_dd_reached:
                max_dd_reached = current_dd_pct

            # Check drawdown failure
            if current_dd_dollars >= max_dd_dollars:
                fail_dd_count += 1
                max_dds_reached.append(max_dd_reached)
                break

            # Check if target hit
            profit = balance - account_size
            if profit >= profit_target_dollars:
                pass_count += 1
                trades_to_pass.append(trade_num + 1)
                max_dds_reached.append(max_dd_reached)
                break

    pass_rate = pass_count / num_sims
    fail_rate = fail_dd_count / num_sims
    avg_trades = np.mean(trades_to_pass) if trades_to_pass else 0
    avg_max_dd = np.mean(max_dds_reached) if max_dds_reached else 0

    return {
        'pass_rate': pass_rate,
        'fail_rate': fail_rate,
        'avg_trades': avg_trades,
        'avg_max_dd': avg_max_dd,
        'pass_count': pass_count,
        'fail_count': fail_dd_count
    }

print("\nRunning simulations with 1.5% risk per trade...")
print()
print(f"{'DD Limit':<12} {'Pass Rate':<12} {'Fail Rate':<12} {'Avg Trades':<12} {'Avg Max DD':<12} {'Days':<10}")
print("-"*90)

results = {}
for dd_pct in drawdown_options:
    result = simulate_challenge_with_dd(dd_pct, risk_pct=0.015)
    results[dd_pct] = result

    days = result['avg_trades'] * 1  # ~1 trade per day

    print(f"{dd_pct*100:<11.0f}%  {result['pass_rate']*100:>6.1f}%      {result['fail_rate']*100:>6.1f}%      {result['avg_trades']:>6.0f}        {result['avg_max_dd']:>6.2f}%      {days:>6.0f}")

# Calculate expected value for each option
print("\n" + "="*90)
print("EXPECTED VALUE ANALYSIS")
print("="*90)

print(f"\n{'DD Limit':<12} {'Pass Rate':<12} {'Exp Payout':<15} {'Exp Cost':<15} {'Net EV':<15} {'ROI':<10}")
print("-"*90)

best_ev = 0
best_dd = 0

for dd_pct in drawdown_options:
    result = results[dd_pct]
    pass_rate = result['pass_rate']

    # Expected payout
    expected_payout = profit_target_dollars * profit_split * pass_rate

    # Expected cost (accounting for multiple attempts)
    expected_attempts = 1 / pass_rate if pass_rate > 0 else 0
    expected_cost = challenge_cost * expected_attempts

    # Net expected value
    net_ev = expected_payout - expected_cost

    # ROI
    roi = (net_ev / expected_cost) * 100 if expected_cost > 0 else 0

    marker = ""
    if net_ev > best_ev:
        best_ev = net_ev
        best_dd = dd_pct
        marker = " <- BEST EV"

    print(f"{dd_pct*100:<11.0f}%  {pass_rate*100:>6.1f}%     ${expected_payout:>12,.0f}  ${expected_cost:>12,.0f}  ${net_ev:>12,.0f}  {roi:>8.0f}%{marker}")

# Two-account probability analysis
print("\n" + "="*90)
print("TWO ACCOUNT STRATEGY ANALYSIS (Buying 2 Challenges)")
print("="*90)

print(f"\n{'DD Limit':<12} {'1 Pass':<12} {'2 Pass':<12} {'>=1 Pass':<12} {'Total Cost':<15} {'Exp Payout':<15} {'Net EV':<15}")
print("-"*90)

best_two_ev = 0
best_two_dd = 0

for dd_pct in drawdown_options:
    result = results[dd_pct]
    pass_rate = result['pass_rate']
    fail_rate = result['fail_rate']

    # Probabilities for 2 accounts
    prob_both_fail = fail_rate ** 2
    prob_one_pass = 2 * pass_rate * fail_rate
    prob_both_pass = pass_rate ** 2
    prob_at_least_one = 1 - prob_both_fail

    # Expected payout
    expected_payout = (prob_one_pass * profit_target_dollars * profit_split) + \
                     (prob_both_pass * 2 * profit_target_dollars * profit_split)

    # Total cost
    total_cost = challenge_cost * 2

    # Net EV
    net_ev = expected_payout - total_cost

    marker = ""
    if net_ev > best_two_ev:
        best_two_ev = net_ev
        best_two_dd = dd_pct
        marker = " <- BEST"

    print(f"{dd_pct*100:<11.0f}%  {prob_one_pass*100:>6.1f}%     {prob_both_pass*100:>6.1f}%     {prob_at_least_one*100:>6.1f}%     ${total_cost:>12,}  ${expected_payout:>12,.0f}  ${net_ev:>12,.0f}{marker}")

# Risk/Reward trade-off analysis
print("\n" + "="*90)
print("RISK/REWARD TRADE-OFF")
print("="*90)

print(f"\n{'DD Limit':<12} {'Safety':<25} {'Speed':<25} {'Recommendation':<30}")
print("-"*90)

for dd_pct in drawdown_options:
    result = results[dd_pct]
    pass_rate = result['pass_rate']
    fail_rate = result['fail_rate']
    avg_trades = result['avg_trades']

    # Safety rating (based on pass rate)
    if pass_rate >= 0.75:
        safety = "Very High"
    elif pass_rate >= 0.70:
        safety = "High"
    elif pass_rate >= 0.65:
        safety = "Moderate"
    elif pass_rate >= 0.60:
        safety = "Low"
    else:
        safety = "Very Low"

    # Speed rating (based on avg trades)
    if avg_trades <= 15:
        speed = "Very Fast"
    elif avg_trades <= 20:
        speed = "Fast"
    elif avg_trades <= 25:
        speed = "Moderate"
    elif avg_trades <= 30:
        speed = "Slow"
    else:
        speed = "Very Slow"

    # Recommendation
    if dd_pct == best_two_dd:
        recommendation = "OPTIMAL (Best EV)"
    elif dd_pct == 0.05:
        recommendation = "Too Conservative (Low EV)"
    elif dd_pct == 0.12:
        recommendation = "Too Risky"
    elif dd_pct == 0.08:
        recommendation = "Standard (Balanced)"
    elif dd_pct == 0.10:
        recommendation = "Aggressive (Fast but risky)"
    else:
        recommendation = "Consider"

    print(f"{dd_pct*100:<11.0f}%  {safety:<24} {speed:<24} {recommendation:<30}")

# Historical max drawdown from backtest
print("\n" + "="*90)
print("HISTORICAL DRAWDOWN DATA (From Your Backtest)")
print("="*90)

# Calculate max drawdown from backtest
equity_curve = [100]
for trade in trades:
    if trade['exit_reason'] in ['TAKE_PROFIT', 'STOP_LOSS']:
        equity_curve.append(equity_curve[-1] * (1 + trade['pnl_pct']))

peak = equity_curve[0]
max_dd = 0
for equity in equity_curve:
    if equity > peak:
        peak = equity
    dd = ((peak - equity) / peak) * 100
    if dd > max_dd:
        max_dd = dd

print(f"\nYour Strategy's Worst Historical Drawdown: {max_dd:.2f}%")
print(f"This occurred during 6-month backtest with {len(trades)} trades")

# Recommendations
print("\n" + "="*90)
print("FINAL RECOMMENDATIONS")
print("="*90)

print(f"\n1. OPTIMAL CHOICE: {best_two_dd*100:.0f}% Drawdown Limit")
print(f"   Pass Rate (2 accounts): {(1 - results[best_two_dd]['fail_rate']**2)*100:.1f}%")
print(f"   Expected Net Profit: ${best_two_ev:,.0f}")
print(f"   Avg Days to Complete: {results[best_two_dd]['avg_trades']:.0f}")
print(f"   Why: Maximizes expected value while maintaining high pass rate")

print(f"\n2. CONSERVATIVE CHOICE: 8% Drawdown Limit (Standard)")
if 0.08 in results:
    result_8 = results[0.08]
    prob_two = 1 - result_8['fail_rate']**2
    print(f"   Pass Rate (2 accounts): {prob_two*100:.1f}%")
    print(f"   Avg Days to Complete: {result_8['avg_trades']:.0f}")
    print(f"   Why: Industry standard, well-tested, balanced approach")

print(f"\n3. AGGRESSIVE CHOICE: {max(drawdown_options)*100:.0f}% Drawdown Limit")
result_max = results[max(drawdown_options)]
prob_max = 1 - result_max['fail_rate']**2
print(f"   Pass Rate (2 accounts): {prob_max*100:.1f}%")
print(f"   Avg Days to Complete: {result_max['avg_trades']:.0f}")
print(f"   Why: Fastest completion, highest pass rate, but requires larger buffer")

# Risk analysis
print("\n" + "="*90)
print("RISK ANALYSIS BY DRAWDOWN LIMIT")
print("="*90)

print(f"\nYour historical max DD was {max_dd:.2f}%")
print(f"This means you need AT LEAST {max_dd:.0f}% DD limit to be safe")
print()

for dd_pct in sorted(drawdown_options):
    buffer = (dd_pct * 100) - max_dd
    if buffer < 0:
        risk_level = "EXTREMELY RISKY - Below historical max DD"
    elif buffer < 1:
        risk_level = "VERY RISKY - Minimal buffer"
    elif buffer < 2:
        risk_level = "RISKY - Small buffer"
    elif buffer < 3:
        risk_level = "MODERATE - Reasonable buffer"
    elif buffer < 4:
        risk_level = "SAFE - Good buffer"
    else:
        risk_level = "VERY SAFE - Large buffer"

    print(f"{dd_pct*100:.0f}% DD Limit: {buffer:+.2f}% buffer -> {risk_level}")

# Cost comparison
print("\n" + "="*90)
print("COST ANALYSIS (Does higher DD limit cost more?)")
print("="*90)

print("\nNote: E8 Markets charges the SAME PRICE regardless of DD limit!")
print(f"All options cost ${challenge_cost} per challenge")
print()
print("This means:")
print("  - Higher DD limit = Better value (same price, higher pass rate)")
print("  - Lower DD limit = Worse value (same price, lower pass rate)")
print()
print("Recommendation: Choose highest DD limit you're comfortable with")

print("\n" + "="*90)
print("OPTIMAL STRATEGY")
print("="*90)

print(f"\nBased on Monte Carlo analysis:")
print()
print(f"[OK] RECOMMENDED: {best_two_dd*100:.0f}% Drawdown Limit")
print(f"  - Highest expected value: ${best_two_ev:,.0f}")
print(f"  - Pass rate (2 accounts): {(1 - results[best_two_dd]['fail_rate']**2)*100:.1f}%")
print(f"  - {(best_two_dd*100 - max_dd):.1f}% buffer above historical max DD")
print(f"  - Complete in ~{results[best_two_dd]['avg_trades']:.0f} days")
print()
print(f"Alternative: 8% DD (Standard) if you prefer industry norm")
print(f"  - Pass rate (2 accounts): {(1 - results[0.08]['fail_rate']**2)*100:.1f}%")
print(f"  - More conservative approach")
print(f"  - Widely accepted by trading community")

print("\n" + "="*90 + "\n")
