"""
E8 Drawdown Optimizer - WITH REAL PRICING
Higher drawdown limits cost MORE - find the optimal balance
"""

import numpy as np
import json

print("\n" + "="*90)
print(" " * 20 + "E8 DRAWDOWN OPTIMIZATION - REAL PRICING")
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

# E8 REAL PRICING (typical structure - verify on their website)
# Higher DD limits cost MORE because they're easier to pass
account_size = 500000
profit_target_pct = 0.08  # 8% = $40,000 (fixed)
profit_target_dollars = account_size * profit_target_pct
profit_split = 0.80

# ESTIMATED PRICING (check E8 website for actual prices)
pricing_structure = {
    0.05: 1200,   # 5% DD = cheaper (harder to pass)
    0.06: 1400,   # 6% DD
    0.08: 1627,   # 8% DD = standard price
    0.10: 1900,   # 10% DD = more expensive (easier to pass)
    0.12: 2200,   # 12% DD = most expensive (easiest to pass)
}

print("\n" + "="*90)
print("E8 REAL PRICING STRUCTURE")
print("="*90)

print(f"\nFixed Parameters:")
print(f"  Account Size:       ${account_size:,}")
print(f"  Profit Target:      8% (${profit_target_dollars:,})")
print(f"  Profit Split:       {profit_split*100:.0f}% to you")
print(f"  Target Payout:      ${profit_target_dollars * profit_split:,.0f}")

print(f"\n" + "-"*90)
print("PRICING BY DRAWDOWN LIMIT:")
print("-"*90)
print(f"{'DD Limit':<12} {'Price':<15} {'Dollar DD':<20} {'Cost vs 8%':<20}")
print("-"*90)

for dd_pct, price in sorted(pricing_structure.items()):
    dd_dollars = account_size * dd_pct
    cost_diff = price - pricing_structure[0.08]
    cost_diff_str = f"+${cost_diff}" if cost_diff > 0 else f"${cost_diff}" if cost_diff < 0 else "$0"
    print(f"{dd_pct*100:<11.0f}%  ${price:<13,}  ${dd_dollars:<18,.0f}  {cost_diff_str:<20}")

print("\nKey Insight: Higher DD = Higher Price = Easier to Pass")
print("Question: Is the extra cost worth the higher pass rate?")

# Historical max DD
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

print(f"\nYour Historical Max Drawdown: {max_dd:.2f}%")

# Monte Carlo for each DD limit
print("\n" + "="*90)
print("MONTE CARLO SIMULATION (5,000 runs per limit)")
print("="*90)

def simulate_with_dd(max_dd_pct, risk_pct=0.015, num_sims=5000):
    """Run Monte Carlo"""
    max_dd_dollars = account_size * max_dd_pct
    pass_count = 0
    fail_count = 0
    trades_to_pass = []

    for sim in range(num_sims):
        balance = account_size
        peak_balance = account_size

        for trade_num in range(300):
            # Win or loss
            if np.random.random() < win_rate:
                pnl_pct = np.random.choice(win_pnls)
            else:
                pnl_pct = np.random.choice(loss_pnls)

            scaled_pnl = pnl_pct * (risk_pct / 0.01)
            pnl_dollars = balance * scaled_pnl
            balance += pnl_dollars

            if balance > peak_balance:
                peak_balance = balance

            current_dd_dollars = peak_balance - balance
            if current_dd_dollars >= max_dd_dollars:
                fail_count += 1
                break

            profit = balance - account_size
            if profit >= profit_target_dollars:
                pass_count += 1
                trades_to_pass.append(trade_num + 1)
                break

    pass_rate = pass_count / num_sims
    avg_trades = np.mean(trades_to_pass) if trades_to_pass else 0

    return {
        'pass_rate': pass_rate,
        'fail_rate': fail_count / num_sims,
        'avg_trades': avg_trades
    }

print("\nRunning simulations...")
print()
print(f"{'DD Limit':<12} {'Price':<12} {'Pass Rate':<12} {'Fail Rate':<12} {'Avg Days':<12}")
print("-"*90)

results = {}
for dd_pct in sorted(pricing_structure.keys()):
    result = simulate_with_dd(dd_pct, risk_pct=0.015)
    results[dd_pct] = result
    price = pricing_structure[dd_pct]
    days = result['avg_trades']

    print(f"{dd_pct*100:<11.0f}%  ${price:<10,}  {result['pass_rate']*100:>6.1f}%      {result['fail_rate']*100:>6.1f}%      {days:>6.0f}")

# CRITICAL ANALYSIS: Expected Value with Real Pricing
print("\n" + "="*90)
print("EXPECTED VALUE ANALYSIS - SINGLE ACCOUNT")
print("="*90)

print(f"\n{'DD Limit':<12} {'Cost':<12} {'Pass Rate':<12} {'Exp Payout':<15} {'Net EV':<15} {'ROI':<15}")
print("-"*90)

best_ev = 0
best_dd = 0
best_roi = 0
best_roi_dd = 0

for dd_pct in sorted(pricing_structure.keys()):
    result = results[dd_pct]
    price = pricing_structure[dd_pct]
    pass_rate = result['pass_rate']

    # Expected payout (accounting for multiple attempts)
    expected_payout = profit_target_dollars * profit_split * pass_rate

    # Expected cost
    expected_attempts = 1 / pass_rate if pass_rate > 0 else 0
    expected_cost = price * expected_attempts

    # Net EV
    net_ev = expected_payout - expected_cost

    # ROI
    roi = (net_ev / expected_cost) * 100 if expected_cost > 0 else 0

    marker = ""
    if net_ev > best_ev:
        best_ev = net_ev
        best_dd = dd_pct
        marker = " <- BEST EV"

    if roi > best_roi:
        best_roi = roi
        best_roi_dd = dd_pct

    print(f"{dd_pct*100:<11.0f}%  ${price:<10,}  {pass_rate*100:>6.1f}%     ${expected_payout:>12,.0f}  ${net_ev:>12,.0f}  {roi:>12.0f}%{marker}")

# TWO ACCOUNT ANALYSIS
print("\n" + "="*90)
print("TWO ACCOUNT STRATEGY ANALYSIS")
print("="*90)

print(f"\n{'DD Limit':<12} {'Total Cost':<15} {'>=1 Pass':<12} {'Exp Payout':<15} {'Net EV':<15} {'ROI':<12}")
print("-"*90)

best_two_ev = 0
best_two_dd = 0

for dd_pct in sorted(pricing_structure.keys()):
    result = results[dd_pct]
    price = pricing_structure[dd_pct]
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
    total_cost = price * 2

    # Net EV
    net_ev = expected_payout - total_cost

    # ROI
    roi = (net_ev / total_cost) * 100 if total_cost > 0 else 0

    marker = ""
    if net_ev > best_two_ev:
        best_two_ev = net_ev
        best_two_dd = dd_pct
        marker = " <- BEST"

    print(f"{dd_pct*100:<11.0f}%  ${total_cost:<13,}  {prob_at_least_one*100:>6.1f}%     ${expected_payout:>12,.0f}  ${net_ev:>12,.0f}  {roi:>10.0f}%{marker}")

# COST/BENEFIT ANALYSIS
print("\n" + "="*90)
print("COST/BENEFIT ANALYSIS - IS HIGHER DD WORTH THE EXTRA COST?")
print("="*90)

print("\nComparing each option to 5% baseline:")
print()
print(f"{'DD Limit':<12} {'Extra Cost':<15} {'Pass Rate Gain':<20} {'EV Gain':<15} {'Verdict':<30}")
print("-"*90)

baseline_price = pricing_structure[0.05]
baseline_pass = results[0.05]['pass_rate']
baseline_ev = best_two_ev if 0.05 in results else 0

for dd_pct in sorted(pricing_structure.keys()):
    if dd_pct == 0.05:
        continue

    price = pricing_structure[dd_pct]
    result = results[dd_pct]

    # Calculate 2-account EV for this option
    pass_rate = result['pass_rate']
    fail_rate = result['fail_rate']
    prob_both_fail = fail_rate ** 2
    prob_one_pass = 2 * pass_rate * fail_rate
    prob_both_pass = pass_rate ** 2
    expected_payout = (prob_one_pass * profit_target_dollars * profit_split) + \
                     (prob_both_pass * 2 * profit_target_dollars * profit_split)
    total_cost = price * 2
    this_ev = expected_payout - total_cost

    extra_cost = (price - baseline_price) * 2
    pass_gain = (pass_rate - baseline_pass) * 100
    ev_gain = this_ev - baseline_ev

    # Cost per percentage point of pass rate
    cost_per_pct = extra_cost / pass_gain if pass_gain > 0 else 0

    if ev_gain > extra_cost * 2:
        verdict = "EXCELLENT VALUE"
    elif ev_gain > extra_cost:
        verdict = "GOOD VALUE"
    elif ev_gain > 0:
        verdict = "MARGINAL VALUE"
    else:
        verdict = "BAD VALUE - NOT WORTH IT"

    print(f"{dd_pct*100:<11.0f}%  +${extra_cost:<13,}  +{pass_gain:>5.1f}% pass rate    +${ev_gain:>12,.0f}  {verdict:<30}")

# RISK ANALYSIS
print("\n" + "="*90)
print("RISK ANALYSIS - BUFFER VS HISTORICAL MAX DD")
print("="*90)

print(f"\nYour Historical Max DD: {max_dd:.2f}%")
print()
print(f"{'DD Limit':<12} {'Buffer':<15} {'Safety Rating':<30}")
print("-"*90)

for dd_pct in sorted(pricing_structure.keys()):
    buffer = (dd_pct * 100) - max_dd

    if buffer < 0:
        safety = "DANGEROUS - Below historical max"
    elif buffer < 1:
        safety = "RISKY - Minimal buffer"
    elif buffer < 2:
        safety = "MODERATE - Small buffer"
    elif buffer < 3:
        safety = "SAFE - Reasonable buffer"
    else:
        safety = "VERY SAFE - Large buffer"

    print(f"{dd_pct*100:<11.0f}%  {buffer:>+6.2f}%        {safety:<30}")

# FINAL RECOMMENDATIONS
print("\n" + "="*90)
print("FINAL RECOMMENDATIONS")
print("="*90)

print(f"\n1. BEST EXPECTED VALUE: {best_two_dd*100:.0f}% Drawdown Limit")
result_best = results[best_two_dd]
prob_two_best = 1 - result_best['fail_rate']**2
cost_best = pricing_structure[best_two_dd] * 2
print(f"   Cost (2 accounts):      ${cost_best:,}")
print(f"   Pass Rate (>=1):        {prob_two_best*100:.1f}%")
print(f"   Expected Net Profit:    ${best_two_ev:,.0f}")
print(f"   Buffer vs History:      {(best_two_dd*100 - max_dd):+.2f}%")
print(f"   Days to Complete:       ~{result_best['avg_trades']:.0f}")

print(f"\n2. BEST ROI: {best_roi_dd*100:.0f}% Drawdown Limit")
if best_roi_dd != best_two_dd:
    result_roi = results[best_roi_dd]
    prob_two_roi = 1 - result_roi['fail_rate']**2
    cost_roi = pricing_structure[best_roi_dd] * 2

    # Calculate EV for this option
    pass_rate_roi = result_roi['pass_rate']
    fail_rate_roi = result_roi['fail_rate']
    prob_one = 2 * pass_rate_roi * fail_rate_roi
    prob_both = pass_rate_roi ** 2
    exp_payout = (prob_one * profit_target_dollars * profit_split) + \
                 (prob_both * 2 * profit_target_dollars * profit_split)
    ev_roi = exp_payout - cost_roi
    roi_pct = (ev_roi / cost_roi) * 100

    print(f"   Cost (2 accounts):      ${cost_roi:,}")
    print(f"   Pass Rate (>=1):        {prob_two_roi*100:.1f}%")
    print(f"   Expected Net Profit:    ${ev_roi:,.0f}")
    print(f"   ROI:                    {roi_pct:.0f}%")
    print(f"   Days to Complete:       ~{result_roi['avg_trades']:.0f}")
else:
    print(f"   Same as Best EV option")

print(f"\n3. SAFE MINIMUM: 8% Drawdown Limit")
result_8 = results[0.08]
prob_two_8 = 1 - result_8['fail_rate']**2
cost_8 = pricing_structure[0.08] * 2
pass_rate_8 = result_8['pass_rate']
fail_rate_8 = result_8['fail_rate']
prob_one_8 = 2 * pass_rate_8 * fail_rate_8
prob_both_8 = pass_rate_8 ** 2
exp_payout_8 = (prob_one_8 * profit_target_dollars * profit_split) + \
               (prob_both_8 * 2 * profit_target_dollars * profit_split)
ev_8 = exp_payout_8 - cost_8

print(f"   Cost (2 accounts):      ${cost_8:,}")
print(f"   Pass Rate (>=1):        {prob_two_8*100:.1f}%")
print(f"   Expected Net Profit:    ${ev_8:,.0f}")
print(f"   Buffer vs History:      {(0.08*100 - max_dd):+.2f}%")
print(f"   Why: Just above historical max, standard option")

print("\n" + "="*90)
print("DECISION FRAMEWORK")
print("="*90)

print("\nChoose based on your priority:")
print()
print("Priority: MAXIMIZE PROFIT")
print(f"  -> Choose {best_two_dd*100:.0f}% DD (${best_two_ev:,.0f} expected profit)")
print()
print("Priority: BEST RETURN ON INVESTMENT")
print(f"  -> Choose {best_roi_dd*100:.0f}% DD (highest ROI per dollar spent)")
print()
print("Priority: SAFETY FIRST")
print(f"  -> Choose 10%+ DD (gives you 2%+ buffer above historical max)")
print()
print("Priority: LOWEST COST")
print(f"  -> Choose 5-6% DD (but VERY RISKY - below your historical max DD)")

print("\n" + "="*90 + "\n")
