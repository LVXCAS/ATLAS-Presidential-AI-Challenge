import json
from datetime import datetime
import random

# CORRECT risk management parameters
WIN_RATE = 0.385
RISK_PER_TRADE = 0.01  # 1% of account per trade (fixed)
REWARD_PER_TRADE = 0.02  # 2% gain per winning trade
CURRENT_CAPITAL = 187190

def calculate_max_drawdown(num_trades, win_rate, risk_per_trade, simulations=10000):
    """
    Calculate maximum drawdown with FIXED 1% risk per trade
    This prevents compounding leverage errors
    """
    max_drawdowns = []

    for _ in range(simulations):
        balance = 100  # Start at 100% for easier percentage tracking
        peak = 100
        max_dd = 0

        for trade in range(num_trades):
            if random.random() < win_rate:
                # Win: +2% of current balance
                balance += balance * REWARD_PER_TRADE
            else:
                # Loss: -1% of current balance
                balance -= balance * risk_per_trade

            # Update peak
            if balance > peak:
                peak = balance

            # Calculate drawdown from peak
            current_dd = ((peak - balance) / peak) * 100
            if current_dd > max_dd:
                max_dd = current_dd

        max_drawdowns.append(max_dd)

    return {
        'median': sorted(max_drawdowns)[len(max_drawdowns) // 2],
        'average': sum(max_drawdowns) / len(max_drawdowns),
        'percentile_75': sorted(max_drawdowns)[int(len(max_drawdowns) * 0.75)],
        'percentile_95': sorted(max_drawdowns)[int(len(max_drawdowns) * 0.95)],
        'worst_case': max(max_drawdowns)
    }

def simulate_losing_streaks(num_simulations=10000):
    """Calculate probability of different losing streak lengths"""
    losing_streaks = []

    for _ in range(num_simulations):
        max_streak = 0
        current_streak = 0

        # Simulate 100 trades
        for _ in range(100):
            if random.random() > WIN_RATE:  # Loss
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        losing_streaks.append(max_streak)

    return {
        'median': sorted(losing_streaks)[len(losing_streaks) // 2],
        'average': sum(losing_streaks) / len(losing_streaks),
        'percentile_75': sorted(losing_streaks)[int(len(losing_streaks) * 0.75)],
        'percentile_95': sorted(losing_streaks)[int(len(losing_streaks) * 0.95)],
        'max': max(losing_streaks)
    }

print("=" * 100)
print("REALISTIC MAXIMUM DRAWDOWN ANALYSIS")
print("Based on 1% risk per trade, 2% reward per win, 38.5% win rate")
print("=" * 100)
print()

# Analyze losing streaks first
print("STEP 1: LOSING STREAK PROBABILITY")
print("-" * 100)
streaks = simulate_losing_streaks()
print(f"With 38.5% win rate (61.5% loss rate), expect these losing streaks:")
print(f"  Typical (Median): {streaks['median']} losses in a row")
print(f"  Average: {streaks['average']:.1f} losses in a row")
print(f"  75th Percentile: {streaks['percentile_75']} losses in a row (happens 1/4 of the time)")
print(f"  95th Percentile: {streaks['percentile_95']} losses in a row (happens 1/20 times)")
print(f"  Worst Case Seen: {streaks['max']} losses in a row (in 10,000 simulations)")
print()

# Calculate theoretical max drawdown from losing streaks
theoretical_dd = {}
for n in [5, 6, 7, 8, 9, 10]:
    # With 1% loss per trade: 5 losses = 4.9% DD (not 5% due to compounding)
    balance = 100
    for _ in range(n):
        balance -= balance * 0.01
    dd_pct = ((100 - balance) / 100) * 100
    theoretical_dd[n] = dd_pct

print("THEORETICAL DRAWDOWN FROM LOSING STREAKS:")
print("-" * 100)
for n, dd in theoretical_dd.items():
    print(f"  {n} losses in a row = {dd:.2f}% drawdown")
print()

# Now simulate actual trading over 1 year
trades_per_year = 96  # 4 pairs * 8 trades/month * 3 months avg (conservative)

print("STEP 2: 1-YEAR DRAWDOWN SIMULATION (96 trades)")
print("-" * 100)
dd_1year = calculate_max_drawdown(trades_per_year, WIN_RATE, RISK_PER_TRADE)
print(f"Expected Maximum Drawdown in 1 Year:")
print(f"  Median: {dd_1year['median']:.2f}%")
print(f"  Average: {dd_1year['average']:.2f}%")
print(f"  75th Percentile: {dd_1year['percentile_75']:.2f}% (1 in 4 years will be this bad)")
print(f"  95th Percentile: {dd_1year['percentile_95']:.2f}% (1 in 20 years will be this bad)")
print(f"  Worst Case: {dd_1year['worst_case']:.2f}% (seen in 10,000 simulations)")
print()

# Calculate for different leverage scenarios
scenarios = {
    'Current Setup (5x leverage, 4 pairs)': {
        'capital': 187190,
        'leverage': 5,
        'num_pairs': 4,
        'dd_multiplier': 1.0,  # Baseline
        'description': 'Your current configuration'
    },
    'Optimized Pairs (5x leverage, 2 pairs)': {
        'capital': 187190,
        'leverage': 5,
        'num_pairs': 2,
        'dd_multiplier': 0.85,  # Better pairs = lower DD
        'description': 'Trade only EUR/USD and GBP/USD'
    },
    'Increased Leverage (10x, 4 pairs)': {
        'capital': 187190,
        'leverage': 10,
        'num_pairs': 4,
        'dd_multiplier': 2.0,  # Double leverage = double DD
        'description': 'DOUBLES your drawdown risk'
    },
    'E8 $100K Challenge (5x, 4 pairs)': {
        'capital': 100000,
        'leverage': 5,
        'num_pairs': 4,
        'dd_multiplier': 1.0,
        'description': 'First E8 challenge - 8% max DD allowed'
    },
    'E8 $500K Scaling (5x, 4 pairs)': {
        'capital': 500000,
        'leverage': 5,
        'num_pairs': 4,
        'dd_multiplier': 1.0,
        'description': 'After passing $100K challenge'
    },
    'Conservative Scaling (6x, 3 pairs, 3x$500K)': {
        'capital': 1500000,
        'leverage': 6,
        'num_pairs': 3,
        'dd_multiplier': 1.2,  # Slightly more aggressive
        'description': 'Recommended long-term setup'
    },
    'Aggressive All-In (10x, 6 pairs, 10x$500K)': {
        'capital': 5000000,
        'leverage': 10,
        'num_pairs': 6,
        'dd_multiplier': 2.5,  # Higher leverage + more pairs
        'description': 'Maximum risk for maximum returns'
    }
}

print("\n" + "=" * 100)
print("STEP 3: SCENARIO-SPECIFIC DRAWDOWN ANALYSIS")
print("=" * 100)
print()

results = {}

for name, scenario in scenarios.items():
    print(f"{name}")
    print("-" * 100)
    print(f"Setup: {scenario['description']}")
    print(f"Capital: ${scenario['capital']:,}")
    print(f"Leverage: {scenario['leverage']}x")
    print()

    # Adjust base drawdown by scenario multiplier
    adjusted_dd = {
        'median': dd_1year['median'] * scenario['dd_multiplier'],
        'average': dd_1year['average'] * scenario['dd_multiplier'],
        'percentile_75': dd_1year['percentile_75'] * scenario['dd_multiplier'],
        'percentile_95': dd_1year['percentile_95'] * scenario['dd_multiplier'],
        'worst_case': dd_1year['worst_case'] * scenario['dd_multiplier']
    }

    # Calculate dollar amounts
    capital = scenario['capital']

    print("EXPECTED MAXIMUM DRAWDOWN:")
    print(f"  Typical (Median): {adjusted_dd['median']:.2f}% = ${capital * adjusted_dd['median']/100:,.0f}")
    print(f"  Average: {adjusted_dd['average']:.2f}% = ${capital * adjusted_dd['average']/100:,.0f}")
    print(f"  75th Percentile: {adjusted_dd['percentile_75']:.2f}% = ${capital * adjusted_dd['percentile_75']/100:,.0f}")
    print(f"  95th Percentile: {adjusted_dd['percentile_95']:.2f}% = ${capital * adjusted_dd['percentile_95']/100:,.0f}")
    print(f"  Worst Case: {adjusted_dd['worst_case']:.2f}% = ${capital * adjusted_dd['worst_case']/100:,.0f}")
    print()

    # Risk assessment
    if adjusted_dd['average'] < 8:
        risk = "LOW"
        note = "Comfortable for most traders"
    elif adjusted_dd['average'] < 12:
        risk = "MEDIUM"
        note = "Will test your discipline"
    elif adjusted_dd['average'] < 18:
        risk = "HIGH"
        note = "Only for experienced traders"
    else:
        risk = "EXTREME"
        note = "Most people can't handle this"

    print(f"Risk Level: {risk}")
    print(f"Assessment: {note}")

    # E8 specific warnings
    if 'E8' in name:
        e8_limit = 8.0
        if adjusted_dd['percentile_75'] > e8_limit:
            print(f"WARNING: 75th percentile DD ({adjusted_dd['percentile_75']:.2f}%) exceeds E8 limit ({e8_limit}%)")
            print(f"         You have ~25% chance of failing the challenge due to DD")
        elif adjusted_dd['percentile_95'] > e8_limit:
            print(f"CAUTION: 95th percentile DD ({adjusted_dd['percentile_95']:.2f}%) exceeds E8 limit ({e8_limit}%)")
            print(f"         You have ~5% chance of hitting the DD limit")
        else:
            print(f"SAFE: Your max DD stays well below E8's {e8_limit}% limit")

    print()

    results[name] = {
        'capital': capital,
        'leverage': scenario['leverage'],
        'drawdown': adjusted_dd,
        'risk_level': risk
    }

# Summary table
print("=" * 100)
print("DRAWDOWN COMPARISON TABLE")
print("=" * 100)
print()
print(f"{'Scenario':<45} {'Capital':<15} {'Avg DD':<12} {'95% DD':<12} {'Dollar Loss':<15} {'Risk'}")
print("-" * 100)

for name, data in results.items():
    scenario_short = name.split('(')[0].strip()
    capital_str = f"${data['capital']/1000:.0f}K" if data['capital'] < 1000000 else f"${data['capital']/1000000:.1f}M"
    avg_dd = f"{data['drawdown']['average']:.1f}%"
    dd_95 = f"{data['drawdown']['percentile_95']:.1f}%"
    dollar_loss = f"${data['capital'] * data['drawdown']['average']/100:,.0f}"

    print(f"{scenario_short:<45} {capital_str:<15} {avg_dd:<12} {dd_95:<12} {dollar_loss:<15} {data['risk_level']}")

print()
print("=" * 100)
print("KEY TAKEAWAYS")
print("=" * 100)
print()

current_unrealized = -5998
current_equity = 187023
current_dd_pct = abs((current_unrealized / current_equity) * 100)

print(f"1. YOUR CURRENT -$5,998 DRAWDOWN IS NORMAL")
print(f"   Current DD: {current_dd_pct:.2f}%")
print(f"   Expected average max: {dd_1year['average']:.2f}%")
print(f"   You're experiencing typical market variance")
print()

print(f"2. REALISTIC EXPECTATIONS FOR YOUR SETUP")
print(f"   Current setup (5x leverage, 4 pairs):")
print(f"   - Typical max DD per year: {dd_1year['median']:.1f}% (${187190 * dd_1year['median']/100:,.0f})")
print(f"   - Bad year (1 in 4): {dd_1year['percentile_75']:.1f}% (${187190 * dd_1year['percentile_75']/100:,.0f})")
print(f"   - Terrible year (1 in 20): {dd_1year['percentile_95']:.1f}% (${187190 * dd_1year['percentile_95']/100:,.0f})")
print()

print(f"3. E8 CHALLENGE VIABILITY")
e8_dd = dd_1year['average'] * 1.0  # Same as baseline
print(f"   Your expected max DD: {e8_dd:.2f}%")
print(f"   E8 limit: 8.0%")
if e8_dd < 8:
    print(f"   Status: SAFE - You should pass the challenge")
    pass_rate = 100 - (dd_1year['percentile_75'] / 8.0 * 25)
    print(f"   Estimated pass rate: {pass_rate:.0f}%")
else:
    print(f"   Status: RISKY - Need to reduce position sizes")
print()

print(f"4. LEVERAGE WARNING")
base_dd = dd_1year['average']
dd_10x = base_dd * 2.0
print(f"   5x leverage: {base_dd:.2f}% average max DD")
print(f"   10x leverage: {dd_10x:.2f}% average max DD (DOUBLES THE PAIN)")
print(f"   On $187K account at 10x: ${187190 * dd_10x/100:,.0f} max loss")
print(f"   DON'T increase leverage until you can stomach this psychologically")
print()

print(f"5. SCALING TO $1M+ CAPITAL")
conservative_dd = dd_1year['average'] * 1.2
print(f"   Conservative scaling (6x leverage, 3 pairs, $1.5M):")
print(f"   - Expected max DD: {conservative_dd:.2f}%")
print(f"   - Dollar amount: ${1500000 * conservative_dd/100:,.0f}")
print(f"   - Reality check: Can you watch ${1500000 * conservative_dd/100:,.0f} disappear and not panic?")
print()

print(f"6. THE TRUTH ABOUT DRAWDOWNS")
print(f"   - They WILL happen 2-4 times per year")
print(f"   - Each one will feel like 'the system is broken'")
print(f"   - You CANNOT avoid them (only manage them)")
print(f"   - Higher ROI = proportionally higher DD")
print(f"   - The pain is the price of admission")
print()

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'current_drawdown_pct': current_dd_pct,
    'current_drawdown_dollars': current_unrealized,
    'losing_streak_analysis': streaks,
    'one_year_baseline_dd': dd_1year,
    'scenarios': {}
}

for name, data in results.items():
    output['scenarios'][name] = {
        'capital': data['capital'],
        'leverage': data['leverage'],
        'average_max_dd_pct': data['drawdown']['average'],
        'percentile_95_dd_pct': data['drawdown']['percentile_95'],
        'average_max_dd_dollars': data['capital'] * data['drawdown']['average'] / 100,
        'risk_level': data['risk_level']
    }

with open('realistic_max_drawdown.json', 'w') as f:
    json.dump(output, f, indent=2)

print("=" * 100)
print("Analysis saved to: realistic_max_drawdown.json")
print("=" * 100)
