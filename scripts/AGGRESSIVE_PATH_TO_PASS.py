"""
AGGRESSIVE PATH TO PASS E8 CHALLENGE
Calculate what you need to reach $20,000 profit target
"""

print("=" * 70)
print("E8 CHALLENGE - AGGRESSIVE PATH TO PASS")
print("=" * 70)

# Current situation
starting_balance = 200000
peak_balance = 208163
current_equity = 200942
max_dd_percent = 0.06
profit_target = 20000

# Calculate current state
current_profit = current_equity - starting_balance
profit_needed = profit_target - current_profit
current_dd = (peak_balance - current_equity) / peak_balance
dd_cushion = peak_balance * (max_dd_percent - current_dd)

print(f"\nCURRENT SITUATION:")
print(f"  Starting Balance: ${starting_balance:,.2f}")
print(f"  Peak Balance: ${peak_balance:,.2f}")
print(f"  Current Equity: ${current_equity:,.2f}")
print(f"  Current Profit: ${current_profit:,.2f}")
print(f"  Profit Target: ${profit_target:,.2f}")
print(f"  PROFIT NEEDED: ${profit_needed:,.2f}")
print(f"  Current DD: {current_dd*100:.2f}%")
print(f"  DD Cushion: ${dd_cushion:,.2f}")

# Calculate trade scenarios
print(f"\n{'=' * 70}")
print("SCENARIOS TO REACH $20,000 PROFIT TARGET")
print("=" * 70)

# Scenario 1: Conservative (score 5.0, current settings)
print(f"\nSCENARIO 1: CONSERVATIVE (Score 5.0+, Current Settings)")
print(f"  Strategy: Only perfect setups")
print(f"  Trade Frequency: 1-2 per week")
print(f"  Position Size: 3-4 lots (DD-constrained)")
print(f"  Avg Win: $1,500 | Avg Loss: $800")
print(f"  Win Rate: 50%")

trades_needed_conservative = profit_needed / ((1500 * 0.5) - (800 * 0.5))
weeks_conservative = trades_needed_conservative / 1.5  # 1.5 trades/week avg

print(f"  Trades Needed: {trades_needed_conservative:.0f} trades")
print(f"  Timeline: {weeks_conservative:.1f} weeks ({weeks_conservative/4:.1f} months)")
print(f"  Pass Probability: 15-20% (very tight DD constraint)")

# Scenario 2: Moderate (score 3.5, more trades)
print(f"\nSCENARIO 2: MODERATE (Score 3.5+, More Opportunities)")
print(f"  Strategy: Strong setups (RSI + trend OR MACD + trend)")
print(f"  Trade Frequency: 3-5 per week")
print(f"  Position Size: 3-5 lots")
print(f"  Avg Win: $2,000 | Avg Loss: $1,000")
print(f"  Win Rate: 55%")

avg_profit_moderate = (2000 * 0.55) - (1000 * 0.45)
trades_needed_moderate = profit_needed / avg_profit_moderate
weeks_moderate = trades_needed_moderate / 4  # 4 trades/week avg

print(f"  Expected Profit per Trade: ${avg_profit_moderate:.2f}")
print(f"  Trades Needed: {trades_needed_moderate:.0f} trades")
print(f"  Timeline: {weeks_moderate:.1f} weeks ({weeks_moderate/4:.1f} months)")
print(f"  Pass Probability: 35-40% (balanced risk/reward)")

# Scenario 3: Aggressive (score 2.5, maximum frequency)
print(f"\nSCENARIO 3: AGGRESSIVE (Score 2.5+, Maximum Frequency)")
print(f"  Strategy: Any decent setup (RSI OR MACD OR trend)")
print(f"  Trade Frequency: 5-10 per week")
print(f"  Position Size: 5-8 lots (risk adjusted)")
print(f"  Avg Win: $2,500 | Avg Loss: $1,300")
print(f"  Win Rate: 52%")

avg_profit_aggressive = (2500 * 0.52) - (1300 * 0.48)
trades_needed_aggressive = profit_needed / avg_profit_aggressive
weeks_aggressive = trades_needed_aggressive / 7.5  # 7.5 trades/week avg

print(f"  Expected Profit per Trade: ${avg_profit_aggressive:.2f}")
print(f"  Trades Needed: {trades_needed_aggressive:.0f} trades")
print(f"  Timeline: {weeks_aggressive:.1f} weeks ({weeks_aggressive/4:.1f} months)")
print(f"  Pass Probability: 45-50% (higher risk of DD violation)")

# Scenario 4: MAXIMUM AGGRESSION (what it takes to pass in 2-4 weeks)
print(f"\nSCENARIO 4: MAXIMUM AGGRESSION (2-4 Week Target)")
print(f"  Strategy: Score 2.0+, all sessions, larger position sizes")
print(f"  Trade Frequency: 10-15 per week")
print(f"  Position Size: 6-10 lots (pushing DD limits)")
print(f"  Avg Win: $3,500 | Avg Loss: $2,000")
print(f"  Win Rate: 50%")

avg_profit_max = (3500 * 0.50) - (2000 * 0.50)
trades_needed_max = profit_needed / avg_profit_max
weeks_max = trades_needed_max / 12.5  # 12.5 trades/week avg

print(f"  Expected Profit per Trade: ${avg_profit_max:.2f}")
print(f"  Trades Needed: {trades_needed_max:.0f} trades")
print(f"  Timeline: {weeks_max:.1f} weeks")
print(f"  Pass Probability: 25-30% (high risk of blowing account)")

# Calculate risk of ruin for each scenario
print(f"\n{'=' * 70}")
print("RISK ANALYSIS")
print("=" * 70)

print(f"\nWith ${dd_cushion:,.2f} DD cushion remaining:")
print(f"  Conservative: Can survive {dd_cushion/800:.0f} consecutive losses")
print(f"  Moderate: Can survive {dd_cushion/1000:.0f} consecutive losses")
print(f"  Aggressive: Can survive {dd_cushion/1300:.0f} consecutive losses")
print(f"  Max Aggression: Can survive {dd_cushion/2000:.0f} consecutive losses")

# Recommendation
print(f"\n{'=' * 70}")
print("RECOMMENDATION")
print("=" * 70)

print(f"""
SCENARIO 2 (MODERATE) is the optimal balance:

WHY:
- Timeline: {weeks_moderate:.1f} weeks (~{weeks_moderate/4:.1f} months) - reasonable
- Pass Probability: 35-40% (3x better than conservative)
- Risk of Ruin: Can survive {dd_cushion/1000:.0f} consecutive losses
- Trade Frequency: 3-5/week (manageable to monitor)
- Need only {trades_needed_moderate:.0f} total trades

CONFIGURATION:
- min_score: 3.5 (was 5.0)
- Position size: Up to 5 lots (from 3-4 lots)
- Scan interval: Keep at 1 hour
- Pairs: EUR_USD, GBP_USD (avoid USD_JPY - price too high)

EXPECTED OUTCOME:
- Week 1: 3-5 trades, +$1,500 to +$3,000
- Week 2: 3-5 trades, +$1,500 to +$3,000
- Week 3: 3-5 trades, +$1,500 to +$3,000
- Week 4-8: Continue until reach $20,000 target

TOTAL TIME: {weeks_moderate:.0f}-{weeks_moderate*1.5:.0f} weeks (variance in win rate)
""")

print("=" * 70)
print("WANT TO SWITCH TO SCENARIO 2 (MODERATE)?")
print("=" * 70)
print("\nI can modify the bot settings right now to:")
print("  1. Lower min_score from 5.0 -> 3.5")
print("  2. Increase position size limits (5-6 lots max)")
print("  3. Restart bot with new aggressive settings")
print("\nThis gives you ~40% pass probability in 6-10 weeks.")
print("=" * 70)
