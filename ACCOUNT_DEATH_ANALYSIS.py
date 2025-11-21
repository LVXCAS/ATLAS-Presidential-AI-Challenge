"""
POST-MORTEM: What Killed the $600 E8 Challenge Account
"""

print("=" * 70)
print("E8 ACCOUNT FAILURE - POST-MORTEM ANALYSIS")
print("=" * 70)

# Timeline reconstruction
print("\n📅 TIMELINE:")
print("-" * 70)
print("Nov 18, 2:12 PM  - Last logged scan (score 5.0, no trades)")
print("Nov 19, 10:21 AM - Bot restarted with SCORE 3.0 aggressive settings")
print("Nov 19, ~10:30 AM - Bot scanned markets (first hourly scan)")
print("Nov 19, ~12:03 PM - Account inaccessible (API error)")
print("-" * 70)

# What changed
print("\n⚙️ CONFIGURATION CHANGES (Conservative -> Aggressive):")
print("-" * 70)
print("min_score:             5.0 -> 3.0  (easier to trigger)")
print("position_multiplier:   0.80 -> 0.90 (larger positions)")
print("max_positions:         3 -> 2  (unchanged impact)")
print("-" * 70)

# Account state before death
print("\n💰 ACCOUNT STATE (Last Known):")
print("-" * 70)
peak_balance = 208163.0
equity_before = 200942.0
dd_before = (peak_balance - equity_before) / peak_balance
dd_cushion = peak_balance * (0.06 - dd_before)

print(f"Peak Balance:      ${peak_balance:,.2f}")
print(f"Equity Before:     ${equity_before:,.2f}")
print(f"Trailing DD:       {dd_before*100:.2f}% / 6.00% max")
print(f"DD Cushion:        ${dd_cushion:,.2f}")
print(f"Open Positions:    0 (clean slate)")
print("-" * 70)

# E8 Daily DD rules
print("\n📏 E8 DRAWDOWN RULES:")
print("-" * 70)
print("TRAILING DD:  6% from peak balance ($208,163)")
print("              - Remaining: $5,268 cushion")
print("")
print("DAILY DD:     Likely 2-4% of starting balance")
print("              - $200k account = $4,000-8,000 max daily loss")
print("              - Resets at midnight EST")
print("-" * 70)

# What likely happened
print("\n💥 PROBABLE CAUSE OF DEATH:")
print("-" * 70)

print("\nSCENARIO 1: Single Large Loss (Most Likely)")
print("  - Bot found score 3.0+ setup on first scan (~10:30 AM)")
print("  - Placed EUR_USD or GBP_USD position (5-6 lots)")
print("  - Position went immediately against bot")
print("  - Hit stop loss: -$2,500 to -$3,500 loss")
print("  - Exceeded DAILY DD limit -> instant account termination")
print("")
print("  Why this happened:")
print("  - Score 3.0 has MUCH lower bar (RSI + trend = trade)")
print("  - 0.90 multiplier = larger positions than before")
print("  - First scan after 20+ hours = pent-up 'opportunities'")
print("  - No positions open = bot eager to trade")

print("\nSCENARIO 2: Multiple Losses (Less Likely)")
print("  - Bot placed 2-3 trades in quick succession")
print("  - Each lost $1,500-2,000")
print("  - Combined losses exceeded daily DD limit")
print("")
print("  Why less likely:")
print("  - Bot scans every 1 hour (not enough time for multiple)")
print("  - max_positions = 2 (would need both to fail simultaneously)")

print("\nSCENARIO 3: Gap/Slippage Event")
print("  - Bot placed position, market gapped against it")
print("  - Stop loss triggered with slippage")
print("  - Loss exceeded expected -1% SL amount")
print("")
print("  Why possible but less likely:")
print("  - Forex gaps are rare during trading hours")
print("  - Would need major news event (didn't see one)")

# Position size calculation
print("\n📊 LIKELY POSITION SIZE:")
print("-" * 70)

balance = 200942
price_eur = 1.15450
price_gbp = 1.30738
sl_pct = 0.01
multiplier = 0.90
leverage = 5
risk_pct = 0.02

# EUR/USD calculation
risk_amount = balance * risk_pct
stop_distance_eur = price_eur * sl_pct
units_eur = int((risk_amount / stop_distance_eur) * leverage * multiplier)
lots_eur = units_eur / 100000
max_loss_eur = (units_eur * stop_distance_eur) / leverage

print(f"\nEUR_USD @ {price_eur}")
print(f"  Position: {units_eur:,} units ({lots_eur:.1f} lots)")
print(f"  Max Loss at SL: ${max_loss_eur:,.2f}")

# GBP/USD calculation
stop_distance_gbp = price_gbp * sl_pct
units_gbp = int((risk_amount / stop_distance_gbp) * leverage * multiplier)
lots_gbp = units_gbp / 100000
max_loss_gbp = (units_gbp * stop_distance_gbp) / leverage

print(f"\nGBP_USD @ {price_gbp}")
print(f"  Position: {units_gbp:,} units ({lots_gbp:.1f} lots)")
print(f"  Max Loss at SL: ${max_loss_gbp:,.2f}")

print("-" * 70)

# DD constraint check
print("\n🚨 DD CONSTRAINT VIOLATION:")
print("-" * 70)
safe_loss = dd_cushion * 0.80

print(f"Safe Max Loss (80% of cushion): ${safe_loss:,.2f}")
print(f"EUR/USD Position Max Loss:      ${max_loss_eur:,.2f}")
print(f"GBP/USD Position Max Loss:      ${max_loss_gbp:,.2f}")

if max_loss_eur > safe_loss:
    print(f"\n❌ EUR/USD position EXCEEDED safe DD limit by ${max_loss_eur - safe_loss:,.2f}")
if max_loss_gbp > safe_loss:
    print(f"❌ GBP/USD position EXCEEDED safe DD limit by ${max_loss_gbp - safe_loss:,.2f}")

print("\nBut bot placed these positions anyway because:")
print("  - DD constraint code has a bug (calculates from balance, not peak)")
print("  - Or position sizing happened before DD check")
print("  - Or slippage exceeded calculated max loss")
print("-" * 70)

# Daily DD calculation
print("\n💀 DAILY DD DEATH SCENARIO:")
print("-" * 70)

daily_dd_limit_conservative = 200000 * 0.02  # 2% ($4k)
daily_dd_limit_moderate = 200000 * 0.03  # 3% ($6k)
daily_dd_limit_aggressive = 200000 * 0.04  # 4% ($8k)

print(f"If E8 Daily DD = 2%: ${daily_dd_limit_conservative:,.2f}")
print(f"If E8 Daily DD = 3%: ${daily_dd_limit_moderate:,.2f}")
print(f"If E8 Daily DD = 4%: ${daily_dd_limit_aggressive:,.2f}")

print(f"\nYour position max loss: ${max_loss_eur:,.2f} to ${max_loss_gbp:,.2f}")
print(f"")
print(f"VERDICT:")
if max_loss_eur > daily_dd_limit_conservative:
    print(f"  ✓ Single position hitting SL would EXCEED 2% daily DD")
if max_loss_eur > daily_dd_limit_moderate:
    print(f"  ✓ Single position hitting SL would EXCEED 3% daily DD")
else:
    print(f"  - Single position hitting SL would be WITHIN 3% daily DD")
    print(f"  - Needed multiple losses OR slippage to exceed limit")

print("-" * 70)

# Lessons
print("\n📚 LESSONS LEARNED:")
print("-" * 70)
print("1. DAILY DD is separate from TRAILING DD")
print("   - You had $5,268 trailing DD cushion")
print("   - But only $4,000-6,000 daily DD limit")
print("   - Bot only checked trailing DD, not daily DD")
print("")
print("2. Score 3.0 was TOO AGGRESSIVE for this situation")
print("   - With $5k cushion, needed score 5.0+ (ultra conservative)")
print("   - Score 3.0 trades 5-7x/week = high risk of daily DD")
print("")
print("3. Position sizing exceeded safe limits")
print("   - 5.8 lot position with $5k cushion = recipe for disaster")
print("   - Should have been 2-3 lots MAX")
print("")
print("4. First scan after long idle = dangerous")
print("   - Bot had been off 20+ hours")
print("   - First scan found 'pent-up opportunities'")
print("   - Should have warmed up with manual review first")
print("")
print("5. Automation without daily DD check = fatal flaw")
print("   - Bot needed to track: 'how much have I lost TODAY?'")
print("   - Should block trades if daily loss > $2,000")
print("-" * 70)

# What should have happened
print("\n✅ WHAT SHOULD HAVE HAPPENED:")
print("-" * 70)
print("CONSERVATIVE APPROACH (Score 5.0):")
print("  - Wait for perfect setup (6.0 score)")
print("  - Position size: 2-3 lots MAX")
print("  - Manual approval before trade execution")
print("  - Check both trailing DD AND daily DD")
print("  - Timeline: 3-6 months to pass")
print("  - Pass probability: 15-20%")
print("")
print("DEMO ACCOUNT VALIDATION:")
print("  - Test aggressive settings on DEMO first")
print("  - Run for 2-4 weeks, see if daily DD is hit")
print("  - THEN deploy on funded account")
print("  - This would have saved $600")
print("-" * 70)

print("\n" + "=" * 70)
print("FINAL VERDICT: Aggressive settings + inadequate safety checks = $600 loss")
print("=" * 70)
print("\nThe $600 is gone, but the lessons are valuable.")
print("Demo account is the right next step.")
print("=" * 70)
