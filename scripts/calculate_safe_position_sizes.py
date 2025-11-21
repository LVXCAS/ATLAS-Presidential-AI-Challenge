from HYBRID_OANDA_TRADELOCKER import HybridAdapter

# Get current account status
adapter = HybridAdapter()
summary = adapter.get_account_summary()

current_equity = summary['NAV']
peak_balance = 208163  # Known peak from previous wins
current_dd_pct = ((peak_balance - current_equity) / peak_balance) * 100
max_dd_pct = 6.0
remaining_dd_pct = max_dd_pct - current_dd_pct
remaining_dd_dollars = (remaining_dd_pct / 100) * peak_balance

print("=" * 70)
print("E8 POSITION SIZING CALCULATOR")
print("=" * 70)
print(f"Current Equity: ${current_equity:,.2f}")
print(f"Peak Balance: ${peak_balance:,.2f}")
print(f"Current Trailing DD: {current_dd_pct:.2f}%")
print(f"Remaining DD Cushion: {remaining_dd_pct:.2f}% (${remaining_dd_dollars:,.2f})")
print()

# Bot settings
risk_per_trade_pct = 0.02  # 2% risk
stop_loss_pct = 0.01  # 1% stop loss
leverage = 5
position_size_multiplier = 0.80  # 80% safety factor

# Standard position sizing (what bot would calculate without DD constraint)
standard_risk_amount = current_equity * risk_per_trade_pct
print("STANDARD POSITION SIZING (WITHOUT DD CONSTRAINT):")
print(f"  Risk per trade: {risk_per_trade_pct*100}% of ${current_equity:,.2f} = ${standard_risk_amount:,.2f}")

# For each pair
pairs = [
    ('EUR_USD', 1.1545),
    ('GBP_USD', 1.3125),
    ('USD_JPY', 156.15)
]

print()
print("=" * 70)
print("STANDARD SIZING (2% risk, NO DD constraint):")
print("=" * 70)

for symbol, price in pairs:
    stop_distance = price * stop_loss_pct
    units = int((standard_risk_amount / stop_distance) * leverage)
    units = int(units * position_size_multiplier)
    
    # Cap at 1M units (10 lots)
    if units > 1000000:
        units = 1000000
    
    lots = units / 100000
    potential_loss = units * stop_distance
    
    print(f"\n{symbol} @ {price:.5f}:")
    print(f"  Position: {units:,} units ({lots:.1f} lots)")
    print(f"  Max Loss (1% SL): ${potential_loss:,.2f}")
    print(f"  Max Profit (2% TP): ${potential_loss * 2:,.2f}")

# Now calculate DD-CONSTRAINED sizing
print()
print("=" * 70)
print("DD-CONSTRAINED SIZING (ACTUAL SAFE SIZES):")
print("=" * 70)
print(f"Maximum safe loss per trade: ${remaining_dd_dollars:,.2f}")
print()

for symbol, price in pairs:
    stop_distance = price * stop_loss_pct
    
    # Calculate max units that keeps loss under DD cushion
    max_safe_units = int(remaining_dd_dollars / stop_distance)
    
    # Apply 80% safety factor
    max_safe_units = int(max_safe_units * 0.80)
    
    # Round down to nearest 10k (mini lot)
    max_safe_units = (max_safe_units // 10000) * 10000
    
    lots = max_safe_units / 100000
    potential_loss = max_safe_units * stop_distance
    potential_profit = potential_loss * 2  # 2:1 R/R
    
    # Calculate reduction vs standard
    standard_units = int((standard_risk_amount / stop_distance) * leverage * position_size_multiplier)
    if standard_units > 1000000:
        standard_units = 1000000
    reduction_pct = ((standard_units - max_safe_units) / standard_units) * 100
    
    print(f"\n{symbol} @ {price:.5f}:")
    print(f"  Safe Position: {max_safe_units:,} units ({lots:.1f} lots)")
    print(f"  Max Loss (1% SL): ${potential_loss:,.2f}")
    print(f"  Max Profit (2% TP): ${potential_profit:,.2f}")
    print(f"  Reduction: -{reduction_pct:.1f}% vs standard sizing")

print()
print("=" * 70)
print("CHALLENGE IMPACT:")
print("=" * 70)
print(f"Profit needed to pass: ${20000 - (current_equity - 200000):,.2f}")
print(f"With DD-constrained sizing:")
print(f"  Avg profit per win: ~${remaining_dd_dollars * 2 * 0.5:,.0f} (50% win rate)")
print(f"  Trades needed: ~{(20000 - (current_equity - 200000)) / (remaining_dd_dollars * 2 * 0.5):.0f} winning trades")
print()
print("RECOMMENDATION:")
print("  1. Wait for HIGH CONVICTION setups (score 4.0+ only)")
print("  2. Focus on pairs with tighter spreads (EUR/USD best)")
print("  3. Consider stopping at loss if equity drops below $198k")
print("     (would leave only 0.87% DD cushion = too risky to continue)")
print("=" * 70)
