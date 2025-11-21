import json

# Load ATLAS options trading data
with open('data/options_completed_trades.json') as f:
    data = json.load(f)

# Calculate metrics
total_pnl = sum(t['realized_pnl'] for t in data)
total_capital_deployed = sum(abs(t['entry_price'] * t['contracts']) for t in data)
avg_capital_per_trade = total_capital_deployed / len(data)

# Time-based metrics
hold_durations = [t['hold_duration_hours'] for t in data]
avg_hold_hours = sum(hold_durations) / len(hold_durations)
avg_hold_days = avg_hold_hours / 24

# Calculate ROI
roi_total = (total_pnl / total_capital_deployed) * 100
roi_per_trade = (total_pnl / len(data)) / avg_capital_per_trade * 100

# Annualized ROI (assuming average hold time)
trades_per_year = 365 / avg_hold_days
annualized_roi = roi_per_trade * trades_per_year

# Win/loss metrics
wins = sum(1 for t in data if t['win_loss'] == 'WIN')
losses = sum(1 for t in data if t['win_loss'] == 'LOSS')
win_rate = wins / len(data) * 100

# Win vs loss amounts
winning_trades = [t['realized_pnl'] for t in data if t['win_loss'] == 'WIN']
losing_trades = [t['realized_pnl'] for t in data if t['win_loss'] == 'LOSS']
avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0

print("=" * 70)
print("ATLAS OPTIONS SYSTEM - ROI ANALYSIS")
print("=" * 70)
print(f"\nCAPITAL METRICS:")
print(f"Total Capital Deployed: ${total_capital_deployed:,.2f}")
print(f"Avg Capital Per Trade: ${avg_capital_per_trade:,.2f}")
print(f"Total P/L: ${total_pnl:,.2f}")

print(f"\nROI METRICS:")
print(f"Total ROI: {roi_total:.2f}%")
print(f"ROI Per Trade: {roi_per_trade:.2f}%")
print(f"Annualized ROI: {annualized_roi:.2f}%")

print(f"\nTIME METRICS:")
print(f"Avg Hold Duration: {avg_hold_hours:.1f} hours ({avg_hold_days:.1f} days)")
print(f"Estimated Trades/Year: {trades_per_year:.1f}")

print(f"\nWIN/LOSS METRICS:")
print(f"Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
print(f"Avg Win: ${avg_win:,.2f}")
print(f"Avg Loss: ${avg_loss:,.2f}")
print(f"Profit Factor: {profit_factor:.2f}x")
print(f"Risk/Reward Ratio: {abs(avg_win / avg_loss):.2f}:1")

print("\n" + "=" * 70)
print("COMPARISON TO BENCHMARKS")
print("=" * 70)
print(f"ATLAS Annualized ROI: {annualized_roi:.2f}%")
print(f"S&P 500 Average:      ~10.0%")
print(f"Hedge Fund Average:   ~8-12%")
print(f"Top Quant Funds:      ~15-25%")
print(f"\nATLAS vs S&P 500:     {annualized_roi / 10:.1f}x better")
print(f"ATLAS vs Hedge Funds: {annualized_roi / 10:.1f}x better")

# Starting capital assumption for concrete example
starting_capital = 10000
ending_capital = starting_capital * (1 + roi_total / 100)

print("\n" + "=" * 70)
print("PRACTICAL EXAMPLE")
print("=" * 70)
print(f"If you started with:  ${starting_capital:,.2f}")
print(f"After 50 trades:      ${ending_capital:,.2f}")
print(f"Net Profit:           ${ending_capital - starting_capital:,.2f}")
print(f"Return:               {roi_total:.2f}%")
