import json

# Load ATLAS options trading data
with open('data/options_completed_trades.json') as f:
    data = json.load(f)

# Calculate performance metrics
total_pnl = sum(t['realized_pnl'] for t in data)
wins = sum(1 for t in data if t['win_loss'] == 'WIN')
losses = sum(1 for t in data if t['win_loss'] == 'LOSS')
win_rate = wins / len(data) * 100

# Strategy breakdown
strategies = {}
for t in data:
    strat = t['strategy_type']
    if strat not in strategies:
        strategies[strat] = {'trades': 0, 'pnl': 0, 'wins': 0}
    strategies[strat]['trades'] += 1
    strategies[strat]['pnl'] += t['realized_pnl']
    if t['win_loss'] == 'WIN':
        strategies[strat]['wins'] += 1

print("=" * 70)
print("ATLAS OPTIONS SYSTEM PERFORMANCE (BACKTEST DATA)")
print("=" * 70)
print(f"Total Trades: {len(data)}")
print(f"Wins: {wins} | Losses: {losses}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Total P/L: ${total_pnl:,.2f}")
print(f"Avg P/L per trade: ${total_pnl/len(data):,.2f}")
print(f"\nProfitable: {'YES' if total_pnl > 0 else 'NO'}")

print("\n" + "=" * 70)
print("STRATEGY BREAKDOWN")
print("=" * 70)
for strat, metrics in strategies.items():
    wr = metrics['wins'] / metrics['trades'] * 100
    print(f"\n{strat}:")
    print(f"  Trades: {metrics['trades']}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total P/L: ${metrics['pnl']:,.2f}")
    print(f"  Avg P/L: ${metrics['pnl']/metrics['trades']:,.2f}")
