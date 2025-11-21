"""
PARSE TRADE HISTORY - Analyze all_transactions.json for complete trade log
"""
import json
from datetime import datetime

# Load the transaction log
with open('all_transactions.json', 'r') as f:
    data = json.load(f)

transactions = data['transactions']

print('='*80)
print('COMPLETE TRADE HISTORY - PARSED FROM OANDA TRANSACTIONS')
print('='*80)

# Build trade dictionary
trades = {}

for txn in transactions:
    txn_type = txn.get('type')

    if txn_type == 'ORDER_FILL':
        # Check what this fill did
        trade_opened = txn.get('tradeOpened')
        trades_closed = txn.get('tradesClosed', [])
        trade_reduced = txn.get('tradeReduced')

        # Opening a trade
        if trade_opened:
            trade_id = trade_opened['tradeID']
            trades[trade_id] = {
                'id': trade_id,
                'instrument': txn['instrument'],
                'entry_time': txn['time'],
                'entry_price': float(trade_opened['price']),
                'units': float(trade_opened['units']),
                'direction': 'LONG' if float(trade_opened['units']) > 0 else 'SHORT',
                'closed': False,
                'entry_txn_id': txn['id']
            }

        # Closing trades
        for closed in trades_closed:
            trade_id = closed['tradeID']
            if trade_id in trades:
                trades[trade_id].update({
                    'closed': True,
                    'exit_time': txn['time'],
                    'exit_price': float(txn['price']),
                    'pnl': float(closed['realizedPL']),
                    'financing': float(closed.get('financing', 0)),
                    'exit_txn_id': txn['id'],
                    'exit_reason': txn.get('reason', 'UNKNOWN')
                })

        # Reduced trades
        if trade_reduced:
            trade_id = trade_reduced['tradeID']
            if trade_id in trades:
                trades[trade_id].update({
                    'closed': True,
                    'exit_time': txn['time'],
                    'exit_price': float(txn['price']),
                    'pnl': float(trade_reduced['realizedPL']),
                    'financing': float(trade_reduced.get('financing', 0)),
                    'exit_txn_id': txn['id'],
                    'exit_reason': 'PARTIAL_CLOSE'
                })

# Separate closed and open trades
closed_trades = [t for t in trades.values() if t['closed']]
open_trades = [t for t in trades.values() if not t['closed']]

# Sort by entry time
closed_trades.sort(key=lambda x: x['entry_time'])

# Calculate stats
total_pnl = 0
winners = 0
losers = 0
win_pnl = 0
loss_pnl = 0

print(f"\n{'='*80}")
print(f"CLOSED TRADES ({len(closed_trades)} total)")
print('='*80)

for i, trade in enumerate(closed_trades, 1):
    entry_dt = datetime.strptime(trade['entry_time'][:19], '%Y-%m-%dT%H:%M:%S')
    exit_dt = datetime.strptime(trade['exit_time'][:19], '%Y-%m-%dT%H:%M:%S')
    duration = exit_dt - entry_dt
    duration_hours = duration.total_seconds() / 3600

    pnl = trade['pnl']
    total_pnl += pnl

    if pnl > 0:
        winners += 1
        win_pnl += pnl
        result = 'WIN'
        symbol = '+'
    else:
        losers += 1
        loss_pnl += pnl
        result = 'LOSS'
        symbol = ''

    # Calculate % return
    position_value = abs(trade['units']) * trade['entry_price']
    pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0

    # Determine why it closed
    reason_map = {
        'STOP_LOSS_ORDER': 'Stop Loss Hit',
        'TAKE_PROFIT_ORDER': 'Take Profit Hit',
        'MARKET_ORDER': 'Manual Close',
        'LIMIT_ORDER': 'Limit Order',
        'PARTIAL_CLOSE': 'Partial Close'
    }
    exit_reason = reason_map.get(trade['exit_reason'], trade['exit_reason'])

    print(f"\n#{i} - Trade {trade['id']}: {trade['instrument']} {trade['direction']} - {result}")
    print(f"  Entry:    {entry_dt.strftime('%b %d %I:%M%p')} @ {trade['entry_price']:.5f}")
    print(f"  Exit:     {exit_dt.strftime('%b %d %I:%M%p')} @ {trade['exit_price']:.5f}")
    print(f"  Reason:   {exit_reason}")
    print(f"  Units:    {abs(trade['units']):,.0f}")
    print(f"  Duration: {duration_hours:.1f} hours")
    print(f"  P/L:      {symbol}${abs(pnl):,.2f} ({pnl_pct:+.3f}%)")
    if trade['financing'] != 0:
        print(f"  Financing: ${trade['financing']:,.2f}")

if open_trades:
    print(f"\n{'='*80}")
    print(f"OPEN TRADES ({len(open_trades)} currently active)")
    print('='*80)
    for trade in open_trades:
        entry_dt = datetime.strptime(trade['entry_time'][:19], '%Y-%m-%dT%H:%M:%S')
        print(f"\nTrade {trade['id']}: {trade['instrument']} {trade['direction']} - OPEN")
        print(f"  Entry: {entry_dt.strftime('%b %d %I:%M%p')} @ {trade['entry_price']:.5f}")
        print(f"  Units: {abs(trade['units']):,.0f}")

print(f"\n{'='*80}")
print('PERFORMANCE SUMMARY')
print('='*80)
print(f'Total Closed Trades: {len(closed_trades)}')

if len(closed_trades) > 0:
    win_rate = (winners / len(closed_trades)) * 100
    print(f'Winners: {winners} ({win_rate:.1f}%)')
    print(f'Losers: {losers} ({(losers/len(closed_trades))*100:.1f}%)')
    print(f'')
    print(f'Total P/L: ${total_pnl:,.2f}')
    print(f'Total Wins: ${win_pnl:,.2f}')
    print(f'Total Losses: ${loss_pnl:,.2f}')

    if winners > 0 and losers > 0:
        avg_win = win_pnl / winners
        avg_loss = loss_pnl / losers
        print(f'')
        print(f'Avg Win: ${avg_win:,.2f}')
        print(f'Avg Loss: ${avg_loss:,.2f}')
        print(f'Win/Loss Ratio: {abs(avg_win / avg_loss):.2f}x')
        print(f'Profit Factor: {abs(win_pnl / loss_pnl):.2f}')
        print(f'')
        print(f'Expected Value per Trade: ${total_pnl / len(closed_trades):,.2f}')

print(f'\nOpen Trades: {len(open_trades)}')
print('='*80)

# Save parsed results
output = {
    'generated_at': datetime.now().isoformat(),
    'closed_trades': closed_trades,
    'open_trades': open_trades,
    'stats': {
        'total_closed': len(closed_trades),
        'winners': winners,
        'losers': losers,
        'win_rate': win_rate if len(closed_trades) > 0 else 0,
        'total_pnl': total_pnl,
        'win_pnl': win_pnl,
        'loss_pnl': loss_pnl,
        'avg_win': win_pnl / winners if winners > 0 else 0,
        'avg_loss': loss_pnl / losers if losers > 0 else 0
    }
}

with open('parsed_trade_history.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nParsed trade history saved to: parsed_trade_history.json")
