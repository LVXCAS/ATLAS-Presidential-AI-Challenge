"""
GET ALL TRADE LOGS - Complete trading history from OANDA
Shows all closed trades with entry/exit times, P/L, and statistics
"""
import os
import json
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.transactions as transactions
from datetime import datetime

# Load credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

token = os.getenv('OANDA_API_KEY')
account_id = os.getenv('OANDA_ACCOUNT_ID', '101-004-29159709-001')

client = API(access_token=token, environment='practice')

print('='*80)
print('COMPLETE TRADE LOG - ALL CLOSED TRADES')
print('='*80)
print('\nFetching transactions...')

# Get transactions using pageSize
params = {'pageSize': 1000}
r = transactions.TransactionList(accountID=account_id, params=params)

try:
    response = client.request(r)
except Exception as e:
    print(f"Error fetching transactions: {e}")
    exit(1)

# Parse response
all_transactions = response.get('transactions', [])

if not all_transactions:
    print("No transactions found!")
    exit(0)

print(f'Found {len(all_transactions)} total transactions')

# Filter for only order fills (actual executions)
order_fills = [t for t in all_transactions if t.get('type') == 'ORDER_FILL']
print(f'Found {len(order_fills)} order fills\n')

# Group by trade ID to match entry/exit pairs
trades_by_id = {}

for fill in order_fills:
    trade_opened = fill.get('tradeOpened')
    trade_closed = fill.get('tradesClosed', [])
    trade_reduced = fill.get('tradeReduced')

    if trade_opened:
        # This is an entry
        trade_id = trade_opened['tradeID']
        trades_by_id[trade_id] = {
            'id': trade_id,
            'instrument': fill['instrument'],
            'entry_time': fill['time'],
            'entry_price': float(trade_opened['price']),
            'units': float(trade_opened['units']),
            'direction': 'LONG' if float(trade_opened['units']) > 0 else 'SHORT',
            'entry_transaction_id': fill['id'],
            'closed': False
        }

    if trade_closed:
        # This is a full exit
        for closed in trade_closed:
            trade_id = closed['tradeID']
            if trade_id in trades_by_id:
                trades_by_id[trade_id]['closed'] = True
                trades_by_id[trade_id]['exit_time'] = fill['time']
                trades_by_id[trade_id]['exit_price'] = float(fill['price'])
                trades_by_id[trade_id]['pnl'] = float(closed['realizedPL'])
                trades_by_id[trade_id]['exit_transaction_id'] = fill['id']
                trades_by_id[trade_id]['exit_reason'] = fill.get('reason', 'UNKNOWN')

    if trade_reduced:
        # Partial exit (treat as full for now)
        trade_id = trade_reduced['tradeID']
        if trade_id in trades_by_id:
            trades_by_id[trade_id]['closed'] = True
            trades_by_id[trade_id]['exit_time'] = fill['time']
            trades_by_id[trade_id]['exit_price'] = float(fill['price'])
            trades_by_id[trade_id]['pnl'] = float(trade_reduced['realizedPL'])
            trades_by_id[trade_id]['exit_transaction_id'] = fill['id']
            trades_by_id[trade_id]['exit_reason'] = fill.get('reason', 'PARTIAL')

# Print closed trades only
closed_trades = [t for t in trades_by_id.values() if t['closed']]
closed_trades.sort(key=lambda x: x['entry_time'], reverse=True)

# Open trades
open_trades = [t for t in trades_by_id.values() if not t['closed']]

total_pnl = 0
winners = 0
losers = 0
win_pnl = 0
loss_pnl = 0

print('='*80)
print(f'CLOSED TRADES ({len(closed_trades)} total) - Most Recent First')
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

    # Calculate P/L percentage based on position size
    position_value = abs(trade['units']) * trade['entry_price']
    pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0

    print(f"\nTrade #{trade['id']} - {trade['instrument']} {trade['direction']} - {result}")
    print(f"  Entry:    {entry_dt.strftime('%b %d %I:%M%p')} @ {trade['entry_price']:.5f}")
    print(f"  Exit:     {exit_dt.strftime('%b %d %I:%M%p')} @ {trade['exit_price']:.5f}")
    print(f"  Reason:   {trade.get('exit_reason', 'UNKNOWN')}")
    print(f"  Units:    {abs(trade['units']):,.0f}")
    print(f"  Duration: {duration_hours:.1f} hours")
    print(f"  P/L:      {symbol}${abs(pnl):,.2f} ({pnl_pct:+.3f}%)")

if open_trades:
    print('\n' + '='*80)
    print(f'OPEN TRADES ({len(open_trades)} currently active)')
    print('='*80)
    for trade in open_trades:
        entry_dt = datetime.strptime(trade['entry_time'][:19], '%Y-%m-%dT%H:%M:%S')
        print(f"\nTrade #{trade['id']} - {trade['instrument']} {trade['direction']} - OPEN")
        print(f"  Entry:  {entry_dt.strftime('%b %d %I:%M%p')} @ {trade['entry_price']:.5f}")
        print(f"  Units:  {abs(trade['units']):,.0f}")

print('\n' + '='*80)
print('SUMMARY STATISTICS')
print('='*80)
print(f'Total Closed Trades: {len(closed_trades)}')
if len(closed_trades) > 0:
    print(f'Winners: {winners} ({winners/len(closed_trades)*100:.1f}%)')
    print(f'Losers: {losers} ({losers/len(closed_trades)*100:.1f}%)')
    print(f'')
    print(f'Total P/L: ${total_pnl:,.2f}')
    print(f'Total Wins: ${win_pnl:,.2f}')
    print(f'Total Losses: ${loss_pnl:,.2f}')
    if winners > 0 and losers > 0:
        print(f'Avg Win: ${win_pnl/winners:,.2f}')
        print(f'Avg Loss: ${loss_pnl/losers:,.2f}')
        print(f'Win/Loss Ratio: {abs(win_pnl/winners)/(abs(loss_pnl/losers)):.2f}x')
        print(f'Profit Factor: {abs(win_pnl/loss_pnl):.2f}')
else:
    print('No closed trades yet.')

print(f'\nOpen Trades: {len(open_trades)}')
print('='*80)

# Save to JSON
output = {
    'generated_at': datetime.now().isoformat(),
    'closed_trades': closed_trades,
    'open_trades': open_trades,
    'stats': {
        'total_closed': len(closed_trades),
        'winners': winners,
        'losers': losers,
        'win_rate': winners/len(closed_trades) if len(closed_trades) > 0 else 0,
        'total_pnl': total_pnl,
        'win_pnl': win_pnl,
        'loss_pnl': loss_pnl,
        'avg_win': win_pnl/winners if winners > 0 else 0,
        'avg_loss': loss_pnl/losers if losers > 0 else 0
    }
}

with open('complete_trade_log.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nTrade log saved to: complete_trade_log.json')
