from HYBRID_OANDA_TRADELOCKER import HybridAdapter
import talib
import numpy as np

adapter = HybridAdapter()
candles = adapter.get_candles('GBP_USD', count=100, granularity='H1')

closes = np.array([float(c['mid']['c']) for c in candles])
highs = np.array([float(c['mid']['h']) for c in candles])
lows = np.array([float(c['mid']['l']) for c in candles])

# Current price
current_price = closes[-1]

# Your entry and exit
entry_price = 1.30738
exit_price = 1.30738 - 0.00104  # Approximately -$1,040 loss = ~104 pips

# Bollinger Bands
upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)

# RSI
rsi = talib.RSI(closes, timeperiod=14)

print('=' * 70)
print('GBP/USD POST-MORTEM: What Happened After You Closed')
print('=' * 70)
print(f'Your Entry (LONG): {entry_price:.5f}')
print(f'Your Exit: ~{exit_price:.5f}')
print(f'Loss: ~$1,040')
print(f'')
print(f'CURRENT PRICE: {current_price:.5f}')
print(f'')

if current_price < exit_price:
    drop_from_exit = (exit_price - current_price) * 10000
    would_be_loss = 1040 + (drop_from_exit * 100)
    print(f'Price DROPPED {drop_from_exit:.1f} pips after you exited')
    print(f'If you held: Would be down ${would_be_loss:,.0f} now')
    print(f'YOU SAVED: ${would_be_loss - 1040:,.0f} by exiting when you did')
    print(f'')
    print('VERDICT: Closing was the RIGHT decision')
elif current_price > entry_price:
    gain_from_entry = (current_price - entry_price) * 10000
    would_be_profit = (gain_from_entry * 100) - 1040
    print(f'Price BOUNCED {gain_from_entry:.1f} pips from your entry')
    print(f'If you held: Would be +${would_be_profit:,.0f} now')
    print(f'YOU MISSED: ${would_be_profit + 1040:,.0f} by exiting early')
    print(f'')
    print('VERDICT: Should have held (hindsight)')
else:
    print(f'Price still near exit level ({current_price:.5f})')
    print(f'No major move yet')

print(f'')
print(f'CURRENT INDICATORS:')
print(f'RSI: {rsi[-1]:.2f}')
print(f'Lower BB: {lower[-1]:.5f}')
print(f'Distance from Lower BB: {((current_price - lower[-1])/lower[-1]*10000):.1f} pips')
print('=' * 70)
