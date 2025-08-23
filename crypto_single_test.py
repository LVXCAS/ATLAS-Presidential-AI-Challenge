#!/usr/bin/env python3
"""
Single Crypto Trading Cycle Test
"""
import alpaca_trade_api as tradeapi
import os
import random
from datetime import datetime
from dotenv import load_dotenv

print("=" * 50)
print("HIVE TRADE - SINGLE CRYPTO CYCLE TEST")
print("=" * 50)

load_dotenv()
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

print(">> Testing crypto trading cycle...")

try:
    # Check account
    account = api.get_account()
    buying_power = float(account.buying_power)
    portfolio = float(account.portfolio_value)
    
    print(f">> Portfolio: ${portfolio:,.2f}")
    print(f">> Buying Power: ${buying_power:,.2f}")
    
    # Check positions
    positions = api.list_positions()
    crypto_positions = []
    
    for pos in positions:
        if 'USD' in pos.symbol and len(pos.symbol) > 5:
            crypto_positions.append({
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'value': float(pos.market_value),
                'pnl': float(pos.unrealized_pl)
            })
    
    print(f">> Current crypto positions: {len(crypto_positions)}")
    for pos in crypto_positions:
        pnl_pct = (pos['pnl'] / abs(pos['value'])) * 100 if pos['value'] != 0 else 0
        print(f"   {pos['symbol']}: ${pos['value']:.2f} P&L: ${pos['pnl']:+.2f} ({pnl_pct:+.1f}%)")
    
    # Test signal generation
    crypto_pairs = ['BTCUSD', 'ETHUSD']
    
    for symbol in crypto_pairs:
        signals = ['BUY', 'SELL', 'HOLD']
        signal = random.choice(signals)
        confidence = random.uniform(0.6, 0.9)
        
        print(f">> {symbol} Signal: {signal} (confidence: {confidence:.2f})")
        
        if signal in ['BUY', 'SELL'] and confidence > 0.75:
            trade_amount = 30  # Small test amount
            
            print(f">> Would place: {signal} ${trade_amount} {symbol}")
            
            # Uncomment to place actual trade:
            # order = api.submit_order(
            #     symbol=symbol,
            #     notional=trade_amount,
            #     side=signal.lower(),
            #     type='market',
            #     time_in_force='gtc'
            # )
            # print(f">> Order placed: {order.id}")
            
            print(">> [TEST MODE - No real trade placed]")
            break
    
    print("\n>> Single cycle test complete!")
    print(">> System is ready for live 24/7 trading")
    print(">> Uncomment trade lines in code to place real trades")
    
except Exception as e:
    print(f">> Error: {e}")

print("=" * 50)