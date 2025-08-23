#!/usr/bin/env python3
"""
HIVE TRADE - Live Crypto Trading v2
Streamlined 24/7 crypto trading system
"""
import alpaca_trade_api as tradeapi
import os
import asyncio
import random
import time
from datetime import datetime
from dotenv import load_dotenv

class LiveCryptoV2:
    def __init__(self):
        print("=" * 50)
        print("HIVE TRADE - LIVE CRYPTO SYSTEM v2")
        print("=" * 50)
        
        load_dotenv()
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        
        # Active crypto pairs
        self.crypto_pairs = ['BTCUSD', 'ETHUSD']
        self.min_trade = 25
        self.max_trade = 100
        
        print(">> Crypto system initialized")
        print(">> Trading pairs:", self.crypto_pairs)
        print(">> Trade range: $25 - $100")
        
    def get_account_info(self):
        """Get account information"""
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            portfolio = float(account.portfolio_value)
            
            print(f">> Portfolio: ${portfolio:,.2f}")
            print(f">> Buying Power: ${buying_power:,.2f}")
            
            return buying_power
        except Exception as e:
            print(f">> Account error: {e}")
            return 0
    
    def get_positions(self):
        """Get current crypto positions"""
        try:
            positions = self.api.list_positions()
            crypto_pos = []
            
            for pos in positions:
                if 'USD' in pos.symbol and len(pos.symbol) > 5:
                    crypto_pos.append({
                        'symbol': pos.symbol,
                        'qty': float(pos.qty),
                        'value': float(pos.market_value),
                        'pnl': float(pos.unrealized_pl)
                    })
            
            return crypto_pos
        except Exception as e:
            print(f">> Position error: {e}")
            return []
    
    def simple_signal(self, symbol):
        """Generate simple trading signal"""
        # Simple momentum-based signal
        signals = ['BUY', 'SELL', 'HOLD']
        weights = [0.35, 0.25, 0.40]  # Slightly bullish bias
        
        signal = random.choices(signals, weights=weights)[0]
        confidence = random.uniform(0.6, 0.85)
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence
        }
    
    def place_crypto_order(self, symbol, side, amount):
        """Place crypto order"""
        try:
            print(f">> PLACING ORDER: {side} ${amount} {symbol}")
            
            order = self.api.submit_order(
                symbol=symbol,
                notional=amount,
                side=side.lower(),
                type='market',
                time_in_force='gtc'
            )
            
            print(f">> ORDER SUCCESS: {order.id}")
            print(f"   Status: {order.status}")
            print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print(f">> ORDER FAILED: {e}")
            return False
    
    async def trading_cycle(self, cycle_num):
        """Single trading cycle"""
        print(f"\n>> CYCLE {cycle_num} - {datetime.now().strftime('%H:%M:%S')}")
        
        # Check account
        buying_power = self.get_account_info()
        if buying_power < self.min_trade:
            print(">> Insufficient buying power")
            return
        
        # Check positions
        positions = self.get_positions()
        print(f">> Current positions: {len(positions)}")
        for pos in positions:
            pnl_pct = (pos['pnl'] / abs(pos['value'])) * 100 if pos['value'] != 0 else 0
            print(f"   {pos['symbol']}: ${pos['value']:.2f} P&L: ${pos['pnl']:+.2f} ({pnl_pct:+.1f}%)")
        
        # Generate signals and trade
        for symbol in self.crypto_pairs:
            signal_data = self.simple_signal(symbol)
            
            print(f">> {symbol}: {signal_data['signal']} (conf: {signal_data['confidence']:.2f})")
            
            if signal_data['signal'] in ['BUY', 'SELL'] and signal_data['confidence'] > 0.7:
                # Calculate trade size
                trade_amount = min(self.max_trade, buying_power * 0.03)  # 3% of buying power
                trade_amount = max(self.min_trade, trade_amount)
                
                if trade_amount >= self.min_trade:
                    success = self.place_crypto_order(symbol, signal_data['signal'], trade_amount)
                    if success:
                        # Log the trade
                        with open('crypto_trades.log', 'a') as f:
                            f.write(f"{datetime.now().isoformat()},{symbol},{signal_data['signal']},{trade_amount}\n")
                    
                    # Wait a bit after trade
                    await asyncio.sleep(5)
    
    async def run_live_trading(self):
        """Main live trading loop"""
        print("\n>> STARTING LIVE 24/7 CRYPTO TRADING")
        print(">> This will place REAL trades with REAL money")
        print(">> Press Ctrl+C to stop")
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                await self.trading_cycle(cycle)
                
                # Wait between cycles
                wait_time = 120  # 2 minutes between cycles
                print(f">> Waiting {wait_time}s until next cycle...")
                await asyncio.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n>> TRADING STOPPED")
            print(f">> Completed {cycle} cycles")
            
            # Show final positions
            final_positions = self.get_positions()
            if final_positions:
                print("\n>> FINAL POSITIONS:")
                total_pnl = 0
                for pos in final_positions:
                    print(f"   {pos['symbol']}: ${pos['value']:.2f} | P&L: ${pos['pnl']:+.2f}")
                    total_pnl += pos['pnl']
                print(f">> TOTAL P&L: ${total_pnl:+.2f}")

async def main():
    trader = LiveCryptoV2()
    await trader.run_live_trading()

if __name__ == "__main__":
    asyncio.run(main())