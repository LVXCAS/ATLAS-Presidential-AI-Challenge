#!/usr/bin/env python3
"""
Test Crypto Trading System - Immediate Start
"""

import alpaca_trade_api as tradeapi
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

class TestCryptoTrader:
    def __init__(self):
        print(">> Initializing Test Crypto Trader...")
        
        load_dotenv()
        
        print(">> Loading API credentials...")
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        
        self.crypto_pairs = ['BTCUSD', 'ETHUSD']
        self.min_trade_amount = 25
        print(">> Initialization complete!")
    
    def test_api_connection(self):
        """Test API connection"""
        try:
            print(">> Testing API connection...")
            account = self.api.get_account()
            print(f">> Account connected! Portfolio: ${float(account.portfolio_value):,.2f}")
            return True
        except Exception as e:
            print(f">> API Error: {e}")
            return False
    
    def analyze_crypto(self, symbol):
        """Simple crypto analysis"""
        print(f">> Analyzing {symbol}...")
        
        # Simple random analysis for testing
        signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.4, 0.3, 0.3])
        strength = np.random.uniform(0.6, 0.9)
        
        print(f"   Signal: {signal}, Strength: {strength:.2f}")
        
        return {
            'symbol': symbol,
            'signal': signal,
            'strength': strength
        }
    
    def place_test_trade(self, symbol, signal, amount):
        """Place a test trade"""
        try:
            print(f">> PLACING TEST TRADE: {signal} ${amount} {symbol}")
            
            # Uncomment to place real trade:
            # order = self.api.submit_order(
            #     symbol=symbol,
            #     notional=amount,
            #     side=signal.lower(),
            #     type='market',
            #     time_in_force='gtc'
            # )
            
            print(f">> TEST TRADE: Would place {signal} ${amount} {symbol}")
            return True
            
        except Exception as e:
            print(f">> Trade Error: {e}")
            return False
    
    async def test_trading_loop(self):
        """Test trading loop"""
        print(">> Starting Test Trading Loop...")
        
        if not self.test_api_connection():
            print(">> API connection failed, stopping...")
            return
        
        cycle = 0
        
        while cycle < 3:  # Run 3 test cycles
            cycle += 1
            print(f"\n>> TEST CYCLE {cycle}")
            print(f">> Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Test each crypto
            for symbol in self.crypto_pairs:
                analysis = self.analyze_crypto(symbol)
                
                if analysis['signal'] in ['BUY', 'SELL']:
                    trade_amount = 50  # Test with $50
                    self.place_test_trade(symbol, analysis['signal'], trade_amount)
                else:
                    print(f"   No trade signal for {symbol}")
            
            print(">> Waiting 30 seconds...")
            await asyncio.sleep(30)
        
        print(">> Test trading complete!")

async def main():
    print("=" * 50)
    print("HIVE TRADE - TEST CRYPTO TRADER")
    print("=" * 50)
    
    trader = TestCryptoTrader()
    await trader.test_trading_loop()

if __name__ == "__main__":
    asyncio.run(main())