#!/usr/bin/env python3
"""
Check what opportunities the bot is currently finding
"""
import asyncio
import sys
sys.path.append('.')

from datetime import datetime
import pytz

async def check_opportunities():
    try:
        from OPTIONS_BOT import TomorrowReadyOptionsBot
        
        print("CHECKING CURRENT TRADING OPPORTUNITIES")
        print("=" * 40)
        
        # Check market status
        et_timezone = pytz.timezone('America/New_York')
        now_et = datetime.now(et_timezone)
        print(f"Current time: {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
        
        bot = TomorrowReadyOptionsBot()
        await bot.initialize_all_systems()
        
        print(f"Current positions: {len(bot.active_positions)}")
        print(f"Max positions: {bot.daily_risk_limits.get('max_positions', 5)}")
        
        # Check if we can find opportunities
        print("\nScanning for opportunities...")
        
        # Test the scan method directly
        opportunities = []
        test_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
        
        for symbol in test_symbols:
            try:
                print(f"  Checking {symbol}...")
                opp = await bot.find_high_quality_opportunity(symbol)
                if opp:
                    opportunities.append(opp)
                    conf = opp.get('confidence', 0)
                    ret = opp.get('expected_return', 0)
                    print(f"    Found: {opp['strategy']} - {conf:.1%} confidence, {ret:.2f}x return")
            except Exception as e:
                print(f"    Error: {e}")
        
        print(f"\nTotal opportunities found: {len(opportunities)}")
        
        # Check 75%+ confidence
        high_conf = [opp for opp in opportunities if opp.get('confidence', 0) >= 0.75]
        print(f"High confidence (75%+): {len(high_conf)}")
        
        for opp in high_conf:
            conf = opp.get('confidence', 0)
            print(f"  EXECUTE: {opp['symbol']} - {opp['strategy']} - {conf:.1%}")
        
        if not high_conf:
            print("\nNo 75%+ confidence opportunities found")
            print("This is why the bot hasn't made trades today.")
            
            # Show all opportunities
            if opportunities:
                print(f"\nAll opportunities found ({len(opportunities)}):")
                for opp in sorted(opportunities, key=lambda x: x.get('confidence', 0), reverse=True):
                    conf = opp.get('confidence', 0)
                    ret = opp.get('expected_return', 0)
                    print(f"  {opp['symbol']} - {opp['strategy']} - {conf:.1%} conf, {ret:.2f}x return")
        
        return opportunities
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_opportunities())