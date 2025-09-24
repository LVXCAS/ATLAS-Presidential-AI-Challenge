#!/usr/bin/env python3
"""
Debug script to understand why the bot isn't making trades
"""

import asyncio
import sys
sys.path.append('.')

from OPTIONS_BOT import TomorrowReadyOptionsBot

async def debug_trading():
    """Debug the trading logic"""
    print("DEBUGGING TRADING ISSUES")
    print("=" * 50)
    
    bot = TomorrowReadyOptionsBot()
    
    # Test market data retrieval
    print("\n1. Testing market data retrieval...")
    test_symbols = ['AAPL', 'SPY', 'QQQ', 'MSFT']
    
    working_symbols = []
    for symbol in test_symbols:
        print(f"  Testing {symbol}...")
        market_data = await bot.get_enhanced_market_data(symbol)
        if market_data:
            working_symbols.append(symbol)
            print(f"    [OK] SUCCESS - Price: ${market_data['current_price']:.2f}")
            print(f"      Volume ratio: {market_data['volume_ratio']:.2f}x")
            print(f"      Price momentum: {market_data['price_momentum']:+.1%}")
            
            # Check updated opportunity criteria
            volume_ok = market_data['volume_ratio'] > 0.8
            momentum_ok = abs(market_data['price_momentum']) > 0.015
            vol_ok = market_data.get('realized_vol', 20) > 15
            price_position = market_data.get('price_position', 0.5)
            
            print(f"      Volume > 0.8x: {volume_ok} ({market_data['volume_ratio']:.2f})")
            print(f"      Momentum > 1.5%: {momentum_ok} ({abs(market_data['price_momentum']):.1%})")
            print(f"      Volatility > 15%: {vol_ok} ({market_data.get('realized_vol', 20):.1f}%)")
            
            meets_criteria = (volume_ok and momentum_ok) or (momentum_ok and vol_ok and price_position > 0.3)
            if meets_criteria:
                print(f"    >>> {symbol} MEETS OPPORTUNITY CRITERIA!")
            else:
                print(f"    >>> {symbol} does not meet criteria")
        else:
            print(f"    [FAIL] - No market data")
    
    print(f"\nWorking symbols: {working_symbols}")
    
    # Test opportunity finding
    print("\n2. Testing opportunity finding...")
    opportunities_found = 0
    
    for symbol in working_symbols[:3]:  # Test first 3 working symbols
        print(f"  Scanning {symbol} for opportunities...")
        opportunity = await bot.find_high_quality_opportunity(symbol)
        
        if opportunity:
            opportunities_found += 1
            print(f"    [OK] OPPORTUNITY FOUND!")
            print(f"      Strategy: {opportunity['strategy']}")
            print(f"      Confidence: {opportunity['confidence']:.1%}")
            print(f"      Expected return: {opportunity['expected_return']}x")
            print(f"      Max profit: ${opportunity['max_profit']}")
            print(f"      Max loss: ${opportunity['max_loss']}")
        else:
            print(f"    [FAIL] No opportunity found")
    
    # Test risk limits
    print(f"\n3. Testing risk management...")
    print(f"  Daily risk limits: {bot.daily_risk_limits}")
    print(f"  Current positions: {len(bot.active_positions)}")
    print(f"  Position limit: {bot.daily_risk_limits.get('max_positions', 5)}")
    print(f"  Max position risk: ${bot.daily_risk_limits.get('max_position_risk', 500)}")
    
    position_limit_ok = len(bot.active_positions) < bot.daily_risk_limits.get('max_positions', 5)
    print(f"  Position limit OK: {position_limit_ok}")
    
    # Test broker connection
    print(f"\n4. Testing broker connection...")
    try:
        await bot.initialize_all_systems()
        if bot.readiness_status.get('broker_connected', False):
            print("  [OK] Broker connected successfully")
            print(f"  Account value: ${bot.risk_manager.account_value:,.2f}")
        else:
            print("  [FAIL] Broker connection failed")
    except Exception as e:
        print(f"  [ERROR] Broker connection error: {e}")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("DIAGNOSIS SUMMARY:")
    print(f"  Working market data: {len(working_symbols)}/{len(test_symbols)} symbols")
    print(f"  Opportunities found: {opportunities_found}")
    print(f"  Position limit reached: {not position_limit_ok}")
    print(f"  Broker connected: {bot.readiness_status.get('broker_connected', False)}")
    
    if len(working_symbols) == 0:
        print("\nPRIMARY ISSUE: No market data available")
        print("   - This happens when markets are closed")
        print("   - Try running during market hours (9:30 AM - 4:00 PM ET)")
    elif opportunities_found == 0:
        print("\nPRIMARY ISSUE: No opportunities meet criteria")
        print("   - Volume ratio must be > 1.2x average")
        print("   - Price momentum must be > 2%")
        print("   - Consider relaxing criteria or waiting for more volatile market conditions")
    elif not bot.readiness_status.get('broker_connected', False):
        print("\nPRIMARY ISSUE: Broker not connected")
        print("   - Check API credentials in .env file")
        print("   - Verify Alpaca account is active")
    else:
        print("\nAll systems appear functional - trading should work!")

if __name__ == "__main__":
    asyncio.run(debug_trading())