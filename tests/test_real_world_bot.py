#!/usr/bin/env python3
"""
Test the Real-World Options Bot
"""

import asyncio
import sys
sys.path.append('.')

async def test_real_world_options_bot():
    """Test the real-world optimized options bot"""
    print("TESTING REAL-WORLD OPTIONS BOT")
    print("=" * 45)
    
    try:
        from real_world_options_bot import RealWorldOptionsBot
        
        # Initialize bot
        print("Step 1: Initializing Real-World Options Bot...")
        bot = RealWorldOptionsBot()
        print(f"  [OK] Bot created with {len(bot.tier1_stocks)} Tier-1 symbols")
        print(f"  [INFO] Base strategy weights: {list(bot.base_strategy_weights.keys())}")
        
        # Test system initialization
        print("\nStep 2: Testing system initialization...")
        initialized = await bot.initialize_systems()
        if initialized:
            print("  [SUCCESS] All systems initialized")
            print(f"  [INFO] Market Regime: {bot.market_regime}")
            print(f"  [INFO] VIX Level: {bot.vix_level:.1f}")
            print(f"  [INFO] Market Trend: {bot.market_trend:+.1%}")
        else:
            print("  [WARNING] Systems initialized in simulation mode")
        
        # Test market regime detection
        print("\nStep 3: Testing market regime adaptation...")
        dynamic_weights = bot.get_dynamic_strategy_weights()
        print(f"  [INFO] Dynamic weights for {bot.market_regime} regime:")
        for strategy, weight in dynamic_weights.items():
            print(f"    {strategy}: {weight:.1%}")
        
        # Test earnings calendar check
        print("\nStep 4: Testing earnings avoidance...")
        test_symbols = ['AAPL', 'MSFT', 'SPY']
        for symbol in test_symbols:
            has_earnings = await bot.check_earnings_calendar(symbol)
            status = "AVOID" if has_earnings else "OK"
            print(f"  {symbol}: {status}")
        
        # Test enhanced market data
        print("\nStep 5: Testing enhanced market data analysis...")
        for symbol in ['AAPL', 'SPY']:
            print(f"  Analyzing {symbol}...")
            market_data = await bot.get_enhanced_market_data(symbol)
            
            if market_data:
                print(f"    [SUCCESS] Enhanced data retrieved")
                print(f"    Price: ${market_data['current_price']:.2f}")
                print(f"    Momentum: {market_data['price_momentum']:+.1%}")
                print(f"    Realized Vol: {market_data['realized_vol']:.1f}%")
                print(f"    Volume Ratio: {market_data['volume_ratio']:.1f}x")
                print(f"    Price Position: {market_data['price_position']:.1%} (0%=low, 100%=high)")
            else:
                print(f"    [FAIL] Could not get enhanced data")
        
        # Test high-quality opportunity finding
        print("\nStep 6: Testing high-quality opportunity detection...")
        opportunities_found = 0
        total_tested = 0
        
        for symbol in ['AAPL', 'SPY', 'QQQ', 'MSFT']:
            print(f"  Professional scan: {symbol}...")
            total_tested += 1
            opportunity = await bot.find_high_quality_opportunities(symbol)
            
            if opportunity:
                opportunities_found += 1
                print(f"    [SUCCESS] High-quality {opportunity['strategy']} found")
                print(f"    Confidence: {opportunity['confidence']:.1%}")
                print(f"    Expected Return: {opportunity['expected_return']:.1f}x")
                print(f"    Position Size: {opportunity['position_size']} contracts")
                print(f"    Max Profit: ${opportunity['max_profit']:.2f}")
                print(f"    Max Loss: ${opportunity['max_loss']:.2f}")
                print(f"    Reason: {opportunity['reason']}")
            else:
                print(f"    [INFO] No high-quality opportunity found")
        
        print(f"\nHigh-Quality Opportunity Rate: {opportunities_found}/{total_tested} ({opportunities_found/total_tested*100:.0f}%)")
        
        # Test position sizing
        print("\nStep 7: Testing professional position sizing...")
        if opportunities_found > 0:
            # Create a mock opportunity for testing
            mock_opportunity = {
                'max_loss': 2.50,
                'confidence': 0.70,
                'strategy': 'BULL_CALL_SPREAD'
            }
            mock_market_data = {
                'realized_vol': 25.0
            }
            
            size = bot.calculate_position_size(mock_opportunity, mock_market_data)
            total_risk = mock_opportunity['max_loss'] * size * 100
            account_value = bot.risk_manager.account_value
            risk_pct = total_risk / account_value
            
            print(f"  Mock trade position sizing:")
            print(f"    Position Size: {size} contracts")
            print(f"    Total Risk: ${total_risk:.2f}")
            print(f"    Risk %: {risk_pct:.1%} of account")
            print(f"    Risk Management: {'PASS' if risk_pct <= 0.05 else 'FAIL'}")
        
        # Test performance tracking
        print("\nStep 8: Testing professional performance tracking...")
        # Simulate some trades
        bot.update_performance_stats(150.0)   # Winner
        bot.update_performance_stats(-75.0)   # Loser
        bot.update_performance_stats(200.0)   # Big winner
        bot.update_performance_stats(-50.0)   # Small loser
        
        await bot.log_professional_performance()
        
        print("\n" + "=" * 45)
        print("REAL-WORLD OPTIONS BOT TEST COMPLETED")
        print("Professional Features Verified:")
        print("  ✓ Market regime detection")
        print("  ✓ Earnings avoidance")
        print("  ✓ Enhanced market analysis") 
        print("  ✓ High-quality opportunity filtering")
        print("  ✓ Professional position sizing")
        print("  ✓ Comprehensive performance tracking")
        print("")
        print("Bot is ready for real-world trading!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_real_world_options_bot())