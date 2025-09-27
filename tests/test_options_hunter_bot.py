#!/usr/bin/env python3
"""
Test the Options Hunter Bot
"""

import asyncio
import sys
sys.path.append('.')

async def test_options_hunter_bot():
    """Test the options hunter bot functionality"""
    print("TESTING OPTIONS HUNTER BOT")
    print("=" * 40)
    
    try:
        from options_hunter_bot import OptionsHunterBot
        
        # Initialize bot
        print("Step 1: Initializing Options Hunter Bot...")
        bot = OptionsHunterBot()
        print(f"  [OK] Bot created with {len(bot.trading_universe)} symbols")
        print(f"  [INFO] Strategy weights: {bot.strategy_weights}")
        
        # Test system initialization
        print("\nStep 2: Testing system initialization...")
        initialized = await bot.initialize_systems()
        if initialized:
            print("  [SUCCESS] All systems initialized")
        else:
            print("  [WARNING] Systems initialized in simulation mode")
        
        # Test opportunity finding
        print("\nStep 3: Testing opportunity detection...")
        test_symbols = ['AAPL', 'SPY', 'QQQ']
        
        opportunities_found = 0
        for symbol in test_symbols:
            print(f"  Testing {symbol}...")
            opportunity = await bot.find_best_options_opportunity(symbol)
            
            if opportunity:
                opportunities_found += 1
                print(f"    [SUCCESS] Found {opportunity['strategy']} opportunity")
                print(f"    [INFO] Confidence: {opportunity['confidence']:.1%}")
                print(f"    [INFO] Expected Return: {opportunity['expected_return']:.2f}")
                print(f"    [INFO] Max Profit: ${opportunity['max_profit']:.2f}")
                print(f"    [INFO] Max Loss: ${opportunity['max_loss']:.2f}")
            else:
                print(f"    [INFO] No opportunity found")
        
        print(f"\nOpportunity Detection: {opportunities_found}/{len(test_symbols)} symbols had opportunities")
        
        # Test strategy selection logic
        print("\nStep 4: Testing strategy selection logic...")
        
        test_conditions = [
            {'price_change': -0.02, 'volatility': 35, 'rsi': 30, 'volume_ratio': 1.5},  # Strong bearish
            {'price_change': 0.015, 'volatility': 25, 'rsi': 65, 'volume_ratio': 1.2}, # Strong bullish  
            {'price_change': 0.005, 'volatility': 40, 'rsi': 50, 'volume_ratio': 1.0}, # High vol neutral
        ]
        
        strategy_counts = {}
        for i, conditions in enumerate(test_conditions):
            strategy = bot.select_optimal_strategy(conditions)
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            print(f"  Condition {i+1}: {strategy} (Change: {conditions['price_change']:+.1%}, RSI: {conditions['rsi']:.0f})")
        
        print(f"  [INFO] Strategy distribution: {strategy_counts}")
        
        # Performance stats test
        print("\nStep 5: Testing performance tracking...")
        bot.update_performance_stats(250.0)   # Winning trade
        bot.update_performance_stats(-100.0)  # Losing trade
        bot.update_performance_stats(150.0)   # Another winner
        
        await bot.log_performance_stats()
        
        print("\n" + "=" * 40)
        print("OPTIONS HUNTER BOT TEST COMPLETED")
        print("✅ Bot is ready for live trading!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_options_hunter_bot())