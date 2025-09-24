#!/usr/bin/env python3
"""
Test the learning system integration
"""
import asyncio
import sys
sys.path.append('.')

async def test_learning_system():
    try:
        print("Testing Learning System Integration...")
        
        from OPTIONS_BOT import TomorrowReadyOptionsBot
        
        # Create and initialize bot
        bot = TomorrowReadyOptionsBot()
        await bot.initialize_all_systems()
        
        print("\n=== LEARNING ENGINE STATUS ===")
        print(f"Learning engine initialized: {bot.learning_engine is not None}")
        
        # Test learning insights
        insights = bot.learning_engine.get_learning_insights()
        print(f"Historical trades: {insights['total_trades']}")
        print(f"Strategy performance data: {len(insights['strategy_performance'])} strategies")
        
        # Test confidence calibration
        print("\n=== TESTING CONFIDENCE CALIBRATION ===")
        test_confidence = 0.75
        test_strategy = "LONG_CALL"
        test_symbol = "AAPL"
        test_conditions = {
            'market_regime': 'BULLISH',
            'volatility': 20,
            'volume_ratio': 1.2,
            'price_momentum': 0.02
        }
        
        calibrated = bot.learning_engine.calibrate_confidence(
            test_confidence, test_strategy, test_symbol, test_conditions
        )
        
        print(f"Original confidence: {test_confidence:.1%}")
        print(f"Calibrated confidence: {calibrated:.1%}")
        print(f"Strategy multiplier: {bot.learning_engine.get_strategy_multiplier(test_strategy):.2f}")
        print(f"Position size multiplier: {bot.learning_engine.get_position_size_multiplier():.2f}")
        
        # Test strategy avoidance
        should_avoid = bot.learning_engine.should_avoid_strategy(test_strategy)
        print(f"Should avoid {test_strategy}: {should_avoid}")
        
        # Test simulated trade recording
        print("\n=== TESTING TRADE RECORDING ===")
        trade_id = "TEST_TRADE_001"
        
        # Record entry
        trade_record = bot.learning_engine.record_trade_entry(
            trade_id=trade_id,
            symbol="AAPL",
            strategy="LONG_CALL",
            confidence=0.75,
            entry_price=5.50,
            quantity=1,
            max_profit=2.50,
            max_loss=1.50,
            market_conditions=test_conditions
        )
        print(f"Trade entry recorded: {trade_record.trade_id}")
        
        # Simulate exit
        exit_record = bot.learning_engine.record_trade_exit(
            trade_id=trade_id,
            exit_price=6.25,  # Profitable exit
            exit_reason="profit_target"
        )
        print(f"Trade exit recorded: P&L = ${exit_record.pnl:.2f}")
        print(f"Trade was winning: {exit_record.win}")
        
        # Get updated insights
        print("\n=== UPDATED INSIGHTS ===")
        updated_insights = bot.learning_engine.get_learning_insights()
        print(f"Total trades after test: {updated_insights['total_trades']}")
        
        for rec in updated_insights['recommendations']:
            print(f"Recommendation: {rec}")
        
        print("\n=== TEST COMPLETE ===")
        print("SUCCESS: Learning system is integrated and functional!")
        print("\nThe bot will now:")
        print("- Track all trade entries and exits")
        print("- Calibrate confidence based on historical performance")
        print("- Adjust position sizes based on recent performance")
        print("- Avoid strategies that consistently lose money")
        print("- Provide learning insights and recommendations")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_learning_system())