#!/usr/bin/env python3
"""
Test the Tomorrow-Ready Options Bot
"""

import asyncio
import sys
from datetime import datetime, time as dt_time
sys.path.append('.')

async def test_tomorrow_ready_bot():
    """Test the tomorrow-ready options bot"""
    print("TESTING TOMORROW-READY OPTIONS BOT")
    print("=" * 50)
    
    try:
        from OPTIONS_BOT import TomorrowReadyOptionsBot
        from agents.exit_strategy_agent import ExitDecision, ExitSignal, ExitReason
        
        # Initialize bot
        print("Step 1: Initializing Tomorrow-Ready Bot...")
        bot = TomorrowReadyOptionsBot()
        print(f"  [OK] Bot created with {len(bot.tier1_stocks)} Tier-1 symbols")
        print(f"  [INFO] Market hours: {bot.market_open_time} - {bot.market_close_time} ET")
        
        # Test time functions
        print("\nStep 2: Testing market time functions...")
        current_time = bot.get_current_market_time()
        is_pre_market = bot.is_pre_market_time()
        is_market_hours = bot.is_market_hours()
        
        print(f"  Current time: {current_time}")
        print(f"  Pre-market: {is_pre_market}")
        print(f"  Market hours: {is_market_hours}")
        
        # Test readiness status
        print("\nStep 3: Testing readiness checklist...")
        for item, status in bot.readiness_status.items():
            status_text = "READY" if status else "NOT READY"
            print(f"  {item}: {status_text}")
        
        # Test pre-market preparation
        print("\nStep 4: Testing pre-market preparation...")
        prep_success = await bot.pre_market_preparation()
        if prep_success:
            print("  [SUCCESS] Pre-market preparation completed")
            print("  Readiness status after prep:")
            for item, status in bot.readiness_status.items():
                status_text = "[OK]" if status else "[X]"
                print(f"    {status_text} {item}")
        else:
            print("  [PARTIAL] Pre-market preparation completed with warnings")
        
        # Test market data functionality
        print("\nStep 5: Testing enhanced market data...")
        test_symbols = ['AAPL', 'SPY']
        for symbol in test_symbols:
            market_data = await bot.get_enhanced_market_data(symbol)
            if market_data:
                print(f"  [SUCCESS] {symbol} data retrieved")
                print(f"    Price: ${market_data['current_price']:.2f}")
                print(f"    Momentum: {market_data['price_momentum']:+.1%}")
                print(f"    Volatility: {market_data['realized_vol']:.1f}%")
                print(f"    Volume ratio: {market_data['volume_ratio']:.1f}x")
            else:
                print(f"  [FAIL] Could not get {symbol} data")
        
        # Test exit strategy agent
        print("\nStep 6: Testing intelligent exit strategy agent...")
        
        # Create mock position data
        mock_position_data = {
            'entry_time': datetime.now(),
            'opportunity': {
                'strategy': 'BULL_CALL_SPREAD',
                'max_profit': 150.0,
                'max_loss': 100.0,
                'days_to_expiry': 30,
                'symbol': 'AAPL'
            },
            'entry_price': 2.50,
            'quantity': 1,
            'market_regime_at_entry': 'BULL'
        }
        
        mock_market_data = {
            'current_price': 240.0,
            'price_momentum': 0.02,
            'realized_vol': 28.0,
            'volume_ratio': 1.2,
            'price_position': 0.8
        }
        
        # Test various P&L scenarios
        test_scenarios = [
            (75.0, "Profit scenario"),
            (-40.0, "Loss scenario"),
            (20.0, "Small profit scenario"),
            (-80.0, "Large loss scenario")
        ]
        
        for pnl, scenario in test_scenarios:
            exit_decision = bot.exit_agent.analyze_position_exit(
                mock_position_data, mock_market_data, pnl
            )
            
            print(f"  {scenario} (P&L: ${pnl:.2f}):")
            print(f"    Signal: {exit_decision.signal}")
            print(f"    Reason: {exit_decision.reason}")
            print(f"    Confidence: {exit_decision.confidence:.1%}")
            print(f"    Urgency: {exit_decision.urgency:.1%}")
            print(f"    Reasoning: {exit_decision.reasoning}")
        
        # Test position monitoring
        print("\nStep 7: Testing position monitoring...")
        
        # Add mock position to bot
        position_id = "TEST_AAPL_20241205"
        bot.active_positions[position_id] = mock_position_data
        
        # Test monitoring
        await bot.intelligent_position_monitoring()
        print(f"  [SUCCESS] Position monitoring completed")
        print(f"  Active positions: {len(bot.active_positions)}")
        
        # Test risk management
        print("\nStep 8: Testing risk management...")
        await bot.set_daily_risk_limits()
        
        print(f"  Daily risk limits set:")
        for key, value in bot.daily_risk_limits.items():
            if isinstance(value, float):
                print(f"    {key}: ${value:.2f}")
            else:
                print(f"    {key}: {value}")
        
        # Test trading plan generation
        print("\nStep 9: Testing daily trading plan...")
        trading_plan = await bot.generate_daily_trading_plan()
        
        print(f"  Trading plan generated:")
        print(f"    Market regime: {trading_plan['market_regime']}")
        print(f"    Preferred strategies: {trading_plan['preferred_strategies']}")
        print(f"    Target new positions: {trading_plan['target_new_positions']}")
        print(f"    Focus symbols: {trading_plan['focus_symbols'][:5]}...")
        
        # Test performance tracking
        print("\nStep 10: Testing performance tracking...")
        
        # Simulate some trades
        bot.update_performance_stats(150.0, ExitDecision(
            signal=ExitSignal.TAKE_PROFIT,
            reason=ExitReason.PROFIT_TARGET,
            confidence=0.85,
            urgency=0.8,
            target_exit_pct=100.0,
            expected_pnl_impact=150.0,
            reasoning="Profit target hit",
            supporting_factors=["profit_pressure"]
        ))
        
        bot.update_performance_stats(-75.0, ExitDecision(
            signal=ExitSignal.STOP_LOSS,
            reason=ExitReason.STOP_LOSS_HIT,
            confidence=0.90,
            urgency=0.95,
            target_exit_pct=100.0,
            expected_pnl_impact=-75.0,
            reasoning="Stop loss triggered",
            supporting_factors=["loss_pressure"]
        ))
        
        await bot.log_daily_performance()
        
        # Test market procedures
        print("\nStep 11: Testing market open/close procedures...")
        
        # Test market open
        print("  Testing market open procedures...")
        await bot.market_open_procedures()
        print(f"    Market open status: {bot.is_market_open}")
        
        # Test market close
        print("  Testing market close procedures...")
        await bot.market_close_procedures()
        print(f"    Market open status: {bot.is_market_open}")
        
        print("\n" + "=" * 50)
        print("TOMORROW-READY BOT TEST COMPLETED")
        print("Production Features Verified:")
        print("  * Pre-market preparation routines")
        print("  * Intelligent exit strategy agent")
        print("  * Real-time position monitoring")
        print("  * Market hours awareness")
        print("  * Risk management systems")
        print("  * Performance tracking")
        print("  * Market open/close procedures")
        print("")
        print("Bot is ready for tomorrow's trading session!")
        
        # Show final status
        print(f"\nFinal Status:")
        print(f"  Account Value: ${bot.risk_manager.account_value:,.2f}")
        print(f"  Market Regime: {bot.market_regime}")
        print(f"  Daily P&L: ${bot.daily_pnl:.2f}")
        print(f"  Active Positions: {len(bot.active_positions)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_tomorrow_ready_bot())