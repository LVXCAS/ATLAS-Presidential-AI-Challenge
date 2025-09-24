#!/usr/bin/env python3
"""
Debug why OPTIONS_BOT isn't making trades
"""
import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

sys.path.append('.')
load_dotenv('.env')

async def debug_options_bot():
    """Debug the complete OPTIONS_BOT workflow"""
    
    print("DEBUGGING OPTIONS_BOT TRADING ISSUES")
    print("=" * 45)
    
    try:
        # Test 1: Check if components can be imported
        print("1. Testing component imports...")
        
        from agents.broker_integration import AlpacaBrokerIntegration
        from agents.options_trading_agent import OptionsTrader, OptionsStrategy
        from agents.options_broker import OptionsBroker
        print("   SUCCESS: All components imported")
        
        # Test 2: Check broker connectivity
        print("\n2. Testing broker connectivity...")
        broker_integration = AlpacaBrokerIntegration()
        account_info = await broker_integration.get_account_info()
        
        if account_info:
            print(f"   SUCCESS: Connected to account {account_info.get('id')}")
            print(f"   Buying Power: ${float(account_info.get('buying_power', 0)):,.2f}")
            print(f"   Trading Blocked: {account_info.get('trading_blocked', False)}")
        else:
            print("   ERROR: Could not get account info")
            return
        
        # Test 3: Check options broker
        print("\n3. Testing options broker...")
        options_broker = OptionsBroker(paper_trading=True)
        print("   SUCCESS: Options broker initialized")
        
        # Test 4: Check if we can get options data
        print("\n4. Testing options data retrieval...")
        options_trader = OptionsTrader(options_broker)
        
        # Try to get options for a popular symbol
        test_symbols = ['SPY', 'QQQ', 'IWM']
        
        for symbol in test_symbols:
            try:
                print(f"   Testing {symbol} options chain...")
                options_chain = await options_trader.get_options_chain(symbol)
                
                if options_chain:
                    print(f"   SUCCESS: Found {len(options_chain)} options for {symbol}")
                    
                    # Show sample option
                    sample_option = options_chain[0]
                    print(f"   Sample: {sample_option.symbol} - Strike: ${sample_option.strike} - Exp: {sample_option.expiration}")
                    break
                else:
                    print(f"   No options found for {symbol}")
            except Exception as e:
                print(f"   ERROR getting {symbol} options: {e}")
                continue
        
        # Test 5: Check if we can find trading opportunities
        print("\n5. Testing opportunity detection...")
        
        try:
            # Simulate what the bot does to find opportunities
            symbol = 'SPY'
            current_price = 550.0  # Approximate SPY price
            volatility = 0.20
            
            strategy = options_trader.find_best_options_strategy(
                symbol=symbol,
                price=current_price,
                volatility=volatility,
                market_sentiment=0.6,  # Bullish
                options_flow={"call_volume": 1000, "put_volume": 500}
            )
            
            if strategy:
                print(f"   SUCCESS: Found strategy - {strategy}")
            else:
                print("   No suitable strategy found")
                
        except Exception as e:
            print(f"   ERROR in strategy detection: {e}")
        
        # Test 6: Test a manual trade execution
        print("\n6. Testing manual trade execution...")
        
        try:
            from agents.options_broker import OptionsOrderRequest, OptionsOrderType, OrderSide
            
            # Create a test order
            test_order = OptionsOrderRequest(
                symbol="SPY250912C00550000",
                underlying="SPY",
                qty=1,
                side=OrderSide.BUY,
                type=OptionsOrderType.LIMIT,
                limit_price=0.01,
                client_order_id="debug_test"
            )
            
            print(f"   Placing test order: {test_order.symbol}")
            response = await options_broker.submit_options_order(test_order)
            
            print(f"   Order ID: {response.id}")
            print(f"   Status: {response.status}")
            
            # Check if it's a real order
            if len(response.id) > 20 and not response.id.startswith('SIM_'):
                print("   SUCCESS: Real order executed!")
            else:
                print("   WARNING: Simulation order used")
                
        except Exception as e:
            print(f"   ERROR in manual trade: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 7: Check market hours and timing
        print("\n7. Testing market timing...")
        
        from datetime import time
        import pytz
        
        et_timezone = pytz.timezone('America/New_York')
        now_et = datetime.now(et_timezone)
        current_time = now_et.time()
        
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        print(f"   Current ET time: {current_time}")
        print(f"   Market hours: {market_open} - {market_close}")
        
        is_market_hours = market_open <= current_time <= market_close
        is_weekday = now_et.weekday() < 5
        
        print(f"   Is market hours: {is_market_hours}")
        print(f"   Is weekday: {is_weekday}")
        
        if not (is_market_hours and is_weekday):
            print("   NOTE: Market is currently closed - bot may not trade")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

async def check_bot_configuration():
    """Check if the bot configuration might be preventing trades"""
    
    print(f"\n\nCHECKING BOT CONFIGURATION")
    print("=" * 30)
    
    # Check environment variables
    required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL']
    
    print("Environment variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"   {var}: {'*' * 10} (set)")
        else:
            print(f"   {var}: NOT SET")
    
    # Check trading settings
    paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    print(f"   PAPER_TRADING: {paper_trading}")
    
    initial_capital = os.getenv('INITIAL_CAPITAL', '100000')
    print(f"   INITIAL_CAPITAL: ${initial_capital}")

async def main():
    print("OPTIONS_BOT DEBUGGING SUITE")
    print("=" * 60)
    
    await debug_options_bot()
    await check_bot_configuration()
    
    print(f"\n{'='*60}")
    print("DEBUGGING COMPLETE")
    print("\nLook for any ERROR messages above to identify issues.")
    print("Common problems:")
    print("- Market is closed (bot won't trade)")
    print("- No suitable opportunities found")
    print("- Broker connectivity issues")
    print("- Options data not available")

if __name__ == "__main__":
    asyncio.run(main())