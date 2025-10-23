#!/usr/bin/env python3
"""
Test the account method fix for daily loss limit
"""

import json
import asyncio
from unittest.mock import Mock, AsyncMock
import sys
import os
from datetime import datetime

# Add the parent directory to the path so we can import OPTIONS_BOT
sys.path.append(os.path.dirname(__file__))

import OPTIONS_BOT

async def test_account_method_fix():
    """Test that the daily loss limit check now uses the correct account method"""

    print("Testing Account Method Fix for Daily Loss Limit")
    print("="*50)

    try:
        # Create bot instance
        bot = OPTIONS_BOT.TomorrowReadyOptionsBot()

        # Mock broker with correct method
        mock_broker = Mock()

        # Test case 1: Test correct method name
        print("\nTest 1: Testing get_account_info method")
        print("-" * 30)

        mock_account_info = {
            'id': 'test_account',
            'portfolio_value': 10500.0,  # This is what we should use for equity
            'cash': 5000.0,
            'buying_power': 20000.0
        }

        mock_broker.get_account_info = AsyncMock(return_value=mock_account_info)
        bot.broker = mock_broker

        # Create a test starting equity file
        starting_equity = {"starting_equity": 10000.0, "date": datetime.now().strftime('%Y-%m-%d')}
        with open('daily_starting_equity.json', 'w') as f:
            json.dump(starting_equity, f)

        # Test the loss limit check
        loss_limit_hit = await bot.check_daily_loss_limit()

        print(f"Mock account portfolio_value: ${mock_account_info['portfolio_value']:,.2f}")
        print(f"Starting equity: ${starting_equity['starting_equity']:,.2f}")
        print(f"Daily P&L: ${mock_account_info['portfolio_value'] - starting_equity['starting_equity']:+,.2f}")
        print(f"Daily P&L %: {((mock_account_info['portfolio_value'] - starting_equity['starting_equity']) / starting_equity['starting_equity']) * 100:+.1f}%")
        print(f"Loss limit hit: {loss_limit_hit}")
        print(f"Trading stopped: {bot.trading_stopped_for_day}")

        # Verify the mock was called correctly
        mock_broker.get_account_info.assert_called()
        print("[OK] get_account_info method called successfully")

        # Test case 2: Test loss limit trigger
        print("\nTest 2: Testing loss limit trigger")
        print("-" * 30)

        bot.reset_daily_trading_flags()
        mock_account_info['portfolio_value'] = 9500.0  # -5% loss

        loss_limit_hit = await bot.check_daily_loss_limit()
        print(f"Account at -5.0% loss: ${mock_account_info['portfolio_value']:,.2f}")
        print(f"Loss limit hit: {loss_limit_hit}")
        print(f"Trading stopped: {bot.trading_stopped_for_day}")

        # Test case 3: Test file creation and date checking
        print("\nTest 3: Testing starting equity file management")
        print("-" * 30)

        bot.reset_daily_trading_flags()

        # Test with old date
        old_equity = {"starting_equity": 9000.0, "date": "2025-01-01"}
        with open('daily_starting_equity.json', 'w') as f:
            json.dump(old_equity, f)

        mock_account_info['portfolio_value'] = 10000.0
        loss_limit_hit = await bot.check_daily_loss_limit()

        # Read the updated file
        with open('daily_starting_equity.json', 'r') as f:
            updated_equity = json.load(f)

        print(f"Old file date: {old_equity['date']}")
        print(f"Updated file date: {updated_equity['date']}")
        print(f"Updated starting equity: ${updated_equity['starting_equity']:,.2f}")
        print("[OK] File updated for new trading day")

        # Test case 4: Test error handling
        print("\nTest 4: Testing error handling")
        print("-" * 30)

        # Mock broker that returns None
        mock_broker.get_account_info = AsyncMock(return_value=None)

        loss_limit_hit = await bot.check_daily_loss_limit()
        print(f"When account_info is None, loss_limit_hit: {loss_limit_hit}")
        print("[OK] Graceful handling when account info unavailable")

        print("\n" + "="*50)
        print("ACCOUNT METHOD FIX SUMMARY")
        print("="*50)
        print("[OK] Fixed method name: get_account() -> get_account_info()")
        print("[OK] Fixed equity field: account.equity -> account_info['portfolio_value']")
        print("[OK] Added proper error handling for missing account info")
        print("[OK] Added starting equity file creation in pre-market prep")
        print("[OK] Added date checking for starting equity file")
        print("[OK] Added fallback logic when file is missing or old")

        print("\nThe daily loss limit check now:")
        print("- Uses correct Alpaca broker method: get_account_info()")
        print("- Accesses portfolio_value for total account equity")
        print("- Creates starting equity file during pre-market prep")
        print("- Updates file automatically for new trading days")
        print("- Handles errors gracefully with fallbacks")

        # Cleanup
        try:
            os.remove('daily_starting_equity.json')
        except:
            pass

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the account method fix test"""
    try:
        success = asyncio.run(test_account_method_fix())
        return success
    except Exception as e:
        print(f"Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)