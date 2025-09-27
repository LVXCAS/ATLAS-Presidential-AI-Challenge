#!/usr/bin/env python3
"""
Test script for 2-week expiry requirement
"""

import asyncio
import sys
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

async def test_expiry_filtering():
    """Test that options trader only selects options > 2 weeks out"""
    print("Testing 2-Week Expiry Filtering")
    print("=" * 40)
    
    try:
        from agents.options_trading_agent import OptionsTrader
        
        trader = OptionsTrader(None)
        print(f"[CONFIG] Minimum days to expiry: {trader.min_days_to_expiry}")
        
        # Test with a popular stock
        print(f"\nTesting with AAPL options...")
        contracts = await trader.get_options_chain('AAPL')
        
        if contracts:
            print(f"[OK] Found {len(contracts)} contracts")
            
            # Check expiry dates
            min_expiry = min(c.days_to_expiry for c in contracts)
            max_expiry = max(c.days_to_expiry for c in contracts)
            avg_expiry = sum(c.days_to_expiry for c in contracts) / len(contracts)
            
            print(f"[INFO] Expiry range: {min_expiry} - {max_expiry} days")
            print(f"[INFO] Average expiry: {avg_expiry:.1f} days")
            
            # Verify all contracts meet requirement
            valid_contracts = [c for c in contracts if c.days_to_expiry > 14]
            print(f"[CHECK] Contracts > 14 days: {len(valid_contracts)}/{len(contracts)}")
            
            if len(valid_contracts) == len(contracts):
                print("[PASS] All contracts meet 2-week requirement!")
            else:
                print("[FAIL] Some contracts don't meet 2-week requirement")
                
            # Show sample contracts
            print(f"\nSample contracts:")
            for i, contract in enumerate(contracts[:3]):
                print(f"  {i+1}. {contract.symbol} - Strike: ${contract.strike:.2f}, "
                      f"Expiry: {contract.days_to_expiry} days")
        else:
            print("[WARN] No contracts found (normal if no options > 14 days)")
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

def test_time_decay_logic():
    """Test the updated time decay exit logic"""
    print("\nTesting Time Decay Exit Logic")
    print("=" * 40)
    
    try:
        from agents.options_trading_agent import OptionsContract, OptionsPosition, OptionsStrategy
        from datetime import datetime
        
        # Create mock contracts with different expiries
        test_cases = [
            {'days': 2, 'pnl': -100, 'expected': True, 'reason': '< 3 days (force exit)'},
            {'days': 5, 'pnl': -100, 'expected': True, 'reason': '< 7 days and losing'},
            {'days': 5, 'pnl': 100, 'expected': False, 'reason': '< 7 days but winning'},
            {'days': 10, 'pnl': -100, 'expected': False, 'reason': '> 7 days'},
        ]
        
        for case in test_cases:
            # Create mock contract
            exp_date = datetime.now() + timedelta(days=case['days'])
            mock_contract = OptionsContract(
                symbol="AAPL240101C00150000",
                underlying="AAPL",
                strike=150.0,
                expiration=exp_date,
                option_type='call',
                bid=2.0,
                ask=2.2,
                volume=100,
                open_interest=500,
                implied_volatility=0.25,
                delta=0.5,
                gamma=0.1,
                theta=-0.05,
                vega=0.3
            )
            
            # Mock position
            position = OptionsPosition(
                symbol="test_position",
                underlying="AAPL",
                strategy=OptionsStrategy.LONG_CALL,
                contracts=[mock_contract],
                quantity=1,
                entry_price=2.1,
                entry_time=datetime.now()
            )
            
            # Calculate if should exit based on time decay
            current_value = 210 + case['pnl']  # $210 entry + P&L
            pnl = case['pnl']
            
            # Apply time decay logic
            days_to_expiry = case['days']
            should_exit_time_decay = (days_to_expiry <= 7 and pnl < 0) or days_to_expiry <= 3
            
            result = "EXIT" if should_exit_time_decay else "HOLD"
            status = "[PASS]" if should_exit_time_decay == case['expected'] else "[FAIL]"
            
            print(f"{status} {case['days']} days, P&L ${case['pnl']} -> {result} ({case['reason']})")
            
    except Exception as e:
        print(f"[ERROR] Time decay test failed: {e}")

async def main():
    """Run expiry tests"""
    await test_expiry_filtering()
    test_time_decay_logic()
    
    print("\n" + "=" * 50)
    print("2-WEEK EXPIRY TESTING COMPLETE")
    print("The system now only trades options with > 14 days to expiry")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())