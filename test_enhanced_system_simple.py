#!/usr/bin/env python3
"""
Simple test script for the Enhanced Trading System
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    try:
        from agents.options_trading_agent import OptionsTrader
        from agents.position_manager import PositionManager
        from agents.risk_management import RiskManager
        print("[OK] All modules imported successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_risk_manager():
    """Test risk management functionality"""
    print("\nTesting Risk Manager...")
    try:
        from agents.risk_management import RiskManager, RiskLevel
        
        manager = RiskManager(RiskLevel.MODERATE)
        manager.update_account_value(100000)
        
        # Test position sizing
        quantity, risk_assessment = manager.calculate_position_size(
            'AAPL', 150.0, 0.75, 0.25
        )
        
        print(f"[OK] Position sizing: {quantity} shares")
        print(f"[OK] Risk score: {risk_assessment.overall_risk_score:.2f}")
        print(f"[OK] Recommendation: {risk_assessment.recommendation}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Risk manager test failed: {e}")
        return False

async def test_position_manager():
    """Test position management"""
    print("\nTesting Position Manager...")
    try:
        from agents.position_manager import PositionManager, PositionType
        
        manager = PositionManager(None)
        
        # Open a position
        position_id = await manager.open_stock_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            position_type=PositionType.STOCK_LONG,
            confidence=0.75
        )
        
        print(f"[OK] Opened position: {position_id}")
        
        # Test price update
        await manager.update_position_prices({'AAPL': 160.0})
        
        # Get portfolio summary
        summary = manager.get_portfolio_summary()
        print(f"[OK] Portfolio has {summary['stock_positions']['total_positions']} positions")
        
        return True
    except Exception as e:
        print(f"[ERROR] Position manager test failed: {e}")
        return False

async def test_options_trader():
    """Test options trading"""
    print("\nTesting Options Trader...")
    try:
        from agents.options_trading_agent import OptionsTrader
        
        trader = OptionsTrader(None)
        
        # Test finding strategy
        result = trader.find_best_options_strategy('AAPL', 150.0, 25.0, 45.0, 0.03)
        
        if result:
            strategy, contracts = result
            print(f"[OK] Found strategy: {strategy}")
        else:
            print("[INFO] No strategy found (normal for test conditions)")
        
        return True
    except Exception as e:
        print(f"[ERROR] Options trader test failed: {e}")
        return False

async def main():
    """Run simplified tests"""
    print("ENHANCED TRADING SYSTEM - SIMPLE TEST")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        return
    
    # Test components
    if not test_risk_manager():
        return
        
    if not await test_position_manager():
        return
        
    if not await test_options_trader():
        return
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("Enhanced trading system is ready to use.")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())