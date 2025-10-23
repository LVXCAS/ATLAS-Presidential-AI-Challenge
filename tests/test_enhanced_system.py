#!/usr/bin/env python3
"""
Test script for the Enhanced Trading System
Validates options trading, position management, and risk controls
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

from agents.options_trading_agent import OptionsTrader, OptionsStrategy
from agents.position_manager import PositionManager, PositionType, ExitRule, ExitReason
from agents.risk_management import RiskManager, RiskLevel

async def test_options_trader():
    """Test the options trading functionality"""
    print("=" * 50)
    print("TESTING OPTIONS TRADER")
    print("=" * 50)
    
    trader = OptionsTrader(None)  # No broker for testing
    
    # Test 1: Get options chain for a popular stock
    print("Test 1: Getting options chain for AAPL...")
    try:
        contracts = await trader.get_options_chain('AAPL')
        if contracts:
            print(f"[OK] Found {len(contracts)} liquid options contracts")
            print(f"Sample contract: {contracts[0].symbol} - Strike: ${contracts[0].strike:.2f}, Bid: ${contracts[0].bid:.2f}")
        else:
            print("[WARN] No options contracts found")
    except Exception as e:
        print(f"[ERROR] Error getting options chain: {e}")
    
    # Test 2: Find best strategy for different market conditions
    print("\nTest 2: Finding best options strategies...")
    test_conditions = [
        {"price": 150.0, "volatility": 25.0, "rsi": 35.0, "price_change": 0.04, "scenario": "Bullish momentum"},
        {"price": 150.0, "volatility": 30.0, "rsi": 65.0, "price_change": -0.035, "scenario": "Bearish momentum"},
        {"price": 150.0, "volatility": 35.0, "rsi": 50.0, "price_change": 0.005, "scenario": "High volatility, neutral"},
    ]
    
    for condition in test_conditions:
        print(f"\n{condition['scenario']}:")
        try:
            result = trader.find_best_options_strategy('AAPL', **{k: v for k, v in condition.items() if k != 'scenario'})
            if result:
                strategy, contracts = result
                print(f"  [OK] Recommended strategy: {strategy}")
                print(f"  [OK] Using {len(contracts)} contracts")
            else:
                print("  [INFO] No strategy recommended for these conditions")
        except Exception as e:
            print(f"  [ERROR] Error finding strategy: {e}")
    
    print("\nOptions trader tests completed.\n")

def test_position_manager():
    """Test the position management system"""
    print("=" * 50)
    print("TESTING POSITION MANAGER")
    print("=" * 50)
    
    manager = PositionManager(None)  # No broker for testing
    
    # Test 1: Open a position with default exit rules
    print("Test 1: Opening stock positions...")
    try:
        position_id_1 = asyncio.run(manager.open_stock_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            position_type=PositionType.STOCK_LONG,
            strategy_source="test",
            confidence=0.75
        ))
        print(f"[OK] Opened long position: {position_id_1}")
        
        position_id_2 = asyncio.run(manager.open_stock_position(
            symbol='GOOGL',
            quantity=-50,  # Short position
            entry_price=2800.0,
            position_type=PositionType.STOCK_SHORT,
            confidence=0.6
        ))
        print(f"[OK] Opened short position: {position_id_2}")
        
    except Exception as e:
        print(f"[ERROR] Error opening positions: {e}")
        return
    
    # Test 2: Update prices and check exit conditions
    print("\nTest 2: Testing exit conditions...")
    try:
        # Update prices
        price_updates = {'AAPL': 160.0, 'GOOGL': 2750.0}  # AAPL up 6.7%, GOOGL down 1.8%
        asyncio.run(manager.update_position_prices(price_updates))
        
        # Check positions
        for pos_id, position in manager.stock_positions.items():
            should_exit, exit_reason, detail = position.should_exit()
            print(f"  Position {position.symbol}: P&L {position.unrealized_pnl:.2f} ({position.unrealized_pnl_percent:.1%})")
            print(f"    Exit needed: {should_exit} - {exit_reason if exit_reason else 'No'} - {detail}")
        
    except Exception as e:
        print(f"[X] Error testing exit conditions: {e}")
    
    # Test 3: Portfolio summary
    print("\nTest 3: Portfolio summary...")
    try:
        summary = manager.get_portfolio_summary()
        print(f"[OK] Total positions: {summary['stock_positions']['total_positions']}")
        print(f"[OK] Total unrealized P&L: ${summary['stock_positions']['total_unrealized_pnl']:.2f}")
        print(f"[OK] Performance: {summary['performance']['total_trades']} trades, "
              f"{summary['performance']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"[X] Error getting portfolio summary: {e}")
    
    print("\nPosition manager tests completed.\n")

def test_risk_manager():
    """Test the risk management system"""
    print("=" * 50)
    print("TESTING RISK MANAGER")
    print("=" * 50)
    
    # Test different risk levels
    risk_levels = [RiskLevel.CONSERVATIVE, RiskLevel.MODERATE, RiskLevel.AGGRESSIVE]
    
    for risk_level in risk_levels:
        print(f"\nTesting {risk_level} risk level:")
        manager = RiskManager(risk_level)
        manager.update_account_value(100000)
        
        # Test 1: Position sizing
        test_stocks = [
            {'symbol': 'AAPL', 'price': 150.0, 'confidence': 0.8, 'volatility': 0.25},
            {'symbol': 'TSLA', 'price': 800.0, 'confidence': 0.6, 'volatility': 0.45},
            {'symbol': 'SPY', 'price': 400.0, 'confidence': 0.9, 'volatility': 0.15},
        ]
        
        print("  Position sizing tests:")
        for stock in test_stocks:
            try:
                quantity, risk_assessment = manager.calculate_position_size(**stock)
                print(f"    {stock['symbol']}: {quantity} shares (${quantity * stock['price']:,.0f})")
                print(f"      Risk Score: {risk_assessment.overall_risk_score:.2f}, "
                      f"Recommendation: {risk_assessment.recommendation}")
            except Exception as e:
                print(f"    [X] Error sizing {stock['symbol']}: {e}")
        
        # Test 2: Risk limits
        print(f"  Risk limits:")
        print(f"    Max position size: {manager.risk_limits.max_position_size_pct}%")
        print(f"    Max portfolio heat: {manager.risk_limits.max_portfolio_heat_pct}%")
        print(f"    Max drawdown: {manager.risk_limits.max_drawdown_pct}%")
    
    print("\nRisk manager tests completed.\n")

async def test_integrated_system():
    """Test the integrated trading system"""
    print("=" * 50)
    print("TESTING INTEGRATED SYSTEM")
    print("=" * 50)
    
    # Create integrated system
    risk_manager = RiskManager(RiskLevel.MODERATE)
    risk_manager.update_account_value(100000)
    position_manager = PositionManager(None)
    
    # Simulate a trading scenario
    print("Simulating trading scenario...")
    
    # Test stock trade
    try:
        position_id = await position_manager.execute_trade_with_management(
            symbol='AAPL',
            signal='BUY',
            quantity=50,
            price=150.0,
            confidence=0.75,
            strategy_source='integrated_test'
        )
        
        if position_id:
            print(f"[OK] Executed managed trade: {position_id}")
            
            # Simulate price movement and monitoring
            await position_manager.update_position_prices({'AAPL': 165.0})  # 10% gain
            actions = await position_manager.monitor_positions()
            
            if actions:
                print(f"[OK] Position monitoring triggered {len(actions)} actions")
                for action in actions:
                    print(f"    {action['action']}: {action['symbol']} - {action['reason']}")
            else:
                print("  No exit actions needed yet")
                
        else:
            print("[X] Failed to execute managed trade")
            
    except Exception as e:
        print(f"[X] Error in integrated system test: {e}")
    
    # Test options trade simulation
    print("\nTesting options integration...")
    try:
        options_id = await position_manager.execute_options_trade(
            symbol='AAPL',
            price=150.0,
            volatility=25.0,
            rsi=45.0,
            price_change=0.03
        )
        
        if options_id:
            print(f"[OK] Executed options strategy: {options_id}")
        else:
            print("  No suitable options strategy found")
            
    except Exception as e:
        print(f"  Error in options test: {e}")
    
    print("\nIntegrated system tests completed.\n")

def test_configuration():
    """Test system configuration and imports"""
    print("=" * 50)
    print("TESTING SYSTEM CONFIGURATION")
    print("=" * 50)
    
    # Test imports
    try:
        from agents.options_trading_agent import OptionsTrader
        from agents.position_manager import PositionManager
        from agents.risk_management import RiskManager
        from agents.broker_integration import AlpacaBrokerIntegration
        print("[OK] All modules imported successfully")
    except Exception as e:
        print(f"[X] Import error: {e}")
        return False
    
    # Test basic instantiation
    try:
        options_trader = OptionsTrader(None)
        position_manager = PositionManager(None)
        risk_manager = RiskManager()
        print("[OK] All components instantiated successfully")
    except Exception as e:
        print(f"[X] Instantiation error: {e}")
        return False
    
    print("[OK] System configuration tests passed\n")
    return True

async def main():
    """Run all tests"""
    print("ENHANCED TRADING SYSTEM TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Configuration tests
    if not test_configuration():
        print("Configuration tests failed - stopping")
        return
    
    # Component tests
    test_risk_manager()
    test_position_manager()
    await test_options_trader()
    await test_integrated_system()
    
    print("=" * 60)
    print("TEST SUITE COMPLETED")
    print("All enhanced trading system components tested!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())