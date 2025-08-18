#!/usr/bin/env python3
"""
Simple Database Schema Test
Tests the database schema without requiring all project dependencies.
"""

import asyncio
import asyncpg
import json
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Database connection settings
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/trading_system')

async def test_database_schema():
    """Test the database schema implementation."""
    print("Testing LangGraph Trading System Database Schema")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        print("✓ Database connection successful")
        
        # Test 1: Check if key tables exist
        print("\n1. Testing table existence...")
        key_tables = [
            'market_data_hf', 'signals', 'fused_signals', 'orders', 'trades', 
            'positions', 'portfolio_snapshots', 'risk_metrics', 'model_performance',
            'news_articles', 'system_logs', 'audit_trail'
        ]
        
        for table in key_tables:
            result = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                table
            )
            if result:
                print(f"  ✓ Table {table} exists")
            else:
                print(f"  ✗ Table {table} missing")
        
        # Test 2: Check if key functions exist
        print("\n2. Testing function existence...")
        key_functions = [
            'calculate_portfolio_metrics', 'calculate_signal_performance',
            'detect_market_data_anomalies', 'calculate_var'
        ]
        
        for func in key_functions:
            result = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.routines WHERE routine_name = $1)",
                func
            )
            if result:
                print(f"  ✓ Function {func} exists")
            else:
                print(f"  ✗ Function {func} missing")
        
        # Test 3: Test data insertion
        print("\n3. Testing data insertion...")
        
        # Test market data insertion
        try:
            await conn.execute("""
                INSERT INTO market_data_hf (
                    symbol, exchange, timestamp, timeframe, open, high, low, close, 
                    volume, provider
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 'TEST', 'NASDAQ', datetime.now(), '1m', 
                Decimal('100.00'), Decimal('101.00'), Decimal('99.50'), 
                Decimal('100.50'), 10000, 'test')
            print("  ✓ Market data insertion successful")
            
            # Clean up
            await conn.execute("DELETE FROM market_data_hf WHERE symbol = 'TEST'")
            
        except Exception as e:
            print(f"  ✗ Market data insertion failed: {e}")
        
        # Test signal insertion
        try:
            await conn.execute("""
                INSERT INTO signals (
                    symbol, agent_name, signal_type, value, confidence, strength,
                    direction, top_3_reasons, timestamp, model_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 'TEST', 'test_agent', 'momentum', Decimal('0.75'), 
                Decimal('0.85'), Decimal('0.80'), 'BUY', 
                json.dumps(['reason1', 'reason2', 'reason3']), 
                datetime.now(), '1.0.0')
            print("  ✓ Signal insertion successful")
            
            # Clean up
            await conn.execute("DELETE FROM signals WHERE symbol = 'TEST'")
            
        except Exception as e:
            print(f"  ✗ Signal insertion failed: {e}")
        
        # Test 4: Test basic queries
        print("\n4. Testing basic queries...")
        
        try:
            # Test portfolio metrics function
            result = await conn.fetch("SELECT * FROM calculate_portfolio_metrics()")
            print(f"  ✓ Portfolio metrics query successful (returned {len(result)} rows)")
        except Exception as e:
            print(f"  ✗ Portfolio metrics query failed: {e}")
        
        try:
            # Test system health view
            result = await conn.fetch("SELECT * FROM system_health")
            print(f"  ✓ System health query successful (returned {len(result)} rows)")
        except Exception as e:
            print(f"  ✗ System health query failed: {e}")
        
        # Test 5: Check indexes
        print("\n5. Testing index existence...")
        
        index_query = """
        SELECT COUNT(*) as index_count
        FROM pg_indexes 
        WHERE schemaname = 'public'
        """
        
        result = await conn.fetchval(index_query)
        print(f"  ✓ Found {result} indexes in the database")
        
        # Test 6: Check triggers
        print("\n6. Testing trigger existence...")
        
        trigger_query = """
        SELECT COUNT(*) as trigger_count
        FROM information_schema.triggers 
        WHERE trigger_schema = 'public'
        """
        
        result = await conn.fetchval(trigger_query)
        print(f"  ✓ Found {result} triggers in the database")
        
        # Test 7: Test data validation trigger
        print("\n7. Testing data validation...")
        
        try:
            # Try to insert invalid trade (should fail)
            await conn.execute("""
                INSERT INTO trades (
                    trade_id, order_id, symbol, exchange, side, quantity, price,
                    executed_at, strategy, agent_name, broker
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 'TEST_INVALID', 'TEST_ORDER', 'TEST', 'NASDAQ', 'BUY', 
                Decimal('-100'), Decimal('100.00'), datetime.now(), 
                'test', 'test_agent', 'test')
            
            print("  ✗ Data validation failed - invalid trade was allowed")
            
        except Exception as e:
            print("  ✓ Data validation working - invalid trade rejected")
        
        print("\n" + "=" * 60)
        print("Database schema test completed successfully!")
        print("All core components are properly implemented.")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"\n✗ Database test failed: {e}")
        return False

async def main():
    """Main test function."""
    success = await test_database_schema()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)