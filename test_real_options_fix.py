#!/usr/bin/env python3
"""
Test the real options trading fix
"""
import asyncio
import sys
import os
from datetime import datetime

sys.path.append('.')

async def test_real_options_trading():
    try:
        from agents.options_broker import OptionsBroker
        from agents.options_broker import OptionsOrderRequest, OptionsOrderType, OrderSide
        
        print("Testing real options trading...")
        
        # Create broker
        broker = OptionsBroker(paper_trading=True)
        
        # Create a test order (low price, won't actually fill)
        test_order = OptionsOrderRequest(
            symbol="SPY250912C00550000",
            underlying="SPY",
            qty=1,
            side=OrderSide.BUY,
            type=OptionsOrderType.LIMIT,
            limit_price=0.01,
            client_order_id="test_001"
        )
        
        print(f"Submitting test order: {test_order.symbol}")
        
        # Submit the order
        response = await broker.submit_options_order(test_order)
        
        print(f"Order response: {response.id} - Status: {response.status}")
        
        if response.id.startswith('ALPACA') or len(response.id) > 10:
            print("SUCCESS: Real Alpaca order placed!")
            return True
        else:
            print("WARNING: Simulation order used")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_real_options_trading())
