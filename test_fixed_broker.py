#!/usr/bin/env python3
"""
Test the fixed options broker
"""
import asyncio
import sys
import os
from dotenv import load_dotenv

sys.path.append('.')
load_dotenv('.env')

async def test_real_order():
    try:
        from agents.options_broker import OptionsBroker, OptionsOrderRequest, OptionsOrderType, OrderSide
        
        print("Testing FIXED options broker...")
        
        broker = OptionsBroker(paper_trading=True)
        
        # Use the format we know works: SPY250912C00550000
        order = OptionsOrderRequest(
            symbol="SPY250912C00550000",
            underlying="SPY",
            qty=1,
            side=OrderSide.BUY,
            type=OptionsOrderType.LIMIT,
            limit_price=0.01,  # Very low price
            client_order_id="test_fixed"
        )
        
        print(f"Placing order: {order.symbol}")
        response = await broker.submit_options_order(order)
        
        print(f"Order ID: {response.id}")
        print(f"Status: {response.status}")
        
        # Check if it's a real Alpaca order ID (not simulation)
        if len(response.id) > 20 and not response.id.startswith('SIM_'):
            print("SUCCESS: Real Alpaca order placed!")
            return True
        else:
            print("WARNING: Simulation order used")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_real_order())
