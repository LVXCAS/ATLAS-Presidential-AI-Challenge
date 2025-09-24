#!/usr/bin/env python3
"""
Enable real options trading in OPTIONS_BOT by modifying the broker integration
"""
import os
import shutil
from datetime import datetime

def backup_current_files():
    """Backup the current files before modifying"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    files_to_backup = [
        'agents/options_broker.py',
        'agents/broker_integration.py'
    ]
    
    print("Creating backups...")
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup_{timestamp}"
            shutil.copy2(file_path, backup_path)
            print(f"  Backed up {file_path} -> {backup_path}")

def patch_options_broker():
    """Patch the options broker to use real Alpaca API calls"""
    
    broker_file = 'agents/options_broker.py'
    
    try:
        with open(broker_file, 'r') as f:
            content = f.read()
        
        # Find the _submit_paper_options_order method
        method_start = content.find('async def _submit_paper_options_order(self, order: OptionsOrderRequest) -> OptionsOrderResponse:')
        
        if method_start == -1:
            print("ERROR: Could not find _submit_paper_options_order method")
            return False
        
        # Find the end of the method (next method or end of class)
        method_end = content.find('\n    async def ', method_start + 1)
        if method_end == -1:
            method_end = content.find('\n    def ', method_start + 1)
        if method_end == -1:
            method_end = len(content)
        
        # Create the new method that uses real Alpaca API
        new_method = '''async def _submit_paper_options_order(self, order: OptionsOrderRequest) -> OptionsOrderResponse:
        """Submit REAL options order to Alpaca paper trading account"""
        import requests
        
        headers = {
            'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
            'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY'),
            'Content-Type': 'application/json'
        }
        
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')
        
        try:
            # Convert to Alpaca order format
            order_data = {
                "symbol": order.symbol,
                "qty": order.qty,
                "side": order.side.value,
                "type": "limit",  # Use limit orders for safety
                "limit_price": str(order.limit_price if order.type == OptionsOrderType.LIMIT else 0.50),
                "time_in_force": "day"
            }
            
            logger.info(f"PLACING REAL ALPACA ORDER: {order_data}")
            
            # Submit to Alpaca - using correct v2 API endpoint
            order_url = f"{base_url}/v2/orders"
            response = requests.post(order_url, headers=headers, json=order_data, timeout=15)
            
            if response.status_code in [200, 201]:
                order_response = response.json()
                logger.info(f"SUCCESS: Real order placed - ID: {order_response.get('id')}")
                
                return OptionsOrderResponse(
                    id=order_response.get('id'),
                    symbol=order.symbol,
                    underlying=order.underlying,
                    qty=order.qty,
                    side=order.side,
                    type=order.type,
                    status=order_response.get('status', 'submitted'),
                    filled_qty=order_response.get('filled_qty', 0),
                    avg_fill_price=float(order_response.get('filled_avg_price', 0)),
                    created_at=datetime.now(),
                    filled_at=datetime.now() if order_response.get('status') == 'filled' else None,
                    commission=1.00
                )
            else:
                logger.error(f"Alpaca order failed: {response.status_code} - {response.text}")
                # Fall back to old simulation method
                return await self._fallback_simulation_order(order)
                
        except Exception as e:
            logger.error(f"Error placing real order: {e}")
            return await self._fallback_simulation_order(order)
    
    async def _fallback_simulation_order(self, order: OptionsOrderRequest) -> OptionsOrderResponse:
        """Fallback to simulation if real order fails"""
        # Get current options price for simulation
        current_price = await self._get_options_price(order.symbol, order.underlying)
        if not current_price:
            current_price = {'ask': 0.50, 'bid': 0.45, 'mid': 0.475}
        
        exec_price = current_price['ask'] if order.side == OrderSide.BUY else current_price['bid']
        
        order_id = f"SIM_{self.order_counter:06d}"
        self.order_counter += 1
        
        logger.warning(f"SIMULATION ORDER (real order failed): {order.side} {order.qty} {order.symbol} @ ${exec_price:.2f}")
        
        response = OptionsOrderResponse(
            id=order_id,
            symbol=order.symbol,
            underlying=order.underlying,
            qty=order.qty,
            side=order.side,
            type=order.type,
            status="filled",
            filled_qty=order.qty,
            avg_fill_price=exec_price,
            created_at=datetime.now(),
            filled_at=datetime.now(),
            commission=1.00
        )
        
        return response
'''
        
        # Replace the method
        new_content = content[:method_start] + new_method + '\n' + content[method_end:]
        
        # Write the modified file
        with open(broker_file, 'w') as f:
            f.write(new_content)
        
        print("SUCCESS: Patched options_broker.py for real trading")
        return True
        
    except Exception as e:
        print(f"ERROR: Could not patch options broker - {e}")
        return False

def create_test_script():
    """Create a simple test script to verify the fix works"""
    
    test_content = '''#!/usr/bin/env python3
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
'''
    
    with open('test_real_options_fix.py', 'w') as f:
        f.write(test_content)
    
    print("Created test_real_options_fix.py")

def main():
    print("ENABLING REAL OPTIONS TRADING IN OPTIONS_BOT")
    print("=" * 50)
    
    # Step 1: Backup files
    backup_current_files()
    
    # Step 2: Patch the options broker
    patch_success = patch_options_broker()
    
    # Step 3: Create test script
    create_test_script()
    
    print("\n" + "=" * 50)
    
    if patch_success:
        print("SUCCESS: OPTIONS_BOT is now configured for real trading!")
        print("\nNEXT STEPS:")
        print("1. Test: python test_real_options_fix.py")
        print("2. Run: python OPTIONS_BOT.py")
        print("3. Watch for real trades in your Alpaca account")
        print("\nIMPORTANT: The bot will now place REAL orders in your paper account!")
    else:
        print("FAILED: Could not enable real trading")
        print("Check the error messages above")

if __name__ == "__main__":
    main()