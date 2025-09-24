"""
TEST REAL OPTIONS ORDER - ALPACA API
Test placing actual options orders using real contract symbols
"""

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os
import requests

load_dotenv(override=True)

def test_real_options_order():
    """Test placing real options orders with valid contract symbols"""

    print("=== TESTING REAL OPTIONS ORDER PLACEMENT ===")

    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    # Get real contract symbols from API
    headers = {
        'Apca-Api-Key-Id': os.getenv('ALPACA_API_KEY'),
        'Apca-Api-Secret-Key': os.getenv('ALPACA_SECRET_KEY')
    }

    base_url = os.getenv('ALPACA_BASE_URL')
    options_url = f"{base_url}/v2/options/contracts"

    params = {
        'underlying_symbols': 'QQQ',
        'status': 'active',
        'limit': 10
    }

    response = requests.get(options_url, headers=headers, params=params)

    if response.status_code == 200:
        options_data = response.json()
        contracts = options_data.get('option_contracts', [])

        if contracts:
            # Use first available contract for test
            test_contract = contracts[0]
            test_symbol = test_contract['symbol']

            print(f"Testing with real contract: {test_symbol}")
            print(f"Strike: ${test_contract['strike_price']}")
            print(f"Expiration: {test_contract['expiration_date']}")
            print(f"Type: {test_contract['type']}")

            try:
                # Test order with very low limit price
                test_order = api.submit_order(
                    symbol=test_symbol,
                    qty=1,
                    side='buy',
                    type='limit',
                    limit_price=0.01,  # Very low price, likely won't fill
                    time_in_force='day'
                )

                print(f"SUCCESS: Options order placed!")
                print(f"Order ID: {test_order.id}")
                print(f"Status: {test_order.status}")

                # Cancel test order immediately
                api.cancel_order(test_order.id)
                print(f"Test order cancelled - options trading confirmed working!")

                return True

            except Exception as e:
                print(f"Order placement failed: {str(e)}")
                return False

        else:
            print("No option contracts found")
            return False

    else:
        print(f"API Error: {response.status_code}")
        return False

if __name__ == "__main__":
    test_real_options_order()