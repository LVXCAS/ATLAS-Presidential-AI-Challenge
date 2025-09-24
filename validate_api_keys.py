"""
API Keys Validation Script

Comprehensive validation of all API keys and broker connections for live trading readiness
"""

import os
import sys
import asyncio
import requests
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class APIValidator:
    """Validate all API connections for trading readiness"""

    def __init__(self):
        self.results = {}
        self.critical_apis = []
        self.optional_apis = []

    def validate_alpaca_connection(self):
        """Validate Alpaca trading API"""
        print("\n1. ALPACA TRADING API")
        print("-" * 40)

        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not api_key or not secret_key:
            print("[ERROR] Alpaca API keys not found in environment")
            self.results['alpaca'] = False
            return False

        print(f"API Key: {api_key[:8]}...")
        print(f"Base URL: {base_url}")

        try:
            # Test account endpoint
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': secret_key
            }

            response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)

            if response.status_code == 200:
                account_data = response.json()
                print(f"[OK] Account Connected")
                print(f"    Account Number: {account_data.get('account_number', 'N/A')}")
                print(f"    Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
                print(f"    Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
                print(f"    Trading Blocked: {account_data.get('trading_blocked', 'Unknown')}")

                # Test positions endpoint
                positions_response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
                if positions_response.status_code == 200:
                    positions = positions_response.json()
                    print(f"    Current Positions: {len(positions)}")

                # Test orders capability
                orders_response = requests.get(f"{base_url}/v2/orders", headers=headers, timeout=10)
                if orders_response.status_code == 200:
                    print(f"[OK] Orders API accessible")

                self.results['alpaca'] = True
                self.critical_apis.append('alpaca')
                return True

            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                self.results['alpaca'] = False
                return False

        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self.results['alpaca'] = False
            return False

    def validate_polygon_api(self):
        """Validate Polygon market data API"""
        print("\n2. POLYGON MARKET DATA API")
        print("-" * 40)

        api_key = os.getenv('POLYGON_API_KEY')

        if not api_key:
            print("[SKIP] Polygon API key not configured")
            self.results['polygon'] = None
            return None

        print(f"API Key: {api_key[:8]}...")

        try:
            # Test with a simple ticker request
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apikey={api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK':
                    print(f"[OK] Polygon API working")
                    if data.get('results'):
                        result = data['results'][0]
                        print(f"    Sample Data: AAPL ${result.get('c', 0):.2f}")
                    self.results['polygon'] = True
                    self.optional_apis.append('polygon')
                    return True
                else:
                    print(f"[WARN] API returned status: {data.get('status')}")
                    self.results['polygon'] = False
                    return False
            else:
                print(f"[ERROR] HTTP {response.status_code}")
                self.results['polygon'] = False
                return False

        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self.results['polygon'] = False
            return False

    def validate_alpha_vantage_api(self):
        """Validate Alpha Vantage API"""
        print("\n3. ALPHA VANTAGE API")
        print("-" * 40)

        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

        if not api_key:
            print("[SKIP] Alpha Vantage API key not configured")
            self.results['alpha_vantage'] = None
            return None

        print(f"API Key: {api_key[:8]}...")

        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
            response = requests.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    price = quote.get('05. price', 'N/A')
                    print(f"[OK] Alpha Vantage API working")
                    print(f"    Sample Data: AAPL ${price}")
                    self.results['alpha_vantage'] = True
                    self.optional_apis.append('alpha_vantage')
                    return True
                elif 'Error Message' in data:
                    print(f"[ERROR] API Error: {data['Error Message']}")
                    self.results['alpha_vantage'] = False
                    return False
                elif 'Note' in data:
                    print(f"[WARN] Rate Limited: {data['Note']}")
                    self.results['alpha_vantage'] = False
                    return False
                else:
                    print(f"[ERROR] Unexpected response format")
                    self.results['alpha_vantage'] = False
                    return False
            else:
                print(f"[ERROR] HTTP {response.status_code}")
                self.results['alpha_vantage'] = False
                return False

        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self.results['alpha_vantage'] = False
            return False

    def validate_yahoo_finance(self):
        """Validate Yahoo Finance (free backup)"""
        print("\n4. YAHOO FINANCE (Backup)")
        print("-" * 40)

        try:
            import yfinance as yf

            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="1d")

            if not data.empty:
                last_price = data['Close'].iloc[-1]
                print(f"[OK] Yahoo Finance working")
                print(f"    Sample Data: AAPL ${last_price:.2f}")
                self.results['yahoo_finance'] = True
                self.optional_apis.append('yahoo_finance')
                return True
            else:
                print(f"[ERROR] No data received")
                self.results['yahoo_finance'] = False
                return False

        except Exception as e:
            print(f"[ERROR] Yahoo Finance failed: {e}")
            self.results['yahoo_finance'] = False
            return False

    def test_options_data_access(self):
        """Test options data access"""
        print("\n5. OPTIONS DATA ACCESS")
        print("-" * 40)

        # Test with Alpaca if available
        if self.results.get('alpaca'):
            try:
                api_key = os.getenv('ALPACA_API_KEY')
                secret_key = os.getenv('ALPACA_SECRET_KEY')
                base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

                headers = {
                    'APCA-API-KEY-ID': api_key,
                    'APCA-API-SECRET-KEY': secret_key
                }

                # Test options endpoint (if available)
                response = requests.get(f"{base_url}/v2/options/contracts", headers=headers, timeout=10)

                if response.status_code == 200:
                    print(f"[OK] Options data accessible via Alpaca")
                    self.results['options_data'] = True
                    return True
                else:
                    print(f"[WARN] Options endpoint not available (Status: {response.status_code})")

            except Exception as e:
                print(f"[WARN] Options test failed: {e}")

        # Fallback to Yahoo Finance for options
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            options_dates = ticker.options

            if options_dates:
                print(f"[OK] Options data available via Yahoo Finance")
                print(f"    Available expiration dates: {len(options_dates)}")
                self.results['options_data'] = True
                return True
            else:
                print(f"[WARN] No options data found")
                self.results['options_data'] = False
                return False

        except Exception as e:
            print(f"[ERROR] Options data access failed: {e}")
            self.results['options_data'] = False
            return False

    def generate_api_summary(self):
        """Generate comprehensive API status summary"""
        print(f"\n{'='*60}")
        print("API VALIDATION SUMMARY")
        print("=" * 60)

        critical_passed = sum(1 for api in self.critical_apis if self.results.get(api, False))
        optional_passed = sum(1 for api in self.optional_apis if self.results.get(api, False))

        print(f"Critical APIs: {critical_passed}/{len(self.critical_apis)} working")
        print(f"Optional APIs: {optional_passed}/{len(self.optional_apis)} working")

        print(f"\nDETAILED STATUS:")
        for api, status in self.results.items():
            if status is True:
                print(f"  [OK] {api.upper()}: Connected")
            elif status is False:
                print(f"  [ERROR] {api.upper()}: Failed")
            else:
                print(f"  [SKIP] {api.upper()}: Not configured")

        # Trading readiness assessment
        alpaca_ready = self.results.get('alpaca', False)
        data_ready = any([
            self.results.get('polygon', False),
            self.results.get('alpha_vantage', False),
            self.results.get('yahoo_finance', False)
        ])

        print(f"\nTRADING READINESS:")
        if alpaca_ready and data_ready:
            print(f"[READY] System ready for live trading!")
            print(f"  - Broker connection: ACTIVE")
            print(f"  - Market data: AVAILABLE")
            print(f"  - Options trading: {'READY' if self.results.get('options_data') else 'LIMITED'}")
            ready_score = 100
        elif alpaca_ready:
            print(f"[PARTIAL] Broker ready, limited market data")
            ready_score = 70
        elif data_ready:
            print(f"[PARTIAL] Market data ready, no broker connection")
            ready_score = 50
        else:
            print(f"[NOT READY] Critical APIs not working")
            ready_score = 20

        print(f"\nREADINESS SCORE: {ready_score}%")

        return ready_score >= 70

async def main():
    """Run comprehensive API validation"""

    print("HIVE TRADING - API KEYS VALIDATION")
    print("=" * 60)
    print(f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    validator = APIValidator()

    # Run all validations
    validator.validate_alpaca_connection()
    validator.validate_polygon_api()
    validator.validate_alpha_vantage_api()
    validator.validate_yahoo_finance()
    validator.test_options_data_access()

    # Generate summary
    is_ready = validator.generate_api_summary()

    print(f"\n{'='*60}")
    print("NEXT STEPS FOR TOMORROW'S TRADING")
    print("=" * 60)

    if is_ready:
        print(f"1. [READY] APIs are configured and working")
        print(f"2. Run: python -m core.main")
        print(f"3. Monitor dashboard for live performance")
        print(f"4. Start with small position sizes")
    else:
        print(f"1. [ACTION] Fix API configuration issues above")
        print(f"2. Verify credentials in .env file")
        print(f"3. Re-run this validation script")
        print(f"4. Contact broker support if needed")

    return is_ready

if __name__ == "__main__":
    asyncio.run(main())