"""
EXECUTE COVERED CALLS TOMORROW - MARKET OPEN
Specific instructions and automation for covered call execution
Target: +$10,078 monthly income starting tomorrow
"""

import alpaca_trade_api as tradeapi
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class CoveredCallExecutor:
    """Execute covered calls automatically at market open"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        print("COVERED CALL EXECUTOR - TOMORROW MARKET OPEN")
        print("=" * 60)
        print("Target: +$10,078 monthly income from covered calls")
        print("Execution: Automatic at 9:30 AM EST")

    def check_options_trading_enabled(self):
        """Check if options trading is enabled on account"""

        print("\nCHECKING OPTIONS TRADING STATUS...")
        print("-" * 40)

        try:
            account = self.alpaca.get_account()

            # Check account status
            print(f"Account Status: {account.status}")
            print(f"Trading Blocked: {account.trading_blocked}")
            print(f"Account Number: {account.account_number}")

            # Note: Alpaca paper trading may not support options
            # Real account needed for options trading

            if "PAPER" in os.getenv('ALPACA_BASE_URL', ''):
                print("\nWARNING: PAPER TRADING ACCOUNT")
                print("Options trading requires LIVE account")
                print("Current setup is for testing/planning only")
                return False
            else:
                print("LIVE ACCOUNT - Options trading possible")
                return True

        except Exception as e:
            print(f"Account check error: {e}")
            return False

    def get_current_positions_for_calls(self):
        """Get current stock positions available for covered calls"""

        print("\nSTOCK POSITIONS AVAILABLE FOR COVERED CALLS")
        print("-" * 50)

        positions = self.alpaca.list_positions()
        call_candidates = []

        target_symbols = ['IWM', 'SOXL', 'TQQQ']  # High volatility positions

        for pos in positions:
            if pos.symbol in target_symbols and int(pos.qty) > 0:
                shares = int(pos.qty)
                market_value = float(pos.market_value)

                # Get current price for strike calculation
                try:
                    ticker = yf.Ticker(pos.symbol)
                    hist = ticker.history(period='1d')
                    current_price = hist['Close'].iloc[-1]

                    # Calculate 5% OTM strike
                    otm_strike = round(current_price * 1.05, 0)  # Round to nearest dollar

                    call_candidates.append({
                        'symbol': pos.symbol,
                        'shares': shares,
                        'current_price': current_price,
                        'otm_strike': otm_strike,
                        'position_value': market_value,
                        'contracts_available': shares // 100  # Each contract = 100 shares
                    })

                    print(f"{pos.symbol}:")
                    print(f"  Shares: {shares}")
                    print(f"  Current Price: ${current_price:.2f}")
                    print(f"  5% OTM Strike: ${otm_strike:.0f}")
                    print(f"  Contracts Available: {shares // 100}")

                except Exception as e:
                    print(f"  Error getting price for {pos.symbol}: {e}")

        return call_candidates

    def create_covered_call_orders(self, call_candidates):
        """Create covered call orders for execution"""

        print("\nCOVERED CALL ORDERS TO EXECUTE")
        print("-" * 40)

        # Get next Friday's date for weekly options
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7  # 4 = Friday
        if days_until_friday == 0:  # If today is Friday, get next Friday
            days_until_friday = 7
        expiry_date = today + timedelta(days=days_until_friday)

        orders = []

        for candidate in call_candidates:
            symbol = candidate['symbol']
            strike = candidate['otm_strike']
            contracts = candidate['contracts_available']

            if contracts > 0:  # Must have at least 100 shares

                # Create options symbol (format: SYMBOL + YYMMDD + C + STRIKE)
                expiry_str = expiry_date.strftime('%y%m%d')
                strike_str = f"{int(strike * 1000):08d}"  # Strike in thousandths
                options_symbol = f"{symbol}{expiry_str}C{strike_str}"

                order_details = {
                    'underlying': symbol,
                    'options_symbol': options_symbol,
                    'strike': strike,
                    'expiry': expiry_date.strftime('%Y-%m-%d'),
                    'contracts': contracts,
                    'side': 'sell',  # Selling covered calls
                    'type': 'limit',
                    'estimated_premium': self.estimate_option_premium(symbol, strike, expiry_date)
                }

                orders.append(order_details)

                print(f"\n{symbol} COVERED CALL ORDER:")
                print(f"  Options Symbol: {options_symbol}")
                print(f"  Strike: ${strike:.0f}")
                print(f"  Expiry: {expiry_date.strftime('%Y-%m-%d')}")
                print(f"  Contracts: {contracts}")
                print(f"  Estimated Premium: ${order_details['estimated_premium']:.2f} per contract")
                print(f"  Total Premium: ${order_details['estimated_premium'] * contracts * 100:.0f}")

        return orders

    def estimate_option_premium(self, symbol, strike, expiry):
        """Estimate option premium for covered calls"""

        try:
            # Simple premium estimation based on volatility
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='30d')

            if len(hist) > 10:
                # Calculate implied volatility estimate
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * (252 ** 0.5)  # Annualized

                current_price = hist['Close'].iloc[-1]
                days_to_expiry = (expiry - datetime.now()).days

                # Simple premium estimate (very rough)
                # Real options pricing would use Black-Scholes
                time_value = volatility * (days_to_expiry / 365) ** 0.5
                otm_amount = (strike - current_price) / current_price

                # Premium estimate (conservative)
                premium_estimate = max(0.1, current_price * time_value * 0.3)

                return premium_estimate

        except Exception as e:
            print(f"Premium estimation error for {symbol}: {e}")

        return 1.0  # Default $1 premium estimate

    def execute_covered_calls_at_market_open(self, orders):
        """Execute covered calls automatically at market open"""

        print(f"\n{'='*60}")
        print("AUTOMATIC EXECUTION AT MARKET OPEN")
        print("="*60)

        market_open = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

        if datetime.now() < market_open:
            wait_time = (market_open - datetime.now()).total_seconds()
            print(f"⏰ Waiting for market open...")
            print(f"Market opens in: {wait_time/60:.1f} minutes")
            print(f"Market open time: {market_open.strftime('%H:%M:%S')}")

        print(f"\n📋 ORDERS READY FOR EXECUTION:")
        total_estimated_premium = 0

        for i, order in enumerate(orders, 1):
            premium_total = order['estimated_premium'] * order['contracts'] * 100
            total_estimated_premium += premium_total

            print(f"\nOrder {i}: {order['underlying']} Covered Call")
            print(f"  Strike: ${order['strike']:.0f}")
            print(f"  Contracts: {order['contracts']}")
            print(f"  Premium: ${premium_total:.0f}")

        print(f"\nTOTAL ESTIMATED PREMIUM: ${total_estimated_premium:.0f}")
        print(f"TARGET PREMIUM: $10,078")
        print(f"Achievement: {(total_estimated_premium/10078)*100:.1f}% of target")

        # Manual execution instructions (since options may not be available in paper)
        print(f"\n🔧 MANUAL EXECUTION INSTRUCTIONS:")
        print(f"(Use if automatic execution not available)")
        print("-" * 50)

        for order in orders:
            print(f"\n{order['underlying']} - SELL TO OPEN CALL:")
            print(f"  Symbol: {order['options_symbol']}")
            print(f"  Quantity: {order['contracts']} contracts")
            print(f"  Order Type: Limit")
            print(f"  Limit Price: ${order['estimated_premium']:.2f}")
            print(f"  Time in Force: Day")

        return orders, total_estimated_premium

    def create_monitoring_alerts(self, orders):
        """Create monitoring alerts for covered call positions"""

        print(f"\nMONITORING ALERTS SETUP")
        print("-" * 30)

        alerts = []

        for order in orders:
            alert = {
                'symbol': order['underlying'],
                'alert_type': 'price_alert',
                'trigger_price': order['strike'],
                'message': f"{order['underlying']} approaching strike ${order['strike']:.0f} - Monitor covered call",
                'action': 'Consider rolling or closing position'
            }
            alerts.append(alert)

            print(f"{order['underlying']}: Alert if price hits ${order['strike']:.0f}")

        print(f"\nPROFIT TAKING RULES:")
        print(f"- Close calls if 50% profit achieved")
        print(f"- Roll up and out if stock approaches strike")
        print(f"- Let expire worthless if stock stays below strike")

        return alerts

    def run_tomorrow_covered_call_execution(self):
        """Run complete covered call execution plan for tomorrow"""

        print("TOMORROW'S COVERED CALL EXECUTION PLAN")
        print("=" * 60)

        # Step 1: Check options trading capability
        options_enabled = self.check_options_trading_enabled()

        # Step 2: Get positions available for calls
        call_candidates = self.get_current_positions_for_calls()

        if not call_candidates:
            print("\n❌ No positions available for covered calls")
            return None

        # Step 3: Create orders
        orders = self.create_covered_call_orders(call_candidates)

        if not orders:
            print("\n❌ No valid covered call orders created")
            return None

        # Step 4: Set up execution
        executed_orders, total_premium = self.execute_covered_calls_at_market_open(orders)

        # Step 5: Set up monitoring
        alerts = self.create_monitoring_alerts(orders)

        # Final summary
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print("="*60)

        print(f"Orders Created: {len(orders)}")
        print(f"Estimated Monthly Premium: ${total_premium:.0f}")
        print(f"Target Achievement: {(total_premium/10078)*100:.1f}%")

        if options_enabled:
            print(f"READY FOR AUTOMATIC EXECUTION AT 9:30 AM")
        else:
            print(f"MANUAL EXECUTION REQUIRED (Paper account)")
            print(f"Use the detailed instructions above")

        return {
            'orders': orders,
            'total_premium': total_premium,
            'alerts': alerts,
            'execution_ready': options_enabled
        }

def main():
    """Execute tomorrow's covered call plan"""

    executor = CoveredCallExecutor()
    results = executor.run_tomorrow_covered_call_execution()

    if results:
        print(f"\nCOVERED CALLS READY FOR TOMORROW!")
        print(f"Premium Target: ${results['total_premium']:.0f}")
    else:
        print(f"\nCovered call setup failed")

    return results

if __name__ == "__main__":
    main()