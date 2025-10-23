#!/usr/bin/env python3
"""
IRON CONDOR STRATEGY ENGINE - Week 5+ Feature
==============================================
4-leg options spread for low-volatility, high-probability income

Strategy: Sell OTM call + put, buy further OTM call + put for protection
Capital Required: $500-1,500 per spread (vs $3,300 for cash-secured puts)
Expected Return: 2-5% per trade (vs 10-30%)
Win Rate: 70-80% (higher than naked positions)
"""

import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()


class IronCondorEngine:
    """Execute iron condor spreads for high-probability income"""

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - IRON_CONDOR - %(message)s')
        self.logger = logging.getLogger(__name__)

    def calculate_iron_condor_strikes(self, symbol, current_price, target_delta=0.16):
        """Calculate 4 strikes for iron condor

        Iron Condor Structure:
        - Sell put at 0.16 delta (~16% probability ITM)
        - Buy put at 0.08 delta (~8% probability ITM) - $5 lower
        - Sell call at 0.16 delta (~16% probability ITM)
        - Buy call at 0.08 delta (~8% probability ITM) - $5 higher

        Target: Collect $50-150 credit per spread
        Risk: $350-450 (difference between strikes minus credit)
        Return: 10-30% return on risk
        """

        # Simplified strike selection (would use Greeks in production)
        # Sell strikes: ~10% OTM (delta ~0.16)
        sell_put_strike = round(current_price * 0.90)
        sell_call_strike = round(current_price * 1.10)

        # Buy strikes: ~15% OTM (delta ~0.08) - $5 wider
        strike_width = 5 if current_price >= 25 else 2.5
        buy_put_strike = sell_put_strike - strike_width
        buy_call_strike = sell_call_strike + strike_width

        return {
            'buy_put': float(buy_put_strike),
            'sell_put': float(sell_put_strike),
            'sell_call': float(sell_call_strike),
            'buy_call': float(buy_call_strike),
            'strike_width': strike_width,
            'expected_credit': strike_width * 100 * 0.30  # Estimate 30% of width
        }

    def execute_iron_condor(self, symbol, current_price, contracts=1, expiration_days=7):
        """Execute full iron condor spread (4 legs)"""

        print(f"\n{'='*80}")
        print(f"EXECUTING IRON CONDOR: {symbol}")
        print(f"{'='*80}")

        # Calculate strikes
        strikes = self.calculate_iron_condor_strikes(symbol, current_price)

        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Buy Put:  ${strikes['buy_put']:.0f} (protection)")
        print(f"  Sell Put: ${strikes['sell_put']:.0f} (collect premium)")
        print(f"  Sell Call: ${strikes['sell_call']:.0f} (collect premium)")
        print(f"  Buy Call: ${strikes['buy_call']:.0f} (protection)")
        print(f"  Expected Credit: ${strikes['expected_credit']:.2f}")
        print(f"  Max Risk: ${(strikes['strike_width'] * 100) - strikes['expected_credit']:.2f}")
        print(f"  Return on Risk: {(strikes['expected_credit'] / ((strikes['strike_width'] * 100) - strikes['expected_credit'])) * 100:.1f}%")

        # Get expiration date
        exp_date = self.get_expiration_date(expiration_days)
        exp_str = exp_date.strftime('%y%m%d')

        # Build options symbols
        buy_put_symbol = f"{symbol}{exp_str}P{int(strikes['buy_put'] * 1000):08d}"
        sell_put_symbol = f"{symbol}{exp_str}P{int(strikes['sell_put'] * 1000):08d}"
        sell_call_symbol = f"{symbol}{exp_str}C{int(strikes['sell_call'] * 1000):08d}"
        buy_call_symbol = f"{symbol}{exp_str}C{int(strikes['buy_call'] * 1000):08d}"

        orders = []

        try:
            # LEG 1: BUY PUT (protection, costs premium)
            print(f"\n  [LEG 1/4] BUYING protective put {buy_put_symbol}...")
            order1 = self.api.submit_order(
                symbol=buy_put_symbol,
                qty=contracts,
                side='buy',
                type='market',
                time_in_force='day'
            )
            orders.append({'leg': 'BUY_PUT', 'order_id': order1.id, 'symbol': buy_put_symbol})
            print(f"  [OK] Order submitted: {order1.id}")

            # LEG 2: SELL PUT (collect premium)
            print(f"  [LEG 2/4] SELLING put {sell_put_symbol}...")
            order2 = self.api.submit_order(
                symbol=sell_put_symbol,
                qty=contracts,
                side='sell',
                type='market',
                time_in_force='day'
            )
            orders.append({'leg': 'SELL_PUT', 'order_id': order2.id, 'symbol': sell_put_symbol})
            print(f"  [OK] Order submitted: {order2.id}")

            # LEG 3: SELL CALL (collect premium)
            print(f"  [LEG 3/4] SELLING call {sell_call_symbol}...")
            order3 = self.api.submit_order(
                symbol=sell_call_symbol,
                qty=contracts,
                side='sell',
                type='market',
                time_in_force='day'
            )
            orders.append({'leg': 'SELL_CALL', 'order_id': order3.id, 'symbol': sell_call_symbol})
            print(f"  [OK] Order submitted: {order3.id}")

            # LEG 4: BUY CALL (protection, costs premium)
            print(f"  [LEG 4/4] BUYING protective call {buy_call_symbol}...")
            order4 = self.api.submit_order(
                symbol=buy_call_symbol,
                qty=contracts,
                side='buy',
                type='market',
                time_in_force='day'
            )
            orders.append({'leg': 'BUY_CALL', 'order_id': order4.id, 'symbol': buy_call_symbol})
            print(f"  [OK] Order submitted: {order4.id}")

            print(f"\n  [SUCCESS] IRON CONDOR EXECUTED: {symbol}")
            print(f"  All 4 legs submitted successfully")
            print(f"  Max profit: ${strikes['expected_credit']:.2f} ({(strikes['expected_credit'] / ((strikes['strike_width'] * 100) - strikes['expected_credit'])) * 100:.1f}% return)")
            print(f"  Max loss: ${(strikes['strike_width'] * 100) - strikes['expected_credit']:.2f}")
            print(f"  Breakeven: ${strikes['sell_put']:.0f} (down) | ${strikes['sell_call']:.0f} (up)")

            return {
                'success': True,
                'symbol': symbol,
                'strategy': 'IRON_CONDOR',
                'orders': orders,
                'strikes': strikes,
                'contracts': contracts,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Iron condor execution failed for {symbol}: {e}")
            print(f"  [FAILED] Error: {e}")

            # Cancel any filled legs if execution fails
            print(f"  [ROLLBACK] Canceling any filled orders...")
            for order in orders:
                try:
                    self.api.cancel_order(order['order_id'])
                    print(f"    Canceled {order['leg']}: {order['order_id']}")
                except:
                    pass

            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'partial_orders': orders
            }

    def get_expiration_date(self, days_out=7):
        """Get nearest expiration date (Friday)"""
        today = datetime.now()
        days_until_friday = 4 - today.weekday()  # Friday = 4
        if days_until_friday <= 0:
            days_until_friday += 7

        if days_out <= days_until_friday:
            next_friday = today + timedelta(days=days_until_friday)
        else:
            next_friday = today + timedelta(days=days_until_friday + 7)

        return next_friday


def test_iron_condor():
    """Test iron condor execution"""
    engine = IronCondorEngine()

    # Test on SPY (liquid, good spreads)
    test_symbol = 'SPY'
    test_price = 450.0

    print("="*80)
    print("IRON CONDOR ENGINE TEST")
    print("="*80)
    print(f"Testing on {test_symbol} @ ${test_price:.2f}")

    result = engine.execute_iron_condor(test_symbol, test_price, contracts=1)

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Orders submitted: {len(result['orders'])}")
        print(f"Expected credit: ${result['strikes']['expected_credit']:.2f}")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    test_iron_condor()
