#!/usr/bin/env python3
"""
BUTTERFLY SPREAD ENGINE - Week 5+ Feature
==========================================
3-leg options spread for defined-risk, neutral strategies

Strategy: Buy 1 ITM, Sell 2 ATM, Buy 1 OTM (calls or puts)
Capital Required: $200-500 per spread
Expected Return: 50-200% at expiration if stock stays at middle strike
Max Loss: Net debit paid
"""

import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


class ButterflySpreadEngine:
    """Execute butterfly spreads for low-risk, high-reward plays"""

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

    def calculate_butterfly_strikes(self, current_price, strike_width=5):
        """Calculate 3 strikes for butterfly

        Butterfly Structure (CALL):
        - Buy 1 call at lower strike (ITM or ATM)
        - Sell 2 calls at middle strike (ATM)
        - Buy 1 call at upper strike (OTM)

        Example: Stock at $100
        - Buy 1x $95 call
        - Sell 2x $100 call
        - Buy 1x $105 call

        Max profit: ($5 - net debit) × 100
        Max loss: Net debit paid
        Breakeven: Lower strike + net debit, Upper strike - net debit
        """

        lower_strike = round(current_price - strike_width)
        middle_strike = round(current_price)
        upper_strike = round(current_price + strike_width)

        # Estimate net debit (typically 30-50% of strike width)
        estimated_debit = strike_width * 100 * 0.40

        return {
            'lower_strike': float(lower_strike),
            'middle_strike': float(middle_strike),
            'upper_strike': float(upper_strike),
            'strike_width': strike_width,
            'estimated_debit': estimated_debit,
            'max_profit': (strike_width * 100) - estimated_debit,
            'max_loss': estimated_debit
        }

    def execute_butterfly(self, symbol, current_price, option_type='CALL', strike_width=5, expiration_days=30):
        """Execute butterfly spread"""

        print(f"\n{'='*80}")
        print(f"EXECUTING BUTTERFLY SPREAD: {symbol}")
        print(f"Type: {option_type} Butterfly")
        print(f"{'='*80}")

        strikes = self.calculate_butterfly_strikes(current_price, strike_width)

        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Lower Strike: ${strikes['lower_strike']:.0f} (BUY 1)")
        print(f"  Middle Strike: ${strikes['middle_strike']:.0f} (SELL 2)")
        print(f"  Upper Strike: ${strikes['upper_strike']:.0f} (BUY 1)")
        print(f"  Estimated Debit: ${strikes['estimated_debit']:.2f}")
        print(f"  Max Profit: ${strikes['max_profit']:.2f} (if stock at ${strikes['middle_strike']:.0f} at expiry)")
        print(f"  Max Loss: ${strikes['max_loss']:.2f}")
        print(f"  Return Potential: {(strikes['max_profit']/strikes['max_loss'])*100:.0f}%")

        # Get expiration
        exp_date = self.get_expiration_date(expiration_days)
        exp_str = exp_date.strftime('%y%m%d')

        # Build symbols
        opt_type = 'C' if option_type == 'CALL' else 'P'
        lower_symbol = f"{symbol}{exp_str}{opt_type}{int(strikes['lower_strike'] * 1000):08d}"
        middle_symbol = f"{symbol}{exp_str}{opt_type}{int(strikes['middle_strike'] * 1000):08d}"
        upper_symbol = f"{symbol}{exp_str}{opt_type}{int(strikes['upper_strike'] * 1000):08d}"

        try:
            # LEG 1: BUY lower strike
            print(f"\n  [LEG 1/3] BUYING {lower_symbol}...")
            order1 = self.api.submit_order(symbol=lower_symbol, qty=1, side='buy', type='market', time_in_force='day')
            print(f"  [OK] Order: {order1.id}")

            # LEG 2: SELL 2x middle strike
            print(f"  [LEG 2/3] SELLING 2x {middle_symbol}...")
            order2 = self.api.submit_order(symbol=middle_symbol, qty=2, side='sell', type='market', time_in_force='day')
            print(f"  [OK] Order: {order2.id}")

            # LEG 3: BUY upper strike
            print(f"  [LEG 3/3] BUYING {upper_symbol}...")
            order3 = self.api.submit_order(symbol=upper_symbol, qty=1, side='buy', type='market', time_in_force='day')
            print(f"  [OK] Order: {order3.id}")

            print(f"\n  [SUCCESS] BUTTERFLY EXECUTED")
            return {'success': True, 'strikes': strikes}

        except Exception as e:
            print(f"  [FAILED] {e}")
            return {'success': False, 'error': str(e)}

    def get_expiration_date(self, days_out=30):
        """Get expiration date"""
        return datetime.now() + timedelta(days=days_out)


if __name__ == "__main__":
    engine = ButterflySpreadEngine()
    # Test butterfly
    engine.execute_butterfly('SPY', 450.0, 'CALL', strike_width=5, expiration_days=30)
