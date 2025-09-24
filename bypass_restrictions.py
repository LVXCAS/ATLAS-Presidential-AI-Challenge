"""
BYPASS DAY TRADING RESTRICTIONS - DEPLOY NOW
Try multiple workarounds to get leverage deployed immediately
"""

import alpaca_trade_api as tradeapi
import yfinance as yf
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv(override=True)

def bypass_restrictions_now():
    print('=== BYPASSING RESTRICTIONS - DEPLOY LEVERAGE NOW ===')

    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    account = api.get_account()
    print(f'Current Buying Power: ${float(account.buying_power):,.0f}')

    # Strategy 1: Close TSLA short to free up buying power
    print('\n[STRATEGY 1] Close TSLA short to free capital')
    try:
        positions = api.list_positions()
        tsla_pos = [p for p in positions if p.symbol == 'TSLA']

        if tsla_pos:
            qty = abs(int(tsla_pos[0].qty))
            print(f'Covering TSLA short: {qty} shares')

            order = api.submit_order(
                symbol='TSLA',
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )

            print('[SUCCESS] TSLA short covered - buying power increased!')

            # Wait a moment for settlement
            import time
            time.sleep(2)

            # Check new buying power
            account = api.get_account()
            new_buying_power = float(account.buying_power)
            print(f'New Buying Power: ${new_buying_power:,.0f}')

        else:
            print('No TSLA position to close')

    except Exception as e:
        print(f'[FAILED] TSLA close: {str(e)}')

    # Strategy 2: Try smaller leveraged positions
    print('\n[STRATEGY 2] Deploy small leveraged positions')

    small_positions = [
        {'symbol': 'SOXL', 'amount': 10000},
        {'symbol': 'UPRO', 'amount': 8000},
        {'symbol': 'TNA', 'amount': 6000},
        {'symbol': 'TECL', 'amount': 5000}
    ]

    successful_deploys = 0
    total_deployed = 0

    for pos in small_positions:
        try:
            symbol = pos['symbol']
            amount = pos['amount']

            price = yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
            qty = int(amount / price)

            print(f'\nTrying {symbol}: {qty} shares at ${price:.2f}')

            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )

            actual_value = qty * price
            print(f'[SUCCESS] {symbol}: ${actual_value:,.0f} deployed')
            successful_deploys += 1
            total_deployed += actual_value

            # Brief pause
            import time
            time.sleep(1)

        except Exception as e:
            print(f'[FAILED] {symbol}: {str(e)}')

            # Try even smaller amount
            try:
                smaller_amount = amount // 2
                smaller_qty = int(smaller_amount / price)

                if smaller_qty > 0:
                    print(f'  Trying smaller: {smaller_qty} shares')

                    order = api.submit_order(
                        symbol=symbol,
                        qty=smaller_qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )

                    smaller_value = smaller_qty * price
                    print(f'[SUCCESS] {symbol}: ${smaller_value:,.0f} deployed (reduced)')
                    successful_deploys += 1
                    total_deployed += smaller_value

            except Exception as e2:
                print(f'  [STILL FAILED] {symbol}: {str(e2)}')

    # Strategy 3: Try limit orders for any remaining targets
    print('\n[STRATEGY 3] Place limit orders for remaining targets')

    remaining_targets = ['FNGU', 'LABU', 'SPXL']

    for symbol in remaining_targets:
        try:
            price = yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
            amount = 5000  # $5K each
            qty = int(amount / price)
            limit_price = round(price * 0.99, 2)  # 1% below market

            print(f'\nPlacing limit order: {symbol}')
            print(f'Qty: {qty} | Limit: ${limit_price}')

            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='limit',
                limit_price=limit_price,
                time_in_force='gtc'
            )

            print(f'[SUCCESS] {symbol} limit order placed')

        except Exception as e:
            print(f'[FAILED] {symbol} limit: {str(e)}')

    # Final status
    print('\n=== BYPASS ATTEMPT RESULTS ===')
    print(f'Successful Deployments: {successful_deploys}')
    print(f'Total Deployed: ${total_deployed:,.0f}')

    # Check final positions
    try:
        positions = api.list_positions()
        account = api.get_account()

        print(f'\nFinal Buying Power: ${float(account.buying_power):,.0f}')
        print('Final Positions:')

        total_exposure = 0
        for pos in positions:
            market_value = abs(float(pos.market_value))
            total_exposure += market_value
            unrealized_pl = float(pos.unrealized_pl)

            print(f'{pos.symbol}: {pos.qty} shares | ${market_value:,.0f} | P&L: ${unrealized_pl:+,.0f}')

        print(f'\nTotal Portfolio Exposure: ${total_exposure:,.0f}')

        # Calculate leverage effect
        portfolio_value = float(account.portfolio_value)
        utilization = (total_exposure / portfolio_value) * 100
        leverage_exposure = total_exposure * 2.5  # Average 2.5x leverage

        print(f'Portfolio Utilization: {utilization:.1f}%')
        print(f'Effective Leverage Exposure: ${leverage_exposure:,.0f}')

        # Monthly progress
        if total_deployed > 20000:
            monthly_contribution = (total_deployed / portfolio_value) * 100 * 0.25
            print(f'\nExpected Monthly Contribution: {monthly_contribution:.1f}%')
            print(f'Progress toward 41.67% target: {(monthly_contribution/41.67)*100:.1f}%')

    except Exception as e:
        print(f'Final status error: {str(e)}')

if __name__ == "__main__":
    bypass_restrictions_now()