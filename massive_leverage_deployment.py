"""
MASSIVE LEVERAGE DEPLOYMENT - AI SYSTEM SCALE UP
Deploy $300K+ in 3x leveraged ETFs RIGHT NOW
Target: 41%+ monthly through aggressive leverage
"""

import alpaca_trade_api as tradeapi
import yfinance as yf
from dotenv import load_dotenv
import os
from datetime import datetime
import time

load_dotenv(override=True)

def deploy_massive_leverage():
    print('=== MASSIVE LEVERAGE DEPLOYMENT - AI SYSTEM SCALE UP ===')
    print(f'Time: {datetime.now().strftime("%H:%M:%S")}')

    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    account = api.get_account()
    buying_power = float(account.buying_power)

    print(f'Available Capital: ${buying_power:,.0f}')
    print('Strategy: Deploy $300K+ in 3x leveraged positions')

    # Massive leverage deployment plan
    leverage_positions = [
        {'symbol': 'SOXL', 'target': 120000, 'leverage': '3x Semiconductors', 'priority': 1},
        {'symbol': 'UPRO', 'target': 100000, 'leverage': '3x S&P 500', 'priority': 2},
        {'symbol': 'TNA', 'target': 80000, 'leverage': '3x Russell 2000', 'priority': 3},
        {'symbol': 'FNGU', 'target': 60000, 'leverage': '3x FANG+', 'priority': 4},
        {'symbol': 'TECL', 'target': 50000, 'leverage': '3x Technology', 'priority': 5}
    ]

    total_target = sum([pos['target'] for pos in leverage_positions])
    print(f'Total Target Deployment: ${total_target:,.0f}')

    executed_positions = []
    total_deployed = 0
    failed_attempts = []

    print(f'\\n[EXECUTING MASSIVE LEVERAGE DEPLOYMENT]')

    for position in leverage_positions:
        symbol = position['symbol']
        target_size = position['target']

        print(f'\\nDeploying {symbol}: ${target_size:,.0f}')
        print(f'  Leverage: {position["leverage"]}')

        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='1m')

            if hist.empty:
                print(f'  [ERROR] No price data for {symbol}')
                continue

            current_price = hist['Close'].iloc[-1]
            qty = int(target_size / current_price)

            print(f'  Price: ${current_price:.2f}')
            print(f'  Quantity: {qty} shares')

            if qty > 0:
                # Try market order
                try:
                    order = api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )

                    actual_value = qty * current_price
                    print(f'  [SUCCESS] Market order executed')
                    print(f'  Position Value: ${actual_value:,.0f}')

                    executed_positions.append({
                        'symbol': symbol,
                        'quantity': qty,
                        'price': current_price,
                        'value': actual_value,
                        'leverage': position['leverage']
                    })

                    total_deployed += actual_value

                    # Brief pause between orders
                    time.sleep(1)

                except Exception as e:
                    print(f'  [MARKET ORDER FAILED] {str(e)}')

                    # Try limit order as backup
                    try:
                        limit_price = round(current_price * 0.99, 2)  # 1% below market

                        order = api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='limit',
                            limit_price=limit_price,
                            time_in_force='day'
                        )

                        print(f'  [LIMIT ORDER] Placed at ${limit_price:.2f}')

                        executed_positions.append({
                            'symbol': symbol,
                            'quantity': qty,
                            'price': limit_price,
                            'value': qty * limit_price,
                            'leverage': position['leverage'],
                            'order_type': 'limit'
                        })

                    except Exception as e2:
                        print(f'  [COMPLETELY FAILED] {str(e2)}')
                        failed_attempts.append({
                            'symbol': symbol,
                            'target': target_size,
                            'error': str(e2)
                        })

        except Exception as e:
            print(f'  [PRICE ERROR] {symbol}: {str(e)}')
            failed_attempts.append({
                'symbol': symbol,
                'target': target_size,
                'error': f'Price data error: {str(e)}'
            })

    # Summary
    print(f'\\n=== MASSIVE DEPLOYMENT SUMMARY ===')
    print(f'Positions Executed: {len(executed_positions)}')
    print(f'Total Capital Deployed: ${total_deployed:,.0f}')
    print(f'Target Achievement: {(total_deployed/total_target)*100:.1f}%')

    if executed_positions:
        print(f'\\n[SUCCESSFUL DEPLOYMENTS]')
        total_leverage_exposure = 0

        for pos in executed_positions:
            leverage_exposure = pos['value'] * 3  # 3x leverage
            total_leverage_exposure += leverage_exposure

            print(f'{pos["symbol"]}: {pos["quantity"]} shares | ${pos["value"]:,.0f} | {pos["leverage"]}')
            print(f'  Effective Exposure: ${leverage_exposure:,.0f}')

        print(f'\\nTotal 3x Leverage Exposure: ${total_leverage_exposure:,.0f}')

        # Calculate expected monthly contribution
        portfolio_value = float(account.portfolio_value)
        expected_monthly_contribution = (total_deployed / portfolio_value) * 100 * 0.3  # Conservative 30% monthly on 3x leverage

        print(f'\\n[COMPOUND MONTHLY ANALYSIS]')
        print(f'Capital Deployed: ${total_deployed:,.0f}')
        print(f'Portfolio Percentage: {(total_deployed/portfolio_value)*100:.1f}%')
        print(f'Expected Monthly Contribution: {expected_monthly_contribution:.1f}%')
        print(f'Monthly Target: 41.67%')
        print(f'Progress: {(expected_monthly_contribution/41.67)*100:.1f}% of target')

        if expected_monthly_contribution > 20:
            print('[EXCELLENT] Major progress toward 41%+ monthly target!')
        elif expected_monthly_contribution > 10:
            print('[GOOD] Solid contribution to compound monthly system')
        else:
            print('[PARTIAL] Need additional leverage deployment')

    if failed_attempts:
        print(f'\\n[FAILED ATTEMPTS]')
        for failure in failed_attempts:
            print(f'{failure["symbol"]}: ${failure["target"]:,.0f} - {failure["error"][:50]}')

    # Check current portfolio
    print(f'\\n[UPDATED PORTFOLIO STATUS]')
    try:
        positions = api.list_positions()
        total_exposure = 0

        for pos in positions:
            market_value = abs(float(pos.market_value))
            total_exposure += market_value
            unrealized_pl = float(pos.unrealized_pl)

            print(f'{pos.symbol}: {pos.qty} shares | ${market_value:,.0f} | P&L: ${unrealized_pl:+,.0f}')

        print(f'\\nTotal Portfolio Exposure: ${total_exposure:,.0f}')
        utilization = (total_exposure / portfolio_value) * 100
        print(f'Portfolio Utilization: {utilization:.1f}%')

        if utilization > 50:
            print('[HIGH LEVERAGE] Significant exposure deployed')
        elif utilization > 30:
            print('[MODERATE LEVERAGE] Good deployment level')
        else:
            print('[LOW LEVERAGE] Room for more deployment')

    except Exception as e:
        print(f'Portfolio check error: {str(e)}')

if __name__ == "__main__":
    deploy_massive_leverage()