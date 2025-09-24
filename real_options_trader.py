"""
REAL OPTIONS TRADING SYSTEM
Executes actual options contracts with Level 3 permissions
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('real_options_trading.log'),
        logging.StreamHandler()
    ]
)

class RealOptionsTrader:
    """Trade actual options contracts using Alpaca Level 3 permissions"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        self.max_options_allocation = 0.50  # 50% of portfolio in options
        self.min_premium = 0.25             # Min $0.25 premium
        self.max_contracts_per_trade = 50   # Max 50 contracts per trade
        self.trading_active = True

    def build_options_symbol(self, ticker, exp_date, option_type, strike):
        """Build proper options symbol format for Alpaca"""
        # Format: AAPL241220C00150000 (AAPL Dec 20 2024 Call $150.00)
        exp_str = exp_date.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"{ticker}{exp_str}{option_type}{strike_str}"

    async def get_options_chain(self, ticker):
        """Get available options for a ticker"""
        try:
            # Get stock price first
            latest_trade = self.alpaca.get_latest_trade(ticker)
            current_price = float(latest_trade.price)

            # Calculate strike prices around current price
            strikes = []
            for offset in [-0.1, -0.05, 0, 0.05, 0.1]:  # ±10% from current
                strike = round(current_price * (1 + offset), 0)
                strikes.append(strike)

            # Use next Friday as expiration (weekly options)
            exp_date = datetime.now() + timedelta(days=(4 - datetime.now().weekday()) % 7 + 7)

            options = []
            for strike in strikes:
                # Create both calls and puts
                for option_type in ['C', 'P']:
                    option_symbol = self.build_options_symbol(ticker, exp_date, option_type, strike)
                    options.append({
                        'symbol': option_symbol,
                        'ticker': ticker,
                        'strike': strike,
                        'exp_date': exp_date,
                        'option_type': option_type,
                        'current_price': current_price
                    })

            return options

        except Exception as e:
            logging.error(f"Error getting options chain for {ticker}: {e}")
            return []

    async def execute_cash_secured_put(self, ticker, strike, premium_estimate):
        """Sell cash-secured puts"""

        try:
            # Get account info
            account = self.alpaca.get_account()
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)

            # Calculate max contracts based on cash to secure - be more aggressive
            cash_required_per_contract = strike * 100
            max_contracts = min(
                int(cash * 0.8 / cash_required_per_contract),  # Use 80% of cash
                self.max_contracts_per_trade
            )

            if max_contracts > 0:
                # Build options symbol for next Friday
                exp_date = datetime.now() + timedelta(days=(4 - datetime.now().weekday()) % 7 + 7)
                options_symbol = self.build_options_symbol(ticker, exp_date, 'P', strike)

                logging.info(f"SELLING CASH-SECURED PUTS: {options_symbol}")
                logging.info(f"Contracts: {max_contracts}")
                logging.info(f"Strike: ${strike}")
                logging.info(f"Cash Required: ${cash_required_per_contract * max_contracts:,.0f}")

                # Execute the actual options trade
                order = self.alpaca.submit_order(
                    symbol=options_symbol,
                    qty=max_contracts,
                    side='sell',  # Sell puts
                    type='market',
                    time_in_force='day'
                )

                logging.info(f"PUT ORDER EXECUTED: {order.id}")

                # Save execution record
                execution = {
                    'timestamp': datetime.now().isoformat(),
                    'strategy': 'CASH_SECURED_PUT',
                    'options_symbol': options_symbol,
                    'ticker': ticker,
                    'contracts': max_contracts,
                    'strike': strike,
                    'side': 'sell',
                    'order_id': order.id,
                    'cash_required': cash_required_per_contract * max_contracts,
                    'status': 'EXECUTED'
                }

                with open('real_options_executions.json', 'a') as f:
                    f.write(json.dumps(execution) + '\n')

                return True
            else:
                logging.warning(f"Insufficient cash for {ticker} puts")
                return False

        except Exception as e:
            logging.error(f"Cash-secured put execution failed: {e}")
            return False

    async def execute_long_calls(self, ticker, strike, premium_estimate):
        """Buy call options for leverage"""

        try:
            account = self.alpaca.get_account()
            portfolio_value = float(account.portfolio_value)

            # Calculate position size for calls
            max_investment = portfolio_value * self.max_options_allocation
            premium_cost_per_contract = premium_estimate * 100  # $100 per point
            max_contracts = min(
                int(max_investment / premium_cost_per_contract),
                self.max_contracts_per_trade
            )

            if max_contracts > 0:
                # Build options symbol
                exp_date = datetime.now() + timedelta(days=(4 - datetime.now().weekday()) % 7 + 7)
                options_symbol = self.build_options_symbol(ticker, exp_date, 'C', strike)

                logging.info(f"BUYING CALL OPTIONS: {options_symbol}")
                logging.info(f"Contracts: {max_contracts}")
                logging.info(f"Strike: ${strike}")
                logging.info(f"Max Investment: ${max_investment:,.0f}")

                # Execute the actual options trade
                order = self.alpaca.submit_order(
                    symbol=options_symbol,
                    qty=max_contracts,
                    side='buy',  # Buy calls
                    type='market',
                    time_in_force='day'
                )

                logging.info(f"CALL ORDER EXECUTED: {order.id}")

                # Save execution record
                execution = {
                    'timestamp': datetime.now().isoformat(),
                    'strategy': 'LONG_CALLS',
                    'options_symbol': options_symbol,
                    'ticker': ticker,
                    'contracts': max_contracts,
                    'strike': strike,
                    'side': 'buy',
                    'order_id': order.id,
                    'max_investment': max_investment,
                    'status': 'EXECUTED'
                }

                with open('real_options_executions.json', 'a') as f:
                    f.write(json.dumps(execution) + '\n')

                return True
            else:
                logging.warning(f"Insufficient funds for {ticker} calls")
                return False

        except Exception as e:
            logging.error(f"Long calls execution failed: {e}")
            return False

    async def get_high_volatility_stocks(self):
        """Get stocks suitable for options trading"""

        try:
            with open('mega_discovery_20250918_2114.json', 'r') as f:
                discovery = json.load(f)

            # Get high-volatility stocks from discovery
            opportunities = discovery.get('top_opportunities', [])
            high_vol_stocks = []

            for opp in opportunities[:10]:
                if opp.get('implied_volatility', 0) > 0.5:  # >50% IV
                    high_vol_stocks.append({
                        'ticker': opp['ticker'],
                        'price': opp['price'],
                        'iv': opp['implied_volatility'],
                        'score': opp['score']
                    })

            return high_vol_stocks[:5]  # Top 5

        except Exception as e:
            logging.warning(f"Using fallback high-vol stocks: {e}")
            return [
                {'ticker': 'TSLA', 'price': 250, 'iv': 0.8, 'score': 95},
                {'ticker': 'RIVN', 'price': 15, 'iv': 1.2, 'score': 90},
                {'ticker': 'LCID', 'price': 3, 'iv': 0.9, 'score': 85},
                {'ticker': 'SNAP', 'price': 8, 'iv': 1.1, 'score': 88},
                {'ticker': 'AMD', 'price': 140, 'iv': 0.7, 'score': 82}
            ]

    async def real_options_trading_loop(self):
        """Main real options trading loop"""

        logging.info("REAL OPTIONS TRADING ACTIVE")
        logging.info("Level 3 Options Permissions - Executing Actual Contracts")
        logging.info("=" * 60)

        cycle = 0

        while self.trading_active and cycle < 10:  # Run 10 aggressive cycles
            cycle += 1
            logging.info(f"OPTIONS CYCLE #{cycle}")

            try:
                # Get account status
                account = self.alpaca.get_account()
                portfolio_value = float(account.portfolio_value)
                cash = float(account.cash)

                logging.info(f"Portfolio: ${portfolio_value:,.0f}")
                logging.info(f"Cash: ${cash:,.0f}")
                logging.info(f"Options Level: 3 (All strategies available)")

                # Get high-volatility stocks for options
                stocks = await self.get_high_volatility_stocks()
                logging.info(f"High-volatility candidates: {len(stocks)}")

                executions = 0

                for stock in stocks[:5]:  # Execute on 5 stocks per cycle
                    ticker = stock['ticker']
                    current_price = stock['price']
                    iv = stock['iv']

                    logging.info(f"ANALYZING: {ticker} at ${current_price}, IV: {iv:.1%}")

                    # Calculate strike prices
                    put_strike = round(current_price * 0.95, 0)  # 5% OTM put
                    call_strike = round(current_price * 1.05, 0)  # 5% OTM call

                    # Estimate premiums based on IV (simplified)
                    put_premium = current_price * 0.02 * iv  # Rough estimate
                    call_premium = current_price * 0.03 * iv  # Rough estimate

                    if put_premium >= self.min_premium:
                        # Execute cash-secured puts
                        success = await self.execute_cash_secured_put(ticker, put_strike, put_premium)
                        if success:
                            executions += 1

                        await asyncio.sleep(5)  # Delay between trades

                    if call_premium >= self.min_premium:
                        # Execute long calls
                        success = await self.execute_long_calls(ticker, call_strike, call_premium)
                        if success:
                            executions += 1

                        await asyncio.sleep(5)  # Delay between trades

                logging.info(f"CYCLE #{cycle} COMPLETE: {executions} options trades executed")

            except Exception as e:
                logging.error(f"Options cycle error: {e}")

            # Wait before next cycle
            await asyncio.sleep(600)  # 10 minutes

async def main():
    print("REAL OPTIONS TRADING SYSTEM")
    print("Level 3 Options - Cash-Secured Puts & Long Calls")
    print("Target: 40% monthly ROI through actual options contracts")
    print("=" * 60)

    trader = RealOptionsTrader()
    await trader.real_options_trading_loop()

if __name__ == "__main__":
    asyncio.run(main())