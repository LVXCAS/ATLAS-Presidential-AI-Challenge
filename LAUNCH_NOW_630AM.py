"""
IMMEDIATE LAUNCH FOR 6:30 AM PT EXECUTION
Run this NOW - it's 6:03 AM PT, market opens in 27 minutes!
"""

import asyncio
import yfinance as yf
import alpaca_trade_api as tradeapi
import numpy as np
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Quick logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class QuickMarketOpenExecution:
    """Emergency 6:30 AM execution system"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        logging.info("QUICK EXECUTION SYSTEM READY")
        logging.info(f"Current time: {datetime.now().strftime('%H:%M:%S PT')}")

    async def quick_regime_check(self):
        """Ultra-fast regime check"""

        logging.info("Quick regime analysis...")

        try:
            # Get SPY data quickly
            spy = yf.download('SPY', period='5d', interval='1d', progress=False)
            current_price = float(spy['Close'].iloc[-1])
            yesterday_price = float(spy['Close'].iloc[-2])

            # Simple momentum check
            daily_momentum = (current_price / yesterday_price - 1) * 100

            if daily_momentum > 0.5:
                confidence = 0.75
                action = "BUY_MORE"
                logging.info(f"BULLISH: {daily_momentum:+.1f}% momentum")
            elif daily_momentum > -0.5:
                confidence = 0.60
                action = "HOLD"
                logging.info(f"NEUTRAL: {daily_momentum:+.1f}% momentum")
            else:
                confidence = 0.45
                action = "REDUCE"
                logging.info(f"BEARISH: {daily_momentum:+.1f}% momentum")

            return {
                'action': action,
                'confidence': confidence,
                'momentum': daily_momentum
            }

        except Exception as e:
            logging.error(f"Quick regime check failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.5, 'momentum': 0}

    async def execute_at_open(self):
        """Execute trades at market open"""

        logging.info("="*50)
        logging.info("EXECUTING AT MARKET OPEN NOW!")
        logging.info("="*50)

        try:
            # Get account status
            account = self.alpaca.get_account()
            portfolio_value = float(account.portfolio_value)
            buying_power = float(account.buying_power)

            logging.info(f"Portfolio: ${portfolio_value:,.0f}")
            logging.info(f"Buying Power: ${buying_power:,.0f}")

            # Quick regime check
            regime = await self.quick_regime_check()

            if regime['confidence'] < 0.55:
                logging.info("Low confidence - holding positions")
                return

            # Get current positions
            positions = self.alpaca.list_positions()

            if regime['action'] == "BUY_MORE" and buying_power > 10000:
                # Buy more TQQQ with available cash
                tqqq_price = yf.Ticker('TQQQ').history(period='1d', interval='1m').iloc[-1]['Close']
                shares_to_buy = int(min(buying_power * 0.5, 25000) / tqqq_price)

                if shares_to_buy > 0:
                    order = self.alpaca.submit_order(
                        symbol='TQQQ',
                        qty=shares_to_buy,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    logging.info(f"BOUGHT {shares_to_buy} TQQQ at market - Order: {order.id}")

            elif regime['action'] == "REDUCE":
                # Reduce leveraged positions by 20%
                for pos in positions:
                    if pos.symbol in ['TQQQ', 'SOXL', 'IWM'] and int(pos.qty) > 100:
                        shares_to_sell = int(int(pos.qty) * 0.2)

                        order = self.alpaca.submit_order(
                            symbol=pos.symbol,
                            qty=shares_to_sell,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        logging.info(f"SOLD {shares_to_sell} {pos.symbol} - Order: {order.id}")

            logging.info("Market open execution completed!")

        except Exception as e:
            logging.error(f"Execution error: {e}")

def main():
    """Run immediate execution"""

    print("IMMEDIATE MARKET OPEN EXECUTION")
    print("===============================")
    print(f"Time: {datetime.now().strftime('%H:%M:%S PT')}")
    print("Market opens at 6:30 AM PT")
    print("===============================")

    executor = QuickMarketOpenExecution()

    # Run immediately if close to market open
    now = datetime.now()
    if now.hour == 6 and now.minute >= 3:  # 6:03 AM or later - RUN NOW!
        print("EXECUTING NOW - MARKET OPENING IN 27 MINUTES!")
        print("Getting ready for optimal entry...")
        asyncio.run(executor.execute_at_open())
    else:
        print(f"Waiting for market open at 6:30 AM PT...")
        print(f"Current time: {now.strftime('%H:%M:%S PT')}")

if __name__ == "__main__":
    main()