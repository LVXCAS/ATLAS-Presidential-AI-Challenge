"""
OPTIONS EXECUTOR READY
Options Executor - Actually places options trades
Adds execution capability to options scanning
"""

import requests
import json
from datetime import datetime, timedelta

class OptionsExecutor:
    def __init__(self):
        self.api_key = 'PKZ7F4B26EOEZ8UN8G8U'
        self.api_secret = 'B1aTbyUpEUsCF1CpxsyshsdUXvGZBqoYEfORpLok'
        self.base_url = "https://paper-api.alpaca.markets"

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        self.max_positions = 3
        self.position_size = 500  # $500 per options position

    def place_options_order(self, symbol, strategy_type, current_price):
        """Actually place an options trade"""

        # Calculate strikes based on strategy
        if strategy_type == 'LONG_CALL':
            strike = round(current_price * 1.02)  # 2% OTM call
            contracts = [{
                'action': 'buy',
                'type': 'call',
                'strike': strike,
                'qty': max(1, int(self.position_size / 100))
            }]

        elif strategy_type == 'LONG_PUT':
            strike = round(current_price * 0.98)  # 2% OTM put
            contracts = [{
                'action': 'buy',
                'type': 'put', 
                'strike': strike,
                'qty': max(1, int(self.position_size / 100))
            }]

        elif strategy_type == 'BULL_PUT_SPREAD':
            # Sell higher strike put, buy lower strike put
            contracts = [
                {'action': 'sell', 'type': 'put', 'strike': round(current_price * 0.98), 'qty': 1},
                {'action': 'buy', 'type': 'put', 'strike': round(current_price * 0.95), 'qty': 1}
            ]

        elif strategy_type == 'IRON_CONDOR':
            # Four-legged strategy
            contracts = [
                {'action': 'sell', 'type': 'put', 'strike': round(current_price * 0.97), 'qty': 1},
                {'action': 'buy', 'type': 'put', 'strike': round(current_price * 0.94), 'qty': 1},
                {'action': 'sell', 'type': 'call', 'strike': round(current_price * 1.03), 'qty': 1},
                {'action': 'buy', 'type': 'call', 'strike': round(current_price * 1.06), 'qty': 1}
            ]
        else:
            return None

        # Get next Friday expiry
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        expiry = today + timedelta(days=days_until_friday)
        expiry_str = expiry.strftime('%Y%m%d')

        # Build order for each leg
        orders_placed = []
        for contract in contracts:
            # Format: SPY241025C450 (symbol + expiry + type + strike)
            option_symbol = f"{symbol}{expiry_str}{contract['type'][0].upper()}{contract['strike']:05d}000"

            order_data = {
                'symbol': option_symbol,
                'qty': contract['qty'],
                'side': 'buy' if contract['action'] == 'buy' else 'sell',
                'type': 'market',
                'time_in_force': 'day',
                'order_class': 'simple'
            }

            try:
                # Place the order
                url = f"{self.base_url}/v2/orders"
                response = requests.post(url, headers=self.headers, json=order_data, timeout=5)

                if response.status_code in [200, 201]:
                    order = response.json()
                    orders_placed.append({
                        'symbol': option_symbol,
                        'side': order_data['side'],
                        'qty': contract['qty'],
                        'order_id': order.get('id')
                    })
                    print(f"OPTIONS ORDER: {order_data['side'].upper()} {contract['qty']} {option_symbol}")
                else:
                    print(f"Failed: {response.text}")

            except Exception as e:
                print(f"Error: {e}")

        return orders_placed if orders_placed else None

    def execute_from_scanner(self, opportunities):
        """Execute trades from scanner opportunities"""
        trades_executed = 0

        for opp in opportunities[:self.max_positions]:  # Limit positions
            if opp['score'] >= 6.0:  # High confidence only
                print(f"\n[EXECUTING] {opp['symbol']} - {opp['strategy']}")

                orders = self.place_options_order(
                    opp['symbol'],
                    opp['strategy'],
                    opp['price']
                )

                if orders:
                    trades_executed += 1

                    # Send Telegram alert
                    message = f"OPTIONS TRADE EXECUTED!\n"
                    message += f"{opp['symbol']} - {opp['strategy']}\n"
                    message += f"Score: {opp['score']}/10\n"
                    message += f"Orders: {len(orders)} legs"
                    self.send_telegram(message)

        return trades_executed

    def send_telegram(self, message):
        """Send trade alerts"""
        try:
            bot_token = "8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ"
            chat_id = "7606409012"
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

            requests.post(url, data={'chat_id': chat_id, 'text': message}, timeout=3)
        except:
            pass

if __name__ == "__main__":
    executor = OptionsExecutor()
    print("Options Executor Ready")
    print("Add this to your options scanner to execute trades!")
