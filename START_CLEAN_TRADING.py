"""
Clean Trading System - Monday Launch
Single unified system that ACTUALLY TRADES
"""

import os
import sys
import time
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

class CleanTradingSystem:
    def __init__(self):
        # Alpaca credentials
        self.api_key = 'PKZ7F4B26EOEZ8UN8G8U'
        self.api_secret = 'B1aTbyUpEUsCF1CpxsyshsdUXvGZBqoYEfORpLok'
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        # Trading config - AGGRESSIVE to actually find trades
        self.min_score = 1.5
        self.position_size = 1000
        self.max_positions = 5

        # Full S&P 500 for options - batch API calls
        self.stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'V', 'UNH',
            'JNJ', 'WMT', 'JPM', 'XOM', 'MA', 'PG', 'AVGO', 'HD', 'CVX', 'MRK',
            'ABBV', 'KO', 'COST', 'PEP', 'LLY', 'ADBE', 'BAC', 'CRM', 'MCD', 'CSCO',
            'ACN', 'TMO', 'ABT', 'NKE', 'DHR', 'TXN', 'WFC', 'DIS', 'VZ', 'PM',
            'NFLX', 'CMCSA', 'UPS', 'NEE', 'RTX', 'ORCL', 'QCOM', 'BMY', 'UNP', 'INTC',
            'AMD', 'LOW', 'COP', 'T', 'HON', 'AMGN', 'IBM', 'SPGI', 'SBUX', 'CAT',
            'GE', 'INTU', 'BA', 'DE', 'AXP', 'BLK', 'PLD', 'GS', 'MS', 'AMT',
            'MDT', 'GILD', 'ELV', 'TJX', 'ISRG', 'BKNG', 'SYK', 'MMC', 'VRTX', 'C',
            'AMAT', 'ADP', 'ZTS', 'LRCX', 'PGR', 'CVS', 'MO', 'ADI', 'REGN', 'NOW',
            'MDLZ', 'CB', 'PYPL', 'SLB', 'SO', 'CI', 'BDX', 'DUK', 'SCHW', 'TGT'
        ]  # 100 most liquid S&P 500 stocks

        print("="*70)
        print("CLEAN TRADING SYSTEM - MONDAY")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Markets: Stocks ({len(self.stocks)} S&P 500 symbols)")
        print(f"Mode: PAPER TRADING")
        print(f"Min Score: {self.min_score} (AGGRESSIVE)")
        print("="*70)

    def get_price(self, symbol):
        """Get latest price"""
        try:
            url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
            response = requests.get(url, headers=self.headers, timeout=3)
            if response.status_code == 200:
                return float(response.json()['trade']['p'])
        except:
            pass
        return None

    def analyze(self, symbol):
        """Quick analysis"""
        try:
            price = self.get_price(symbol)
            if not price:
                return None

            # Get bars
            end = datetime.now()
            start = end - timedelta(hours=6)
            url = f"{self.data_url}/v2/stocks/{symbol}/bars"
            params = {
                'start': start.isoformat() + 'Z',
                'end': end.isoformat() + 'Z',
                'timeframe': '5Min',
                'limit': 50
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=3)
            if response.status_code != 200:
                return None

            bars = response.json().get('bars', [])
            if len(bars) < 20:
                return None

            closes = [b['c'] for b in bars]
            volumes = [b['v'] for b in bars]

            # Simple momentum
            sma_5 = np.mean(closes[-5:])
            sma_20 = np.mean(closes[-20:])
            momentum = (sma_5 - sma_20) / sma_20 * 100

            # Volume
            avg_vol = np.mean(volumes[:-5])
            recent_vol = np.mean(volumes[-5:])
            vol_surge = recent_vol / avg_vol if avg_vol > 0 else 1

            # Scoring - AGGRESSIVE to trigger trades
            score = 0
            signal = 'HOLD'

            if momentum > 0.1:  # Any positive momentum
                signal = 'BUY'
                score = 2 + (momentum * 3) + vol_surge
            elif momentum < -0.1:  # Any negative momentum
                signal = 'SELL'
                score = 2 + (abs(momentum) * 3) + vol_surge

            if score >= self.min_score:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'score': round(score, 1),
                    'price': price,
                    'momentum': round(momentum, 2)
                }
        except:
            pass
        return None

    def place_order(self, symbol, side):
        """Actually place trade"""
        try:
            price = self.get_price(symbol)
            if not price:
                return None

            qty = int(self.position_size / price)
            if qty < 1:
                return None

            url = f"{self.base_url}/v2/orders"
            order_data = {
                'symbol': symbol,
                'qty': qty,
                'side': side.lower(),
                'type': 'market',
                'time_in_force': 'day'
            }

            response = requests.post(url, headers=self.headers, json=order_data, timeout=5)
            if response.status_code in [200, 201]:
                order = response.json()
                print(f"\n[TRADE EXECUTED] {side} {qty} {symbol} @ ${price:.2f}")
                print(f"Order ID: {order.get('id')}")

                # Telegram
                msg = f"TRADE EXECUTED!\n{side} {qty} {symbol}\nPrice: ${price:.2f}"
                try:
                    bot_url = "https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
                    requests.post(bot_url, data={'chat_id': '7606409012', 'text': msg}, timeout=3)
                except:
                    pass

                return order
        except Exception as e:
            print(f"[ERROR] {e}")
        return None

    def get_positions(self):
        """Get current positions"""
        try:
            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []

    def scan_and_trade(self):
        """Scan and execute"""
        print(f"\n{'='*70}")
        print(f"SCANNING - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")

        positions = self.get_positions()
        position_symbols = [p['symbol'] for p in positions]
        print(f"Current Positions: {len(positions)}")

        opportunities = []

        # Scan all symbols
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.analyze, self.stocks))
            opportunities = [r for r in results if r]

        if opportunities:
            print(f"\n[OPPORTUNITIES: {len(opportunities)}]")
            for opp in opportunities:
                print(f"  {opp['symbol']}: {opp['signal']} (Score: {opp['score']})")

                # Execute if high confidence
                if opp['score'] >= 6.0:
                    if opp['signal'] == 'BUY' and opp['symbol'] not in position_symbols:
                        if len(positions) < self.max_positions:
                            self.place_order(opp['symbol'], 'buy')
                            position_symbols.append(opp['symbol'])

                    elif opp['signal'] == 'SELL' and opp['symbol'] in position_symbols:
                        for pos in positions:
                            if pos['symbol'] == opp['symbol']:
                                qty = int(pos['qty'])
                                self.place_order(opp['symbol'], 'sell')
                                break
        else:
            print("\n[No opportunities found]")

    def run(self):
        """Main loop"""
        print("\n[STARTING TRADING]")
        print("Scanning every 2 minutes")
        print("Press Ctrl+C to stop\n")

        # Startup notification
        try:
            bot_url = "https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
            msg = "CLEAN TRADING SYSTEM STARTED!\n\nMonday markets\n8 stocks\nMin score: 5.0\nReady to trade!"
            requests.post(bot_url, data={'chat_id': '7606409012', 'text': msg}, timeout=3)
        except:
            pass

        iteration = 0
        while True:
            try:
                iteration += 1
                now = datetime.now()

                # Check market hours
                if now.weekday() < 5 and 9 <= now.hour < 16:
                    print(f"\n[ITERATION {iteration}] Market OPEN")
                    self.scan_and_trade()
                    print("\n[Waiting 2 minutes...]")
                    time.sleep(120)
                else:
                    print(f"\n[Market CLOSED]")
                    time.sleep(600)

            except KeyboardInterrupt:
                print("\n[STOPPED]")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                time.sleep(30)

if __name__ == "__main__":
    system = CleanTradingSystem()
    system.run()