"""
S&P 500 Efficient Scanner - Smart API Usage
Scans entire S&P 500 without hitting rate limits
"""

import os
import time
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

class SP500EfficientScanner:
    def __init__(self):
        self.api_key = 'PKZ7F4B26EOEZ8UN8G8U'
        self.api_secret = 'B1aTbyUpEUsCF1CpxsyshsdUXvGZBqoYEfORpLok'
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        # S&P 500 symbols (top 100 most liquid for starters)
        self.sp500 = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'V', 'UNH',
            'JNJ', 'WMT', 'JPM', 'XOM', 'MA', 'PG', 'AVGO', 'HD', 'CVX', 'MRK',
            'ABBV', 'KO', 'COST', 'PEP', 'LLY', 'ADBE', 'BAC', 'CRM', 'MCD', 'CSCO',
            'ACN', 'TMO', 'ABT', 'NKE', 'DHR', 'TXN', 'WFC', 'DIS', 'VZ', 'PM',
            'NFLX', 'CMCSA', 'UPS', 'NEE', 'RTX', 'ORCL', 'QCOM', 'BMY', 'UNP', 'INTC',
            'AMD', 'LOW', 'COP', 'T', 'HON', 'AMGN', 'IBM', 'SPGI', 'SBUX', 'CAT',
            'GE', 'INTU', 'BA', 'DE', 'AXP', 'BLK', 'PLD', 'GS', 'MS', 'AMT',
            'MDT', 'GILD', 'ELV', 'TJX', 'ISRG', 'BKNG', 'SYK', 'MMC', 'VRTX', 'C',
            'AMAT', 'ADP', 'ZTS', 'LRCX', 'PGR', 'CVS', 'MO', 'ADI', 'REGN', 'NOW',
            'MDLZ', 'CB', 'PYPL', 'SLB', 'SO', 'CI', 'BDX', 'DUK', 'SCHW', 'TGT'
        ]

        # Cache for data (reduce API calls)
        self.price_cache = {}
        self.cache_ttl = 60  # 1 minute cache
        self.cache_timestamps = {}

        # Rate limiting
        self.api_calls_per_minute = 200  # Alpaca limit
        self.api_call_count = 0
        self.api_call_reset_time = time.time() + 60

        # Trading config
        self.min_score = 5.5
        self.position_size = 1000
        self.max_positions = 10

        print("="*70)
        print("S&P 500 EFFICIENT SCANNER")
        print("="*70)
        print(f"Symbols: {len(self.sp500)}")
        print(f"API Rate Limit: {self.api_calls_per_minute}/min")
        print(f"Cache TTL: {self.cache_ttl}s")
        print(f"Min Score: {self.min_score}")
        print("="*70)

    def check_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()

        if current_time >= self.api_call_reset_time:
            self.api_call_count = 0
            self.api_call_reset_time = current_time + 60

        if self.api_call_count >= self.api_calls_per_minute:
            sleep_time = self.api_call_reset_time - current_time
            if sleep_time > 0:
                print(f"[RATE LIMIT] Sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.api_call_count = 0
                self.api_call_reset_time = time.time() + 60

        self.api_call_count += 1

    def get_cached_price(self, symbol):
        """Get price from cache or API"""
        current_time = time.time()

        # Check cache
        if symbol in self.price_cache:
            if current_time - self.cache_timestamps.get(symbol, 0) < self.cache_ttl:
                return self.price_cache[symbol]

        # Fetch from API
        self.check_rate_limit()
        try:
            url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
            response = requests.get(url, headers=self.headers, timeout=2)
            if response.status_code == 200:
                price = float(response.json()['trade']['p'])
                self.price_cache[symbol] = price
                self.cache_timestamps[symbol] = current_time
                return price
        except:
            pass
        return None

    def get_batch_quotes(self, symbols):
        """Get multiple quotes in one API call - MOST EFFICIENT"""
        self.check_rate_limit()
        try:
            # Alpaca allows up to 100 symbols in one request
            symbols_str = ','.join(symbols[:100])
            url = f"{self.data_url}/v2/stocks/quotes/latest"
            params = {'symbols': symbols_str}

            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                quotes = data.get('quotes', {})

                # Cache all results
                current_time = time.time()
                prices = {}
                for symbol, quote in quotes.items():
                    if 'ap' in quote and quote['ap']:  # ask price
                        price = float(quote['ap'])
                        self.price_cache[symbol] = price
                        self.cache_timestamps[symbol] = current_time
                        prices[symbol] = price

                return prices
        except Exception as e:
            print(f"[ERROR] Batch quotes: {e}")
        return {}

    def get_snapshots(self, symbols):
        """Get snapshots with OHLCV - SINGLE API CALL for multiple symbols"""
        self.check_rate_limit()
        try:
            symbols_str = ','.join(symbols[:100])
            url = f"{self.data_url}/v2/stocks/snapshots"
            params = {'symbols': symbols_str}

            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"[ERROR] Snapshots: {e}")
        return {}

    def quick_filter(self, snapshots):
        """Pre-filter stocks by volume and volatility to reduce analysis"""
        filtered = []

        for symbol, data in snapshots.items():
            try:
                # Get daily bar
                daily_bar = data.get('dailyBar', {})
                minute_bar = data.get('minuteBar', {})

                if not daily_bar or not minute_bar:
                    continue

                # Volume filter - must have significant volume
                volume = daily_bar.get('v', 0)
                if volume < 500000:  # Less than 500k shares
                    continue

                # Price movement filter
                prev_close = data.get('prevDailyBar', {}).get('c', 0)
                current = minute_bar.get('c', 0)

                if prev_close and current:
                    change_pct = abs((current - prev_close) / prev_close * 100)
                    if change_pct > 0.5:  # Moving at least 0.5%
                        filtered.append({
                            'symbol': symbol,
                            'price': current,
                            'volume': volume,
                            'change': change_pct
                        })
            except:
                continue

        return filtered

    def analyze_symbol_fast(self, symbol_data):
        """Fast analysis using cached snapshot data"""
        try:
            symbol = symbol_data['symbol']
            price = symbol_data['price']
            volume = symbol_data['volume']
            change_pct = symbol_data['change']

            # Simple scoring based on movement and volume
            score = 0

            # High volume = confidence
            if volume > 5000000:  # 5M+ shares
                score += 2
            elif volume > 2000000:
                score += 1

            # Price movement
            if abs(change_pct) > 2:
                score += 3
            elif abs(change_pct) > 1:
                score += 2
            elif abs(change_pct) > 0.5:
                score += 1

            # Determine signal
            signal = 'BUY' if change_pct > 0 else 'SELL' if change_pct < 0 else 'HOLD'

            if score >= self.min_score:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'score': score,
                    'price': price,
                    'volume': volume,
                    'change': round(change_pct, 2)
                }
        except:
            pass
        return None

    def scan_efficient(self):
        """Efficient scan using batch API calls"""
        print(f"\n{'='*70}")
        print(f"EFFICIENT SCAN - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")

        # STEP 1: Get snapshots in batches (minimize API calls)
        all_snapshots = {}
        batch_size = 100

        for i in range(0, len(self.sp500), batch_size):
            batch = self.sp500[i:i+batch_size]
            print(f"[BATCH {i//batch_size + 1}] Fetching {len(batch)} symbols...")
            snapshots = self.get_snapshots(batch)
            all_snapshots.update(snapshots)

            # Small delay between batches
            if i + batch_size < len(self.sp500):
                time.sleep(0.5)

        print(f"[FETCHED] {len(all_snapshots)} snapshots")

        # STEP 2: Quick filter to reduce processing
        filtered = self.quick_filter(all_snapshots)
        print(f"[FILTERED] {len(filtered)} high-volume movers")

        # STEP 3: Analyze only filtered stocks
        opportunities = []
        for stock_data in filtered:
            result = self.analyze_symbol_fast(stock_data)
            if result:
                opportunities.append(result)

        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        # Display results
        if opportunities:
            print(f"\n[OPPORTUNITIES: {len(opportunities)}]")
            for opp in opportunities[:10]:  # Top 10
                print(f"  {opp['symbol']}: {opp['signal']} "
                      f"(Score: {opp['score']}, Change: {opp['change']:+.2f}%, "
                      f"Vol: {opp['volume']/1000000:.1f}M)")
        else:
            print("\n[No opportunities found]")

        return opportunities

    def place_order(self, symbol, side):
        """Execute trade"""
        try:
            price = self.get_cached_price(symbol)
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

            self.check_rate_limit()
            response = requests.post(url, headers=self.headers, json=order_data, timeout=5)

            if response.status_code in [200, 201]:
                order = response.json()
                print(f"\n[EXECUTED] {side} {qty} {symbol} @ ${price:.2f}")

                # Telegram
                try:
                    msg = f"SP500 TRADE!\n{side} {qty} {symbol}\nPrice: ${price:.2f}"
                    bot_url = "https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
                    requests.post(bot_url, data={'chat_id': '7606409012', 'text': msg}, timeout=3)
                except:
                    pass

                return order
        except Exception as e:
            print(f"[ERROR] Order: {e}")
        return None

    def get_positions(self):
        """Get current positions"""
        self.check_rate_limit()
        try:
            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []

    def execute_opportunities(self, opportunities):
        """Execute top opportunities"""
        positions = self.get_positions()
        position_symbols = [p['symbol'] for p in positions]

        executed = 0
        for opp in opportunities[:5]:  # Top 5
            if opp['score'] >= 6.5:  # High confidence
                if opp['signal'] == 'BUY' and opp['symbol'] not in position_symbols:
                    if len(positions) < self.max_positions:
                        if self.place_order(opp['symbol'], 'buy'):
                            executed += 1
                            position_symbols.append(opp['symbol'])

        print(f"\n[TRADES EXECUTED: {executed}]")
        return executed

    def run(self):
        """Main loop"""
        print("\n[STARTING EFFICIENT S&P 500 SCANNER]")
        print("Scans every 3 minutes")
        print("Uses batch API calls to minimize requests\n")

        # Startup notification
        try:
            msg = "SP500 EFFICIENT SCANNER STARTED!\n\n100 symbols\nBatch API calls\nSmart caching\nRate limiting\n\nReady to scan!"
            bot_url = "https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
            requests.post(bot_url, data={'chat_id': '7606409012', 'text': msg}, timeout=3)
        except:
            pass

        iteration = 0
        while True:
            try:
                iteration += 1
                now = datetime.now()

                # Market hours check
                if now.weekday() < 5 and 9 <= now.hour < 16:
                    print(f"\n[ITERATION {iteration}]")
                    opportunities = self.scan_efficient()
                    self.execute_opportunities(opportunities)

                    print(f"\n[API Calls: {self.api_call_count}/{self.api_calls_per_minute}]")
                    print("[Waiting 3 minutes...]")
                    time.sleep(180)
                else:
                    print("\n[Market CLOSED]")
                    time.sleep(600)

            except KeyboardInterrupt:
                print("\n[STOPPED]")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                time.sleep(30)

if __name__ == "__main__":
    scanner = SP500EfficientScanner()
    scanner.run()