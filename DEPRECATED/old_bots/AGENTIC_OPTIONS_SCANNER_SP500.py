"""
Agentic Options Scanner - Full S&P 500 Coverage
Autonomous AI-driven options discovery across all 500 stocks
Enhanced with institutional-grade quant libraries (TA-Lib, pandas-ta)
"""

import time
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# QUANT LIBRARIES - Institutional-grade technical analysis
try:
    import talib  # TA-Lib: 200+ technical indicators
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARNING] TA-Lib not available - using basic calculations")

try:
    import pandas_ta as ta  # pandas-ta: 130+ indicators
    PANDASTA_AVAILABLE = True
except ImportError:
    PANDASTA_AVAILABLE = False
    print("[WARNING] pandas-ta not available - using basic calculations")

class AgenticOptionsScanner:
    def __init__(self):
        # WORKING API KEYS - Trading enabled
        self.api_key = os.getenv('ALPACA_API_KEY', 'PKOWU7D6JANXP47ZU72X72757D')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY', '52QKewJCoafjLsFPKJJSTZs7BG7XBa6mLwi3e1W3Z7Tq')
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        # Top 20 most liquid symbols for FAST OPTIONS scanning (yfinance compatible)
        self.sp500_symbols = [
            'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'META',
            'AMZN', 'GOOGL', 'NFLX', 'JPM', 'BAC', 'XOM', 'DIS', 'F',
            'NIO', 'PLTR', 'SOFI', 'BABA'
        ]

        # Agentic parameters
        self.min_score = 1.0  # AGGRESSIVE - will execute trades
        self.max_concurrent = 20  # Parallel processing
        self.learning_memory = {}  # Track successful patterns
        self.strategy_weights = {
            'IRON_CONDOR': 1.2,      # Prefer market neutral
            'BULL_PUT_SPREAD': 1.1,  # Income generation
            'LONG_CALL': 1.0,        # Directional plays
            'BUTTERFLY': 1.3,        # Low risk, high reward
            'CALENDAR_SPREAD': 1.15  # Time decay advantage
        }

        print("=" * 70)
        print("AGENTIC OPTIONS SCANNER - S&P 500")
        print("=" * 70)
        print(f"Coverage: {len(self.sp500_symbols)} stocks")
        print(f"Min Score: {self.min_score}")
        print(f"Max Concurrent: {self.max_concurrent}")
        print("Mode: AUTONOMOUS AI-DRIVEN")
        if TALIB_AVAILABLE:
            print("Quant Libraries: TA-Lib ENABLED (Professional)")
        if PANDASTA_AVAILABLE:
            print("Quant Libraries: pandas-ta ENABLED (130+ indicators)")
        print("=" * 70)

    def get_market_data_batch(self, symbol):
        """Get market data using POLYGON API (FAST <1ms latency!)"""
        try:
            import requests
            from datetime import datetime, timedelta

            polygon_key = os.getenv('POLYGON_API_KEY')

            # Get 60 days of data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

            # Polygon aggregates endpoint (fast!)
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {'apiKey': polygon_key}

            response = requests.get(url, params=params, timeout=5)

            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get('results', [])

            if len(results) < 20:
                return None

            # Current price = most recent close
            price = results[-1]['c']

            # Convert Polygon format to expected format
            bars = []
            for bar in results:
                bars.append({
                    'c': bar['c'],  # close
                    'h': bar['h'],  # high
                    'l': bar['l'],  # low
                    'o': bar['o'],  # open
                    'v': bar['v']   # volume
                })

            return {
                'symbol': symbol,
                'price': price,
                'bars': bars
            }
        except Exception as e:
            return None

    def calculate_advanced_metrics(self, data):
        """Calculate advanced options metrics using INSTITUTIONAL QUANT LIBRARIES"""
        bars = data['bars']

        # Convert to numpy arrays for TA-Lib
        closes = np.array([b['c'] for b in bars], dtype=float)
        highs = np.array([b['h'] for b in bars], dtype=float)
        lows = np.array([b['l'] for b in bars], dtype=float)
        volumes = np.array([b['v'] for b in bars], dtype=float)

        current = data['price']

        # QUANT LIBRARY CALCULATIONS
        if TALIB_AVAILABLE and len(closes) >= 50:
            # TA-Lib: Professional-grade indicators
            sma_20 = talib.SMA(closes, timeperiod=20)[-1]
            sma_50 = talib.SMA(closes, timeperiod=50)[-1]
            rsi = talib.RSI(closes, timeperiod=14)[-1]
            macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(closes, timeperiod=20)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]

            # Advanced momentum using TA-Lib
            momentum = ((current - closes[-20]) / closes[-20] * 100)

            # Volatility using ATR (more accurate than std dev)
            volatility = (atr / current * 100) * np.sqrt(252)

            # Trend strength using ADX
            trend_strength = adx

        else:
            # Fallback: Basic calculations
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
            rsi = 50  # Neutral
            macd_hist = np.array([0])
            bbands_upper = closes[-1:] * 1.02
            bbands_lower = closes[-1:] * 0.98
            momentum = ((current - closes[-20]) / closes[-20] * 100) if len(closes) >= 20 else 0
            volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100 * np.sqrt(252) if len(closes) >= 20 else 20
            trend_strength = abs(momentum)
            adx = 25

        # Price movement analysis
        returns = np.diff(closes) / closes[:-1]

        # IV Rank (enhanced with TA-Lib volatility)
        recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
        hist_vol = np.std(returns)
        iv_rank = min(100, (recent_vol / hist_vol * 50) if hist_vol > 0 else 50)

        # Support/Resistance using recent highs/lows
        resistance = np.max(highs[-20:])
        support = np.min(lows[-20:])

        # Volume analysis
        avg_volume = np.mean(volumes[-20:])
        recent_volume = volumes[-1]
        volume_surge = (recent_volume / avg_volume) if avg_volume > 0 else 1

        # Trend determination using multiple indicators
        if TALIB_AVAILABLE and len(closes) >= 50:
            trend = 'BULLISH' if (current > sma_20 > sma_50 and rsi > 50 and macd_hist[-1] > 0) else \
                    'BEARISH' if (current < sma_20 < sma_50 and rsi < 50 and macd_hist[-1] < 0) else \
                    'NEUTRAL'
        else:
            trend = 'BULLISH' if current > sma_20 > sma_50 else 'BEARISH' if current < sma_20 < sma_50 else 'NEUTRAL'

        return {
            'volatility': volatility,
            'iv_rank': iv_rank,
            'momentum': momentum,
            'trend': trend,
            'trend_strength': trend_strength,  # NEW: ADX-based trend strength
            'rsi': rsi if TALIB_AVAILABLE else 50,  # NEW: RSI for overbought/oversold
            'resistance': resistance,
            'support': support,
            'volume_surge': volume_surge,
            'price_position': (current - support) / (resistance - support) if resistance > support else 0.5
        }

    def select_optimal_strategy(self, metrics):
        """AI-driven strategy selection based on market conditions"""
        strategies = []

        # High IV - Sell premium strategies
        if metrics['iv_rank'] > 70:
            if metrics['trend'] == 'NEUTRAL':
                strategies.append(('IRON_CONDOR', 8.0))
                strategies.append(('BUTTERFLY', 7.5))
            elif metrics['trend'] == 'BULLISH':
                strategies.append(('BULL_PUT_SPREAD', 7.0))
                strategies.append(('SHORT_PUT', 6.5))
            else:  # BEARISH
                strategies.append(('BEAR_CALL_SPREAD', 7.0))
                strategies.append(('SHORT_CALL', 6.5))

        # Low IV - Buy premium strategies
        elif metrics['iv_rank'] < 30:
            if abs(metrics['momentum']) > 5:
                if metrics['momentum'] > 0:
                    strategies.append(('LONG_CALL', 7.5))
                    strategies.append(('CALL_DEBIT_SPREAD', 6.5))
                else:
                    strategies.append(('LONG_PUT', 7.5))
                    strategies.append(('PUT_DEBIT_SPREAD', 6.5))
            else:
                strategies.append(('CALENDAR_SPREAD', 6.0))

        # Medium IV with directional bias
        else:
            if metrics['volume_surge'] > 1.5:  # High volume interest
                if metrics['trend'] == 'BULLISH':
                    strategies.append(('BULL_CALL_SPREAD', 6.5))
                elif metrics['trend'] == 'BEARISH':
                    strategies.append(('BEAR_PUT_SPREAD', 6.5))
                else:
                    strategies.append(('STRADDLE', 6.0))

        # Apply strategy weights from learning
        weighted_strategies = []
        for strategy, base_score in strategies:
            weight = self.strategy_weights.get(strategy, 1.0)
            weighted_score = base_score * weight

            # Boost score based on market conditions
            if metrics['volume_surge'] > 2:
                weighted_score += 0.5
            if metrics['price_position'] < 0.2 or metrics['price_position'] > 0.8:
                weighted_score += 0.3  # Near support/resistance

            weighted_strategies.append((strategy, weighted_score))

        # Return best strategy
        if weighted_strategies:
            best = max(weighted_strategies, key=lambda x: x[1])
            return best
        return None

    def scan_symbol(self, symbol):
        """Scan individual symbol for options opportunities"""
        try:
            data = self.get_market_data_batch(symbol)
            if not data:
                return None

            metrics = self.calculate_advanced_metrics(data)
            strategy_result = self.select_optimal_strategy(metrics)

            if strategy_result and strategy_result[1] >= self.min_score:
                strategy, score = strategy_result

                # Calculate strike recommendations
                price = data['price']
                if 'PUT' in strategy:
                    strike_low = round(price * 0.95, 0)
                    strike_high = round(price * 0.98, 0)
                else:
                    strike_low = round(price * 1.02, 0)
                    strike_high = round(price * 1.05, 0)

                return {
                    'symbol': symbol,
                    'price': price,
                    'strategy': strategy,
                    'score': round(score, 1),
                    'iv_rank': round(metrics['iv_rank'], 1),
                    'volatility': round(metrics['volatility'], 1),
                    'momentum': round(metrics['momentum'], 2),
                    'trend': metrics['trend'],
                    'volume_surge': round(metrics['volume_surge'], 2),
                    'strikes': f"${strike_low}-${strike_high}",
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return None

    def autonomous_scan(self):
        """Autonomous parallel scanning of all symbols"""
        print(f"\n{'='*70}")
        print(f"AUTONOMOUS SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        opportunities = []
        failed_count = 0

        # Use ThreadPoolExecutor for parallel scanning
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.scan_symbol, symbol): symbol
                for symbol in self.sp500_symbols
            }

            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    result = future.result(timeout=5)
                    if result:
                        opportunities.append(result)
                        print(f"  [FOUND] {result['symbol']}: {result['strategy']} (Score: {result['score']})")

                        # Update learning memory
                        strategy = result['strategy']
                        if strategy in self.strategy_weights:
                            self.strategy_weights[strategy] *= 1.01  # Reinforce successful patterns
                except:
                    failed_count += 1

                # Progress update every 20 symbols
                if completed % 20 == 0:
                    print(f"  [PROGRESS] {completed}/{len(self.sp500_symbols)} symbols scanned...")

        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n[SCAN COMPLETE]")
        print(f"  Scanned: {len(self.sp500_symbols)} symbols")
        print(f"  Opportunities: {len(opportunities)}")
        print(f"  Failed: {failed_count}")

        # Save top opportunities
        if opportunities:
            # Save detailed file
            filename = f"agentic_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(opportunities[:20], f, indent=2)  # Top 20

            # Display top 5
            print(f"\n[TOP 5 OPPORTUNITIES]")
            for opp in opportunities[:5]:
                print(f"\n  {opp['symbol']} @ ${opp['price']:.2f}")
                print(f"    Strategy: {opp['strategy']}")
                print(f"    Score: {opp['score']}/10")
                print(f"    IV Rank: {opp['iv_rank']}%")
                print(f"    Trend: {opp['trend']}")
                print(f"    Strikes: {opp['strikes']}")

            print(f"\n[Saved {min(20, len(opportunities))} opportunities to {filename}]")

            # EXECUTE TOP OPPORTUNITIES
            self.execute_top_opportunities(opportunities)

        return opportunities

    def execute_top_opportunities(self, opportunities):
        """Execute trades on top opportunities"""
        executed = 0
        max_trades = 3  # Limit to 3 options positions

        for opp in opportunities[:max_trades]:
            if opp['score'] >= 5.0:  # Lowered from 7.0 to capture more trades
                print(f"\n[EXECUTING OPTIONS TRADE]")
                print(f"  {opp['symbol']}: {opp['strategy']} (Score: {opp['score']})")

                # Execute the trade
                result = self.place_options_order(opp)
                if result:
                    executed += 1
                    print(f"  [SUCCESS] Trade #{executed} executed")
                else:
                    print(f"  [FAILED] Execution failed - check logs above")

        print(f"\n[OPTIONS TRADES EXECUTED: {executed}]")
        return executed

    def place_options_order(self, opportunity):
        """Place actual options order"""
        try:
            symbol = opportunity['symbol']
            strategy = opportunity['strategy']
            price = opportunity['price']

            # Simplified execution - place market order
            url = f"https://paper-api.alpaca.markets/v2/orders"
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.api_secret
            }

            # Calculate contract details
            if strategy == 'LONG_CALL':
                strike = round(price * 1.02)
                option_type = 'C'
                side = 'buy'
                print(f"  [LONG_CALL] Strike ${strike}, Side: {side}")
            elif strategy == 'LONG_PUT':
                strike = round(price * 0.98)
                option_type = 'P'
                side = 'buy'
                print(f"  [LONG_PUT] Strike ${strike}, Side: {side}")
            elif strategy == 'BUTTERFLY':
                # Butterfly: Buy 1 ITM call, Sell 2 ATM calls, Buy 1 OTM call
                # Simplified: Execute ATM call as proxy for butterfly
                strike = round(price)
                option_type = 'C'
                side = 'buy'
                print(f"  [BUTTERFLY->SIMPLIFIED] Executing ATM call at ${strike} as proxy")
            elif 'SPREAD' in strategy or 'CALENDAR' in strategy:
                # For complex spreads, execute ITM call as proxy
                strike = round(price * 0.98)
                option_type = 'C'
                side = 'buy'
                print(f"  [{strategy}->SIMPLIFIED] Executing ITM call at ${strike} as proxy")
            else:
                print(f"  [UNKNOWN STRATEGY: {strategy}]")
                return None

            # Get next Friday expiry
            from datetime import timedelta
            today = datetime.now()
            days_until_friday = (4 - today.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            expiry = today + timedelta(days=days_until_friday)
            expiry_str = expiry.strftime('%Y%m%d')

            # Build option symbol: SPY241025C450
            option_symbol = f"{symbol}{expiry_str}{option_type}{strike:05d}000"

            order_data = {
                'symbol': option_symbol,
                'qty': 1,
                'side': side,
                'type': 'market',
                'time_in_force': 'day'
            }

            response = requests.post(url, headers=headers, json=order_data, timeout=5)

            if response.status_code in [200, 201]:
                order = response.json()
                print(f"  [OK] ORDER PLACED: {side.upper()} 1 {option_symbol}")
                print(f"  Order ID: {order.get('id')}")

                # Send Telegram alert
                message = f"🎯 OPTIONS TRADE!\n{symbol} {strategy}\nStrike: ${strike}\nScore: {opportunity['score']}/10"
                try:
                    bot_url = f"https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
                    requests.post(bot_url, data={'chat_id': '7606409012', 'text': message}, timeout=3)
                except:
                    pass

                return order
            else:
                error_msg = response.text if response.text else "No error message"
                print(f"  [FAILED] Order rejected: HTTP {response.status_code}")
                print(f"  Response: {error_msg[:200]}")  # First 200 chars
                return None

        except Exception as e:
            print(f"  [ERROR] Exception during order placement: {e}")
            import traceback
            print(f"  Stack trace: {traceback.format_exc()[:300]}")
            return None

    def send_telegram_alert(self, opportunities):
        """Send top opportunities to Telegram"""
        if not opportunities:
            return

        message = "🎯 *TOP OPTIONS OPPORTUNITIES*\n\n"

        for opp in opportunities[:5]:
            message += f"*{opp['symbol']}* - {opp['strategy']}\n"
            message += f"  Score: {opp['score']}/10 | IV: {opp['iv_rank']}%\n"
            message += f"  Strikes: {opp['strikes']}\n\n"

        try:
            bot_token = "8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ"
            chat_id = "7606409012"
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

            requests.post(url, data={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }, timeout=5)
        except:
            pass

    def run_continuous(self):
        """Run continuous autonomous scanning"""
        print("\n" + "=" * 70)
        print("STARTING AUTONOMOUS AGENTIC SCANNER")
        print("Full S&P 500 Coverage | AI-Driven Strategy Selection")
        print("Scan Interval: 30 minutes during market hours")
        print("Press Ctrl+C to stop")
        print("=" * 70)

        iteration = 0
        while True:
            iteration += 1

            try:
                now = datetime.now()

                # Check if market is open (simplified)
                if now.weekday() < 5 and 9 <= now.hour < 16:
                    print(f"\n[ITERATION #{iteration}]")
                    opportunities = self.autonomous_scan()

                    # Send Telegram alert for top opportunities
                    if opportunities:
                        self.send_telegram_alert(opportunities)

                    print(f"\n[Next scan: {(now + timedelta(minutes=30)).strftime('%H:%M:%S')}]")
                    time.sleep(1800)  # 30 minutes
                else:
                    print(f"\n[MARKET CLOSED - Waiting... Next check in 15 minutes]")
                    time.sleep(900)  # 15 minutes

            except KeyboardInterrupt:
                print("\n[Agentic Scanner stopped by user]")
                break
            except Exception as e:
                print(f"\n[ERROR] {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    scanner = AgenticOptionsScanner()
    scanner.run_continuous()