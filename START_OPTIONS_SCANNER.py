"""
Options Scanner - Live Market Monitor
Scans for high-probability options trades in real-time
"""

import time
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import os

class OptionsScanner:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY', 'PKZ7F4B26EOEZ8UN8G8U')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY', 'B1aTbyUpEUsCF1CpxsyshsdUXvGZBqoYEfORpLok')
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        # Top liquid options stocks
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'AMZN']

        # Score thresholds
        self.min_score = 6.0  # Balanced threshold
        self.trades_log = []

        print("=" * 70)
        print("OPTIONS SCANNER INITIALIZED")
        print("=" * 70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Min Score: {self.min_score}")
        print(f"Mode: PAPER TRADING")
        print("=" * 70)

    def get_price(self, symbol):
        """Get current price for a symbol"""
        try:
            url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['trade']['p'])
        except:
            pass
        return None

    def get_market_data(self, symbol):
        """Get bars data for analysis"""
        try:
            end = datetime.now()
            start = end - timedelta(days=30)

            url = f"{self.data_url}/v2/stocks/{symbol}/bars"
            params = {
                'start': start.isoformat() + 'Z',
                'end': end.isoformat() + 'Z',
                'timeframe': '1Day',
                'limit': 30
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('bars', [])
        except:
            pass
        return []

    def calculate_iv_rank(self, symbol):
        """Calculate simple IV rank based on price movement"""
        bars = self.get_market_data(symbol)
        if len(bars) < 20:
            return 50  # Default middle value

        # Calculate daily returns
        returns = []
        for i in range(1, len(bars)):
            ret = (bars[i]['c'] - bars[i-1]['c']) / bars[i-1]['c']
            returns.append(abs(ret))

        # Current volatility (last 5 days)
        current_vol = sum(returns[-5:]) / 5 if len(returns) >= 5 else 0

        # Historical volatility (all days)
        hist_vol = sum(returns) / len(returns) if returns else 0

        # IV Rank approximation (0-100)
        if hist_vol > 0:
            iv_rank = min(100, (current_vol / hist_vol) * 50)
        else:
            iv_rank = 50

        return iv_rank

    def analyze_momentum(self, symbol):
        """Analyze price momentum"""
        bars = self.get_market_data(symbol)
        if len(bars) < 10:
            return 0

        # Simple momentum: compare current to 10 days ago
        current = bars[-1]['c']
        past = bars[-10]['c']

        momentum = ((current - past) / past) * 100
        return momentum

    def scan_for_opportunities(self):
        """Scan all symbols for options opportunities"""
        opportunities = []

        print("\n" + "=" * 70)
        print(f"SCANNING AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        for symbol in self.symbols:
            price = self.get_price(symbol)
            if not price:
                continue

            iv_rank = self.calculate_iv_rank(symbol)
            momentum = self.analyze_momentum(symbol)

            # Scoring logic
            score = 0
            strategy = None

            # High IV - good for selling premium
            if iv_rank > 70:
                score += 3
                if momentum > 2:  # Bullish
                    strategy = "BULL_PUT_SPREAD"
                    score += 2
                elif momentum < -2:  # Bearish
                    strategy = "BEAR_CALL_SPREAD"
                    score += 2
                else:  # Neutral
                    strategy = "IRON_CONDOR"
                    score += 3

            # Low IV - good for buying options
            elif iv_rank < 30:
                score += 2
                if momentum > 5:  # Strong bullish
                    strategy = "LONG_CALL"
                    score += 3
                elif momentum < -5:  # Strong bearish
                    strategy = "LONG_PUT"
                    score += 3

            # Medium IV with strong momentum
            elif abs(momentum) > 3:
                score += 1
                if momentum > 0:
                    strategy = "CALL_DEBIT_SPREAD"
                    score += 2
                else:
                    strategy = "PUT_DEBIT_SPREAD"
                    score += 2

            # Additional scoring
            if abs(momentum) > 10:
                score += 1  # Strong trend
            if 40 < iv_rank < 60:
                score += 0.5  # Balanced IV

            if score >= self.min_score and strategy:
                opportunity = {
                    'symbol': symbol,
                    'price': price,
                    'iv_rank': round(iv_rank, 1),
                    'momentum': round(momentum, 2),
                    'strategy': strategy,
                    'score': round(score, 1),
                    'timestamp': datetime.now().isoformat()
                }
                opportunities.append(opportunity)

                print(f"\n[OPPORTUNITY FOUND]")
                print(f"  Symbol: {symbol}")
                print(f"  Price: ${price:.2f}")
                print(f"  IV Rank: {iv_rank:.1f}%")
                print(f"  Momentum: {momentum:.2f}%")
                print(f"  Strategy: {strategy}")
                print(f"  Score: {score:.1f}/10")

        if not opportunities:
            print("\n[No opportunities meeting threshold]")
        else:
            # Save to file
            filename = f"options_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(opportunities, f, indent=2)
            print(f"\n[Saved {len(opportunities)} opportunities to {filename}]")

        return opportunities

    def run_continuous(self):
        """Run continuous scanning"""
        print("\n" + "=" * 70)
        print("STARTING CONTINUOUS OPTIONS SCANNING")
        print("Scan Interval: 15 minutes")
        print("Press Ctrl+C to stop")
        print("=" * 70)

        iteration = 0
        while True:
            iteration += 1

            try:
                # Check if market is open (simplified)
                now = datetime.now()
                if now.weekday() < 5 and 9 <= now.hour < 16:
                    opportunities = self.scan_for_opportunities()

                    # Log summary
                    print(f"\n[ITERATION #{iteration} COMPLETE]")
                    print(f"  Found: {len(opportunities)} opportunities")
                    print(f"  Next scan: {(now + timedelta(minutes=15)).strftime('%H:%M:%S')}")
                else:
                    print(f"\n[MARKET CLOSED - Waiting...]")

                # Wait for next scan
                time.sleep(900)  # 15 minutes

            except KeyboardInterrupt:
                print("\n[Scanner stopped by user]")
                break
            except Exception as e:
                print(f"\n[ERROR] {str(e)}")
                time.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    scanner = OptionsScanner()
    scanner.run_continuous()