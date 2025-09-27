#!/usr/bin/env python3
"""
OVERNIGHT GAP SCANNER - PRE-MARKET WEST COAST PREP
Scans for pre-market gaps and movers before 6:30 AM PST market open
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - GAP SCANNER - %(message)s')

class OvernightGapScanner:
    """Scan for overnight gaps and pre-market movers"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # West Coast focused watchlist
        self.watchlist = [
            'AAPL', 'TSLA', 'NVDA', 'AMD', 'AMZN', 'GOOGL', 'META', 'NFLX',
            'MSFT', 'QQQ', 'SPY', 'IWM', 'TQQQ', 'SQQQ', 'SPXL', 'SPXS',
            'LCID', 'RIVN', 'NTLA', 'SNAP', 'INTC', 'AMC', 'ARM'  # Current positions
        ]

        self.gap_thresholds = {
            'major_gap': 3.0,    # >3% gap = major move
            'minor_gap': 1.5,    # >1.5% gap = worth watching
            'volume_spike': 2.0   # 2x average volume
        }

        logging.info("OVERNIGHT GAP SCANNER INITIALIZED")
        logging.info(f"Watching {len(self.watchlist)} symbols for pre-market gaps")

    async def get_overnight_gaps(self):
        """Scan for overnight gaps"""

        gaps = []
        now = datetime.now()

        print(f"\n=== OVERNIGHT GAP SCAN - {now.strftime('%Y-%m-%d %H:%M:%S PST')} ===")

        for symbol in self.watchlist:
            try:
                # Get latest quote
                quote = self.alpaca.get_latest_quote(symbol)
                current_price = float(quote.bid_price)

                # Get previous close
                bars = self.alpaca.get_bars(
                    symbol,
                    '1Day',
                    start=(datetime.now() - timedelta(days=2)).date(),
                    end=datetime.now().date(),
                    adjustment='all'
                )

                if len(bars) > 0:
                    prev_close = float(bars[-1].close)
                    gap_pct = ((current_price - prev_close) / prev_close) * 100

                    if abs(gap_pct) >= self.gap_thresholds['minor_gap']:
                        gaps.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'prev_close': prev_close,
                            'gap_pct': gap_pct,
                            'gap_type': 'GAP UP' if gap_pct > 0 else 'GAP DOWN',
                            'magnitude': 'MAJOR' if abs(gap_pct) >= self.gap_thresholds['major_gap'] else 'MINOR'
                        })

                        status = "MAJOR GAP" if abs(gap_pct) >= self.gap_thresholds['major_gap'] else "Minor gap"
                        direction = "UP" if gap_pct > 0 else "DOWN"
                        print(f"{status} {direction}: {symbol} {gap_pct:+.1f}% (${current_price:.2f} vs ${prev_close:.2f})")

            except Exception as e:
                logging.error(f"Error scanning {symbol}: {e}")

        return gaps

    async def analyze_pre_market_volume(self, gaps):
        """Analyze pre-market volume for gap stocks"""

        volume_analysis = []

        for gap in gaps:
            try:
                # Get pre-market bars
                bars = self.alpaca.get_bars(
                    gap['symbol'],
                    '5Min',
                    start=(datetime.now() - timedelta(hours=4)),
                    end=datetime.now(),
                    adjustment='all'
                )

                if len(bars) > 0:
                    total_volume = sum(bar.volume for bar in bars)
                    avg_price = sum(bar.close for bar in bars) / len(bars)

                    volume_analysis.append({
                        'symbol': gap['symbol'],
                        'premarket_volume': total_volume,
                        'avg_premarket_price': avg_price,
                        'gap_pct': gap['gap_pct']
                    })

            except Exception as e:
                logging.error(f"Volume analysis error for {gap['symbol']}: {e}")

        return volume_analysis

    async def generate_trading_signals(self, gaps):
        """Generate trading signals based on gaps"""

        signals = []

        for gap in gaps:
            symbol = gap['symbol']
            gap_pct = gap['gap_pct']

            # Signal generation logic
            if abs(gap_pct) >= self.gap_thresholds['major_gap']:
                if gap_pct > 0:
                    # Major gap up - consider fade or momentum
                    signals.append({
                        'symbol': symbol,
                        'signal': 'FADE_GAP_UP',
                        'entry_price': gap['current_price'],
                        'target': gap['prev_close'],
                        'stop': gap['current_price'] * 1.02,
                        'confidence': 0.7,
                        'reason': f"Major gap up {gap_pct:.1f}% - fade opportunity"
                    })
                else:
                    # Major gap down - consider bounce
                    signals.append({
                        'symbol': symbol,
                        'signal': 'BOUNCE_GAP_DOWN',
                        'entry_price': gap['current_price'],
                        'target': gap['prev_close'] * 0.99,
                        'stop': gap['current_price'] * 0.98,
                        'confidence': 0.6,
                        'reason': f"Major gap down {gap_pct:.1f}% - bounce play"
                    })

            elif abs(gap_pct) >= self.gap_thresholds['minor_gap']:
                # Minor gaps - momentum plays
                direction = 'BUY' if gap_pct > 0 else 'SELL'
                signals.append({
                    'symbol': symbol,
                    'signal': f'MOMENTUM_{direction}',
                    'entry_price': gap['current_price'],
                    'target': gap['current_price'] * (1 + gap_pct/100 * 0.5),
                    'stop': gap['current_price'] * (1 - abs(gap_pct)/100 * 0.3),
                    'confidence': 0.5,
                    'reason': f"Minor gap {gap_pct:.1f}% - momentum continuation"
                })

        return signals

    async def run_overnight_scan(self):
        """Run comprehensive overnight gap scan"""

        print("OVERNIGHT GAP SCANNER - WEST COAST PRE-MARKET PREP")
        print("="*60)
        print(f"Market opens in: {self.time_to_market_open()}")
        print("="*60)

        # Scan for gaps
        gaps = await self.get_overnight_gaps()

        if not gaps:
            print("No significant overnight gaps detected")
            return

        # Sort by gap magnitude
        gaps.sort(key=lambda x: abs(x['gap_pct']), reverse=True)

        print(f"\nFOUND {len(gaps)} OVERNIGHT GAPS:")
        print("-" * 40)

        for gap in gaps:
            print(f"{gap['magnitude']} {gap['gap_type']}: {gap['symbol']} {gap['gap_pct']:+.1f}%")

        # Volume analysis
        volume_data = await self.analyze_pre_market_volume(gaps)

        # Generate signals
        signals = await self.generate_trading_signals(gaps)

        if signals:
            print(f"\nTRADING SIGNALS GENERATED ({len(signals)}):")
            print("-" * 40)
            for signal in signals:
                print(f"{signal['signal']}: {signal['symbol']} @ ${signal['entry_price']:.2f}")
                print(f"  Target: ${signal['target']:.2f} | Stop: ${signal['stop']:.2f} | Confidence: {signal['confidence']:.1f}")
                print(f"  Reason: {signal['reason']}")
                print()

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'gaps': gaps,
            'volume_analysis': volume_data,
            'signals': signals
        }

        with open(f'gap_scan_{datetime.now().strftime("%Y%m%d_%H%M")}.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def time_to_market_open(self):
        """Calculate time until market open (6:30 AM PST)"""
        now = datetime.now()
        market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)

        if now > market_open:
            market_open += timedelta(days=1)

        time_diff = market_open - now
        hours, remainder = divmod(time_diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        return f"{hours}h {minutes}m"

    async def continuous_gap_monitoring(self):
        """Run continuous gap monitoring"""

        logging.info("Starting continuous overnight gap monitoring")

        while True:
            try:
                await self.run_overnight_scan()
                await asyncio.sleep(300)  # Scan every 5 minutes

            except Exception as e:
                logging.error(f"Gap scan error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

async def main():
    """Run overnight gap scanner"""
    scanner = OvernightGapScanner()
    await scanner.continuous_gap_monitoring()

if __name__ == "__main__":
    asyncio.run(main())