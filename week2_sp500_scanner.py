#!/usr/bin/env python3
"""
WEEK 2 S&P 500 SCANNER - MOMENTUM ENHANCED
==========================================
Scans 123+ S&P 500 stocks for high-probability options opportunities

Week 2 Upgrades:
- 123 S&P 500 tickers (vs 5-8 in Week 1)
- 5-10 trades per day (vs 2 in Week 1)
- 10-15% weekly target (vs 5-8% in Week 1)
- All strategies: Intel dual, straddles, iron condors, butterflies
"""

import asyncio
import json
from datetime import datetime, timedelta
from time_series_momentum_strategy import TimeSeriesMomentumStrategy
from autonomous_trading_empire import AutonomousTradingEmpire
from mission_control_logger import MissionControlLogger
from ml_activation_system import MLActivationSystem

class Week2SP500Scanner:
    """Week 2 scanner with full S&P 500 universe"""

    def __init__(self):
        # Load S&P 500 ticker universe
        with open('sp500_options_filtered.json', 'r') as f:
            universe_data = json.load(f)
            self.sp500_tickers = universe_data['tickers']

        print("=" * 70)
        print("WEEK 2 S&P 500 MOMENTUM SCANNER")
        print("=" * 70)
        print(f"Universe: {len(self.sp500_tickers)} S&P 500 stocks")
        print("Target: 10-15% weekly ROI")
        print("Max trades: 5-10 per day")
        print("=" * 70)

        # Initialize systems
        self.system = AutonomousTradingEmpire()
        self.momentum_strategy = TimeSeriesMomentumStrategy()
        self.mission_control = MissionControlLogger()

        # Activate ML/DL/RL systems
        print("\n[ACTIVATING] ML/DL/RL Systems...")
        ml_system = MLActivationSystem()
        ml_system.activate_all_systems()

        # Week 2 settings - REALISTIC for actual trading
        self.confidence_threshold = 3.2  # Lower threshold for real opportunities (was 4.0)
        self.max_trades_per_day = 3  # Start conservative: 3 trades (scale to 10 later)
        self.risk_per_trade = 0.015  # 1.5% risk per trade (conservative start)
        self.trades_today = []
        self.scans_completed = 0

        # Week 2 REALITY CHECK settings
        self.min_volume = 1_000_000  # Minimum daily volume for liquidity
        self.max_positions = 5  # Don't overextend
        self.simulation_mode = True  # Paper trade first! Set False for live

        print(f"\n[WEEK 2 SETTINGS - REALISTIC]")
        print(f"  Confidence threshold: {self.confidence_threshold}+ (lowered from 4.0 to find opportunities)")
        print(f"  Max trades per day: {self.max_trades_per_day} (starting conservative)")
        print(f"  Risk per trade: {self.risk_per_trade*100}%")
        print(f"  Tickers scanning: {len(self.sp500_tickers)}")
        print(f"  Min volume: {self.min_volume:,}")
        print(f"  Max positions: {self.max_positions}")
        print(f"  Mode: {'SIMULATION (Paper Trading)' if self.simulation_mode else 'LIVE TRADING'}")

    async def scan_sp500_opportunities(self):
        """Scan entire S&P 500 for momentum-enhanced opportunities"""

        print(f"\n{'='*70}")
        print(f"SCAN #{self.scans_completed + 1} - S&P 500 MOMENTUM SCAN")
        print(f"{'='*70}")
        print(f"Time: {datetime.now().strftime('%I:%M:%S %p')}")
        print(f"Scanning {len(self.sp500_tickers)} tickers...")

        opportunities = []

        # Scan all S&P 500 tickers
        for i, symbol in enumerate(self.sp500_tickers, 1):
            if i % 25 == 0:
                print(f"  Progress: {i}/{len(self.sp500_tickers)} tickers scanned...")

            try:
                # Get market data
                bars = self.system.api.get_bars(symbol, '1Day', limit=30).df

                if bars.empty:
                    continue

                current_price = float(bars['close'].iloc[-1])
                volume = float(bars['volume'].iloc[-1])

                # Calculate volatility
                returns = bars['close'].pct_change().dropna()
                volatility = float(returns.std()) if len(returns) > 0 else 0

                # Base score from autonomous system
                base_score = self._calculate_opportunity_score(
                    current_price, volume, volatility
                )

                # ML enhancement (simulated - in production would use actual models)
                ml_boost = self._calculate_ml_enhancement(bars)
                ml_score = base_score + ml_boost

                # MOMENTUM ENHANCEMENT
                momentum_signal = self.momentum_strategy.calculate_momentum_signal(
                    symbol, lookback_days=21
                )

                final_score = ml_score

                if momentum_signal:
                    momentum_pct = momentum_signal['momentum']
                    signal_direction = momentum_signal['signal']['direction']

                    # Strong bullish momentum
                    if signal_direction == 'BULLISH' and momentum_pct > 0.05:
                        momentum_boost = 0.5
                        final_score += momentum_boost
                    # Moderate bullish
                    elif signal_direction == 'BULLISH' and momentum_pct > 0.02:
                        momentum_boost = 0.3
                        final_score += momentum_boost
                    # Neutral (good for spreads)
                    elif abs(momentum_pct) < 0.02:
                        momentum_boost = 0.2
                        final_score += momentum_boost

                    # Create opportunity if qualified
                    if final_score >= self.confidence_threshold:
                        opportunity = {
                            'symbol': symbol,
                            'price': current_price,
                            'volume': volume,
                            'volatility': volatility,
                            'score': final_score,
                            'base_score': base_score,
                            'ml_score': ml_score,
                            'momentum': momentum_pct if momentum_signal else 0,
                            'momentum_direction': signal_direction if momentum_signal else 'UNKNOWN',
                            'strategy': self._select_strategy(momentum_signal, volatility),
                            'timestamp': datetime.now().isoformat()
                        }

                        opportunities.append(opportunity)

            except Exception as e:
                # Skip errors, continue scanning
                pass

        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n{'='*70}")
        print(f"SCAN COMPLETE - Found {len(opportunities)} qualified opportunities")
        print(f"{'='*70}")

        # Display top 10
        if opportunities:
            print(f"\nTOP 10 OPPORTUNITIES:")
            for i, opp in enumerate(opportunities[:10], 1):
                print(f"\n{i}. {opp['symbol']}: ${opp['price']:.2f}")
                print(f"   Score: {opp['score']:.2f} | Momentum: {opp['momentum']:+.1%} ({opp['momentum_direction']})")
                print(f"   Strategy: {opp['strategy']}")

        self.scans_completed += 1
        return opportunities

    def _calculate_opportunity_score(self, price, volume, volatility):
        """Calculate base opportunity score"""

        score = 3.0  # Base

        # Volume score
        if volume > 5_000_000:
            score += 0.5
        elif volume > 1_000_000:
            score += 0.3

        # Volatility score (higher vol = better premiums)
        if volatility > 0.03:
            score += 0.4
        elif volatility > 0.02:
            score += 0.2

        return score

    def _calculate_ml_enhancement(self, bars):
        """Simulated ML enhancement (in production would use actual models)"""

        # Calculate technical indicators
        df = bars.copy()

        # Simple momentum
        returns_5d = (df['close'].iloc[-1] / df['close'].iloc[-6]) - 1 if len(df) >= 6 else 0
        returns_10d = (df['close'].iloc[-1] / df['close'].iloc[-11]) - 1 if len(df) >= 11 else 0

        ml_boost = 0

        # Positive momentum
        if returns_5d > 0.02 and returns_10d > 0.05:
            ml_boost += 0.5
        elif returns_5d > 0 and returns_10d > 0:
            ml_boost += 0.3

        # Volume trend
        avg_volume_recent = df['volume'].iloc[-5:].mean()
        avg_volume_older = df['volume'].iloc[-20:-5].mean() if len(df) >= 20 else avg_volume_recent

        if avg_volume_recent > avg_volume_older * 1.2:
            ml_boost += 0.2

        return ml_boost

    def _select_strategy(self, momentum_signal, volatility):
        """Select optimal strategy based on momentum and volatility"""

        if not momentum_signal:
            return "Hold (insufficient data)"

        momentum_pct = momentum_signal['momentum']
        direction = momentum_signal['signal']['direction']

        # Strong directional momentum → Directional strategies
        if direction == 'BULLISH' and momentum_pct > 0.05:
            return "Bull Call Spread or Long Calls"
        elif direction == 'BEARISH' and momentum_pct < -0.05:
            return "Bear Put Spread or Long Puts"

        # Moderate momentum → Premium collection
        elif direction == 'BULLISH' and momentum_pct > 0.02:
            return "Bull Put Spread (collect premium)"
        elif direction == 'BEARISH' and momentum_pct < -0.02:
            return "Bear Call Spread (collect premium)"

        # Low momentum, high volatility → Iron condor
        elif abs(momentum_pct) < 0.02 and volatility > 0.03:
            return "Iron Condor (high prob income)"

        # Low momentum, low volatility → Butterfly
        elif abs(momentum_pct) < 0.02:
            return "Butterfly Spread (defined risk)"

        else:
            return "Hold (no clear edge)"

    async def execute_top_opportunities(self, opportunities):
        """Execute top opportunities up to daily limit"""

        if len(self.trades_today) >= self.max_trades_per_day:
            print(f"\n[LIMIT] Already executed {len(self.trades_today)} trades today (max {self.max_trades_per_day})")
            return

        remaining_trades = self.max_trades_per_day - len(self.trades_today)
        to_execute = opportunities[:remaining_trades]

        print(f"\n{'='*70}")
        print(f"EXECUTING TOP {len(to_execute)} OPPORTUNITIES")
        print(f"{'='*70}")

        for opp in to_execute:
            print(f"\n[EXECUTE] {opp['symbol']}: {opp['strategy']}")
            print(f"  Score: {opp['score']:.2f}")
            print(f"  Momentum: {opp['momentum']:+.1%} ({opp['momentum_direction']})")

            # In Week 2, would execute real orders here
            # For now, just log the opportunity
            self.trades_today.append(opp)

        print(f"\n[SUMMARY] Executed {len(to_execute)} trades")
        print(f"[REMAINING] {self.max_trades_per_day - len(self.trades_today)} trades available today")

    async def run_continuous_scanning(self):
        """Run continuous scanning every 5 minutes during market hours"""

        print(f"\n{'='*70}")
        print("STARTING WEEK 2 CONTINUOUS S&P 500 SCANNING")
        print(f"{'='*70}\n")

        # Display mission control
        self.mission_control.display_full_dashboard()

        while True:
            try:
                # Check if market is open (6:30 AM - 1:00 PM PDT)
                now = datetime.now()
                market_open = now.replace(hour=6, minute=30, second=0)
                market_close = now.replace(hour=13, minute=0, second=0)

                if market_open <= now <= market_close:
                    # Scan S&P 500
                    opportunities = await self.scan_sp500_opportunities()

                    # Execute top opportunities
                    if opportunities and len(self.trades_today) < self.max_trades_per_day:
                        await self.execute_top_opportunities(opportunities)

                    # Wait 5 minutes
                    print(f"\n[WAITING] Next scan in 5 minutes...")
                    await asyncio.sleep(300)

                else:
                    print(f"\n[MARKET CLOSED] Waiting for next trading day...")
                    print(f"Market hours: 6:30 AM - 1:00 PM PDT")

                    # Generate end-of-day report
                    self._generate_end_of_day_report()

                    break

            except KeyboardInterrupt:
                print(f"\n\n[STOPPED] Scanner stopped by user")
                self._generate_end_of_day_report()
                break

            except Exception as e:
                print(f"\n[ERROR] Scan error: {e}")
                await asyncio.sleep(60)

    def _generate_end_of_day_report(self):
        """Generate Week 2 end-of-day report"""

        print(f"\n{'='*70}")
        print("WEEK 2 END OF DAY REPORT")
        print(f"{'='*70}")
        print(f"Scans completed: {self.scans_completed}")
        print(f"Trades executed: {len(self.trades_today)}/{self.max_trades_per_day}")

        if self.trades_today:
            print(f"\nTRADES EXECUTED:")
            for i, trade in enumerate(self.trades_today, 1):
                print(f"  {i}. {trade['symbol']} - {trade['strategy']} (Score: {trade['score']:.2f})")

        # Save report
        report = {
            'date': datetime.now().isoformat(),
            'scans_completed': self.scans_completed,
            'trades_executed': len(self.trades_today),
            'trades': self.trades_today
        }

        filename = f"week2_sp500_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n[SAVED] Report saved to {filename}")
        print(f"{'='*70}\n")


async def main():
    """Run Week 2 S&P 500 scanner"""

    scanner = Week2SP500Scanner()
    await scanner.run_continuous_scanning()


if __name__ == "__main__":
    asyncio.run(main())
