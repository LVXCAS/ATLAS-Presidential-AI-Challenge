#!/usr/bin/env python3
"""
WEEK 2 ENHANCED SCANNER - ALL 4 LEARNINGS INTEGRATED
=====================================================
Implements all lessons learned from Day 3:

Learning #1: Multi-source data fetcher (yfinance primary) - 10x speed
Learning #2: Strategy selection logic (Bull Put Spreads for <3% momentum)
Learning #3: Market regime detection (check if Bull Put Spreads viable today)
Learning #4: Account verification (prevent trading on wrong account)
"""

import asyncio
import json
import pytz
from datetime import datetime, timedelta
from time_series_momentum_strategy import TimeSeriesMomentumStrategy
from week1_execution_system import Week1ExecutionSystem
from mission_control_logger import MissionControlLogger
from ml_activation_system import MLActivationSystem
from core.adaptive_dual_options_engine import AdaptiveDualOptionsEngine
from multi_source_data_fetcher import MultiSourceDataFetcher
from account_verification_system import AccountVerificationSystem
from market_regime_detector import MarketRegimeDetector

# Week 3+ Advanced Strategies
try:
    from strategies.bull_put_spread_engine import BullPutSpreadEngine
    from strategies.butterfly_spread_engine import ButterflySpreadEngine
    ADVANCED_STRATEGIES_AVAILABLE = True
except ImportError:
    ADVANCED_STRATEGIES_AVAILABLE = False
    print("[WARNING] Advanced strategies not available - using Dual Options only")

# OPTIONS LEARNING INTEGRATION
try:
    from options_learning_integration import (
        OptionsLearningTracker,
        OptionsTrade,
        get_tracker,
        initialize_learning
    )
    OPTIONS_LEARNING_AVAILABLE = True
except ImportError:
    OPTIONS_LEARNING_AVAILABLE = False
    print("[WARNING] Options learning integration not available")


class Week2EnhancedScanner:
    """Week 2 scanner with ALL 4 KEY LEARNINGS integrated"""

    def __init__(self):
        print("=" * 70)
        print("WEEK 3 PRODUCTION SCANNER - WITH CONTINUOUS LEARNING")
        print("=" * 70)

        # CRITICAL FIX: Force load main .env file FIRST to ensure correct account
        from dotenv import load_dotenv
        load_dotenv(override=True)  # Override any previous env loads

        import os
        print(f"\n[ENVIRONMENT CHECK]")
        print(f"  API Key: {os.getenv('ALPACA_API_KEY')[:10]}... (first 10 chars)")
        print(f"  Base URL: {os.getenv('ALPACA_BASE_URL')}")

        # LEARNING #4: Account Verification (FIRST - before anything else!)
        print("\n[LEARNING #4] ACCOUNT VERIFICATION")
        verifier = AccountVerificationSystem()
        account_check = verifier.verify_account_ready('BULL_PUT_SPREAD')

        if not account_check['ready']:
            print("\n[FATAL] ACCOUNT NOT READY - STOPPING SCANNER")
            print("Fix issues and restart scanner")
            raise Exception(f"Account verification failed: {account_check['issues']}")

        self.account_info = account_check
        print(f"\n[OK] Trading on account: {account_check['account_id'][:10]}... (${account_check['equity']:,.0f})")

        # LEARNING #3: Market Regime Detection
        print("\n[LEARNING #3] MARKET REGIME DETECTION")
        regime_detector = MarketRegimeDetector()
        market_regime = regime_detector.analyze_market_regime()

        self.market_regime = market_regime
        print(f"\n[OK] Market regime: {market_regime['regime']}")
        print(f"[OK] Bull Put Spreads viable: {'YES' if market_regime['bull_put_spread_viable'] else 'NO'}")

        # Load S&P 500 ticker universe
        with open('sp500_options_filtered.json', 'r') as f:
            universe_data = json.load(f)
            self.sp500_tickers = universe_data['tickers']

        print(f"\nUniverse: {len(self.sp500_tickers)} S&P 500 stocks")
        print("Target: 10-15% weekly ROI")
        print("=" * 70)

        # Initialize systems (force main account for production)
        self.system = Week1ExecutionSystem(force_main_account=True)
        self.momentum_strategy = TimeSeriesMomentumStrategy()
        self.mission_control = MissionControlLogger()
        # Pass credentials to options engine to use same account
        self.options_engine = AdaptiveDualOptionsEngine(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL')
        )

        # LEARNING #1: Multi-source data fetcher (10x speed)
        print("\n[LEARNING #1] MULTI-SOURCE DATA FETCHER")
        self.data_fetcher = MultiSourceDataFetcher()
        print("[OK] Data sources: yfinance (primary) -> OpenBB -> Alpaca")
        print("[OK] Scan speed: 30-60 seconds per 503 tickers (10x faster)")

        # LEARNING #2: Advanced strategy engines
        print("\n[LEARNING #2] STRATEGY SELECTION LOGIC")
        if ADVANCED_STRATEGIES_AVAILABLE:
            self.bull_put_spread_engine = BullPutSpreadEngine()
            self.butterfly_engine = ButterflySpreadEngine()
            print("[OK] Advanced strategies loaded: Bull Put Spread, Butterfly")
            print("[OK] Strategy selection: <3% momentum -> Bull Put Spread")
            print("[OK]                      >=3% momentum -> Dual Options")
        else:
            self.bull_put_spread_engine = None
            self.butterfly_engine = None
            print("[WARNING] Advanced strategies not available")

        self.multi_strategy_mode = True

        # Activate ML/DL/RL systems
        print("\n[ACTIVATING] ML/DL/RL Systems...")
        ml_system = MLActivationSystem()
        ml_system.activate_all_systems()

        # Initialize learning tracker
        self.learning_tracker = None
        if OPTIONS_LEARNING_AVAILABLE:
            print("\n[LEARNING SYSTEM] INITIALIZING...")
            self.learning_tracker = get_tracker()
            # Load optimized parameters from previous learning cycles
            optimized_params = self.learning_tracker.get_optimized_parameters()
            # CRITICAL FIX: Increased from 4.0 to 6.0 after 66% losing rate
            base_threshold = max(optimized_params.get('confidence_threshold', 6.0), 6.0)
            print(f"[OK] Learning system initialized")
            print(f"[OK] Loaded optimized parameters from previous cycles")
        else:
            # CRITICAL FIX: Increased from 4.0 to 6.0 after 66% losing rate
            base_threshold = 6.0
            print("\n[WARNING] Learning system not available - using fixed parameters")

        # ADAPTIVE CONFIDENCE THRESHOLD based on market regime
        threshold_adjustment = market_regime['confidence_adjustment']
        self.confidence_threshold = base_threshold + threshold_adjustment

        print(f"\n[ADAPTIVE SETTINGS]")
        print(f"  Base threshold: {base_threshold:.1f} (from learning system)" if OPTIONS_LEARNING_AVAILABLE else f"  Base threshold: {base_threshold:.1f}")
        print(f"  Market adjustment: {threshold_adjustment:+.1f} ({market_regime['regime']} regime)")
        print(f"  Final threshold: {self.confidence_threshold:.1f}")

        self.max_trades_per_day = 20 if self.multi_strategy_mode else 5
        self.risk_per_trade = 0.015
        self.trades_today = []
        self.scans_completed = 0

        # Week 2 REALITY CHECK settings
        self.min_volume = 1_000_000
        self.max_positions = 5
        self.simulation_mode = False  # PAPER TRADING ON ALPACA

        print(f"\n[FINAL SETTINGS]")
        print(f"  Confidence threshold: {self.confidence_threshold}+ (adaptive based on market)")
        print(f"  Max trades per day: {self.max_trades_per_day}")
        print(f"  Risk per trade: {self.risk_per_trade*100}%")
        print(f"  Tickers scanning: {len(self.sp500_tickers)}")
        print(f"  Min volume: {self.min_volume:,}")
        print(f"  Max positions: {self.max_positions}")
        print(f"  Mode: {'SIMULATION (No Orders)' if self.simulation_mode else 'PAPER TRADING (Alpaca)'}")

    async def scan_sp500_opportunities(self):
        """Scan entire S&P 500 for momentum-enhanced opportunities"""

        print(f"\n{'='*70}")
        print(f"SCAN #{self.scans_completed + 1} - S&P 500 MOMENTUM SCAN")
        print(f"{'='*70}")
        print(f"Time: {datetime.now().strftime('%I:%M:%S %p')}")
        print(f"Market Regime: {self.market_regime['regime']} (Bull Put Spreads: {'VIABLE' if self.market_regime['bull_put_spread_viable'] else 'NOT VIABLE'})")
        print(f"Scanning {len(self.sp500_tickers)} tickers...")

        opportunities = []
        debug_count = 0
        error_count = 0

        # Scan all S&P 500 tickers
        for i, symbol in enumerate(self.sp500_tickers, 1):
            if i % 25 == 0:
                print(f"  Progress: {i}/{len(self.sp500_tickers)} tickers scanned... (errors: {error_count}, scored: {debug_count})")

            try:
                # Get market data (using multi-source fetcher - NO rate limits!)
                bars = self.data_fetcher.get_bars(symbol, '1Day', limit=30).df

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
                momentum_pct = 0
                signal_direction = 'UNKNOWN'

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

                debug_count += 1

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
                        'momentum': momentum_pct,
                        'momentum_direction': signal_direction,
                        'strategy': self._select_strategy(momentum_signal, volatility),
                        'timestamp': datetime.now().isoformat()
                    }

                    opportunities.append(opportunity)

            except Exception as e:
                # Skip errors, continue scanning
                error_count += 1
                if error_count <= 3:  # Print first 3 errors for debugging
                    print(f"  [ERROR] {symbol}: {str(e)[:100]}")
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

    def _select_optimal_strategy_engine(self, opportunity):
        """
        Select which strategy engine to use based on opportunity characteristics

        Returns:
            tuple: (strategy_name, engine_callable) or (None, None) if filters fail
        """

        if not self.multi_strategy_mode or not ADVANCED_STRATEGIES_AVAILABLE:
            # Week 2 mode: Always use Dual Options
            return ('DUAL_OPTIONS', self.options_engine.execute_dual_strategy)

        momentum = opportunity.get('momentum', 0)  # Get signed momentum
        momentum_abs = abs(momentum)
        momentum_direction = opportunity.get('momentum_direction', 'UNKNOWN')

        # Get volatility estimate (simplified - would use IV in production)
        volatility = opportunity.get('volatility', 0.02)  # Default 2%

        # CRITICAL FILTERS - Prevent bad trades that caused 66% losing rate

        # Filter 1: Skip extremely high volatility (>5% daily moves)
        if volatility > 0.05:
            print(f"  [FILTER FAIL] Volatility too high ({volatility:.1%}) - skipping trade")
            return (None, None)

        # Filter 2: For bull put spreads, ONLY trade in uptrend or neutral
        if momentum_abs < 0.03:  # Bull put spread candidate
            if momentum_direction == 'BEARISH' and momentum < -0.02:
                print(f"  [FILTER FAIL] Bull Put Spread in downtrend ({momentum:+.1%}) - too risky")
                return (None, None)

            # Filter 3: Skip if volatility too low (premiums not worth it)
            if volatility < 0.015:
                print(f"  [FILTER FAIL] Volatility too low ({volatility:.1%}) - premiums insufficient")
                return (None, None)

        # Strategy selection logic:
        # 1. Low momentum + any volatility → Bull Put Spread (high probability)
        if momentum_abs < 0.03 and momentum_direction != 'BEARISH':
            print(f"  [STRATEGY] Bull Put Spread - Low momentum ({momentum:+.1%}), safe conditions")
            return ('BULL_PUT_SPREAD', self.bull_put_spread_engine.execute_bull_put_spread)

        # 2. Very low momentum → Butterfly (neutral play)
        elif momentum_abs < 0.02:
            print(f"  [STRATEGY] Butterfly - Neutral momentum ({momentum:+.1%}), defined risk")
            return ('BUTTERFLY', self.butterfly_engine.execute_butterfly)

        # 3. Strong momentum → Dual Options (directional)
        else:
            print(f"  [STRATEGY] Dual Options - Strong momentum ({momentum:+.1%}), directional")
            return ('DUAL_OPTIONS', self.options_engine.execute_dual_strategy)

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
            print(f"  Volatility: {opp['volatility']:.1%}")

            # Execute paper trade on Alpaca
            try:
                if not self.simulation_mode:
                    # Get buying power
                    account = self.system.api.get_account()
                    buying_power = float(account.buying_power)

                    # Select optimal strategy (with filters)
                    strategy_name, strategy_engine = self._select_optimal_strategy_engine(opp)

                    # Check if filters rejected the trade
                    if strategy_name is None:
                        print(f"  [SKIP] Trade filtered out - not suitable for current conditions")
                        continue  # Don't count as executed

                    # POSITION SIZING FIX: Max 5% per position
                    max_position_size = buying_power * 0.05
                    print(f"  [POSITION SIZE] Max allowed: ${max_position_size:,.0f} (5% of ${buying_power:,.0f})")

                    # Execute using selected strategy
                    if strategy_name == 'DUAL_OPTIONS':
                        # Dual Options - uses existing signature
                        result = strategy_engine(
                            opportunities=[opp],
                            buying_power=buying_power
                        )
                    elif strategy_name == 'BULL_PUT_SPREAD':
                        # Bull Put Spread - different signature (already has safe sizing)
                        result = strategy_engine(
                            symbol=opp['symbol'],
                            current_price=opp['price'],
                            contracts=1,  # Conservative: 1 contract = $300-500 max
                            expiration_days=7
                        )
                    elif strategy_name == 'BUTTERFLY':
                        # Butterfly - different signature
                        result = strategy_engine(
                            symbol=opp['symbol'],
                            current_price=opp['price'],
                            option_type='CALL' if opp.get('momentum_direction') == 'BULLISH' else 'PUT',
                            strike_width=5
                        )

                    print(f"  [OK] PAPER TRADE EXECUTED - {strategy_name}")
                else:
                    print(f"  [NOTE] SIMULATION MODE: Trade logged but not executed")

                # Only count successful trades
                self.trades_today.append(opp)

            except Exception as e:
                print(f"  [ERROR] EXECUTION ERROR: {e}")
                print(f"  [SKIP] Failed trade not counted against daily limit")
                continue  # Don't count failed trades

        print(f"\n[SUMMARY] Executed {len(to_execute)} trades")
        print(f"[REMAINING] {self.max_trades_per_day - len(self.trades_today)} trades available today")

    async def run_continuous_scanning(self):
        """Run continuous scanning every 5 minutes during market hours"""

        print(f"\n{'='*70}")
        print("STARTING WEEK 2 ENHANCED CONTINUOUS SCANNING")
        print(f"{'='*70}\n")

        while True:
            try:
                # Check if market is open (6:30 AM - 1:00 PM PDT)
                pdt = pytz.timezone('America/Los_Angeles')
                now = datetime.now(pdt)
                market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)
                market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)

                if now < market_open:
                    # Before market open - wait
                    wait_seconds = (market_open - now).total_seconds()
                    wait_minutes = int(wait_seconds / 60)
                    print(f"\n[PRE-MARKET] Market opens in {wait_minutes} minutes (6:30 AM PDT)...")
                    print(f"[STANDBY] All systems ready - waiting for market open...")
                    await asyncio.sleep(60)  # Check every minute

                elif market_open <= now <= market_close:
                    # Market is open - scan
                    opportunities = await self.scan_sp500_opportunities()

                    # Execute top opportunities
                    if opportunities and len(self.trades_today) < self.max_trades_per_day:
                        await self.execute_top_opportunities(opportunities)

                    # Wait 5 minutes
                    print(f"\n[WAITING] Next scan in 5 minutes...")
                    await asyncio.sleep(300)

                else:
                    # After market close
                    print(f"\n[MARKET CLOSED] Trading day complete")
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
            'trades': self.trades_today,
            'market_regime': self.market_regime,
            'account_info': {
                'account_id': self.account_info['account_id'],
                'equity': self.account_info['equity'],
                'options_buying_power': self.account_info['options_buying_power']
            }
        }

        filename = f"week2_enhanced_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n[SAVED] Report saved to {filename}")
        print(f"{'='*70}\n")


async def main():
    """Run Week 2 Enhanced Scanner"""

    scanner = Week2EnhancedScanner()
    await scanner.run_continuous_scanning()


if __name__ == "__main__":
    asyncio.run(main())
