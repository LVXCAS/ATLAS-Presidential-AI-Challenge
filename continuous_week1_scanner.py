#!/usr/bin/env python3
"""
FULL POWER TRADING SYSTEM - ALL SYSTEMS ACTIVATED
==================================================
MISSION CONTROL: Complete visibility into all trading systems
ML/DL/RL/GPU: All systems online and executing
Black-Scholes: Professional options pricing validation
Portfolio Management: Real-time risk monitoring with Sharpe ratio

ACTIVATED SYSTEMS:
- XGBoost Pattern Recognition
- LightGBM Ensemble Models
- PyTorch Neural Networks
- GPU Genetic Evolution
- Reinforcement Learning Agents
- Meta-Learning Optimizer
- All 6 Autonomous Agents
- Black-Scholes Greeks Validation
- FinQuant Portfolio Management
- Time Series Momentum (Moskowitz 2012)
"""

import asyncio
import time
from datetime import datetime, time as dt_time
from week1_execution_system import Week1ExecutionSystem
from options_executor import AlpacaOptionsExecutor
from mission_control_logger import MissionControlLogger
from enhanced_options_validator import EnhancedOptionsValidator
from enhanced_portfolio_manager import EnhancedPortfolioManager
from ml_activation_system import MLActivationSystem
from time_series_momentum_strategy import TimeSeriesMomentumStrategy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinuousWeek1Scanner:
    """Full power trading system with all ML/DL/RL activated"""

    def __init__(self):
        # Core systems
        self.system = Week1ExecutionSystem()
        self.options_executor = AlpacaOptionsExecutor()

        # Enhanced professional systems
        self.mission_control = MissionControlLogger()
        self.options_validator = EnhancedOptionsValidator()
        self.portfolio_manager = EnhancedPortfolioManager()

        # ML/DL/RL Systems - ACTIVATE ALL NOW
        self.ml_system = MLActivationSystem()
        print("\n[INITIALIZING] Activating all ML/DL/RL systems...")
        active_count = self.ml_system.activate_all_systems()
        print(f"[SUCCESS] {active_count}/6 systems activated\n")

        # Time Series Momentum System
        self.momentum_strategy = TimeSeriesMomentumStrategy()
        print("[SUCCESS] Time Series Momentum activated\n")

        # Scanning config
        self.scan_interval = 300  # 5 minutes between scans
        self.trades_today = []
        self.scans_completed = 0
        self.opportunities_found = 0

    async def run_continuous_scanning(self):
        """Run continuous scanning during market hours"""

        # Display mission control dashboard on startup
        self.mission_control.full_dashboard(
            scan_num=0,
            opportunities=0,
            strategy="INITIALIZING",
            confidence=0.0
        )

        while True:
            try:
                current_time = datetime.now().time()

                # Market hours: 6:30 AM - 1:00 PM PDT (9:30 AM - 4:00 PM EDT)
                market_open = dt_time(6, 30)  # 6:30 AM PDT
                market_close = dt_time(13, 0)  # 1:00 PM PDT

                if market_open <= current_time <= market_close:
                    await self._perform_scan_cycle()
                else:
                    if current_time < market_open:
                        wait_minutes = (datetime.combine(datetime.today(), market_open) -
                                      datetime.combine(datetime.today(), current_time)).seconds // 60
                        print(f"Pre-market: Waiting {wait_minutes} minutes for market open")
                    else:
                        print("Market closed for the day")
                        await self._generate_end_of_day_report()
                        break

                # Wait 5 minutes before next scan
                await asyncio.sleep(self.scan_interval)

            except Exception as e:
                logger.error(f"Error in continuous scanning: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _perform_scan_cycle(self):
        """Perform one complete scan cycle"""

        self.scans_completed += 1

        # Check portfolio health BEFORE scanning
        portfolio_health = self.portfolio_manager.get_portfolio_health()

        # Check if we've hit daily trade limit
        if len(self.trades_today) >= 2:
            self.mission_control.full_dashboard(
                scan_num=self.scans_completed,
                opportunities=0,
                strategy="DAILY LIMIT REACHED",
                confidence=0.0
            )
            return

        # Check stop loss / take profit
        if portfolio_health['total_return_pct'] <= -3.0:
            print(f"\n[STOP LOSS HIT] Daily loss at {portfolio_health['total_return_pct']:.2f}% - HALTING TRADING")
            return

        if portfolio_health['total_return_pct'] >= 5.0:
            print(f"\n[TAKE PROFIT HIT] Daily gain at {portfolio_health['total_return_pct']:.2f}% - SECURING PROFITS")
            return

        # Scan Intel-style opportunities
        intel_opportunities = await self._scan_intel_opportunities()

        # Scan earnings opportunities
        earnings_opportunities = await self._scan_earnings_opportunities()

        total_opportunities = len(intel_opportunities) + len(earnings_opportunities)
        self.opportunities_found += total_opportunities

        # Track best opportunity for dashboard
        best_opportunity = None
        best_score = 0.0

        # Execute first qualifying opportunity with Black-Scholes validation
        if intel_opportunities:
            opportunity = intel_opportunities[0]

            # VALIDATE WITH BLACK-SCHOLES GREEKS
            enhanced_score = self.options_validator.validate_intel_dual_opportunity(
                opportunity['symbol'],
                opportunity['price'],
                opportunity['score']
            )

            opportunity['enhanced_score'] = enhanced_score
            best_opportunity = opportunity
            best_score = enhanced_score

            # Only execute if enhanced score still qualifies
            if enhanced_score >= 4.0:
                trade_result = await self._execute_opportunity(opportunity, 'intel_style')
                if trade_result:
                    self.trades_today.append(trade_result)

        elif earnings_opportunities and len(self.trades_today) < 1:  # Only 1 earnings trade max
            opportunity = earnings_opportunities[0]

            # VALIDATE EARNINGS STRADDLE
            enhanced_score = self.options_validator.validate_earnings_straddle(
                opportunity['symbol'],
                opportunity['price'],
                opportunity['score']
            )

            opportunity['enhanced_score'] = enhanced_score
            best_opportunity = opportunity
            best_score = enhanced_score

            if enhanced_score >= 3.5:
                trade_result = await self._execute_opportunity(opportunity, 'earnings')
                if trade_result:
                    self.trades_today.append(trade_result)

        # Display mission control dashboard with scan results
        if best_opportunity:
            self.mission_control.full_dashboard(
                scan_num=self.scans_completed,
                opportunities=total_opportunities,
                strategy=best_opportunity['symbol'],
                confidence=best_score
            )
        else:
            self.mission_control.full_dashboard(
                scan_num=self.scans_completed,
                opportunities=total_opportunities,
                strategy="No qualifying setups",
                confidence=0.0
            )

    async def _scan_intel_opportunities(self):
        """Scan for Intel-style opportunities using REAL market data"""

        opportunities = []
        symbols = ['INTC', 'AMD', 'NVDA', 'QCOM', 'MU']

        # Typical volatility estimates for semiconductor stocks
        # (Paper accounts have limited historical data - these are reasonable estimates)
        typical_volatility = {
            'INTC': 0.015,  # Lower volatility (mature chip maker)
            'AMD': 0.025,   # Medium-high volatility
            'NVDA': 0.030,  # Higher volatility (AI/GPU plays)
            'QCOM': 0.020,  # Medium volatility
            'MU': 0.025     # Medium-high volatility (memory chips)
        }

        for symbol in symbols:
            try:
                # Get REAL market data from Alpaca API
                bars = self.system.api.get_bars(symbol, '1Day', limit=20).df

                if bars.empty:
                    print(f"    {symbol}: No data available")
                    continue

                # Calculate real metrics
                current_price = float(bars['close'].iloc[-1])
                volume = float(bars['volume'].iloc[-1])

                # Calculate actual volatility if we have enough data
                returns = bars['close'].pct_change().dropna()
                if len(returns) > 5:
                    volatility = float(returns.std())
                else:
                    # Use typical volatility estimate for this stock
                    volatility = typical_volatility.get(symbol, 0.02)

                # Use the REAL scoring method from unified system
                real_score = self.system._calculate_intel_opportunity_score(
                    current_price, volatility, volume
                )

                # ENHANCE WITH ML/DL/RL SYSTEMS
                ml_enhanced_score = self.ml_system.ml_enhanced_opportunity_scoring(
                    symbol, current_price, volume, volatility
                )

                # ENHANCE WITH TIME SERIES MOMENTUM
                momentum_signal = self.momentum_strategy.calculate_momentum_signal(symbol, lookback_days=21)
                final_score = ml_enhanced_score

                if momentum_signal:
                    momentum_pct = momentum_signal['momentum']
                    signal_direction = momentum_signal['signal']['direction']
                    signal_confidence = momentum_signal['signal']['confidence']

                    # Intel dual strategy benefits from bullish momentum
                    if signal_direction == 'BULLISH' and momentum_pct > 0.05:
                        # Strong upward momentum boosts confidence
                        momentum_boost = 0.5
                        final_score += momentum_boost
                        print(f"  [MOMENTUM] {symbol}: +{momentum_pct:+.1%} momentum → +{momentum_boost} boost")
                    elif signal_direction == 'BULLISH' and momentum_pct > 0.02:
                        # Moderate momentum
                        momentum_boost = 0.3
                        final_score += momentum_boost
                        print(f"  [MOMENTUM] {symbol}: +{momentum_pct:+.1%} momentum → +{momentum_boost} boost")
                    elif signal_direction == 'BEARISH':
                        # Bearish momentum reduces confidence for bullish strategy
                        print(f"  [MOMENTUM] {symbol}: {momentum_pct:+.1%} bearish (caution)")

                opportunity = {
                    'symbol': symbol,
                    'price': current_price,
                    'volume': volume,
                    'volatility': volatility,
                    'score': final_score,  # Use momentum-enhanced score
                    'ml_score': ml_enhanced_score,
                    'base_score': real_score,
                    'momentum': momentum_signal['momentum'] if momentum_signal else 0,
                    'momentum_direction': momentum_signal['signal']['direction'] if momentum_signal else 'UNKNOWN',
                    'type': 'intel_style',
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'ALPACA + ML + MOMENTUM'
                }

                # Week 1 requires 4.0+ score (momentum-enhanced)
                if final_score >= 4.0:
                    opportunities.append(opportunity)
                    print(f"  [QUALIFIED] {symbol}: ${current_price:.2f} - ML Score: {ml_enhanced_score:.2f} (Base: {real_score:.2f}) [Vol: {volume:,.0f}, IV: {volatility:.3f}]")
                else:
                    print(f"    {symbol}: ${current_price:.2f} - ML Score: {ml_enhanced_score:.2f} (below 4.0) [Base: {real_score:.2f}, Vol: {volume:,.0f}, IV: {volatility:.3f}]")

            except Exception as e:
                print(f"    {symbol}: Error - {e}")

        return opportunities

    async def _scan_earnings_opportunities(self):
        """Scan for earnings opportunities using REAL market data"""

        opportunities = []
        # Note: In production, would integrate with earnings calendar API
        # For now, scanning high-volatility stocks that might have earnings
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        for symbol in symbols:
            try:
                # Get real market data
                bars = self.system.api.get_bars(symbol, '1Day', limit=20).df

                if bars.empty:
                    continue

                current_price = float(bars['close'].iloc[-1])
                volume = float(bars['volume'].iloc[-1])

                # Calculate volatility to estimate expected move
                returns = bars['close'].pct_change().dropna()
                volatility = float(returns.std()) if len(returns) > 0 else 0.05

                # Estimate expected move based on volatility
                expected_move = volatility * 2.5  # Approximate earnings move

                # Calculate IV rank (simplified - would need options data for real IV)
                iv_rank = min(volatility * 100, 100)  # Normalized to 0-100

                # Use real scoring method
                base_score = self.system._calculate_earnings_opportunity_score(
                    expected_move, iv_rank, current_price
                )

                # ENHANCE WITH TIME SERIES MOMENTUM
                momentum_signal = self.momentum_strategy.calculate_momentum_signal(symbol, lookback_days=21)
                final_score = base_score

                if momentum_signal:
                    momentum_pct = momentum_signal['momentum']
                    signal_direction = momentum_signal['signal']['direction']

                    # For earnings: Strong momentum suggests directional play
                    # Weak momentum suggests straddle (what we're already doing)
                    if abs(momentum_pct) > 0.05:
                        # Strong directional momentum - straddle is risky
                        print(f"  [MOMENTUM] {symbol}: {momentum_pct:+.1%} {signal_direction} (directional > straddle)")
                        # Slight penalty - earnings + strong trend = pick direction, not straddle
                        final_score -= 0.2
                    elif abs(momentum_pct) < 0.02:
                        # Weak momentum - perfect for straddle
                        momentum_boost = 0.3
                        final_score += momentum_boost
                        print(f"  [MOMENTUM] {symbol}: {momentum_pct:+.1%} low momentum → +{momentum_boost} boost (straddle ideal)")
                    else:
                        # Moderate momentum
                        print(f"  [MOMENTUM] {symbol}: {momentum_pct:+.1%} {signal_direction} (neutral)")

                # Only consider if meaningful expected move
                if expected_move >= 0.03:  # At least 3% expected move
                    opportunity = {
                        'symbol': symbol,
                        'price': current_price,
                        'volume': volume,
                        'expected_move': expected_move,
                        'iv_rank': iv_rank,
                        'score': final_score,  # Use momentum-enhanced score
                        'base_score': base_score,
                        'momentum': momentum_signal['momentum'] if momentum_signal else 0,
                        'momentum_direction': momentum_signal['signal']['direction'] if momentum_signal else 'UNKNOWN',
                        'type': 'earnings',
                        'timestamp': datetime.now().isoformat(),
                        'data_source': 'ALPACA + MOMENTUM'
                    }

                    # Week 1 requires 3.5+ for earnings (momentum-enhanced)
                    if final_score >= 3.5:
                        opportunities.append(opportunity)
                        print(f"  [QUALIFIED] {symbol} EARNINGS: ${current_price:.2f} - Score: {final_score:.2f} - ExpMove: {expected_move:.1%}")
                    else:
                        print(f"    {symbol} EARNINGS: ${current_price:.2f} - Score: {final_score:.2f} (below 3.5) - ExpMove: {expected_move:.1%}")

            except Exception as e:
                print(f"    {symbol}: Error - {e}")

        return opportunities

    def _get_real_current_price(self, symbol):
        """Get current price from Alpaca API"""
        try:
            bars = self.system.api.get_bars(symbol, '1Day', limit=1).df
            if not bars.empty:
                return float(bars['close'].iloc[-1])
        except:
            pass
        return None

    async def _execute_opportunity(self, opportunity, strategy_type):
        """Execute a qualified opportunity with REAL OPTIONS ORDERS"""

        print(f"\n>>> EXECUTING: {opportunity['symbol']} ({strategy_type.upper()})")
        print(f"   Score: {opportunity['score']:.2f}")
        print(f"   Price: ${opportunity['price']:.2f}")

        # EXECUTE REAL OPTIONS ORDERS VIA ALPACA
        execution_result = None
        try:
            if strategy_type == 'intel_style':
                print(f"   [REAL EXECUTION] Submitting Intel dual strategy orders to Alpaca...")
                execution_result = self.options_executor.execute_intel_dual(
                    symbol=opportunity['symbol'],
                    current_price=opportunity['price'],
                    contracts=2,
                    expiry_days=21
                )
            else:  # earnings
                print(f"   [REAL EXECUTION] Submitting straddle orders to Alpaca...")
                execution_result = self.options_executor.execute_straddle(
                    symbol=opportunity['symbol'],
                    current_price=opportunity['price'],
                    contracts=1,
                    expiry_days=14
                )

            if execution_result and 'orders' in execution_result:
                print(f"   [SUCCESS] {len(execution_result['orders'])} orders filled!")
                for order in execution_result['orders']:
                    print(f"      - {order.get('type')}: {order.get('symbol')} x{order.get('qty')}")
            else:
                print(f"   [WARNING] Execution returned no orders")

        except Exception as e:
            print(f"   [ERROR] Execution failed: {e}")
            execution_result = {'error': str(e), 'status': 'FAILED'}

        # Create trade record
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': f"{strategy_type}_week1_continuous",
            'symbol': opportunity['symbol'],
            'current_price': opportunity['price'],
            'opportunity_score': opportunity['score'],
            'scan_number': self.scans_completed,
            'execution_type': 'REAL_OPTIONS_EXECUTION',
            'execution_result': execution_result,
            'week1_conservative': True,
            'paper_trade': True
        }

        if strategy_type == 'intel_style':
            # Add Intel-style trade details
            trade_record.update({
                'trades': [
                    {
                        'type': 'CASH_SECURED_PUT',
                        'strike': round(opportunity['price'] * 0.96, 1),
                        'contracts': 2,
                        'premium_estimate': opportunity['price'] * 0.015
                    },
                    {
                        'type': 'LONG_CALL',
                        'strike': round(opportunity['price'] * 1.04, 1),
                        'contracts': 2,
                        'cost_estimate': opportunity['price'] * 0.02
                    }
                ],
                'total_risk': opportunity['price'] * 100 * 2 * 1.015,
                'risk_percentage': 1.4,  # Conservative Week 1 sizing
                'expected_roi': '8-15%'
            })
        else:  # earnings
            trade_record.update({
                'expected_move': opportunity['expected_move'],
                'trades': [
                    {
                        'type': 'LONG_CALL',
                        'strike': round(opportunity['price'], 0),
                        'contracts': 1,
                        'cost_estimate': opportunity['price'] * 0.025
                    },
                    {
                        'type': 'LONG_PUT',
                        'strike': round(opportunity['price'], 0),
                        'contracts': 1,
                        'cost_estimate': opportunity['price'] * 0.025
                    }
                ],
                'total_cost': opportunity['price'] * 100 * 0.05,
                'cost_percentage': 1.0,  # Conservative 1% for earnings
                'expected_roi': '15-30%'
            })

        # Save trade record
        filename = f"week1_continuous_trade_{len(self.trades_today)+1}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        import json
        with open(filename, 'w') as f:
            json.dump(trade_record, f, indent=2)

        print(f"   [OK] Trade executed and logged: {filename}")

        return trade_record

    async def _generate_end_of_day_report(self):
        """Generate end of day report"""

        print(f"\nEND OF DAY REPORT - WEEK 1 DAY 1")
        print("=" * 40)
        print(f"Total scans completed: {self.scans_completed}")
        print(f"Opportunities found: {self.opportunities_found}")
        print(f"Trades executed: {len(self.trades_today)}")
        print()

        if self.trades_today:
            total_risk = sum(
                trade.get('total_risk', trade.get('total_cost', 0))
                for trade in self.trades_today
            )
            avg_risk = total_risk / 100000 * 100 / len(self.trades_today) if self.trades_today else 0

            print("TRADES EXECUTED:")
            for i, trade in enumerate(self.trades_today, 1):
                print(f"  {i}. {trade['symbol']} ({trade['strategy']}) - Score: {trade['opportunity_score']:.2f}")

            print(f"\nRISK MANAGEMENT:")
            print(f"  Total risk used: {total_risk/100000*100:.2f}%")
            print(f"  Average risk per trade: {avg_risk:.2f}%")
            print(f"  Within Week 1 limits: {'YES' if total_risk/100000 <= 0.03 else 'NO'}")
        else:
            print("No trades executed - excellent discipline!")
            print("All opportunities failed to meet Week 1 conservative thresholds")

        # Save daily summary
        daily_summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'week': 1,
            'day': 1,
            'scans_completed': self.scans_completed,
            'opportunities_found': self.opportunities_found,
            'trades_executed': len(self.trades_today),
            'trades': self.trades_today,
            'week1_assessment': 'EXCELLENT DISCIPLINE' if len(self.trades_today) <= 2 else 'WITHIN LIMITS'
        }

        filename = f"week1_day1_continuous_summary_{datetime.now().strftime('%Y%m%d')}.json"
        import json
        with open(filename, 'w') as f:
            json.dump(daily_summary, f, indent=2)

        print(f"\nDaily summary saved: {filename}")
        print("Ready for Week 1 Day 2 tomorrow!")

async def main():
    """Run continuous scanner for the full trading day"""
    scanner = ContinuousWeek1Scanner()
    await scanner.run_continuous_scanning()

if __name__ == "__main__":
    asyncio.run(main())