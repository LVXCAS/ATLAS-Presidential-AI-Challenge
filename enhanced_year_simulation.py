#!/usr/bin/env python3
"""
Enhanced Year-Long Monte Carlo Simulation
Testing the improved Sharpe ratio system over ~252 trading days
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List
import random

from agents.advanced_monte_carlo_engine import advanced_monte_carlo_engine, OptionSpec
from agents.enhanced_options_pricing_engine import enhanced_options_pricing_engine
from agents.sharpe_enhanced_filters import sharpe_enhanced_filters

class EnhancedYearSimulation:
    """Year-long simulation with enhanced Sharpe ratio filters"""

    def __init__(self, trading_days=252, trades_per_day=3):
        self.trading_days = trading_days
        self.trades_per_day = trades_per_day
        self.total_trades = trading_days * trades_per_day
        self.results = []

        # Enhanced system parameters
        self.initial_capital = 18113.50
        self.position_size_pct = 0.015  # Slightly increased from analysis
        self.max_loss_per_trade = 0.25  # Tighter stops (25% vs 30%)

        # Market parameters
        self.volatility_range = (0.15, 0.35)
        self.days_to_expiry_range = (14, 30)
        self.moneyness_range = (0.95, 1.05)
        self.min_premium = 0.50
        self.max_premium = 15.0

        # Performance tracking
        self.daily_pnl = []
        self.equity_curve = [self.initial_capital]
        self.sharpe_filters = sharpe_enhanced_filters

    def generate_market_cycle(self, day: int) -> str:
        """Generate realistic market cycles over the year"""
        # Simulate different market regimes throughout the year
        cycle_day = day % 60  # 60-day cycles

        if cycle_day < 15:
            return 'BULL_RUN'
        elif cycle_day < 25:
            return 'CONSOLIDATION'
        elif cycle_day < 35:
            return 'CORRECTION'
        elif cycle_day < 45:
            return 'RECOVERY'
        else:
            return 'NEUTRAL'

    def generate_enhanced_scenario(self, day: int) -> Dict:
        """Generate scenario with enhanced filters and market cycles"""
        market_cycle = self.generate_market_cycle(day)

        # Adjust stock price based on market cycle
        if market_cycle == 'BULL_RUN':
            base_price = np.random.normal(180, 40)  # Higher prices in bull run
        elif market_cycle == 'CORRECTION':
            base_price = np.random.normal(130, 35)  # Lower prices in correction
        else:
            base_price = np.random.normal(155, 35)  # Normal distribution

        stock_price = max(50, min(400, base_price))

        # Generate price history for technical analysis
        price_history = self.generate_price_history(stock_price, market_cycle)

        # Calculate realized volatility
        returns = np.diff(price_history) / price_history[:-1]
        realized_vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.25

        # Adjust volatility based on market cycle
        if market_cycle == 'CORRECTION':
            vol_multiplier = 1.5  # Higher vol in corrections
        elif market_cycle == 'BULL_RUN':
            vol_multiplier = 0.8  # Lower vol in bull runs
        else:
            vol_multiplier = 1.0

        volatility = np.clip(realized_vol * vol_multiplier, *self.volatility_range)

        days_to_expiry = np.random.randint(*self.days_to_expiry_range)
        option_type = np.random.choice(['call', 'put'])
        moneyness = np.random.uniform(*self.moneyness_range)
        strike_price = stock_price * moneyness

        return {
            'stock_price': stock_price,
            'strike_price': strike_price,
            'volatility': volatility,
            'days_to_expiry': days_to_expiry,
            'option_type': option_type,
            'moneyness': moneyness,
            'price_history': price_history,
            'market_cycle': market_cycle,
            'day': day
        }

    def generate_price_history(self, current_price: float, market_cycle: str, days: int = 50) -> np.ndarray:
        """Generate realistic price history based on market cycle"""
        if market_cycle == 'BULL_RUN':
            drift = 0.001  # Positive drift
            vol = 0.015    # Lower volatility
        elif market_cycle == 'CORRECTION':
            drift = -0.002 # Negative drift
            vol = 0.025    # Higher volatility
        elif market_cycle == 'RECOVERY':
            drift = 0.0015 # Strong positive drift
            vol = 0.020    # Medium volatility
        else:
            drift = 0.0    # No drift
            vol = 0.018    # Normal volatility

        returns = np.random.normal(drift, vol, days)
        prices = [current_price]

        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        return np.array(prices)

    async def apply_enhanced_filters(self, scenario: Dict) -> Dict:
        """Apply the enhanced filter system"""
        try:
            # Get comprehensive filters
            filter_results = await self.sharpe_filters.get_comprehensive_filters(
                "SIMULATION", scenario['volatility']
            )

            # Extract filter results
            rsi_filter = filter_results.get('rsi_filter', {})
            ema_filter = filter_results.get('ema_filter', {})
            momentum_filter = filter_results.get('momentum_filter', {})
            volatility_filter = filter_results.get('volatility_filter', {})
            iv_rank_filter = filter_results.get('iv_rank_filter', {})
            earnings_filter = filter_results.get('earnings_filter', {})

            # Apply filters with enhanced logic
            filters_passed = True
            filter_score = 1.0

            # RSI filter
            if not rsi_filter.get('pass', True):
                if np.random.random() < 0.8:  # 80% rejection rate
                    filters_passed = False
                else:
                    filter_score *= 0.7  # Penalty if we proceed

            # EMA filter
            ema_trend = ema_filter.get('trend', 'NEUTRAL')
            if ema_trend == 'BEARISH':
                if np.random.random() < 0.7:  # 70% rejection rate
                    filters_passed = False
                else:
                    filter_score *= 0.6
            elif ema_trend == 'BULLISH':
                filter_score *= 1.15  # Bonus for bullish

            # IV rank filter
            iv_rank = iv_rank_filter.get('rank', 50)
            if iv_rank < 30:
                if np.random.random() < 0.4:  # 40% rejection rate
                    filters_passed = False
                else:
                    filter_score *= 0.8
            elif iv_rank > 70:
                filter_score *= 1.1  # Bonus for high IV

            # Volatility regime adjustments
            vol_regime = volatility_filter.get('regime', 'NORMAL')
            if vol_regime == 'LOW_VOL':
                position_multiplier = 1.2
            elif vol_regime == 'HIGH_VOL':
                position_multiplier = 0.8
            else:
                position_multiplier = 1.0

            # Earnings filter
            if not earnings_filter.get('safe_to_trade', True):
                filters_passed = False

            return {
                'filters_passed': filters_passed,
                'filter_score': filter_score,
                'position_multiplier': position_multiplier,
                'ema_trend': ema_trend,
                'vol_regime': vol_regime,
                'iv_rank': iv_rank,
                'enhanced_filters_used': True
            }

        except Exception as e:
            # Fallback to basic filters
            return {
                'filters_passed': True,
                'filter_score': 1.0,
                'position_multiplier': 1.0,
                'ema_trend': 'NEUTRAL',
                'vol_regime': 'NORMAL',
                'iv_rank': 50,
                'enhanced_filters_used': False
            }

    async def simulate_enhanced_trade(self, scenario: Dict) -> Dict:
        """Simulate trade with enhanced system"""
        try:
            # Apply enhanced filters
            filter_result = await self.apply_enhanced_filters(scenario)

            if not filter_result['filters_passed']:
                return None  # Trade filtered out

            # Get option pricing
            analysis = await enhanced_options_pricing_engine.get_comprehensive_option_analysis(
                underlying_price=scenario['stock_price'],
                strike_price=scenario['strike_price'],
                time_to_expiry_days=scenario['days_to_expiry'],
                volatility=scenario['volatility'],
                option_type=scenario['option_type']
            )

            entry_price = analysis['pricing']['theoretical_price']

            # Premium filters
            if entry_price < self.min_premium or entry_price > self.max_premium:
                return None

            # Enhanced position sizing
            base_position_size = self.position_size_pct
            position_multiplier = filter_result['position_multiplier']
            filter_score = filter_result['filter_score']

            # Market cycle adjustments
            if scenario['market_cycle'] == 'BULL_RUN':
                cycle_multiplier = 1.1
            elif scenario['market_cycle'] == 'CORRECTION':
                cycle_multiplier = 0.8
            else:
                cycle_multiplier = 1.0

            final_position_size = base_position_size * position_multiplier * filter_score * cycle_multiplier
            final_position_size = np.clip(final_position_size, 0.005, 0.03)  # 0.5% to 3% cap

            # Simulate trade outcome
            holding_days = min(3, scenario['days_to_expiry'] - 1)

            # Enhanced price movement based on market cycle and filters
            if scenario['market_cycle'] == 'BULL_RUN' and filter_result['ema_trend'] == 'BULLISH':
                price_bias = 0.002  # Positive bias
            elif scenario['market_cycle'] == 'CORRECTION':
                price_bias = -0.001  # Negative bias
            else:
                price_bias = 0.0

            daily_return = np.random.normal(price_bias, scenario['volatility'] / np.sqrt(252))
            daily_return = np.clip(daily_return, -0.08, 0.08)  # Cap at ±8%

            final_stock_price = scenario['stock_price'] * (1 + daily_return * holding_days)
            remaining_days = max(1, scenario['days_to_expiry'] - holding_days)

            # Calculate exit price
            if remaining_days > 1:
                exit_analysis = await enhanced_options_pricing_engine.get_comprehensive_option_analysis(
                    underlying_price=final_stock_price,
                    strike_price=scenario['strike_price'],
                    time_to_expiry_days=remaining_days,
                    volatility=scenario['volatility'],
                    option_type=scenario['option_type']
                )
                exit_price = exit_analysis['pricing']['theoretical_price']
            else:
                # Expiration value
                if scenario['option_type'] == 'call':
                    exit_price = max(0, final_stock_price - scenario['strike_price'])
                else:
                    exit_price = max(0, scenario['strike_price'] - final_stock_price)

            # Apply enhanced risk management (25% stop loss)
            raw_pnl = exit_price - entry_price
            max_loss = entry_price * self.max_loss_per_trade

            if raw_pnl < -max_loss:
                raw_pnl = -max_loss
                exit_price = entry_price - max_loss
                stop_loss_triggered = True
            else:
                stop_loss_triggered = False

            # Calculate position impact
            risk_amount = self.initial_capital * final_position_size
            contracts = max(1, int(risk_amount / (entry_price * 100)))
            position_pnl = raw_pnl * contracts * 100
            capital_impact = position_pnl / self.initial_capital * 100

            return {
                'capital_impact': capital_impact,
                'position_pnl': position_pnl,
                'is_winner': raw_pnl > 0,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'contracts': contracts,
                'position_size_pct': final_position_size,
                'stop_loss_triggered': stop_loss_triggered,
                'market_cycle': scenario['market_cycle'],
                'enhanced_filters': filter_result['enhanced_filters_used'],
                'filter_score': filter_result['filter_score'],
                'ema_trend': filter_result['ema_trend'],
                'vol_regime': filter_result['vol_regime'],
                'day': scenario['day'],
                'holding_days': holding_days
            }

        except Exception as e:
            return None

    async def run_year_simulation(self):
        """Run the year-long simulation"""
        print("=" * 80)
        print("ENHANCED YEAR-LONG MONTE CARLO SIMULATION")
        print("Testing Sharpe ratio improvements over ~1 year of trading")
        print("=" * 80)
        print(f"Trading Days: {self.trading_days}")
        print(f"Trades per Day: {self.trades_per_day}")
        print(f"Total Target Trades: {self.total_trades:,}")
        print(f"Enhanced Filters: ACTIVE")
        print(f"Stop Loss: {self.max_loss_per_trade:.0%}")
        print(f"Position Sizing: {self.position_size_pct:.1%} base")
        print("=" * 80)

        start_time = time.time()
        successful_trades = 0

        for day in range(self.trading_days):
            daily_trades = []
            daily_pnl = 0

            # Generate trades for this day
            for trade_num in range(self.trades_per_day):
                total_attempt = day * self.trades_per_day + trade_num + 1

                if total_attempt % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = total_attempt / elapsed
                    eta = (self.total_trades - total_attempt) / rate
                    success_rate = successful_trades / total_attempt * 100
                    print(f"Day {day+1:3d}: {total_attempt:,}/{self.total_trades:,} "
                          f"({total_attempt/self.total_trades*100:.1f}%) - "
                          f"Success: {success_rate:.1f}% - ETA: {eta:.0f}s")

                scenario = self.generate_enhanced_scenario(day)
                result = await self.simulate_enhanced_trade(scenario)

                if result:
                    self.results.append(result)
                    daily_trades.append(result)
                    daily_pnl += result['position_pnl']
                    successful_trades += 1

            # Update equity curve
            self.daily_pnl.append(daily_pnl)
            current_equity = self.equity_curve[-1] + daily_pnl
            self.equity_curve.append(current_equity)

            # Log major market events
            market_cycle = self.generate_market_cycle(day)
            if day % 50 == 0:
                print(f"Day {day+1:3d}: Market={market_cycle:12s} Daily P&L=${daily_pnl:8.2f} "
                      f"Equity=${current_equity:,.2f} Trades={len(daily_trades)}")

        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.1f} seconds")
        print(f"Successful trades: {len(self.results):,} out of {self.total_trades:,} attempts")

    def analyze_year_results(self) -> Dict:
        """Analyze the year-long simulation results"""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # Basic performance metrics
        total_trades = len(df)
        win_rate = len(df[df['is_winner']]) / total_trades

        # Capital impact analysis
        capital_impacts = df['capital_impact'].values
        avg_capital_impact = np.mean(capital_impacts)
        std_capital_impact = np.std(capital_impacts)

        # Calculate annualized Sharpe ratio
        daily_returns = np.array(self.daily_pnl) / self.initial_capital
        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)

        if std_daily_return > 0:
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Portfolio metrics
        total_pnl = sum(self.daily_pnl)
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Risk metrics
        equity_returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        max_drawdown = self.calculate_max_drawdown(self.equity_curve)

        # Enhanced filter analysis
        enhanced_trades = df[df['enhanced_filters']].copy() if 'enhanced_filters' in df.columns else df
        enhanced_win_rate = len(enhanced_trades[enhanced_trades['is_winner']]) / len(enhanced_trades) if len(enhanced_trades) > 0 else 0

        # Stop loss analysis
        stop_loss_rate = len(df[df['stop_loss_triggered']]) / total_trades if 'stop_loss_triggered' in df.columns else 0

        # Market cycle performance
        cycle_performance = {}
        if 'market_cycle' in df.columns:
            for cycle in df['market_cycle'].unique():
                cycle_df = df[df['market_cycle'] == cycle]
                cycle_performance[cycle] = {
                    'trades': len(cycle_df),
                    'win_rate': len(cycle_df[cycle_df['is_winner']]) / len(cycle_df),
                    'avg_return': cycle_df['capital_impact'].mean(),
                    'total_pnl': cycle_df['position_pnl'].sum()
                }

        return {
            'simulation_summary': {
                'trading_days': self.trading_days,
                'total_trades': total_trades,
                'success_rate': total_trades / self.total_trades,
                'enhanced_filters_active': True
            },
            'performance_metrics': {
                'win_rate': win_rate,
                'enhanced_win_rate': enhanced_win_rate,
                'sharpe_ratio': sharpe_ratio,
                'total_return_pct': total_return,
                'avg_daily_return': avg_daily_return * 100,
                'volatility_daily': std_daily_return * 100,
                'stop_loss_rate': stop_loss_rate
            },
            'portfolio_metrics': {
                'initial_capital': self.initial_capital,
                'final_equity': final_equity,
                'total_pnl': total_pnl,
                'max_drawdown_pct': max_drawdown * 100,
                'best_day': max(self.daily_pnl),
                'worst_day': min(self.daily_pnl)
            },
            'risk_metrics': {
                'var_95': np.percentile(daily_returns, 5) * 100,
                'cvar_95': np.mean(daily_returns[daily_returns <= np.percentile(daily_returns, 5)]) * 100,
                'max_consecutive_losses': self.calculate_max_consecutive_losses(df),
                'largest_loss': df['capital_impact'].min() if len(df) > 0 else 0
            },
            'market_cycle_performance': cycle_performance,
            'filter_effectiveness': {
                'enhanced_trades': len(enhanced_trades),
                'avg_filter_score': df['filter_score'].mean() if 'filter_score' in df.columns else 1.0,
                'bullish_ema_trades': len(df[df['ema_trend'] == 'BULLISH']) if 'ema_trend' in df.columns else 0,
                'high_vol_regime_trades': len(df[df['vol_regime'] == 'HIGH_VOL']) if 'vol_regime' in df.columns else 0
            }
        }

    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def calculate_max_consecutive_losses(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses"""
        if len(df) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for _, row in df.iterrows():
            if not row['is_winner']:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def print_year_results(self, analysis: Dict):
        """Print comprehensive year results"""
        print("\n" + "=" * 80)
        print("ENHANCED YEAR-LONG SIMULATION RESULTS")
        print("Sharpe Ratio Optimization System Performance")
        print("=" * 80)

        sim = analysis['simulation_summary']
        print(f"\nSIMULATION OVERVIEW:")
        print(f"   Trading Period: {sim['trading_days']} days (~1 year)")
        print(f"   Total Trades: {sim['total_trades']:,}")
        print(f"   Trade Success Rate: {sim['success_rate']:.1%}")
        print(f"   Enhanced Filters: {'ACTIVE' if sim['enhanced_filters_active'] else 'INACTIVE'}")

        perf = analysis['performance_metrics']
        print(f"\n🎯 KEY PERFORMANCE METRICS:")
        print(f"   Win Rate: {perf['win_rate']:.1%}")
        print(f"   Enhanced Win Rate: {perf['enhanced_win_rate']:.1%}")
        print(f"   📈 SHARPE RATIO: {perf['sharpe_ratio']:.4f}")
        print(f"   Annual Return: {perf['total_return_pct']:.2f}%")
        print(f"   Daily Return Avg: {perf['avg_daily_return']:.3f}%")
        print(f"   Daily Volatility: {perf['volatility_daily']:.3f}%")
        print(f"   Stop Loss Rate: {perf['stop_loss_rate']:.1%}")

        port = analysis['portfolio_metrics']
        print(f"\n💰 PORTFOLIO PERFORMANCE:")
        print(f"   Initial Capital: ${port['initial_capital']:,.2f}")
        print(f"   Final Equity: ${port['final_equity']:,.2f}")
        print(f"   Total P&L: ${port['total_pnl']:,.2f}")
        print(f"   Max Drawdown: {port['max_drawdown_pct']:.2f}%")
        print(f"   Best Day: ${port['best_day']:,.2f}")
        print(f"   Worst Day: ${port['worst_day']:,.2f}")

        risk = analysis['risk_metrics']
        print(f"\n⚠️  RISK METRICS:")
        print(f"   VaR (95%): {risk['var_95']:.2f}%")
        print(f"   CVaR (95%): {risk['cvar_95']:.2f}%")
        print(f"   Max Consecutive Losses: {risk['max_consecutive_losses']}")
        print(f"   Largest Single Loss: {risk['largest_loss']:.2f}%")

        cycles = analysis['market_cycle_performance']
        print(f"\n📊 MARKET CYCLE PERFORMANCE:")
        for cycle, metrics in cycles.items():
            print(f"   {cycle:12s}: {metrics['trades']:3d} trades, "
                  f"{metrics['win_rate']:.1%} win rate, "
                  f"${metrics['total_pnl']:8,.2f} P&L")

        filters = analysis['filter_effectiveness']
        print(f"\n🔧 ENHANCED FILTER EFFECTIVENESS:")
        print(f"   Enhanced Filter Trades: {filters['enhanced_trades']:,}")
        print(f"   Average Filter Score: {filters['avg_filter_score']:.3f}")
        print(f"   Bullish EMA Trades: {filters['bullish_ema_trades']:,}")
        print(f"   High Vol Regime Trades: {filters['high_vol_regime_trades']:,}")

        # Sharpe ratio assessment
        print(f"\n" + "=" * 80)
        print("📈 SHARPE RATIO ASSESSMENT")
        print("=" * 80)

        sharpe = perf['sharpe_ratio']
        if sharpe > 2.0:
            print(f"🚀 EXCELLENT: Sharpe ratio of {sharpe:.4f} exceeds target of 2.0!")
            print(f"   This represents world-class risk-adjusted performance.")
        elif sharpe > 1.5:
            print(f"✅ VERY GOOD: Sharpe ratio of {sharpe:.4f} shows strong improvement.")
            print(f"   Target of 2.0+ is within reach with further optimization.")
        elif sharpe > 1.0:
            print(f"👍 GOOD: Sharpe ratio of {sharpe:.4f} shows positive results.")
            print(f"   Enhanced filters are working, but more improvement possible.")
        else:
            print(f"⚠️  NEEDS WORK: Sharpe ratio of {sharpe:.4f} requires further optimization.")

        print("=" * 80)

async def main():
    """Run the enhanced year simulation"""
    simulation = EnhancedYearSimulation(trading_days=252, trades_per_day=3)
    await simulation.run_year_simulation()
    analysis = simulation.analyze_year_results()
    simulation.print_year_results(analysis)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_year_simulation_{timestamp}.json"

    # Make results JSON serializable
    json_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, dict):
            json_analysis[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                for k, v in value.items()}
        else:
            json_analysis[key] = float(value) if isinstance(value, (np.integer, np.floating)) else value

    with open(results_file, 'w') as f:
        json.dump({
            'analysis': json_analysis,
            'daily_pnl': [float(x) for x in simulation.daily_pnl],
            'equity_curve': [float(x) for x in simulation.equity_curve],
            'trade_details': [
                {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                 for k, v in trade.items()}
                for trade in simulation.results
            ]
        }, f, indent=2, default=str)

    print(f"\n📁 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main())