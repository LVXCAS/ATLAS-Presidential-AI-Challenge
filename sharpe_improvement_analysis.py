#!/usr/bin/env python3
"""
Analysis: How to Improve Sharpe Ratio Beyond 1.38
Testing EMA and other strategy improvements
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List
from scipy import stats
import ta  # Technical analysis library

from agents.advanced_monte_carlo_engine import advanced_monte_carlo_engine, OptionSpec
from agents.enhanced_options_pricing_engine import enhanced_options_pricing_engine

class SharpeImprovementAnalysis:
    """Test various improvements to boost Sharpe ratio"""

    def __init__(self, trades_per_test=1000):
        self.trades_per_test = trades_per_test
        self.improvement_tests = {}

        # Base parameters (current system)
        self.base_params = {
            'position_size_pct': 0.01,
            'max_loss_per_trade': 0.30,
            'volatility_range': (0.18, 0.28),
            'days_to_expiry_range': (14, 30),
            'moneyness_range': (0.95, 1.05),
            'min_premium': 0.50,
            'max_premium': 10.0
        }

    def generate_price_history(self, initial_price: float, days: int = 50) -> np.ndarray:
        """Generate realistic price history for technical analysis"""
        returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
        prices = [initial_price]

        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        return np.array(prices)

    def calculate_ema_signal(self, prices: np.ndarray, fast_period: int = 12, slow_period: int = 26) -> str:
        """Calculate EMA crossover signal"""
        if len(prices) < slow_period:
            return 'NEUTRAL'

        # Calculate EMAs
        fast_ema = ta.trend.EMAIndicator(close=pd.Series(prices), window=fast_period).ema_indicator().iloc[-1]
        slow_ema = ta.trend.EMAIndicator(close=pd.Series(prices), window=slow_period).ema_indicator().iloc[-1]

        # Previous EMAs for crossover detection
        fast_ema_prev = ta.trend.EMAIndicator(close=pd.Series(prices[:-1]), window=fast_period).ema_indicator().iloc[-1]
        slow_ema_prev = ta.trend.EMAIndicator(close=pd.Series(prices[:-1]), window=slow_period).ema_indicator().iloc[-1]

        # Detect crossovers
        if fast_ema > slow_ema and fast_ema_prev <= slow_ema_prev:
            return 'BULLISH'
        elif fast_ema < slow_ema and fast_ema_prev >= slow_ema_prev:
            return 'BEARISH'
        elif fast_ema > slow_ema:
            return 'BULLISH_CONT'
        else:
            return 'BEARISH_CONT'

    def calculate_rsi_signal(self, prices: np.ndarray, period: int = 14) -> str:
        """Calculate RSI-based signal"""
        if len(prices) < period + 1:
            return 'NEUTRAL'

        rsi = ta.momentum.RSIIndicator(close=pd.Series(prices), window=period).rsi().iloc[-1]

        if rsi < 30:
            return 'OVERSOLD'
        elif rsi > 70:
            return 'OVERBOUGHT'
        elif 40 <= rsi <= 60:
            return 'NEUTRAL'
        else:
            return 'TRENDING'

    def calculate_volatility_regime(self, prices: np.ndarray, window: int = 20) -> str:
        """Determine current volatility regime"""
        if len(prices) < window:
            return 'NORMAL'

        returns = np.diff(prices) / prices[:-1]
        current_vol = np.std(returns[-window:]) * np.sqrt(252)

        if current_vol < 0.15:
            return 'LOW_VOL'
        elif current_vol > 0.35:
            return 'HIGH_VOL'
        else:
            return 'NORMAL'

    async def test_base_strategy(self) -> Dict:
        """Test current strategy (baseline)"""
        return await self.run_strategy_test("Base Strategy", self.base_params)

    async def test_ema_filter(self) -> Dict:
        """Test adding EMA filter for direction bias"""
        params = self.base_params.copy()
        params['use_ema_filter'] = True
        return await self.run_strategy_test("EMA Filter", params)

    async def test_rsi_filter(self) -> Dict:
        """Test adding RSI for entry timing"""
        params = self.base_params.copy()
        params['use_rsi_filter'] = True
        return await self.run_strategy_test("RSI Filter", params)

    async def test_volatility_sizing(self) -> Dict:
        """Test volatility-based position sizing"""
        params = self.base_params.copy()
        params['volatility_sizing'] = True
        return await self.run_strategy_test("Vol Sizing", params)

    async def test_momentum_filter(self) -> Dict:
        """Test momentum-based entry filter"""
        params = self.base_params.copy()
        params['momentum_filter'] = True
        return await self.run_strategy_test("Momentum Filter", params)

    async def test_iv_rank_filter(self) -> Dict:
        """Test implied volatility rank filter"""
        params = self.base_params.copy()
        params['iv_rank_filter'] = True
        return await self.run_strategy_test("IV Rank Filter", params)

    async def test_combined_filters(self) -> Dict:
        """Test combination of best filters"""
        params = self.base_params.copy()
        params.update({
            'use_ema_filter': True,
            'volatility_sizing': True,
            'iv_rank_filter': True,
            'position_size_pct': 0.015,  # Slightly increase size
            'max_loss_per_trade': 0.25   # Tighter stop loss
        })
        return await self.run_strategy_test("Combined Strategy", params)

    async def run_strategy_test(self, strategy_name: str, params: Dict) -> Dict:
        """Run a single strategy test"""
        print(f"Testing {strategy_name}...")

        results = []
        for _ in range(self.trades_per_test):
            scenario = self.generate_enhanced_scenario(params)
            if scenario:
                result = await self.simulate_enhanced_trade(scenario, params)
                if result:
                    results.append(result)

        if len(results) < 50:
            return {'strategy': strategy_name, 'valid': False}

        df = pd.DataFrame(results)

        # Calculate metrics
        win_rate = len(df[df['is_winner']]) / len(df)
        avg_return = df['capital_impact'].mean()
        vol_return = df['capital_impact'].std()
        sharpe = (avg_return / vol_return) * np.sqrt(252 / 3) if vol_return > 0 else 0

        total_return = df['capital_impact'].sum()
        max_dd = df['capital_impact'].min()

        return {
            'strategy': strategy_name,
            'valid': True,
            'trades': len(df),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'avg_return': avg_return,
            'volatility': vol_return,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'profit_factor': self.calculate_profit_factor(df)
        }

    def generate_enhanced_scenario(self, params: Dict) -> Dict:
        """Generate scenario with filters applied"""
        # Base scenario generation
        stock_price = np.random.normal(150, 30)
        stock_price = max(50, min(300, stock_price))

        # Generate price history for technical analysis
        price_history = self.generate_price_history(stock_price, 50)

        # Apply filters
        if params.get('use_ema_filter'):
            ema_signal = self.calculate_ema_signal(price_history)
            # Skip trades against EMA trend
            if ema_signal in ['BEARISH', 'BEARISH_CONT']:
                if np.random.random() > 0.7:  # Reduce bearish trades
                    return None

        if params.get('use_rsi_filter'):
            rsi_signal = self.calculate_rsi_signal(price_history)
            # Skip trades in extreme RSI conditions
            if rsi_signal in ['OVERBOUGHT', 'OVERSOLD']:
                if np.random.random() > 0.5:  # Reduce extreme RSI trades
                    return None

        if params.get('iv_rank_filter'):
            # Simulate IV rank filter (prefer higher IV)
            iv_rank = np.random.uniform(0, 100)
            if iv_rank < 30:  # Skip low IV environments
                if np.random.random() > 0.6:
                    return None

        # Enhanced volatility selection
        if params.get('volatility_sizing'):
            vol_regime = self.calculate_volatility_regime(price_history)
            if vol_regime == 'LOW_VOL':
                volatility = np.random.uniform(0.15, 0.22)
            elif vol_regime == 'HIGH_VOL':
                volatility = np.random.uniform(0.25, 0.35)
            else:
                volatility = np.random.uniform(*params['volatility_range'])
        else:
            volatility = np.random.uniform(*params['volatility_range'])

        days_to_expiry = np.random.randint(*params['days_to_expiry_range'])
        option_type = np.random.choice(['call', 'put'])

        # Momentum-based moneyness adjustment
        if params.get('momentum_filter'):
            recent_returns = (price_history[-5:] - price_history[-6:-1]) / price_history[-6:-1]
            momentum = np.mean(recent_returns)

            if momentum > 0.01:  # Strong positive momentum
                moneyness_range = (0.98, 1.08) if option_type == 'call' else (0.92, 1.02)
            elif momentum < -0.01:  # Strong negative momentum
                moneyness_range = (0.92, 1.02) if option_type == 'call' else (0.98, 1.08)
            else:
                moneyness_range = params['moneyness_range']
        else:
            moneyness_range = params['moneyness_range']

        moneyness = np.random.uniform(*moneyness_range)
        strike_price = stock_price * moneyness

        return {
            'stock_price': stock_price,
            'strike_price': strike_price,
            'volatility': volatility,
            'days_to_expiry': days_to_expiry,
            'option_type': option_type,
            'moneyness': moneyness,
            'price_history': price_history
        }

    async def simulate_enhanced_trade(self, scenario: Dict, params: Dict) -> Dict:
        """Simulate trade with enhanced parameters"""
        try:
            # Get option pricing
            analysis = await enhanced_options_pricing_engine.get_comprehensive_option_analysis(
                underlying_price=scenario['stock_price'],
                strike_price=scenario['strike_price'],
                time_to_expiry_days=scenario['days_to_expiry'],
                volatility=scenario['volatility'],
                option_type=scenario['option_type']
            )

            entry_price = analysis['pricing']['theoretical_price']

            # Premium filter
            if entry_price < params['min_premium'] or entry_price > params['max_premium']:
                return None

            # Volatility-based position sizing
            if params.get('volatility_sizing'):
                vol_regime = self.calculate_volatility_regime(scenario['price_history'])
                if vol_regime == 'LOW_VOL':
                    position_size = params['position_size_pct'] * 1.2  # Increase size in low vol
                elif vol_regime == 'HIGH_VOL':
                    position_size = params['position_size_pct'] * 0.8  # Decrease size in high vol
                else:
                    position_size = params['position_size_pct']
            else:
                position_size = params['position_size_pct']

            # Simulate trade outcome (simplified)
            holding_days = 3
            daily_return = np.random.normal(0, scenario['volatility'] / np.sqrt(252))
            daily_return = np.clip(daily_return * 0.5, -0.05, 0.05)

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
                if scenario['option_type'] == 'call':
                    exit_price = max(0, final_stock_price - scenario['strike_price'])
                else:
                    exit_price = max(0, scenario['strike_price'] - final_stock_price)

            # Calculate P&L with stop loss
            raw_pnl = exit_price - entry_price
            max_loss = entry_price * params['max_loss_per_trade']

            if raw_pnl < -max_loss:
                raw_pnl = -max_loss
                exit_price = entry_price - max_loss

            # Position sizing
            risk_amount = 18113.50 * position_size
            contracts = max(1, int(risk_amount / (entry_price * 100)))
            position_pnl = raw_pnl * contracts * 100
            capital_impact = position_pnl / 18113.50 * 100

            return {
                'capital_impact': capital_impact,
                'is_winner': raw_pnl > 0,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size
            }

        except Exception:
            return None

    def calculate_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor"""
        winners = df[df['is_winner']]['capital_impact'].sum()
        losers = abs(df[~df['is_winner']]['capital_impact'].sum())
        return winners / losers if losers > 0 else float('inf')

    async def run_all_tests(self):
        """Run all improvement tests"""
        print("=" * 80)
        print("SHARPE RATIO IMPROVEMENT ANALYSIS")
        print("Testing various enhancements to boost performance beyond 1.38")
        print("=" * 80)

        tests = [
            ("Base Strategy", self.test_base_strategy),
            ("EMA Filter", self.test_ema_filter),
            ("RSI Filter", self.test_rsi_filter),
            ("Volatility Sizing", self.test_volatility_sizing),
            ("Momentum Filter", self.test_momentum_filter),
            ("IV Rank Filter", self.test_iv_rank_filter),
            ("Combined Strategy", self.test_combined_filters)
        ]

        results = []
        for name, test_func in tests:
            result = await test_func()
            if result.get('valid'):
                results.append(result)

        # Analyze and rank results
        self.analyze_improvements(results)

    def analyze_improvements(self, results: List[Dict]):
        """Analyze and rank all improvements"""
        df = pd.DataFrame(results)
        df_sorted = df.sort_values('sharpe_ratio', ascending=False)

        print(f"\nSTRATEGY COMPARISON RESULTS:")
        print("-" * 100)
        print(f"{'Strategy':<20} {'Sharpe':<8} {'Win%':<8} {'Avg Ret%':<10} {'Vol%':<8} {'Total%':<10} {'Max DD%':<10}")
        print("-" * 100)

        for _, row in df_sorted.iterrows():
            print(f"{row['strategy']:<20} {row['sharpe_ratio']:<8.4f} "
                  f"{row['win_rate']:<8.1%} {row['avg_return']:<10.4f} "
                  f"{row['volatility']:<8.4f} {row['total_return']:<10.2f} {row['max_drawdown']:<10.2f}")

        # Find best strategy
        best_strategy = df_sorted.iloc[0]
        base_strategy = df[df['strategy'] == 'Base Strategy'].iloc[0] if len(df[df['strategy'] == 'Base Strategy']) > 0 else None

        print("\n" + "=" * 80)
        print("IMPROVEMENT ANALYSIS")
        print("=" * 80)

        if base_strategy is not None:
            improvement = ((best_strategy['sharpe_ratio'] - base_strategy['sharpe_ratio']) /
                          base_strategy['sharpe_ratio'] * 100)

            print(f"BEST STRATEGY: {best_strategy['strategy']}")
            print(f"  Sharpe Ratio: {best_strategy['sharpe_ratio']:.4f}")
            print(f"  Improvement: +{improvement:.1f}% over base strategy")
            print(f"  New Sharpe: {best_strategy['sharpe_ratio']:.4f} vs Base: {base_strategy['sharpe_ratio']:.4f}")

        print(f"\nTOP 3 IMPROVEMENTS:")
        for i, (_, row) in enumerate(df_sorted.head(3).iterrows()):
            print(f"  {i+1}. {row['strategy']}: {row['sharpe_ratio']:.4f} Sharpe")

        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if df_sorted.iloc[0]['sharpe_ratio'] > 1.5:
            print(f"  🚀 EXCELLENT: Implementing these improvements could boost")
            print(f"     your Sharpe ratio to {df_sorted.iloc[0]['sharpe_ratio']:.4f}")
        elif df_sorted.iloc[0]['sharpe_ratio'] > 1.4:
            print(f"  ✅ GOOD: Modest improvement possible")
        else:
            print(f"  ⚠️  LIMITED: Current strategy already well-optimized")

async def main():
    analyzer = SharpeImprovementAnalysis(trades_per_test=500)
    await analyzer.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())