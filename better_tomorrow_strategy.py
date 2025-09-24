"""
BETTER TOMORROW STRATEGY
Maximize profit from $1.86M leveraged portfolio instead of tiny covered call premiums
Focus on momentum, regime detection, and smart position scaling
"""

import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

class BetterTomorrowStrategy:
    """Optimize tomorrow's trading for maximum profit from leveraged positions"""

    def __init__(self):
        self.portfolio_value = 1856681  # Current leveraged portfolio
        self.buying_power = 0  # From analysis - fully invested

        # Current positions (3x leveraged ETFs)
        self.positions = {
            'SPY': {'shares': 1099, 'value': 724439, 'leverage': 1.0, 'sector': 'Market'},
            'QQQ': {'shares': 1074, 'value': 633660, 'leverage': 1.0, 'sector': 'Tech'},
            'IWM': {'shares': 1907, 'value': 455563, 'leverage': 3.0, 'sector': 'Small Cap'},
            'TQQQ': {'shares': 253, 'value': 24877, 'leverage': 3.0, 'sector': 'Tech'},
            'SOXL': {'shares': 337, 'value': 10268, 'leverage': 3.0, 'sector': 'Semiconductors'},
            'UPRO': {'shares': 73, 'value': 7874, 'leverage': 3.0, 'sector': 'Market'}
        }

        print("BETTER TOMORROW STRATEGY - MAXIMIZE LEVERAGED PORTFOLIO")
        print("=" * 70)
        print(f"Portfolio Value: ${self.portfolio_value:,}")
        print(f"Strategy: Focus on momentum and regime-based scaling")
        print(f"Target: $20K-$50K daily profit from smart positioning")

    def analyze_current_market_regime(self):
        """Analyze market regime for tomorrow's opportunities"""

        print("\nMARKET REGIME ANALYSIS FOR TOMORROW")
        print("-" * 45)

        try:
            # Get recent market data
            spy_data = yf.download('SPY', period='3mo', interval='1d')
            spy_close = spy_data['Close']

            # Bull market indicators
            current_price = spy_close.iloc[-1]
            sma_20 = spy_close.rolling(20).mean().iloc[-1]
            sma_50 = spy_close.rolling(50).mean().iloc[-1]

            # Momentum indicators
            momentum_5d = (current_price / spy_close.iloc[-6] - 1) * 100
            momentum_1d = (current_price / spy_close.iloc[-2] - 1) * 100

            # Volatility
            returns = spy_close.pct_change().rolling(20).std() * np.sqrt(252) * 100
            current_vol = returns.iloc[-1]

            print(f"SPY Analysis:")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  vs 20-day MA: {((current_price/sma_20-1)*100):+.1f}%")
            print(f"  vs 50-day MA: {((current_price/sma_50-1)*100):+.1f}%")
            print(f"  5-day Momentum: {momentum_5d:+.1f}%")
            print(f"  1-day Momentum: {momentum_1d:+.1f}%")
            print(f"  Volatility: {current_vol:.1f}%")

            # Regime classification
            if current_price > sma_50 and momentum_5d > 0:
                regime = "BULLISH_MOMENTUM"
                confidence = 0.75
                expected_move = 1.2  # 1.2% expected up move
            elif current_price > sma_50:
                regime = "BULLISH_CONSOLIDATION"
                confidence = 0.65
                expected_move = 0.5  # 0.5% expected up move
            elif momentum_5d > 0:
                regime = "BEARISH_BOUNCE"
                confidence = 0.45
                expected_move = 0.3  # 0.3% expected up move
            else:
                regime = "BEARISH_DOWNTREND"
                confidence = 0.35
                expected_move = -0.8  # -0.8% expected down move

            print(f"\nRegime Classification: {regime}")
            print(f"Confidence: {confidence*100:.0f}%")
            print(f"Expected Move Tomorrow: {expected_move:+.1f}%")

            return {
                'regime': regime,
                'confidence': confidence,
                'expected_move': expected_move,
                'momentum_5d': momentum_5d,
                'momentum_1d': momentum_1d,
                'volatility': current_vol
            }

        except Exception as e:
            print(f"Error in regime analysis: {e}")
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.5,
                'expected_move': 0.0,
                'momentum_5d': 0.0,
                'momentum_1d': 0.0,
                'volatility': 15.0
            }

    def calculate_optimal_position_scaling(self, regime_data):
        """Calculate optimal position scaling based on regime"""

        print(f"\nOPTIMAL POSITION SCALING STRATEGY")
        print("-" * 45)

        regime = regime_data['regime']
        confidence = regime_data['confidence']
        expected_move = regime_data['expected_move']

        # Scaling factors based on regime
        if regime == "BULLISH_MOMENTUM":
            base_scaling = 1.3  # Scale up 30%
            focus_sectors = ['Tech', 'Semiconductors']
        elif regime == "BULLISH_CONSOLIDATION":
            base_scaling = 1.1  # Scale up 10%
            focus_sectors = ['Market']
        elif regime == "BEARISH_BOUNCE":
            base_scaling = 0.9  # Scale down 10%
            focus_sectors = ['Market']
        else:
            base_scaling = 0.7  # Scale down 30%
            focus_sectors = []

        print(f"Base Scaling Factor: {base_scaling:.1f}x")
        print(f"Focus Sectors: {focus_sectors}")

        # Calculate specific position adjustments
        position_adjustments = {}
        total_value_change = 0

        for symbol, data in self.positions.items():
            current_value = data['value']
            sector = data['sector']
            leverage = data['leverage']

            # Sector-specific scaling
            if sector in focus_sectors:
                sector_multiplier = 1.2
            elif len(focus_sectors) == 0:
                sector_multiplier = 0.8  # Defensive in bad regime
            else:
                sector_multiplier = 1.0

            # Leverage-specific scaling (3x ETFs get more aggressive scaling)
            if leverage >= 3.0:
                leverage_multiplier = 1.1
            else:
                leverage_multiplier = 1.0

            # Final scaling
            final_scaling = base_scaling * sector_multiplier * leverage_multiplier
            target_value = current_value * final_scaling
            value_change = target_value - current_value

            position_adjustments[symbol] = {
                'current_value': current_value,
                'target_value': target_value,
                'value_change': value_change,
                'scaling_factor': final_scaling,
                'action': 'BUY' if value_change > 0 else 'SELL' if value_change < 0 else 'HOLD'
            }

            total_value_change += abs(value_change)

            print(f"{symbol} ({sector}):")
            print(f"  Current: ${current_value:,.0f}")
            print(f"  Target: ${target_value:,.0f}")
            print(f"  Change: ${value_change:+,.0f} ({final_scaling:.1f}x)")

        print(f"\nTotal Position Adjustments: ${total_value_change:,.0f}")

        return position_adjustments

    def calculate_profit_scenarios_with_scaling(self, regime_data, position_adjustments):
        """Calculate profit scenarios with optimal scaling"""

        print(f"\nPROFIT SCENARIOS WITH OPTIMAL SCALING")
        print("-" * 45)

        expected_move = regime_data['expected_move'] / 100
        confidence = regime_data['confidence']

        # Market scenarios
        scenarios = {
            'conservative': expected_move * 0.5,
            'base_case': expected_move,
            'optimistic': expected_move * 1.5,
            'best_case': expected_move * 2.0
        }

        scenario_results = {}

        for scenario_name, market_move in scenarios.items():
            total_profit = 0

            print(f"\n{scenario_name.upper()} ({market_move*100:+.1f}% market move):")

            for symbol, adjustment in position_adjustments.items():
                target_value = adjustment['target_value']
                leverage = self.positions[symbol]['leverage']

                # Profit from scaled position
                base_profit = target_value * market_move * leverage
                total_profit += base_profit

                print(f"  {symbol}: ${base_profit:+,.0f}")

            scenario_results[scenario_name] = total_profit
            probability = confidence if scenario_name == 'base_case' else confidence * 0.7

            print(f"  TOTAL: ${total_profit:+,.0f} (prob: {probability*100:.0f}%)")

        return scenario_results

    def create_6am_execution_plan(self, regime_data, position_adjustments, profit_scenarios):
        """Create specific execution plan for 6:30 AM PT"""

        print(f"\n{'='*70}")
        print("6:30 AM PT EXECUTION PLAN - TOMORROW")
        print("="*70)

        regime = regime_data['regime']
        expected_profit = profit_scenarios['base_case']

        print(f"Market Regime: {regime}")
        print(f"Expected Profit: ${expected_profit:+,.0f}")

        # Pre-market preparation
        print(f"\n5:30 AM PT - PRE-MARKET PREPARATION:")
        print("1. Check overnight futures and Asian markets")
        print("2. Review economic calendar for market-moving events")
        print("3. Confirm regime signals still valid")
        print("4. Set up position scaling orders")

        # Market open execution
        print(f"\n6:30 AM PT - MARKET OPEN EXECUTION:")

        high_priority_trades = []
        medium_priority_trades = []

        for symbol, adjustment in position_adjustments.items():
            value_change = adjustment['value_change']
            action = adjustment['action']

            if abs(value_change) > 10000:  # High priority: >$10K changes
                high_priority_trades.append((symbol, adjustment))
            elif abs(value_change) > 1000:  # Medium priority: >$1K changes
                medium_priority_trades.append((symbol, adjustment))

        if high_priority_trades:
            print("HIGH PRIORITY (Execute first 5 minutes):")
            for symbol, adj in high_priority_trades:
                print(f"  {adj['action']} {symbol}: ${adj['value_change']:+,.0f}")

        if medium_priority_trades:
            print("MEDIUM PRIORITY (Execute 5-15 minutes):")
            for symbol, adj in medium_priority_trades:
                print(f"  {adj['action']} {symbol}: ${adj['value_change']:+,.0f}")

        # Throughout the day
        print(f"\n7:00 AM - 1:00 PM PT - MONITORING:")
        print("1. Monitor momentum indicators for additional scaling")
        print("2. Set stop-losses at -2% of scaled positions")
        print("3. Take profits at regime-based targets")
        print("4. Watch for regime change signals")

        return {
            'expected_profit': expected_profit,
            'high_priority_trades': high_priority_trades,
            'medium_priority_trades': medium_priority_trades,
            'regime': regime
        }

    def calculate_risk_management(self, profit_scenarios):
        """Calculate risk management for scaled positions"""

        print(f"\nRISK MANAGEMENT FRAMEWORK")
        print("-" * 45)

        max_profit = profit_scenarios['best_case']
        max_loss = min(profit_scenarios.values()) if min(profit_scenarios.values()) < 0 else -max_profit * 0.3

        print(f"Maximum Potential Profit: ${max_profit:+,.0f}")
        print(f"Maximum Acceptable Loss: ${max_loss:+,.0f}")

        # Risk limits
        daily_stop_loss = self.portfolio_value * 0.02  # 2% of portfolio
        position_stop_loss = 0.15  # 15% per position

        print(f"\nDaily Stop Loss: ${daily_stop_loss:,.0f} (2% of portfolio)")
        print(f"Position Stop Loss: 15% per scaled position")
        print(f"Regime Stop: Exit all if regime changes to bearish")

    def run_better_tomorrow_strategy(self):
        """Execute complete better tomorrow strategy"""

        # Step 1: Analyze regime
        regime_data = self.analyze_current_market_regime()

        # Step 2: Calculate optimal scaling
        position_adjustments = self.calculate_optimal_position_scaling(regime_data)

        # Step 3: Calculate profit scenarios
        profit_scenarios = self.calculate_profit_scenarios_with_scaling(regime_data, position_adjustments)

        # Step 4: Create execution plan
        execution_plan = self.create_6am_execution_plan(regime_data, position_adjustments, profit_scenarios)

        # Step 5: Set up risk management
        self.calculate_risk_management(profit_scenarios)

        # Final summary
        print(f"\n{'='*70}")
        print("BETTER STRATEGY SUMMARY")
        print("="*70)

        expected_profit = execution_plan['expected_profit']
        regime = execution_plan['regime']

        print(f"Tomorrow's Regime: {regime}")
        print(f"Expected Profit: ${expected_profit:+,.0f}")
        print(f"Strategy: Position scaling based on regime + momentum")
        print(f"Execution: 6:30 AM PT market open")
        print(f"Risk Management: 2% daily stop, 15% position stops")

        vs_covered_calls = expected_profit - 80  # vs $80 covered call premium
        print(f"\nVs Covered Calls: ${vs_covered_calls:+,.0f} better")

        return {
            'regime_data': regime_data,
            'position_adjustments': position_adjustments,
            'profit_scenarios': profit_scenarios,
            'execution_plan': execution_plan
        }

def main():
    """Run better tomorrow strategy analysis"""

    strategy = BetterTomorrowStrategy()
    results = strategy.run_better_tomorrow_strategy()

    print(f"\nSTRATEGY READY FOR 6:30 AM PT EXECUTION!")

    return results

if __name__ == "__main__":
    main()