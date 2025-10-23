#!/usr/bin/env python3
"""
COMPREHENSIVE FOREX OPTIMIZATION
Tests multiple timeframes (4H, Daily) and parameter sets
Goal: Find 65%+ win rate with reasonable trade frequency

FIXES:
1. Tests 4-HOUR and DAILY timeframes (not 1-hour)
2. Correct pip calculation for JPY pairs
3. More realistic entry criteria
4. Comprehensive parameter grid
"""

from data.oanda_data_fetcher import OandaDataFetcher
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json

class ComprehensiveForexOptimizer:
    """
    Comprehensive forex strategy optimizer
    Tests multiple timeframes and parameters
    """

    def __init__(self):
        self.fetcher = OandaDataFetcher()

        # Parameter combinations to test (LOOSER FILTERS)
        self.parameter_sets = [
            # Original parameters
            {'fast': 10, 'slow': 20, 'trend': 200, 'rsi_long': 50, 'rsi_short': 50, 'score_min': 7.0, 'name': '10/20/200 RSI50'},

            # Fibonacci EMAs
            {'fast': 8, 'slow': 21, 'trend': 200, 'rsi_long': 50, 'rsi_short': 50, 'score_min': 7.0, 'name': '8/21/200 RSI50'},

            # MACD-based
            {'fast': 12, 'slow': 26, 'trend': 200, 'rsi_long': 50, 'rsi_short': 50, 'score_min': 7.0, 'name': '12/26/200 RSI50'},

            # Shorter trend (more trades)
            {'fast': 10, 'slow': 20, 'trend': 100, 'rsi_long': 50, 'rsi_short': 50, 'score_min': 6.0, 'name': '10/20/100 Loose'},
            {'fast': 8, 'slow': 21, 'trend': 100, 'rsi_long': 50, 'rsi_short': 50, 'score_min': 6.0, 'name': '8/21/100 Loose'},

            # Stricter RSI
            {'fast': 10, 'slow': 20, 'trend': 200, 'rsi_long': 55, 'rsi_short': 45, 'score_min': 8.0, 'name': '10/20/200 RSI55/45'},
            {'fast': 8, 'slow': 21, 'trend': 200, 'rsi_long': 55, 'rsi_short': 45, 'score_min': 8.0, 'name': '8/21/200 RSI55/45'},

            # Very strict (quality over quantity)
            {'fast': 10, 'slow': 20, 'trend': 200, 'rsi_long': 60, 'rsi_short': 40, 'score_min': 9.0, 'name': '10/20/200 Strict'},

            # Wide EMA spread
            {'fast': 10, 'slow': 30, 'trend': 200, 'rsi_long': 50, 'rsi_short': 50, 'score_min': 7.0, 'name': '10/30/200 Wide'},
            {'fast': 13, 'slow': 48, 'trend': 200, 'rsi_long': 50, 'rsi_short': 50, 'score_min': 7.0, 'name': '13/48/200 Wide'},
        ]

        # Forex pairs
        self.pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

        # Timeframes to test
        self.timeframes = [
            {'tf': 'H4', 'name': '4-Hour', 'candles': 2000},  # ~330 days of 4H data
            {'tf': 'D', 'name': 'Daily', 'candles': 365}
        ]

    def get_pip_multiplier(self, pair: str) -> float:
        """Get correct pip multiplier based on pair"""
        if 'JPY' in pair:
            return 100  # JPY pairs: 2 decimals
        else:
            return 10000  # Other pairs: 5 decimals

    def calculate_indicators(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate EMAs, RSI, and ATR"""

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=params['fast'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=params['slow'], adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=params['trend'], adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()

        return df

    def backtest_strategy(self, df: pd.DataFrame, pair: str, params: Dict) -> Dict:
        """Run backtest with given parameters"""

        # Calculate indicators
        df = self.calculate_indicators(df.copy(), params)

        # Get pip multiplier
        pip_multiplier = self.get_pip_multiplier(pair)

        trades = []
        in_position = False
        current_trade = None

        # Start after enough data
        start_idx = params['trend'] + 20

        for i in range(start_idx, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]

            price = current['close']
            ema_fast_curr = current['ema_fast']
            ema_slow_curr = current['ema_slow']
            ema_trend_curr = current['ema_trend']
            rsi = current['rsi']
            atr = current['atr']

            ema_fast_prev = previous['ema_fast']
            ema_slow_prev = previous['ema_slow']

            # Skip if NaN
            if pd.isna(rsi) or pd.isna(atr):
                continue

            # Check for crossovers
            bullish_cross = (ema_fast_curr > ema_slow_curr) and (ema_fast_prev <= ema_slow_prev)
            bearish_cross = (ema_fast_curr < ema_slow_curr) and (ema_fast_prev >= ema_slow_prev)

            # Scoring (more lenient)
            if not in_position:
                score = 5.0
                direction = None

                # LONG Setup
                if bullish_cross:
                    if price > ema_trend_curr:
                        score += 2.0
                    if rsi > params['rsi_long']:
                        score += 2.0
                    if rsi > 60:
                        score += 1.0

                    if score >= params['score_min']:
                        direction = 'LONG'

                # SHORT Setup
                elif bearish_cross:
                    if price < ema_trend_curr:
                        score += 2.0
                    if rsi < params['rsi_short']:
                        score += 2.0
                    if rsi < 40:
                        score += 1.0

                    if score >= params['score_min']:
                        direction = 'SHORT'

                # Enter trade
                if direction:
                    if direction == 'LONG':
                        entry = price
                        stop_loss = entry - (2 * atr)
                        take_profit = entry + (3 * atr)
                    else:
                        entry = price
                        stop_loss = entry + (2 * atr)
                        take_profit = entry - (3 * atr)

                    current_trade = {
                        'entry_time': current.name,
                        'entry_price': entry,
                        'direction': direction,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'rsi': rsi,
                        'score': score,
                        'entry_idx': i
                    }
                    in_position = True

            # Check exits
            if in_position and current_trade:
                exit_triggered = False
                exit_price = None
                outcome = None

                if current_trade['direction'] == 'LONG':
                    if price <= current_trade['stop_loss']:
                        exit_price = current_trade['stop_loss']
                        outcome = 'LOSS'
                        exit_triggered = True
                    elif price >= current_trade['take_profit']:
                        exit_price = current_trade['take_profit']
                        outcome = 'WIN'
                        exit_triggered = True

                else:  # SHORT
                    if price >= current_trade['stop_loss']:
                        exit_price = current_trade['stop_loss']
                        outcome = 'LOSS'
                        exit_triggered = True
                    elif price <= current_trade['take_profit']:
                        exit_price = current_trade['take_profit']
                        outcome = 'WIN'
                        exit_triggered = True

                if exit_triggered:
                    # Calculate pips
                    if current_trade['direction'] == 'LONG':
                        profit_pips = (exit_price - current_trade['entry_price']) * pip_multiplier
                    else:
                        profit_pips = (current_trade['entry_price'] - exit_price) * pip_multiplier

                    current_trade['exit_time'] = current.name
                    current_trade['exit_price'] = exit_price
                    current_trade['outcome'] = outcome
                    current_trade['profit_pips'] = profit_pips
                    current_trade['bars_held'] = i - current_trade['entry_idx']

                    trades.append(current_trade)
                    in_position = False
                    current_trade = None

        # Calculate statistics
        if len(trades) == 0:
            return {
                'pair': pair,
                'params': params,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pips': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0
            }

        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']

        win_rate = (len(wins) / len(trades)) * 100
        total_pips = sum([t['profit_pips'] for t in trades])
        avg_win = sum([t['profit_pips'] for t in wins]) / len(wins) if wins else 0
        avg_loss = sum([t['profit_pips'] for t in losses]) / len(losses) if losses else 0

        gross_profit = sum([t['profit_pips'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['profit_pips'] for t in losses])) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Expectancy (average pips per trade)
        expectancy = total_pips / len(trades)

        return {
            'pair': pair,
            'params': params,
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }

    def optimize(self):
        """Run comprehensive optimization"""

        print("\n" + "="*80)
        print("COMPREHENSIVE FOREX OPTIMIZATION")
        print("="*80)
        print(f"\nPairs: {', '.join(self.pairs)}")
        print(f"Timeframes: {', '.join([tf['name'] for tf in self.timeframes])}")
        print(f"Parameter Sets: {len(self.parameter_sets)}")
        print("\n")

        all_results = {}

        for tf_info in self.timeframes:
            tf = tf_info['tf']
            tf_name = tf_info['name']
            candles = tf_info['candles']

            print(f"\n{'='*80}")
            print(f"TIMEFRAME: {tf_name} ({tf})")
            print(f"{'='*80}")

            tf_results = []

            for pair in self.pairs:
                print(f"\n  Testing {pair}...")

                # Fetch data
                df = self.fetcher.get_bars(pair, tf, limit=candles)

                if df is None or df.empty:
                    print(f"  [ERROR] Could not fetch data for {pair}")
                    continue

                print(f"  [OK] Fetched {len(df)} candles ({df.index[0]} to {df.index[-1]})")

                # Test each parameter set
                for params in self.parameter_sets:
                    result = self.backtest_strategy(df, pair, params)
                    result['timeframe'] = tf_name
                    tf_results.append(result)

            all_results[tf_name] = tf_results

            # Show summary for this timeframe
            self.show_timeframe_summary(tf_name, tf_results)

        return all_results

    def show_timeframe_summary(self, tf_name: str, results: List[Dict]):
        """Show summary for a timeframe"""

        print(f"\n  --- {tf_name} Summary ---")

        # Group by parameter set
        param_groups = {}
        for r in results:
            name = r['params']['name']
            if name not in param_groups:
                param_groups[name] = []
            param_groups[name].append(r)

        # Calculate aggregate stats
        for name, group in param_groups.items():
            total_trades = sum([g['trades'] for g in group])
            total_wins = sum([g['wins'] for g in group])
            total_losses = sum([g['losses'] for g in group])
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            total_pips = sum([g['total_pips'] for g in group])

            if total_trades >= 5:  # Only show if at least 5 trades
                print(f"  {name}: {win_rate:.1f}% WR, {total_trades} trades, {total_pips:.0f} pips")

    def analyze_results(self, all_results: Dict) -> Dict:
        """Analyze all results and find best parameters"""

        print("\n" + "="*80)
        print("FINAL ANALYSIS")
        print("="*80)

        best_overall = None
        best_overall_score = 0

        for tf_name, results in all_results.items():
            print(f"\n{tf_name} Timeframe:")
            print("-" * 80)

            # Group by parameter set
            param_groups = {}
            for r in results:
                name = r['params']['name']
                if name not in param_groups:
                    param_groups[name] = {
                        'results': [],
                        'total_trades': 0,
                        'total_wins': 0,
                        'total_losses': 0,
                        'total_pips': 0
                    }

                param_groups[name]['results'].append(r)
                param_groups[name]['total_trades'] += r['trades']
                param_groups[name]['total_wins'] += r['wins']
                param_groups[name]['total_losses'] += r['losses']
                param_groups[name]['total_pips'] += r['total_pips']

            # Rank by combined score (win_rate * 0.7 + profit_factor * 0.3)
            ranked = []
            for name, data in param_groups.items():
                if data['total_trades'] >= 5:  # Minimum trades threshold
                    win_rate = (data['total_wins'] / data['total_trades']) * 100

                    # Calculate profit factor
                    wins_pips = sum([r['avg_win'] * r['wins'] for r in data['results']])
                    loss_pips = abs(sum([r['avg_loss'] * r['losses'] for r in data['results']]))
                    profit_factor = wins_pips / loss_pips if loss_pips > 0 else 0

                    # Combined score
                    score = (win_rate * 0.6) + (profit_factor * 20) + (min(data['total_trades'], 30) * 0.5)

                    ranked.append({
                        'name': name,
                        'timeframe': tf_name,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'total_trades': data['total_trades'],
                        'total_pips': data['total_pips'],
                        'score': score,
                        'params': data['results'][0]['params']
                    })

            # Sort by score
            ranked.sort(key=lambda x: x['score'], reverse=True)

            # Show top 3
            for i, r in enumerate(ranked[:3], 1):
                print(f"{i}. {r['name']}")
                print(f"   Win Rate: {r['win_rate']:.1f}% | Trades: {r['total_trades']} | Pips: {r['total_pips']:.0f} | PF: {r['profit_factor']:.2f}x")

            # Track best overall
            if ranked and ranked[0]['score'] > best_overall_score:
                best_overall = ranked[0]
                best_overall_score = ranked[0]['score']

        return best_overall

    def save_results(self, best: Dict, all_results: Dict):
        """Save results to markdown"""

        content = f"""# COMPREHENSIVE FOREX OPTIMIZATION RESULTS

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Pairs Tested:** {', '.join(self.pairs)}
**Timeframes:** {', '.join([tf['name'] for tf in self.timeframes])}

---

## BEST PARAMETERS FOUND

### {best['name']} - {best['timeframe']}

**Configuration:**
- Fast EMA: **{best['params']['fast']}**
- Slow EMA: **{best['params']['slow']}**
- Trend EMA: **{best['params']['trend']}**
- RSI Long: >{best['params']['rsi_long']}
- RSI Short: <{best['params']['rsi_short']}
- Min Score: {best['params']['score_min']}

**Performance:**
- Win Rate: **{best['win_rate']:.1f}%**
- Total Trades: {best['total_trades']}
- Total Pips: {best['total_pips']:.1f}
- Profit Factor: {best['profit_factor']:.2f}x
- Timeframe: **{best['timeframe']}**

### Goal Achievement

"""

        if best['win_rate'] >= 65:
            content += "**STATUS:** SUCCESS - 65%+ win rate achieved!\n\n"
        elif best['win_rate'] >= 60:
            content += "**STATUS:** GOOD - Win rate above 60%\n\n"
        elif best['win_rate'] >= 55:
            content += "**STATUS:** ACCEPTABLE - Win rate above 55%\n\n"
        else:
            content += "**STATUS:** NEEDS IMPROVEMENT - Continue optimization\n\n"

        content += """---

## KEY FINDINGS

### Critical Insights:

1. **Timeframe Matters:** The strategy performs differently on 4H vs Daily
2. **Trade Frequency:** Too strict = no trades, too loose = poor win rate
3. **USD/JPY pip calculation fixed:** JPY pairs now calculate correctly
4. **Parameter sensitivity:** Small EMA changes have big impacts

### Recommendations:

"""

        if best['win_rate'] >= 65:
            content += """- **READY FOR PAPER TRADING** with recommended parameters
- Start with minimum position sizes
- Monitor for 30 days before scaling
- Recalibrate if win rate drops below 60%
"""
        else:
            content += """- **CONTINUE OPTIMIZATION** - Goal not yet achieved
- Consider additional parameter combinations
- Test longer time periods
- Evaluate other entry criteria (volume, volatility, etc.)
"""

        content += """
---

## NEXT STEPS

1. [ ] Implement best parameters in strategy file
2. [ ] Run 30-day forward test
3. [ ] Paper trade with OANDA practice account
4. [ ] Monitor and adjust as needed
5. [ ] Scale up gradually when consistent

---

*Generated by forex_comprehensive_optimization.py*
"""

        with open('C:\\Users\\lucas\\PC-HIVE-TRADING\\FOREX_OPTIMIZATION_RESULTS.md', 'w', encoding='utf-8') as f:
            f.write(content)

        print("\n[OK] Results saved to: FOREX_OPTIMIZATION_RESULTS.md")


def main():
    """Run comprehensive optimization"""

    optimizer = ComprehensiveForexOptimizer()

    # Run optimization
    all_results = optimizer.optimize()

    if not all_results:
        print("\n[ERROR] No results generated")
        return

    # Analyze and find best
    best = optimizer.analyze_results(all_results)

    if best:
        print("\n" + "="*80)
        print("RECOMMENDED PARAMETERS")
        print("="*80)
        print(f"\n{best['name']} on {best['timeframe']}")
        print(f"Win Rate: {best['win_rate']:.1f}%")
        print(f"Total Trades: {best['total_trades']}")
        print(f"Total Pips: {best['total_pips']:.1f}")
        print(f"Profit Factor: {best['profit_factor']:.2f}x")

        print(f"\nConfiguration:")
        print(f"  Fast EMA: {best['params']['fast']}")
        print(f"  Slow EMA: {best['params']['slow']}")
        print(f"  Trend EMA: {best['params']['trend']}")
        print(f"  RSI Thresholds: {best['params']['rsi_long']}/{best['params']['rsi_short']}")

        if best['win_rate'] >= 65:
            print("\n SUCCESS! 65%+ win rate achieved!")
        elif best['win_rate'] >= 60:
            print("\n GOOD! Win rate above 60%")
        else:
            print("\n NEEDS WORK - Continue optimization")

        # Save results
        optimizer.save_results(best, all_results)
    else:
        print("\n[WARNING] Could not determine best parameters")


if __name__ == "__main__":
    main()
