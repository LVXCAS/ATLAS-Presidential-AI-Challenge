"""
SIMPLIFIED BACKTEST - Safe vs Aggressive

Quick test to see which strategy avoids daily DD violations.
"""

import backtrader as bt
import pandas as pd
from pathlib import Path

class SafeStrategy(bt.Strategy):
    """Conservative: 2 lots, moderate threshold"""

    params = (
        ('max_lots', 2.0),
        ('score_threshold', 5.5),
        ('stop_pips', 20),
        ('max_trades_week', 3),
    )

    def __init__(self):
        self.daily_dd_violations = 0
        self.daily_losses = {}
        self.trades_this_week = 0
        self.weekly_reset_date = None
        self.orders = {}  # Track open orders

        # Indicators for each pair
        self.indicators = {}
        for data in self.datas:
            self.indicators[data._name] = {
                'rsi': bt.indicators.RSI(data.close, period=14),
                'ema50': bt.indicators.EMA(data.close, period=50),
                'ema200': bt.indicators.EMA(data.close, period=200),
            }

    def notify_trade(self, trade):
        if trade.isclosed:
            today = self.data.datetime.date(0)
            if today not in self.daily_losses:
                self.daily_losses[today] = 0
            self.daily_losses[today] += trade.pnl

            # Check daily DD violation
            if self.daily_losses[today] < -3000:
                self.daily_dd_violations += 1
                print(f'[DD VIOLATION] {today}: ${self.daily_losses[today]:.2f}')

    def next(self):
        # Reset weekly counter
        today = self.data.datetime.date(0)
        if self.weekly_reset_date is None or (today - self.weekly_reset_date).days >= 7:
            self.trades_this_week = 0
            self.weekly_reset_date = today

        # Limit trades
        if self.trades_this_week >= self.params.max_trades_week:
            return

        # Scan each pair
        for data in self.datas:
            if self.getposition(data).size != 0:
                continue

            if len(data) < 200:
                continue

            ind = self.indicators[data._name]
            score = self.score_setup(data, ind)

            if score >= self.params.score_threshold:
                # Buy with stop-loss and take-profit
                size = int(self.params.max_lots * 100000)

                # Calculate stop and target prices
                entry_price = data.close[0]
                stop_distance = self.params.stop_pips * 0.0001  # pips to price
                stop_price = entry_price - stop_distance
                target_price = entry_price + (stop_distance * 2)  # 2R target

                # Enter with bracket order
                order = self.buy_bracket(
                    data=data,
                    size=size,
                    stopprice=stop_price,
                    limitprice=target_price
                )

                self.trades_this_week += 1
                print(f'{today} BUY {data._name}: {size} units, score={score:.1f}, SL={stop_price:.5f}, TP={target_price:.5f}')

    def score_setup(self, data, ind):
        """Simple scoring"""
        score = 0
        price = data.close[0]
        rsi = ind['rsi'][0]
        ema50 = ind['ema50'][0]
        ema200 = ind['ema200'][0]

        # Trend
        if price > ema200:
            score += 2.0

        # RSI pullback
        if 40 <= rsi <= 60:
            score += 2.0

        # Price above EMA50
        if price > ema50:
            score += 1.5

        return score


class AggressiveStrategy(SafeStrategy):
    """Aggressive: 3 lots, looser threshold"""

    params = (
        ('max_lots', 3.0),
        ('score_threshold', 4.5),
        ('stop_pips', 15),
        ('max_trades_week', 6),
    )


def run_simple_backtest(strategy_class, name):
    """Run simplified backtest"""

    print(f"\n{'='*80}")
    print(f"TESTING: {name}")
    print(f"{'='*80}")

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(200000.0)
    cerebro.broker.setcommission(commission=0.0001)

    # Load data
    data_dir = Path(__file__).parent / 'data'
    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    for pair in pairs:
        csv_file = data_dir / f"{pair}_H1_6M.csv"
        df = pd.read_csv(csv_file, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)

        data = bt.feeds.PandasData(dataname=df, name=pair)
        cerebro.adddata(data)
        print(f"[LOADED] {pair}: {len(df)} candles")

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Run
    print(f"[RUNNING] Backtest...")
    results = cerebro.run()
    strat = results[0]

    # Results
    final_value = cerebro.broker.getvalue()
    profit = final_value - 200000
    roi = profit / 200000 * 100

    trades = strat.analyzers.trades.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()

    # Safely extract trade stats
    try:
        total_trades = trades.total.total if trades.total.total else 0
    except:
        total_trades = 0

    try:
        won = trades.won.total if trades.won.total else 0
    except:
        won = 0

    try:
        lost = trades.lost.total if trades.lost.total else 0
    except:
        lost = 0

    win_rate = (won / total_trades * 100) if total_trades > 0 else 0

    print(f"\n[RESULTS]")
    print(f"  Final Balance: ${final_value:,.2f}")
    print(f"  Profit: ${profit:+,.2f} ({roi:+.2f}%)")
    print(f"  Max Drawdown: {dd.max.drawdown:.2f}%")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Daily DD Violations: {strat.daily_dd_violations}")

    # Verdict
    passed = (
        profit >= 20000 and
        dd.max.drawdown <= 6.0 and
        strat.daily_dd_violations == 0
    )

    print(f"\n[E8 VERDICT]")
    if passed:
        print(f"  [PASS] Would pass E8 challenge")
    else:
        print(f"  [FAIL] Would fail E8 challenge")
        if profit < 20000:
            print(f"    - Didn't reach $20k target (only ${profit:,.2f})")
        if dd.max.drawdown > 6.0:
            print(f"    - Exceeded 6% DD limit ({dd.max.drawdown:.2f}%)")
        if strat.daily_dd_violations > 0:
            print(f"    - {strat.daily_dd_violations} daily DD violations")

    print(f"{'='*80}\n")

    return {
        'name': name,
        'final_value': final_value,
        'profit': profit,
        'roi': roi,
        'trades': total_trades,
        'win_rate': win_rate,
        'dd': dd.max.drawdown,
        'dd_violations': strat.daily_dd_violations,
        'passed': passed,
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("E8 BACKTEST: SAFE VS AGGRESSIVE")
    print("="*80)
    print("\nData: 6 months OANDA (May-Nov 2025)")
    print("Pairs: EUR/USD, GBP/USD, USD/JPY")
    print("\n" + "="*80)

    # Test both strategies
    safe = run_simple_backtest(SafeStrategy, "Safe (2 lots, 5.5 score, 3 trades/week)")
    agg = run_simple_backtest(AggressiveStrategy, "Aggressive (3 lots, 4.5 score, 6 trades/week)")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<25} {'Safe':<20} {'Aggressive':<20}")
    print("-"*80)
    print(f"{'Final Balance':<25} ${safe['final_value']:>18,.2f} ${agg['final_value']:>18,.2f}")
    print(f"{'Profit':<25} ${safe['profit']:>+18,.2f} ${agg['profit']:>+18,.2f}")
    print(f"{'ROI':<25} {safe['roi']:>18.2f}% {agg['roi']:>18.2f}%")
    print(f"{'Trades':<25} {safe['trades']:>18} {agg['trades']:>18}")
    print(f"{'Win Rate':<25} {safe['win_rate']:>17.1f}% {agg['win_rate']:>17.1f}%")
    print(f"{'Max Drawdown':<25} {safe['dd']:>17.2f}% {agg['dd']:>17.2f}%")
    print(f"{'DD Violations':<25} {safe['dd_violations']:>18} {agg['dd_violations']:>18}")
    print(f"{'E8 Verdict':<25} {'[PASS]' if safe['passed'] else '[FAIL]':>18} {'[PASS]' if agg['passed'] else '[FAIL]':>18}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if safe['passed'] and not agg['passed']:
        print("\n[WINNER] Safe-Optimized")
        print("  - Passed E8 challenge")
        print("  - Zero daily DD violations")
        print(f"  - ROI: {safe['roi']:.2f}%")
        print("\n  Next: Build Safe-Optimized ATLAS bot")

    elif agg['passed'] and not safe['passed']:
        print("\n[WINNER] Aggressive-Optimized")
        print("  - Passed E8 challenge (surprising!)")
        print(f"  - ROI: {agg['roi']:.2f}%")
        print("\n  Next: Build Aggressive-Optimized ATLAS bot")

    elif safe['passed'] and agg['passed']:
        winner = "Aggressive" if agg['roi'] > safe['roi'] else "Safe"
        print(f"\n[BOTH PASSED] Build {winner} (higher ROI)")

    else:
        print("\n[BOTH FAILED]")
        print("  Need ULTRA-conservative:")
        print("  - 1.5 lots max")
        print("  - 6.0 score threshold")
        print("  - 1-2 trades/week")

    print("\n" + "="*80 + "\n")
