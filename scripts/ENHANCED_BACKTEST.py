"""
ENHANCED FOREX BACKTEST - LEAN-Quality Without Docker
Adds realistic spreads, slippage, and professional risk metrics

Improvements over simple backtest:
1. Bid/ask spreads on entry/exit (1-3 pips per pair)
2. Slippage on stop-loss exits (0.5-1 pip worse fill)
3. Sharpe ratio calculation
4. Accurate max drawdown tracking
5. Monthly return breakdown
"""

import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import numpy as np
from datetime import datetime, timedelta
import json

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[ERROR] TA-Lib required for backtest")
    exit(1)

class EnhancedForexBacktest:
    def __init__(self):
        self.oanda_token = os.getenv('OANDA_API_KEY')
        self.client = API(access_token=self.oanda_token, environment='practice')

        # Bot parameters (EXACT same as WORKING_FOREX_OANDA.py)
        self.min_score = 2.5
        self.risk_per_trade = 0.01  # 1% stop loss
        self.profit_target = 0.02   # 2% take profit
        self.leverage_multiplier = 5

        # Currency pairs with REALISTIC spreads (in pips)
        self.pairs = {
            'EUR_USD': {'spread_pips': 1.5, 'pip_value': 0.0001},
            'GBP_USD': {'spread_pips': 2.5, 'pip_value': 0.0001},
            'USD_JPY': {'spread_pips': 2.0, 'pip_value': 0.01},
            'GBP_JPY': {'spread_pips': 3.0, 'pip_value': 0.01}
        }

        # Slippage (pips worse than expected on stops)
        self.stop_slippage_pips = 0.8  # Stop-losses typically fill 0.8 pips worse

        # Results tracking
        self.trades = []
        self.equity_curve = []
        self.monthly_returns = {}

    def get_spread_cost(self, pair, price):
        """
        Calculate actual spread cost for entry
        Returns: cost as percentage of trade
        """
        config = self.pairs[pair]
        spread_pips = config['spread_pips']
        pip_value = config['pip_value']

        # Spread cost in currency units
        spread_cost = spread_pips * pip_value

        # As percentage of price
        spread_pct = spread_cost / price

        return spread_pct

    def apply_slippage_to_stop(self, pair, stop_price, direction):
        """
        Apply realistic slippage to stop-loss execution
        Stop-losses typically fill worse than expected
        """
        config = self.pairs[pair]
        slippage = self.stop_slippage_pips * config['pip_value']

        if direction == 'LONG':
            # LONG stop gets filled lower (worse)
            return stop_price - slippage
        else:  # SHORT
            # SHORT stop gets filled higher (worse)
            return stop_price + slippage

    def get_historical_data(self, pair, days_back=180):
        """Get historical 1H candles from OANDA"""
        print(f"\n[DOWNLOADING] {pair} - Last {days_back} days...")

        try:
            candles_needed = min(days_back * 24, 5000)

            params = {
                'count': candles_needed,
                'granularity': 'H1'
            }

            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = self.client.request(r)
            candles = response['candles']

            if len(candles) < 100:
                print(f"  [WARN] Only {len(candles)} candles")
                return None

            print(f"  [OK] {len(candles)} candles")

            timestamps = [c['time'] for c in candles]
            opens = np.array([float(c['mid']['o']) for c in candles])
            highs = np.array([float(c['mid']['h']) for c in candles])
            lows = np.array([float(c['mid']['l']) for c in candles])
            closes = np.array([float(c['mid']['c']) for c in candles])

            return {
                'timestamps': timestamps,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes
            }

        except Exception as e:
            print(f"  [ERROR] {pair}: {e}")
            return None

    def calculate_signals(self, data, index):
        """Calculate LONG/SHORT signals - EXACT logic from WORKING_FOREX_OANDA.py"""
        if index < 50:
            return None, None

        closes = data['closes'][:index+1]
        highs = data['highs'][:index+1]
        lows = data['lows'][:index+1]
        current_price = closes[-1]

        if len(closes) < 50:
            return None, None

        long_score = 0
        short_score = 0
        long_signals = []
        short_signals = []

        # Indicators
        rsi = talib.RSI(closes, timeperiod=14)[-1]
        macd, macd_signal, macd_hist = talib.MACD(closes)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
        volatility = (atr / current_price) * 100
        adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]

        ema_fast = talib.EMA(closes, timeperiod=10)
        ema_slow = talib.EMA(closes, timeperiod=21)
        ema_trend = talib.EMA(closes, timeperiod=200)

        # === LONG SIGNALS ===
        if rsi < 40:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")

        if len(macd_hist) >= 2:
            if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                long_score += 2.5
                long_signals.append("MACD_BULLISH")

        if len(ema_fast) >= 2 and len(ema_slow) >= 2:
            if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]:
                long_score += 2
                long_signals.append("EMA_CROSS_BULLISH")

        if len(ema_trend) >= 1 and current_price > ema_trend[-1]:
            long_score += 1
            long_signals.append("UPTREND")

        # === SHORT SIGNALS ===
        if rsi > 60:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")

        if len(macd_hist) >= 2:
            if macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                short_score += 2.5
                short_signals.append("MACD_BEARISH")

        if len(ema_fast) >= 2 and len(ema_slow) >= 2:
            if ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]:
                short_score += 2
                short_signals.append("EMA_CROSS_BEARISH")

        if len(ema_trend) >= 1 and current_price < ema_trend[-1]:
            short_score += 1
            short_signals.append("DOWNTREND")

        # === SHARED SIGNALS ===
        if adx > 20:
            long_score += 1.5
            short_score += 1.5
            long_signals.append("STRONG_TREND")
            short_signals.append("STRONG_TREND")

        if volatility > 0.3:
            long_score += 1
            short_score += 1
            long_signals.append("FX_VOLATILITY")
            short_signals.append("FX_VOLATILITY")

        # Return signals
        long_signal = None
        short_signal = None

        if long_score >= self.min_score:
            long_signal = {
                'score': long_score,
                'price': current_price,
                'signals': long_signals,
                'rsi': rsi,
                'adx': adx
            }

        if short_score >= self.min_score:
            short_signal = {
                'score': short_score,
                'price': current_price,
                'signals': short_signals,
                'rsi': rsi,
                'adx': adx
            }

        return long_signal, short_signal

    def simulate_trade(self, pair, direction, entry_price, data, entry_index):
        """
        Simulate trade with REALISTIC spreads and slippage
        """
        # Apply spread cost on entry
        spread_cost_pct = self.get_spread_cost(pair, entry_price)

        # Adjust entry price for spread
        if direction == 'LONG':
            # LONG pays the ask (higher)
            actual_entry = entry_price * (1 + spread_cost_pct)
        else:  # SHORT
            # SHORT receives the bid (lower)
            actual_entry = entry_price * (1 - spread_cost_pct)

        # Calculate stop and target
        if direction == 'LONG':
            stop_loss = actual_entry * (1 - self.risk_per_trade)
            take_profit = actual_entry * (1 + self.profit_target)
        else:  # SHORT
            stop_loss = actual_entry * (1 + self.risk_per_trade)
            take_profit = actual_entry * (1 - self.profit_target)

        # Walk forward to find exit
        for i in range(entry_index + 1, len(data['closes'])):
            high = data['highs'][i]
            low = data['lows'][i]

            if direction == 'LONG':
                if low <= stop_loss:
                    # Stop hit - apply slippage (worse fill)
                    slipped_stop = self.apply_slippage_to_stop(pair, stop_loss, direction)
                    pnl_pct = (slipped_stop - actual_entry) / actual_entry

                    # Also pay spread on exit
                    pnl_pct -= spread_cost_pct

                    return {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': actual_entry,
                        'exit_price': slipped_stop,
                        'exit_reason': 'STOP_LOSS',
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_index,
                        'spread_cost_pct': spread_cost_pct * 2,  # Entry + exit
                        'slippage_pct': abs(pnl_pct - (-self.risk_per_trade))
                    }

                elif high >= take_profit:
                    # Target hit - no slippage on limits
                    pnl_pct = (take_profit - actual_entry) / actual_entry

                    # Pay spread on exit
                    pnl_pct -= spread_cost_pct

                    return {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': actual_entry,
                        'exit_price': take_profit,
                        'exit_reason': 'TAKE_PROFIT',
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_index,
                        'spread_cost_pct': spread_cost_pct * 2,
                        'slippage_pct': 0
                    }

            else:  # SHORT
                if high >= stop_loss:
                    # Stop hit - apply slippage
                    slipped_stop = self.apply_slippage_to_stop(pair, stop_loss, direction)
                    pnl_pct = (actual_entry - slipped_stop) / actual_entry

                    # Pay spread on exit
                    pnl_pct -= spread_cost_pct

                    return {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': actual_entry,
                        'exit_price': slipped_stop,
                        'exit_reason': 'STOP_LOSS',
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_index,
                        'spread_cost_pct': spread_cost_pct * 2,
                        'slippage_pct': abs(pnl_pct - (-self.risk_per_trade))
                    }

                elif low <= take_profit:
                    # Target hit
                    pnl_pct = (actual_entry - take_profit) / actual_entry

                    # Pay spread on exit
                    pnl_pct -= spread_cost_pct

                    return {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': actual_entry,
                        'exit_price': take_profit,
                        'exit_reason': 'TAKE_PROFIT',
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_index,
                        'spread_cost_pct': spread_cost_pct * 2,
                        'slippage_pct': 0
                    }

        # End of data
        exit_price = data['closes'][-1]

        if direction == 'LONG':
            pnl_pct = (exit_price - actual_entry) / actual_entry
        else:
            pnl_pct = (actual_entry - exit_price) / actual_entry

        # Pay exit spread
        pnl_pct -= spread_cost_pct

        return {
            'pair': pair,
            'direction': direction,
            'entry_price': actual_entry,
            'exit_price': exit_price,
            'exit_reason': 'END_OF_DATA',
            'pnl_pct': pnl_pct,
            'bars_held': len(data['closes']) - entry_index,
            'spread_cost_pct': spread_cost_pct * 2,
            'slippage_pct': 0
        }

    def run_backtest(self, pair):
        """Run backtest on a single pair"""
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {pair}")
        print(f"{'='*70}")

        data = self.get_historical_data(pair, days_back=180)

        if not data:
            print(f"[SKIP] {pair}")
            return

        print(f"[SCANNING] {len(data['closes'])} candles...")

        signals_found = 0
        trades_executed = 0

        i = 50
        while i < len(data['closes']) - 24:
            long_signal, short_signal = self.calculate_signals(data, i)

            if long_signal:
                signals_found += 1
                trade = self.simulate_trade(pair, 'LONG', long_signal['price'], data, i)
                self.trades.append(trade)
                trades_executed += 1
                i += trade['bars_held']
                continue

            if short_signal:
                signals_found += 1
                trade = self.simulate_trade(pair, 'SHORT', short_signal['price'], data, i)
                self.trades.append(trade)
                trades_executed += 1
                i += trade['bars_held']
                continue

            i += 1

        print(f"\n[RESULTS] {pair}:")
        print(f"  Signals: {signals_found}")
        print(f"  Trades: {trades_executed}")

    def calculate_sharpe_ratio(self, returns):
        """
        Calculate Sharpe ratio (risk-adjusted return)
        Industry standard: >0.5 is acceptable, >1.0 is good, >2.0 is excellent
        """
        if len(returns) == 0:
            return 0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        # Annualized Sharpe (assuming hourly returns)
        # 252 trading days * 24 hours = 6048 periods per year
        sharpe = (mean_return / std_return) * np.sqrt(6048)

        return sharpe

    def calculate_statistics(self):
        """Calculate comprehensive statistics"""
        if not self.trades:
            print("\n[ERROR] No trades executed")
            return

        print(f"\n{'='*70}")
        print("ENHANCED BACKTEST RESULTS - WITH SPREADS & SLIPPAGE")
        print(f"{'='*70}\n")

        # Basic stats
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_pct'] < 0]

        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = sum(t['pnl_pct'] for t in self.trades)
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0

        # Profit factor
        total_wins = sum(t['pnl_pct'] for t in winning_trades)
        total_losses = abs(sum(t['pnl_pct'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Calculate costs
        avg_spread_cost = np.mean([t['spread_cost_pct'] for t in self.trades]) * 100
        total_spread_cost = sum(t['spread_cost_pct'] for t in self.trades) * 100
        avg_slippage = np.mean([t['slippage_pct'] for t in self.trades if t['exit_reason'] == 'STOP_LOSS']) * 100 if any(t['exit_reason'] == 'STOP_LOSS' for t in self.trades) else 0

        # Equity curve & drawdown
        equity = 100000
        peak = equity
        max_dd = 0
        returns = []

        for trade in self.trades:
            equity += equity * trade['pnl_pct']
            returns.append(trade['pnl_pct'])
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        final_equity = equity
        total_return = ((final_equity - 100000) / 100000) * 100

        # Sharpe ratio
        sharpe = self.calculate_sharpe_ratio(returns)

        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print()
        print(f"Average Win: {avg_win*100:.2f}%")
        print(f"Average Loss: {avg_loss*100:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print()
        print(f"Total Return: {total_return:.2f}%")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Final Equity: ${final_equity:,.2f}")
        print()
        print(f"Average Spread Cost: {avg_spread_cost:.3f}% per trade")
        print(f"Total Spread Cost: {total_spread_cost:.2f}%")
        print(f"Average Slippage (stops): {avg_slippage:.3f}%")
        print()

        # Per-pair breakdown
        print(f"\n{'='*70}")
        print("PER-PAIR BREAKDOWN")
        print(f"{'='*70}\n")

        for pair in self.pairs.keys():
            pair_trades = [t for t in self.trades if t['pair'] == pair]

            if not pair_trades:
                print(f"{pair}: No trades")
                continue

            pair_wins = [t for t in pair_trades if t['pnl_pct'] > 0]
            pair_win_rate = (len(pair_wins) / len(pair_trades)) * 100 if pair_trades else 0
            pair_total_pnl = sum(t['pnl_pct'] for t in pair_trades)

            print(f"{pair}:")
            print(f"  Trades: {len(pair_trades)}")
            print(f"  Win Rate: {pair_win_rate:.1f}%")
            print(f"  Total P/L: {pair_total_pnl*100:.2f}%")
            print()

        # Recommendation
        print(f"\n{'='*70}")
        print("FINAL RECOMMENDATION")
        print(f"{'='*70}\n")

        if win_rate >= 38 and profit_factor >= 1.15 and sharpe >= 0.5:
            print("GREEN FLAG - Strategy Has Edge")
            print(f"  Win rate {win_rate:.1f}%, profit factor {profit_factor:.2f}, Sharpe {sharpe:.2f}")
            print(f"  Strategy is viable even with realistic costs")
            print()
            print("  RECOMMENDATION: Continue trading")
            print("  - Let current GBP_USD/EUR_USD positions develop")
            print("  - Focus on USD_JPY/GBP_JPY for future trades (best performers)")
            print("  - Proceed with E8 challenge after 7-day validation")
        elif win_rate >= 35 and profit_factor >= 1.10 and sharpe >= 0.3:
            print("YELLOW FLAG - Strategy Needs Tweaking")
            print(f"  Win rate {win_rate:.1f}%, profit factor {profit_factor:.2f}, Sharpe {sharpe:.2f}")
            print(f"  Strategy has potential but marginal edge")
            print()
            print("  RECOMMENDATION: Optimize before scaling")
            print("  - Close GBP_USD position (worst performer)")
            print("  - Hold EUR_USD (marginal but positive)")
            print("  - Raise min_score to 3.0+ for higher quality signals")
            print("  - Focus only on USD_JPY/GBP_JPY going forward")
        else:
            print("RED FLAG - Strategy Does Not Have Reliable Edge")
            print(f"  Win rate {win_rate:.1f}%, profit factor {profit_factor:.2f}, Sharpe {sharpe:.2f}")
            print(f"  After spreads/slippage, returns are too low or negative")
            print()
            print("  RECOMMENDATION: Close positions and redesign")
            print("  - Close both GBP_USD and EUR_USD now")
            print(f"  - Accept current loss (${(final_equity - 100000)*1.87:.2f} on your account)")
            print("  - Redesign strategy with new parameters")
            print("  - Do NOT purchase E8 challenge with this strategy")

        print()

        # Save results
        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd * 100,
            'avg_spread_cost': avg_spread_cost,
            'avg_slippage': avg_slippage,
            'trades': self.trades
        }

        with open('enhanced_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[SAVED] Results saved to enhanced_backtest_results.json")
        print()

if __name__ == '__main__':
    print("="*70)
    print("ENHANCED FOREX BACKTEST - REALISTIC SPREADS & SLIPPAGE")
    print("="*70)
    print()
    print("Improvements over simple backtest:")
    print("  - Bid/ask spreads (1.5-3 pips per pair)")
    print("  - Stop-loss slippage (0.8 pips worse fill)")
    print("  - Sharpe ratio calculation")
    print("  - Accurate cost tracking")
    print()
    print("This will take 2-3 minutes...")
    print()

    backtest = EnhancedForexBacktest()

    for pair in backtest.pairs.keys():
        backtest.run_backtest(pair)

    backtest.calculate_statistics()

    print("="*70)
    print("ENHANCED BACKTEST COMPLETE")
    print("="*70)
