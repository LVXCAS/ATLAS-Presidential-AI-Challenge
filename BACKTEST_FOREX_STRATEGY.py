"""
COMPREHENSIVE BACKTEST - WORKING_FOREX_OANDA.py Strategy
Tests the exact signal logic on 6 months of historical data

Answers:
- Does this strategy actually work?
- What's the win rate?
- What's the profit factor?
- Max drawdown?
- Best/worst currency pairs?
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

# Check for TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
    print("[OK] TA-Lib imported successfully")
except ImportError:
    TALIB_AVAILABLE = False
    print("[ERROR] TA-Lib not available - cannot run backtest")
    exit(1)

class ForexBacktest:
    def __init__(self):
        self.oanda_token = os.getenv('OANDA_API_KEY')
        self.client = API(access_token=self.oanda_token, environment='practice')

        # Bot parameters (EXACT same as WORKING_FOREX_OANDA.py)
        self.min_score = 2.5
        self.risk_per_trade = 0.01  # 1% stop loss
        self.profit_target = 0.02   # 2% take profit
        self.leverage_multiplier = 5

        # Currency pairs to test
        self.pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'GBP_JPY']

        # Results
        self.trades = []
        self.equity_curve = []

    def get_historical_data(self, pair, days_back=180):
        """
        Get historical 1H candles from OANDA
        Default: 6 months (180 days * 24 hours = 4320 candles max)
        """
        print(f"\n[DOWNLOADING] {pair} - Last {days_back} days of hourly data...")

        try:
            # OANDA limits to 5000 candles per request
            candles_needed = min(days_back * 24, 5000)

            params = {
                'count': candles_needed,
                'granularity': 'H1'  # 1-hour candles
            }

            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = self.client.request(r)

            candles = response['candles']

            if len(candles) < 100:
                print(f"  [WARN] Only got {len(candles)} candles - not enough data")
                return None

            print(f"  [OK] Downloaded {len(candles)} candles")

            # Extract OHLC data
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
            print(f"  [ERROR] Failed to download {pair}: {e}")
            return None

    def calculate_signals(self, data, index):
        """
        Calculate LONG/SHORT signals at a specific candle index
        Uses EXACT same logic as WORKING_FOREX_OANDA.py
        """
        # Need at least 50 candles for indicators
        if index < 50:
            return None, None

        # Get data up to current index (simulate real-time)
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

        # Calculate indicators
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

        # Return signals if they meet threshold
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
        Simulate a trade with 1% stop-loss and 2% take-profit
        Returns trade result
        """
        # Calculate stop and target
        if direction == 'LONG':
            stop_loss = entry_price * (1 - self.risk_per_trade)
            take_profit = entry_price * (1 + self.profit_target)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.risk_per_trade)
            take_profit = entry_price * (1 - self.profit_target)

        # Walk forward through candles until stop or target hit
        for i in range(entry_index + 1, len(data['closes'])):
            high = data['highs'][i]
            low = data['lows'][i]
            close = data['closes'][i]

            # Check if stop or target hit
            if direction == 'LONG':
                if low <= stop_loss:
                    # Stop loss hit
                    return {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'exit_reason': 'STOP_LOSS',
                        'pnl_pct': -self.risk_per_trade,
                        'bars_held': i - entry_index
                    }
                elif high >= take_profit:
                    # Take profit hit
                    return {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'exit_reason': 'TAKE_PROFIT',
                        'pnl_pct': self.profit_target,
                        'bars_held': i - entry_index
                    }

            else:  # SHORT
                if high >= stop_loss:
                    # Stop loss hit
                    return {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'exit_reason': 'STOP_LOSS',
                        'pnl_pct': -self.risk_per_trade,
                        'bars_held': i - entry_index
                    }
                elif low <= take_profit:
                    # Take profit hit
                    return {
                        'pair': pair,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'exit_reason': 'TAKE_PROFIT',
                        'pnl_pct': self.profit_target,
                        'bars_held': i - entry_index
                    }

        # If we reach end of data, close at current price
        exit_price = data['closes'][-1]

        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - exit_price) / entry_price

        return {
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': 'END_OF_DATA',
            'pnl_pct': pnl_pct,
            'bars_held': len(data['closes']) - entry_index
        }

    def run_backtest(self, pair):
        """
        Run backtest on a single currency pair
        """
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {pair}")
        print(f"{'='*70}")

        # Download historical data
        data = self.get_historical_data(pair, days_back=180)

        if not data:
            print(f"[SKIP] {pair} - No data available")
            return

        print(f"[SCANNING] {len(data['closes'])} candles for signals...")

        signals_found = 0
        trades_executed = 0

        # Walk through each candle (starting at 50 to have enough indicator data)
        i = 50
        while i < len(data['closes']) - 24:  # Leave 24 hours for trade to develop

            # Check for signals
            long_signal, short_signal = self.calculate_signals(data, i)

            # If we get a signal, simulate the trade
            if long_signal:
                signals_found += 1
                trade = self.simulate_trade(pair, 'LONG', long_signal['price'], data, i)
                self.trades.append(trade)
                trades_executed += 1

                # Skip ahead to avoid overlapping trades
                i += trade['bars_held']
                continue

            if short_signal:
                signals_found += 1
                trade = self.simulate_trade(pair, 'SHORT', short_signal['price'], data, i)
                self.trades.append(trade)
                trades_executed += 1

                # Skip ahead to avoid overlapping trades
                i += trade['bars_held']
                continue

            i += 1

        print(f"\n[RESULTS] {pair}:")
        print(f"  Signals Found: {signals_found}")
        print(f"  Trades Executed: {trades_executed}")

    def calculate_statistics(self):
        """
        Calculate overall backtest statistics
        """
        if not self.trades:
            print("\n[ERROR] No trades executed - cannot calculate statistics")
            return

        print(f"\n{'='*70}")
        print("BACKTEST RESULTS - OVERALL STATISTICS")
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

        # Max drawdown
        equity = 100000  # Starting capital
        peak = equity
        max_dd = 0

        for trade in self.trades:
            equity += equity * trade['pnl_pct']
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        final_equity = equity
        total_return = ((final_equity - 100000) / 100000) * 100

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
        print(f"Final Equity: ${final_equity:,.2f}")
        print()

        # Per-pair breakdown
        print(f"\n{'='*70}")
        print("PER-PAIR BREAKDOWN")
        print(f"{'='*70}\n")

        for pair in self.pairs:
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
        print("RECOMMENDATION")
        print(f"{'='*70}\n")

        if win_rate >= 55 and profit_factor >= 1.5:
            print("✅ CONTINUE - Strategy shows positive edge")
            print(f"   Win rate {win_rate:.1f}% and profit factor {profit_factor:.2f} are solid")
            print(f"   Current open trades have statistical probability of success")
        elif win_rate >= 45 and profit_factor >= 1.2:
            print("⚠️  TWEAK - Strategy has potential but needs optimization")
            print(f"   Win rate {win_rate:.1f}% is marginal, consider:")
            print("   - Raising min_score threshold (filter for higher quality)")
            print("   - Adding time-based filters (avoid choppy hours)")
            print("   - Testing different pairs (some may perform better)")
        else:
            print("❌ ABANDON - Strategy does not show reliable edge")
            print(f"   Win rate {win_rate:.1f}% and profit factor {profit_factor:.2f} are too low")
            print(f"   Current open trades are -$1,090 and backtest confirms weakness")
            print("   Recommendation: Close current positions and redesign strategy")

        print()

        # Save results
        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_dd * 100,
            'trades': self.trades
        }

        with open('backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[SAVED] Full results saved to backtest_results.json")
        print()

if __name__ == '__main__':
    print("="*70)
    print("FOREX STRATEGY BACKTEST - 6 MONTH HISTORICAL VALIDATION")
    print("="*70)
    print()
    print("Testing: WORKING_FOREX_OANDA.py signal logic")
    print("Period: Last 180 days")
    print("Timeframe: 1H candles")
    print("Pairs: EUR_USD, GBP_USD, USD_JPY, GBP_JPY")
    print()
    print("This will take 2-3 minutes to download data and simulate trades...")
    print()

    backtest = ForexBacktest()

    # Run backtest on each pair
    for pair in backtest.pairs:
        backtest.run_backtest(pair)

    # Calculate overall statistics
    backtest.calculate_statistics()

    print("="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
