"""
E8 MASSIVE.COM BACKTEST - 4 NEW FOREX PAIRS
Uses Massive.com API to fetch historical OHLC data and Backtrader for backtesting
Tests: AUD/USD, USD/CAD, NZD/USD, EUR/GBP with 54 parameter configurations
"""

import requests
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta
import time
import json

# MASSIVE.COM API KEY
MASSIVE_API_KEY = "BilU0h9fY61tfvunw8K4VG2H6PvxjGnV"
MASSIVE_BASE_URL = "https://api.massive.com"

# 4 NEW PAIRS TO TEST
FOREX_PAIRS = {
    'AUD/USD': 'C:AUDUSD',
    'USD/CAD': 'C:USDCAD',
    'NZD/USD': 'C:NZDUSD',
    'EUR/GBP': 'C:EURGBP'
}

# E8 CONSTRAINTS
MAX_DRAWDOWN = 0.06  # 6% max drawdown limit
STARTING_BALANCE = 200000  # $200K E8 account


def fetch_massive_data(ticker, days=90):
    """Fetch hourly OHLC data from Massive.com API"""
    print(f"\n[FETCH] Getting {days} days of hourly data for {ticker}...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Format dates as YYYY-MM-DD
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    # Massive.com endpoint: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/hour/{from_date}/{to_date}"

    params = {
        'apiKey': MASSIVE_API_KEY,
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'results' not in data or not data['results']:
            print(f"[ERROR] No results returned for {ticker}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data['results'])

        # Rename columns to Backtrader format
        df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }, inplace=True)

        # Convert timestamp from milliseconds to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)

        # Select only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        print(f"[SUCCESS] Fetched {len(df)} hourly bars for {ticker}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch {ticker}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Processing {ticker} data: {e}")
        return None


class E8Strategy(bt.Strategy):
    """E8 Forex Strategy with configurable parameters"""

    params = (
        ('min_score', 4.0),
        ('risk_per_trade', 0.02),
        ('profit_target_pct', 0.02),
        ('stop_loss_pct', 0.01),
    )

    def __init__(self):
        # Technical Indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data)
        self.ema10 = bt.indicators.EMA(self.data.close, period=10)
        self.ema21 = bt.indicators.EMA(self.data.close, period=21)
        self.ema200 = bt.indicators.EMA(self.data.close, period=200)
        self.atr = bt.indicators.ATR(self.data, period=14)

        self.order = None
        self.entry_price = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
            elif order.issell():
                self.entry_price = None

        self.order = None

    def next(self):
        # Skip if order pending
        if self.order:
            return

        # Skip if insufficient data
        if len(self.data) < 200:
            return

        current_price = self.data.close[0]

        # If no position, look for entry
        if not self.position:
            score = self.calculate_score()

            if score >= self.params.min_score:
                # Calculate position size
                risk_amount = self.broker.getvalue() * self.params.risk_per_trade
                stop_distance = current_price * self.params.stop_loss_pct
                position_size = int(risk_amount / stop_distance) if stop_distance > 0 else 0

                if position_size > 0:
                    self.order = self.buy(size=position_size)

        # If in position, check exit conditions
        else:
            if self.entry_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price

                # Take profit
                if pnl_pct >= self.params.profit_target_pct:
                    self.order = self.close()

                # Stop loss
                elif pnl_pct <= -self.params.stop_loss_pct:
                    self.order = self.close()

    def calculate_score(self):
        """Calculate entry signal score (0-10)"""
        score = 0

        # RSI (0-2 points)
        if 30 <= self.rsi[0] <= 70:
            score += 2
        elif 40 <= self.rsi[0] <= 60:
            score += 1

        # MACD (0-2 points)
        if self.macd.macd[0] > self.macd.signal[0]:
            score += 2
        elif self.macd.macd[0] > 0:
            score += 1

        # ADX Trend Strength (0-2 points)
        if self.adx[0] > 25:
            score += 2
        elif self.adx[0] > 20:
            score += 1

        # EMA Alignment (0-2 points)
        if self.ema10[0] > self.ema21[0] > self.ema200[0]:
            score += 2
        elif self.ema10[0] > self.ema21[0]:
            score += 1

        # Price above EMA200 (0-2 points)
        if self.data.close[0] > self.ema200[0]:
            score += 2

        return score


def run_backtest(pair_name, ticker, data, config):
    """Run backtest for one pair with specific config"""

    cerebro = bt.Cerebro()

    # Add data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Add strategy with parameters
    cerebro.addstrategy(
        E8Strategy,
        min_score=config['min_score'],
        risk_per_trade=config['risk_per_trade'],
        profit_target_pct=config['profit_target_pct'],
        stop_loss_pct=config['stop_loss_pct']
    )

    # Set starting cash
    cerebro.broker.setcash(STARTING_BALANCE)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run backtest
    results = cerebro.run()
    strat = results[0]

    # Extract metrics
    final_value = cerebro.broker.getvalue()
    profit = final_value - STARTING_BALANCE
    roi_pct = (profit / STARTING_BALANCE) * 100

    # Drawdown
    dd_analyzer = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analyzer.get('max', {}).get('drawdown', 0) / 100

    # Trades
    trade_analyzer = strat.analyzers.trades.get_analysis()
    total_trades = trade_analyzer.get('total', {}).get('closed', 0)
    won_trades = trade_analyzer.get('won', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    # Returns
    returns_analyzer = strat.analyzers.returns.get_analysis()
    annual_return = returns_analyzer.get('rnorm100', 0)

    return {
        'pair': pair_name,
        'ticker': ticker,
        'config': config,
        'final_value': final_value,
        'profit': profit,
        'roi_pct': roi_pct,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'won_trades': won_trades,
        'win_rate': win_rate,
        'annual_return': annual_return,
        'passes_e8': max_drawdown <= MAX_DRAWDOWN
    }


def main():
    print("=" * 70)
    print("E8 MASSIVE.COM BACKTEST - 4 NEW FOREX PAIRS")
    print("=" * 70)
    print(f"Testing: {', '.join(FOREX_PAIRS.keys())}")
    print(f"E8 Constraint: Max {MAX_DRAWDOWN*100}% drawdown")
    print(f"Starting Balance: ${STARTING_BALANCE:,}")
    print("=" * 70)

    # STEP 1: Fetch historical data for all pairs
    print("\n[STEP 1] FETCHING HISTORICAL DATA FROM MASSIVE.COM")
    print("-" * 70)

    forex_data = {}
    for pair_name, ticker in FOREX_PAIRS.items():
        data = fetch_massive_data(ticker, days=90)
        if data is not None:
            forex_data[pair_name] = data
        time.sleep(1)  # Rate limiting

    if not forex_data:
        print("\n[ERROR] No data fetched. Exiting.")
        return

    print(f"\n[SUCCESS] Fetched data for {len(forex_data)} pairs")

    # STEP 2: Generate parameter configurations
    print("\n[STEP 2] GENERATING PARAMETER CONFIGURATIONS")
    print("-" * 70)

    configs = []

    # Test different min_score thresholds
    for min_score in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        # Test different risk levels
        for risk in [0.015, 0.02, 0.025, 0.03]:
            # Test profit/stop ratios
            for profit_target in [0.02, 0.03]:
                for stop_loss in [0.01, 0.015]:
                    configs.append({
                        'min_score': min_score,
                        'risk_per_trade': risk,
                        'profit_target_pct': profit_target,
                        'stop_loss_pct': stop_loss
                    })

    print(f"Generated {len(configs)} parameter configurations")

    # STEP 3: Run backtests
    print("\n[STEP 3] RUNNING BACKTESTS WITH BACKTRADER")
    print("-" * 70)

    results = []
    total_tests = len(forex_data) * len(configs)
    test_count = 0

    for pair_name, data in forex_data.items():
        for config in configs:
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] Testing {pair_name} - Score:{config['min_score']} Risk:{config['risk_per_trade']*100}%")

            result = run_backtest(pair_name, FOREX_PAIRS[pair_name], data, config)
            results.append(result)

            print(f"  ROI: {result['roi_pct']:.2f}% | DD: {result['max_drawdown']*100:.2f}% | Trades: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}%")

    # STEP 4: Analyze results
    print("\n" + "=" * 70)
    print("[STEP 4] BACKTEST RESULTS ANALYSIS")
    print("=" * 70)

    # Filter for E8-compliant results
    e8_compliant = [r for r in results if r['passes_e8']]

    print(f"\nTotal backtests: {len(results)}")
    print(f"E8-compliant (<=6% DD): {len(e8_compliant)}")

    if not e8_compliant:
        print("\n[WARNING] No configurations passed E8 drawdown limit!")
        return

    # Find best config for each pair
    best_per_pair = {}
    for pair in FOREX_PAIRS.keys():
        pair_results = [r for r in e8_compliant if r['pair'] == pair]
        if pair_results:
            best = max(pair_results, key=lambda x: x['roi_pct'])
            best_per_pair[pair] = best

    print("\n" + "-" * 70)
    print("BEST CONFIGURATION PER PAIR (E8-COMPLIANT)")
    print("-" * 70)

    total_roi = 0
    for pair, result in best_per_pair.items():
        total_roi += result['roi_pct']
        print(f"\n{pair}:")
        print(f"  ROI: {result['roi_pct']:.2f}% (90 days)")
        print(f"  Annual ROI: {result['annual_return']:.2f}%")
        print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
        print(f"  Trades: {result['total_trades']} ({result['win_rate']:.1f}% win rate)")
        print(f"  Config: Score>={result['config']['min_score']}, Risk={result['config']['risk_per_trade']*100}%, TP={result['config']['profit_target_pct']*100}%, SL={result['config']['stop_loss_pct']*100}%")

    # FINAL RECOMMENDATION
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    combined_roi_90d = total_roi / len(best_per_pair)
    combined_annual = combined_roi_90d * 4

    print(f"\nCombined 4-Pair Performance:")
    print(f"  90-day ROI: {combined_roi_90d:.2f}%")
    print(f"  Annual ROI: {combined_annual:.2f}%")

    print(f"\nCurrent 3-Pair System:")
    print(f"  90-day ROI: 25.16%")
    print(f"  Annual ROI: 102%")

    if combined_roi_90d > 30:
        print(f"\n[YES] RECOMMENDATION: ADD THE 4 NEW PAIRS")
        print(f"   Combined 7-pair potential: {(25.16 + combined_roi_90d)/2:.2f}% per 90 days")
        print(f"   Projected annual ROI: {((25.16 + combined_roi_90d)/2)*4:.2f}%")
    else:
        print(f"\n[NO] RECOMMENDATION: KEEP CURRENT 3-PAIR SYSTEM")
        print(f"   New pairs don't significantly improve ROI")
        print(f"   Current system already delivers 102% annual")

    # Save results to JSON
    output_file = 'e8_massive_backtest_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_tests': len(results),
            'e8_compliant': len(e8_compliant),
            'best_per_pair': best_per_pair,
            'combined_roi_90d': combined_roi_90d,
            'combined_annual': combined_annual,
            'all_results': results
        }, f, indent=2, default=str)

    print(f"\n[SAVED] Results saved to {output_file}")


if __name__ == "__main__":
    main()
