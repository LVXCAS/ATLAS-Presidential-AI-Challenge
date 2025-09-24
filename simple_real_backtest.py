"""
SIMPLE REAL BACKTEST
===================
Direct backtest using our real historical data without LEAN CLI
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class SimpleRealBacktest:
    """Simple backtesting engine using our real historical data"""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.portfolio_value_history = []
        self.trade_log = []
        self.data_cache = {}

    def load_historical_data(self, symbol):
        """Load our real historical data"""
        data_file = f"lean_engine/Data/equity/usa/daily/{symbol.lower()}.csv"

        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            return None

        # Load CSV data (no headers in our format)
        df = pd.read_csv(data_file, header=None, names=['date', 'open', 'high', 'low', 'close', 'volume'])

        # Convert LEAN scaled prices back to actual prices
        df['open'] = df['open'] / 10000
        df['high'] = df['high'] / 10000
        df['low'] = df['low'] / 10000
        df['close'] = df['close'] / 10000

        # Convert date format
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df.set_index('date', inplace=True)

        self.data_cache[symbol] = df
        return df

    def calculate_momentum_score(self, symbol, current_date):
        """Calculate 12-1 month momentum score"""
        if symbol not in self.data_cache:
            return None

        df = self.data_cache[symbol]

        # Get data up to current date
        available_data = df[df.index <= current_date]

        if len(available_data) < 252:  # Need at least 12 months
            return None

        # Calculate momentum
        price_12m_ago = available_data['close'].iloc[-252] if len(available_data) >= 252 else available_data['close'].iloc[0]
        price_1m_ago = available_data['close'].iloc[-22] if len(available_data) >= 22 else available_data['close'].iloc[-1]
        current_price = available_data['close'].iloc[-1]

        # 12-1 month momentum
        momentum_return = (price_1m_ago / price_12m_ago) - 1

        # Recent performance penalty
        recent_return = (current_price / price_1m_ago) - 1

        momentum_score = momentum_return - (0.5 * max(0, -recent_return))

        return momentum_score

    def calculate_mean_reversion_score(self, symbol, current_date):
        """Calculate mean reversion score using RSI-like logic"""
        if symbol not in self.data_cache:
            return None

        df = self.data_cache[symbol]
        available_data = df[df.index <= current_date]

        if len(available_data) < 50:
            return None

        # Simple RSI calculation
        recent_data = available_data.tail(14)
        gains = recent_data['close'].diff().clip(lower=0)
        losses = -recent_data['close'].diff().clip(upper=0)

        avg_gain = gains.mean()
        avg_loss = losses.mean()

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Mean reversion signal (buy oversold, sell overbought)
        if rsi < 30:
            return 1  # Buy signal
        elif rsi > 70:
            return -1  # Sell signal
        else:
            return 0  # Hold

    def run_momentum_strategy(self):
        """Run momentum strategy backtest"""
        print("RUNNING MOMENTUM STRATEGY BACKTEST")
        print("=" * 50)

        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

        # Load all data
        for symbol in symbols:
            self.load_historical_data(symbol)

        # Get date range from SPY data
        if 'SPY' not in self.data_cache:
            print("SPY data not available")
            return None

        dates = self.data_cache['SPY'].index
        start_date = dates[252]  # Start after 12 months for momentum calculation

        monthly_dates = [start_date + timedelta(days=30*i) for i in range(48)]  # 4 years monthly
        monthly_dates = [d for d in monthly_dates if d in dates]

        print(f"Backtesting from {monthly_dates[0]} to {monthly_dates[-1]}")
        print(f"Monthly rebalancing dates: {len(monthly_dates)}")

        for i, current_date in enumerate(monthly_dates):
            # Calculate momentum scores
            momentum_scores = {}

            for symbol in symbols:
                score = self.calculate_momentum_score(symbol, current_date)
                if score is not None:
                    momentum_scores[symbol] = score

            # Select top 5 momentum stocks
            top_symbols = sorted(momentum_scores.keys(),
                               key=lambda x: momentum_scores[x],
                               reverse=True)[:5]

            # Rebalance portfolio
            self.rebalance_portfolio(top_symbols, current_date, equal_weight=True)

            if i % 6 == 0:  # Log every 6 months
                portfolio_value = self.calculate_portfolio_value(current_date)
                print(f"{current_date.strftime('%Y-%m')}: Portfolio ${portfolio_value:,.0f}, Top: {top_symbols[:3]}")

        # Calculate final results
        final_value = self.calculate_portfolio_value(monthly_dates[-1])
        return self.calculate_performance_metrics(monthly_dates)

    def run_mean_reversion_strategy(self):
        """Run mean reversion strategy backtest"""
        print("RUNNING MEAN REVERSION STRATEGY BACKTEST")
        print("=" * 50)

        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

        # Load all data
        for symbol in symbols:
            self.load_historical_data(symbol)

        dates = self.data_cache['SPY'].index
        start_date = dates[50]  # Start after 50 days for indicators

        weekly_dates = [start_date + timedelta(days=7*i) for i in range(200)]  # Weekly for 4 years
        weekly_dates = [d for d in weekly_dates if d in dates]

        print(f"Backtesting from {weekly_dates[0]} to {weekly_dates[-1]}")

        for i, current_date in enumerate(weekly_dates):
            # Calculate mean reversion signals
            buy_signals = []
            sell_signals = []

            for symbol in symbols:
                signal = self.calculate_mean_reversion_score(symbol, current_date)
                if signal == 1:
                    buy_signals.append(symbol)
                elif signal == -1 and symbol in self.positions:
                    sell_signals.append(symbol)

            # Execute trades
            self.execute_mean_reversion_trades(buy_signals, sell_signals, current_date)

            if i % 13 == 0:  # Log every quarter
                portfolio_value = self.calculate_portfolio_value(current_date)
                positions = list(self.positions.keys())
                print(f"{current_date.strftime('%Y-%m')}: Portfolio ${portfolio_value:,.0f}, Positions: {len(positions)}")

        return self.calculate_performance_metrics(weekly_dates)

    def rebalance_portfolio(self, target_symbols, current_date, equal_weight=True):
        """Rebalance portfolio to target symbols"""
        # Liquidate positions not in target
        for symbol in list(self.positions.keys()):
            if symbol not in target_symbols:
                self.liquidate_position(symbol, current_date)

        # Calculate target allocation
        if target_symbols:
            target_weight = 0.9 / len(target_symbols)  # 90% invested

            for symbol in target_symbols:
                current_value = self.calculate_portfolio_value(current_date)
                target_value = current_value * target_weight

                if symbol in self.data_cache:
                    current_price = self.data_cache[symbol].loc[current_date, 'close']
                    target_shares = int(target_value / current_price)

                    current_shares = self.positions.get(symbol, 0)

                    if abs(target_shares - current_shares) > 0:
                        self.execute_trade(symbol, target_shares - current_shares, current_price, current_date)

    def execute_mean_reversion_trades(self, buy_signals, sell_signals, current_date):
        """Execute mean reversion trades"""
        # Sell first
        for symbol in sell_signals:
            self.liquidate_position(symbol, current_date)

        # Buy with equal allocation
        if buy_signals:
            available_cash = self.cash
            allocation_per_symbol = available_cash / len(buy_signals) * 0.9  # 90% of cash

            for symbol in buy_signals:
                if symbol in self.data_cache:
                    current_price = self.data_cache[symbol].loc[current_date, 'close']
                    shares_to_buy = int(allocation_per_symbol / current_price)

                    if shares_to_buy > 0:
                        self.execute_trade(symbol, shares_to_buy, current_price, current_date)

    def execute_trade(self, symbol, shares, price, date):
        """Execute a trade"""
        if shares == 0:
            return

        trade_value = shares * price

        if shares > 0:  # Buy
            if self.cash >= trade_value:
                self.cash -= trade_value
                self.positions[symbol] = self.positions.get(symbol, 0) + shares
                self.trade_log.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': trade_value
                })
        else:  # Sell
            current_shares = self.positions.get(symbol, 0)
            shares_to_sell = min(abs(shares), current_shares)

            if shares_to_sell > 0:
                self.cash += shares_to_sell * price
                self.positions[symbol] = current_shares - shares_to_sell

                if self.positions[symbol] == 0:
                    del self.positions[symbol]

                self.trade_log.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'value': shares_to_sell * price
                })

    def liquidate_position(self, symbol, current_date):
        """Liquidate entire position in a symbol"""
        if symbol in self.positions and symbol in self.data_cache:
            shares = self.positions[symbol]
            current_price = self.data_cache[symbol].loc[current_date, 'close']
            self.execute_trade(symbol, -shares, current_price, current_date)

    def calculate_portfolio_value(self, current_date):
        """Calculate current portfolio value"""
        total_value = self.cash

        for symbol, shares in self.positions.items():
            if symbol in self.data_cache:
                current_price = self.data_cache[symbol].loc[current_date, 'close']
                total_value += shares * current_price

        return total_value

    def calculate_performance_metrics(self, dates):
        """Calculate strategy performance metrics"""
        # Calculate portfolio value over time
        portfolio_values = []
        daily_returns = []

        for date in dates:
            portfolio_value = self.calculate_portfolio_value(date)
            portfolio_values.append(portfolio_value)

            if len(portfolio_values) > 1:
                daily_return = (portfolio_value / portfolio_values[-2]) - 1
                daily_returns.append(daily_return)

        # Performance metrics
        total_return = (portfolio_values[-1] / self.initial_capital) - 1
        annual_return = np.mean(daily_returns) * 252 if daily_returns else 0
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative_returns = np.cumprod(np.array(daily_returns) + 1) if daily_returns else [1]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # Win rate
        winning_periods = len([r for r in daily_returns if r > 0])
        win_rate = winning_periods / len(daily_returns) if daily_returns else 0

        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_log),
            'final_value': portfolio_values[-1],
            'initial_capital': self.initial_capital
        }

        return results

def main():
    """Run real backtests using our historical data"""
    print("REAL BACKTESTING WITH ACTUAL HISTORICAL DATA")
    print("=" * 60)

    # Test momentum strategy
    momentum_backtest = SimpleRealBacktest(100000)
    momentum_results = momentum_backtest.run_momentum_strategy()

    print("\\nMOMENTUM STRATEGY RESULTS:")
    print("-" * 30)
    if momentum_results:
        print(f"Total Return: {momentum_results['total_return']:.2%}")
        print(f"Annual Return: {momentum_results['annual_return']:.2%}")
        print(f"Sharpe Ratio: {momentum_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {momentum_results['max_drawdown']:.2%}")
        print(f"Win Rate: {momentum_results['win_rate']:.1%}")
        print(f"Total Trades: {momentum_results['total_trades']}")
        print(f"Final Value: ${momentum_results['final_value']:,.0f}")

    print("\\n" + "=" * 60)

    # Test mean reversion strategy
    mean_reversion_backtest = SimpleRealBacktest(100000)
    mean_reversion_results = mean_reversion_backtest.run_mean_reversion_strategy()

    print("\\nMEAN REVERSION STRATEGY RESULTS:")
    print("-" * 30)
    if mean_reversion_results:
        print(f"Total Return: {mean_reversion_results['total_return']:.2%}")
        print(f"Annual Return: {mean_reversion_results['annual_return']:.2%}")
        print(f"Sharpe Ratio: {mean_reversion_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {mean_reversion_results['max_drawdown']:.2%}")
        print(f"Win Rate: {mean_reversion_results['win_rate']:.1%}")
        print(f"Total Trades: {mean_reversion_results['total_trades']}")
        print(f"Final Value: ${mean_reversion_results['final_value']:,.0f}")

    # Save results
    all_results = {
        'momentum_strategy': momentum_results,
        'mean_reversion_strategy': mean_reversion_results,
        'backtest_date': datetime.now().isoformat(),
        'data_period': '2020-01-01 to 2024-09-18'
    }

    results_file = f"real_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\\nResults saved to: {results_file}")
    print("\\n[SUCCESS] Real backtesting completed with actual historical data!")

if __name__ == "__main__":
    main()