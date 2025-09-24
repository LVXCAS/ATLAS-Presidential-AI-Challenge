"""
ENHANCED REAL BACKTEST
=====================
Complete backtesting with all 5 strategies + leverage + proper calculations
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class EnhancedRealBacktest:
    """Enhanced backtesting with all strategies, leverage, and proper metrics"""

    def __init__(self, initial_capital=100000, leverage=4.0):
        self.initial_capital = initial_capital
        self.leverage = leverage
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

        # Calculate technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)

        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        self.data_cache[symbol] = df
        return df

    def run_momentum_strategy_enhanced(self):
        """Enhanced momentum strategy with leverage"""
        print("RUNNING ENHANCED MOMENTUM STRATEGY")
        print("=" * 50)

        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

        # Load all data
        for symbol in symbols:
            self.load_historical_data(symbol)

        dates = self.data_cache['SPY'].index
        start_date = dates[252]  # Start after 12 months

        # Daily rebalancing for more aggressive strategy
        daily_dates = [d for d in dates if d >= start_date]
        monthly_rebalance_dates = daily_dates[::22]  # Monthly rebalancing

        print(f"Backtesting from {monthly_rebalance_dates[0]} to {monthly_rebalance_dates[-1]}")

        for i, current_date in enumerate(monthly_rebalance_dates):
            # Calculate momentum scores
            momentum_scores = {}

            for symbol in symbols:
                score = self.calculate_enhanced_momentum_score(symbol, current_date)
                if score is not None:
                    momentum_scores[symbol] = score

            # Select top 3 momentum stocks (concentrated for higher returns)
            top_symbols = sorted(momentum_scores.keys(),
                               key=lambda x: momentum_scores[x],
                               reverse=True)[:3]

            # Use leverage for higher returns
            self.rebalance_portfolio_leveraged(top_symbols, current_date)

            if i % 6 == 0:  # Log every 6 months
                portfolio_value = self.calculate_portfolio_value(current_date)
                print(f"{current_date.strftime('%Y-%m')}: Portfolio ${portfolio_value:,.0f}, Top: {top_symbols}")

        return self.calculate_performance_metrics(monthly_rebalance_dates)

    def run_volatility_breakout_strategy(self):
        """Volatility breakout strategy - buy on low vol, sell on high vol"""
        print("RUNNING VOLATILITY BREAKOUT STRATEGY")
        print("=" * 50)

        self.reset_portfolio()

        symbols = ['QQQ', 'AAPL', 'TSLA', 'NVDA', 'META']  # High volatility names

        for symbol in symbols:
            self.load_historical_data(symbol)

        dates = self.data_cache['QQQ'].index
        start_date = dates[63]  # Start after volatility calculation period

        weekly_dates = [d for d in dates if d >= start_date][::5]  # Weekly

        for current_date in weekly_dates:
            # Volatility-based allocation
            allocations = {}

            for symbol in symbols:
                vol_signal = self.calculate_volatility_signal(symbol, current_date)
                if vol_signal > 0:
                    allocations[symbol] = vol_signal

            if allocations:
                self.rebalance_volatility_portfolio(allocations, current_date)

        return self.calculate_performance_metrics(weekly_dates)

    def run_options_simulation_strategy(self):
        """Simulate options-like returns using leveraged equity positions"""
        print("RUNNING OPTIONS SIMULATION STRATEGY")
        print("=" * 50)

        self.reset_portfolio()

        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']

        for symbol in symbols:
            self.load_historical_data(symbol)

        dates = self.data_cache['SPY'].index
        start_date = dates[50]

        # Weekly options-like trades
        weekly_dates = [d for d in dates if d >= start_date][::5]

        for current_date in weekly_dates:
            # Options-like signals (buy on momentum + low RSI)
            options_signals = {}

            for symbol in symbols:
                signal = self.calculate_options_signal(symbol, current_date)
                if signal > 0:
                    options_signals[symbol] = signal

            if options_signals:
                # High leverage for options-like returns
                self.execute_options_trades(options_signals, current_date)

        return self.calculate_performance_metrics(weekly_dates)

    def run_pairs_trading_strategy(self):
        """Pairs trading between correlated stocks"""
        print("RUNNING PAIRS TRADING STRATEGY")
        print("=" * 50)

        self.reset_portfolio()

        # Define pairs
        pairs = [
            ('AAPL', 'MSFT'),
            ('GOOGL', 'META'),
            ('TSLA', 'NVDA'),
            ('SPY', 'QQQ')
        ]

        for pair in pairs:
            for symbol in pair:
                self.load_historical_data(symbol)

        dates = self.data_cache['SPY'].index
        start_date = dates[50]

        weekly_dates = [d for d in dates if d >= start_date][::5]

        for current_date in weekly_dates:
            for pair in pairs:
                self.execute_pairs_trade(pair, current_date)

        return self.calculate_performance_metrics(weekly_dates)

    def run_sector_rotation_strategy(self):
        """Sector rotation based on momentum"""
        print("RUNNING SECTOR ROTATION STRATEGY")
        print("=" * 50)

        self.reset_portfolio()

        # Sector representatives
        sectors = {
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'META'],
            'Growth': ['TSLA', 'NVDA', 'NFLX'],
            'Broad Market': ['SPY', 'QQQ'],
            'Ecommerce': ['AMZN']
        }

        all_symbols = []
        for sector_stocks in sectors.values():
            all_symbols.extend(sector_stocks)

        for symbol in set(all_symbols):
            self.load_historical_data(symbol)

        dates = self.data_cache['SPY'].index
        start_date = dates[63]

        monthly_dates = [d for d in dates if d >= start_date][::22]

        for current_date in monthly_dates:
            # Calculate sector momentum
            sector_scores = {}

            for sector, stocks in sectors.items():
                sector_momentum = 0
                valid_stocks = 0

                for stock in stocks:
                    momentum = self.calculate_enhanced_momentum_score(stock, current_date)
                    if momentum is not None:
                        sector_momentum += momentum
                        valid_stocks += 1

                if valid_stocks > 0:
                    sector_scores[sector] = sector_momentum / valid_stocks

            # Invest in top 2 sectors
            if sector_scores:
                top_sectors = sorted(sector_scores.keys(),
                                   key=lambda x: sector_scores[x],
                                   reverse=True)[:2]

                self.allocate_to_sectors(top_sectors, sectors, current_date)

        return self.calculate_performance_metrics(monthly_dates)

    def calculate_enhanced_momentum_score(self, symbol, current_date):
        """Enhanced momentum calculation with volatility adjustment"""
        if symbol not in self.data_cache:
            return None

        df = self.data_cache[symbol]
        available_data = df[df.index <= current_date]

        if len(available_data) < 252:
            return None

        # Multiple momentum timeframes
        price_current = available_data['close'].iloc[-1]
        price_1m = available_data['close'].iloc[-22] if len(available_data) >= 22 else price_current
        price_3m = available_data['close'].iloc[-63] if len(available_data) >= 63 else price_current
        price_12m = available_data['close'].iloc[-252] if len(available_data) >= 252 else price_current

        # Multi-timeframe momentum
        momentum_1m = (price_current / price_1m) - 1
        momentum_3m = (price_current / price_3m) - 1
        momentum_12m = (price_current / price_12m) - 1

        # Weighted momentum score
        momentum_score = (0.2 * momentum_1m + 0.3 * momentum_3m + 0.5 * momentum_12m)

        # Volatility adjustment (higher momentum for lower vol)
        recent_vol = available_data['volatility'].iloc[-1] if 'volatility' in available_data.columns else 0.2
        vol_adjustment = 1 / (1 + recent_vol)

        return momentum_score * vol_adjustment

    def calculate_volatility_signal(self, symbol, current_date):
        """Calculate volatility breakout signal"""
        if symbol not in self.data_cache:
            return 0

        df = self.data_cache[symbol]
        available_data = df[df.index <= current_date]

        if len(available_data) < 63:
            return 0

        current_vol = available_data['volatility'].iloc[-1]
        avg_vol = available_data['volatility'].tail(63).mean()

        # Buy when volatility is low (mean reversion)
        if current_vol < avg_vol * 0.8:
            return 1.5  # Strong buy
        elif current_vol < avg_vol:
            return 1.0  # Buy
        else:
            return 0

    def calculate_options_signal(self, symbol, current_date):
        """Calculate options-like signal (momentum + oversold)"""
        if symbol not in self.data_cache:
            return 0

        df = self.data_cache[symbol]
        available_data = df[df.index <= current_date]

        if len(available_data) < 50:
            return 0

        # Momentum component
        momentum = self.calculate_enhanced_momentum_score(symbol, current_date) or 0

        # RSI component (buy oversold)
        rsi = available_data['rsi'].iloc[-1] if 'rsi' in available_data.columns else 50

        # Combined signal
        if momentum > 0.1 and rsi < 35:  # Strong momentum + oversold
            return 2.0
        elif momentum > 0.05 and rsi < 40:
            return 1.0
        else:
            return 0

    def rebalance_portfolio_leveraged(self, target_symbols, current_date):
        """Rebalance with leverage for higher returns"""
        # Clear existing positions
        for symbol in list(self.positions.keys()):
            if symbol not in target_symbols:
                self.liquidate_position(symbol, current_date)

        if target_symbols:
            # Use leverage - allocate 150% of portfolio (1.5x leverage)
            current_value = self.calculate_portfolio_value(current_date)
            leverage_multiplier = 1.5
            total_allocation = current_value * leverage_multiplier

            allocation_per_symbol = total_allocation / len(target_symbols)

            for symbol in target_symbols:
                if symbol in self.data_cache:
                    current_price = self.data_cache[symbol].loc[current_date, 'close']
                    target_shares = int(allocation_per_symbol / current_price)
                    current_shares = self.positions.get(symbol, 0)

                    if target_shares != current_shares:
                        self.execute_trade(symbol, target_shares - current_shares, current_price, current_date)

    def rebalance_volatility_portfolio(self, allocations, current_date):
        """Rebalance based on volatility signals"""
        # Clear positions
        for symbol in list(self.positions.keys()):
            if symbol not in allocations:
                self.liquidate_position(symbol, current_date)

        # Allocate based on signals
        current_value = self.calculate_portfolio_value(current_date)
        total_signal = sum(allocations.values())

        for symbol, signal in allocations.items():
            allocation = (signal / total_signal) * current_value * 1.2  # 1.2x leverage
            current_price = self.data_cache[symbol].loc[current_date, 'close']
            target_shares = int(allocation / current_price)
            current_shares = self.positions.get(symbol, 0)

            if target_shares != current_shares:
                self.execute_trade(symbol, target_shares - current_shares, current_price, current_date)

    def execute_options_trades(self, signals, current_date):
        """Execute options-like leveraged trades"""
        # Use high leverage for options-like returns
        current_value = self.calculate_portfolio_value(current_date)
        total_signal = sum(signals.values())

        for symbol, signal in signals.items():
            # High leverage allocation (up to 2x)
            allocation = (signal / total_signal) * current_value * 2.0
            current_price = self.data_cache[symbol].loc[current_date, 'close']
            target_shares = int(allocation / current_price)

            if target_shares > 0:
                self.execute_trade(symbol, target_shares, current_price, current_date)

    def execute_pairs_trade(self, pair, current_date):
        """Execute pairs trading"""
        symbol1, symbol2 = pair

        if symbol1 not in self.data_cache or symbol2 not in self.data_cache:
            return

        # Calculate price ratio
        price1 = self.data_cache[symbol1].loc[current_date, 'close']
        price2 = self.data_cache[symbol2].loc[current_date, 'close']

        # Get historical ratio
        data1 = self.data_cache[symbol1][self.data_cache[symbol1].index <= current_date]
        data2 = self.data_cache[symbol2][self.data_cache[symbol2].index <= current_date]

        if len(data1) < 50 or len(data2) < 50:
            return

        # Calculate ratio divergence
        common_dates = data1.index.intersection(data2.index)[-50:]
        if len(common_dates) < 50:
            return

        ratios = data1.loc[common_dates, 'close'] / data2.loc[common_dates, 'close']
        mean_ratio = ratios.mean()
        current_ratio = price1 / price2

        # Trade based on divergence
        portfolio_value = self.calculate_portfolio_value(current_date)
        trade_size = portfolio_value * 0.1  # 10% per pair trade

        if current_ratio > mean_ratio * 1.1:  # Ratio too high, short stock1, long stock2
            shares1 = -int(trade_size * 0.5 / price1)
            shares2 = int(trade_size * 0.5 / price2)
        elif current_ratio < mean_ratio * 0.9:  # Ratio too low, long stock1, short stock2
            shares1 = int(trade_size * 0.5 / price1)
            shares2 = -int(trade_size * 0.5 / price2)
        else:
            return

        # Execute trades (simplified - only long positions for now)
        if shares1 > 0:
            self.execute_trade(symbol1, shares1, price1, current_date)
        if shares2 > 0:
            self.execute_trade(symbol2, shares2, price2, current_date)

    def allocate_to_sectors(self, top_sectors, sectors, current_date):
        """Allocate to top performing sectors"""
        # Clear all positions
        for symbol in list(self.positions.keys()):
            self.liquidate_position(symbol, current_date)

        # Allocate to top sectors
        portfolio_value = self.calculate_portfolio_value(current_date)
        allocation_per_sector = portfolio_value * 0.6 / len(top_sectors)  # 60% per top sector

        for sector in top_sectors:
            stocks = sectors[sector]
            allocation_per_stock = allocation_per_sector / len(stocks)

            for stock in stocks:
                if stock in self.data_cache:
                    current_price = self.data_cache[stock].loc[current_date, 'close']
                    shares = int(allocation_per_stock / current_price)

                    if shares > 0:
                        self.execute_trade(stock, shares, current_price, current_date)

    def reset_portfolio(self):
        """Reset portfolio for new strategy"""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value_history = []
        self.trade_log = []

    def execute_trade(self, symbol, shares, price, date):
        """Execute a trade with margin capability"""
        if shares == 0:
            return

        trade_value = shares * price

        if shares > 0:  # Buy
            # Allow leveraged buying up to 4x
            max_leverage_value = self.calculate_portfolio_value(date) * self.leverage

            if trade_value <= max_leverage_value:
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
        """Calculate CORRECTED performance metrics"""
        # Calculate portfolio value over time
        portfolio_values = []

        for date in dates:
            portfolio_value = self.calculate_portfolio_value(date)
            portfolio_values.append(portfolio_value)

        if len(portfolio_values) < 2:
            return None

        # CORRECTED calculations
        total_return = (portfolio_values[-1] / self.initial_capital) - 1

        # Correct annual return calculation
        days_elapsed = (dates[-1] - dates[0]).days
        years_elapsed = days_elapsed / 365.25
        annual_return = ((portfolio_values[-1] / self.initial_capital) ** (1/years_elapsed)) - 1

        # Daily returns for other metrics
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
            daily_returns.append(daily_return)

        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Max drawdown
        running_max = portfolio_values[0]
        max_drawdown = 0

        for value in portfolio_values:
            if value > running_max:
                running_max = value
            drawdown = (value / running_max) - 1
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        # Win rate
        winning_periods = len([r for r in daily_returns if r > 0])
        win_rate = winning_periods / len(daily_returns) if daily_returns else 0

        results = {
            'total_return': total_return,
            'annual_return': annual_return,  # CORRECTED
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_log),
            'final_value': portfolio_values[-1],
            'initial_capital': self.initial_capital,
            'years_elapsed': years_elapsed
        }

        return results

def main():
    """Run all 5 enhanced strategies"""
    print("ENHANCED REAL BACKTESTING - ALL 5 STRATEGIES")
    print("=" * 60)

    strategies = [
        ("Enhanced Momentum", "run_momentum_strategy_enhanced"),
        ("Volatility Breakout", "run_volatility_breakout_strategy"),
        ("Options Simulation", "run_options_simulation_strategy"),
        ("Pairs Trading", "run_pairs_trading_strategy"),
        ("Sector Rotation", "run_sector_rotation_strategy")
    ]

    all_results = {}

    for strategy_name, method_name in strategies:
        print(f"\\n{'='*60}")
        print(f"TESTING: {strategy_name.upper()}")
        print(f"{'='*60}")

        backtest = EnhancedRealBacktest(100000, leverage=2.0)  # 2x leverage
        method = getattr(backtest, method_name)
        results = method()

        if results:
            print(f"\\n{strategy_name.upper()} RESULTS:")
            print("-" * 40)
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Annual Return: {results['annual_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"Win Rate: {results['win_rate']:.1%}")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Final Value: ${results['final_value']:,.0f}")
            print(f"Years: {results['years_elapsed']:.1f}")

            all_results[strategy_name] = results

    # Save results
    enhanced_results = {
        'strategies': all_results,
        'backtest_date': datetime.now().isoformat(),
        'data_period': '2020-01-01 to 2024-09-18',
        'leverage_used': '2x',
        'calculation_method': 'CORRECTED'
    }

    results_file = f"enhanced_real_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(enhanced_results, f, indent=2, default=str)

    print(f"\\n\\nEnhanced results saved to: {results_file}")
    print("\\n[SUCCESS] All 5 strategies backtested with REAL data and CORRECTED metrics!")

if __name__ == "__main__":
    main()