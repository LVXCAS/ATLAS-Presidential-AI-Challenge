"""
REAL MEAN REVERSION STRATEGY - ACTUAL LEAN ALGORITHM
===================================================
Genuine mean reversion strategy with real market logic
"""

from AlgorithmImports import *

class RealMeanReversionStrategy(QCAlgorithm):
    """
    Real mean reversion strategy using actual market data
    Buy oversold stocks, sell overbought stocks based on RSI and price deviation
    """

    def Initialize(self):
        """Initialize the algorithm"""
        # Set timeframe for backtest
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 9, 18)
        self.SetCash(100000)

        # Universe selection
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

        # Add securities
        self.securities_data = {}
        for symbol in self.symbols:
            equity = self.AddEquity(symbol, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Adjusted)

            # Initialize indicators for each symbol
            self.securities_data[symbol] = {
                'rsi': self.RSI(symbol, 14),  # 14-day RSI
                'sma_50': self.SMA(symbol, 50),  # 50-day moving average
                'sma_200': self.SMA(symbol, 200),  # 200-day moving average
                'bollinger': self.BB(symbol, 20, 2),  # Bollinger Bands
                'price_history': []
            }

        # Strategy parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.lookback_period = 20
        self.rebalance_frequency = 5  # Weekly rebalancing

        # Risk management
        self.max_position_size = 0.15  # 15% max per position
        self.stop_loss = -0.12  # 12% stop loss
        self.take_profit = 0.08  # 8% take profit

        # Performance tracking
        self.previous_portfolio_value = self.Portfolio.TotalPortfolioValue
        self.daily_returns = []
        self.trades_count = 0

        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.WeekStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )

        # Daily performance tracking
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.BeforeMarketClose("SPY", 0),
            self.TrackPerformance
        )

        self.Debug("Real Mean Reversion Strategy initialized")

    def Rebalance(self):
        """Execute weekly rebalancing based on mean reversion signals"""
        self.Debug(f"Rebalancing on {self.Time}")

        # Calculate mean reversion scores
        buy_signals = []
        sell_signals = []

        for symbol in self.symbols:
            signal = self.CalculateMeanReversionSignal(symbol)

            if signal == 'BUY':
                buy_signals.append(symbol)
            elif signal == 'SELL':
                sell_signals.append(symbol)

        self.Debug(f"Buy signals: {buy_signals}")
        self.Debug(f"Sell signals: {sell_signals}")

        # Execute sell signals first
        for symbol in sell_signals:
            if self.Portfolio[symbol].Invested and self.Portfolio[symbol].IsLong:
                self.Liquidate(symbol)
                self.trades_count += 1
                self.Debug(f"Sold {symbol} - mean reversion sell signal")

        # Execute buy signals
        if buy_signals:
            position_size = min(self.max_position_size, 0.8 / len(buy_signals))  # Max 80% invested

            for symbol in buy_signals:
                if not self.Portfolio[symbol].Invested:
                    self.SetHoldings(symbol, position_size)
                    self.trades_count += 1
                    self.Debug(f"Bought {symbol} - mean reversion buy signal ({position_size:.1%})")

    def CalculateMeanReversionSignal(self, symbol):
        """Calculate mean reversion signal for a symbol"""
        try:
            security_data = self.securities_data[symbol]
            current_price = self.Securities[symbol].Price

            # Ensure indicators are ready
            if not (security_data['rsi'].IsReady and
                   security_data['sma_50'].IsReady and
                   security_data['bollinger'].IsReady):
                return 'HOLD'

            rsi = security_data['rsi'].Current.Value
            sma_50 = security_data['sma_50'].Current.Value
            sma_200 = security_data['sma_200'].Current.Value
            bb_upper = security_data['bollinger'].UpperBand.Current.Value
            bb_lower = security_data['bollinger'].LowerBand.Current.Value
            bb_middle = security_data['bollinger'].MiddleBand.Current.Value

            # Get recent price action
            history = self.History(symbol, 5, Resolution.Daily)
            if history.empty:
                return 'HOLD'

            recent_low = history['low'].min()
            recent_high = history['high'].max()

            # Buy conditions (oversold)
            buy_conditions = [
                rsi < self.rsi_oversold,  # RSI oversold
                current_price < bb_lower,  # Below lower Bollinger Band
                current_price > sma_200,  # Above long-term trend
                current_price < sma_50 * 0.95  # Significantly below medium-term average
            ]

            # Sell conditions (overbought)
            sell_conditions = [
                rsi > self.rsi_overbought,  # RSI overbought
                current_price > bb_upper,  # Above upper Bollinger Band
                current_price > sma_50 * 1.05  # Significantly above medium-term average
            ]

            # Decision logic
            if sum(buy_conditions) >= 3:  # At least 3 buy conditions
                return 'BUY'
            elif sum(sell_conditions) >= 2:  # At least 2 sell conditions
                return 'SELL'
            else:
                return 'HOLD'

        except Exception as e:
            self.Debug(f"Error calculating signal for {symbol}: {e}")
            return 'HOLD'

    def OnData(self, data):
        """Handle incoming data and risk management"""
        # Risk management - stop losses and take profits
        for holding in self.Portfolio.Values:
            if holding.Invested and holding.IsLong:
                unrealized_pnl = holding.UnrealizedProfitPercent

                # Stop loss
                if unrealized_pnl < self.stop_loss:
                    self.Liquidate(holding.Symbol)
                    self.trades_count += 1
                    self.Debug(f"Stop loss triggered for {holding.Symbol}: {unrealized_pnl:.1%}")

                # Take profit
                elif unrealized_pnl > self.take_profit:
                    self.Liquidate(holding.Symbol)
                    self.trades_count += 1
                    self.Debug(f"Take profit triggered for {holding.Symbol}: {unrealized_pnl:.1%}")

    def TrackPerformance(self):
        """Track daily performance metrics"""
        current_value = self.Portfolio.TotalPortfolioValue
        daily_return = (current_value / self.previous_portfolio_value) - 1
        self.daily_returns.append(daily_return)
        self.previous_portfolio_value = current_value

        # Log performance weekly
        if len(self.daily_returns) % 5 == 0 and len(self.daily_returns) > 0:
            recent_returns = self.daily_returns[-5:]
            weekly_return = sum(recent_returns)
            self.Debug(f"Weekly return: {weekly_return:.2%}")

    def OnEndOfAlgorithm(self):
        """Calculate final performance metrics"""
        if len(self.daily_returns) > 0:
            # Calculate metrics
            total_return = (self.Portfolio.TotalPortfolioValue / 100000) - 1
            daily_returns_array = np.array(self.daily_returns)

            annual_return = np.mean(daily_returns_array) * 252
            volatility = np.std(daily_returns_array) * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

            # Calculate max drawdown
            cumulative_returns = np.cumprod(1 + daily_returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns / running_max) - 1
            max_drawdown = np.min(drawdowns)

            # Win rate
            winning_days = len([r for r in daily_returns_array if r > 0])
            win_rate = winning_days / len(daily_returns_array)

            # Log final results
            self.Debug("=== REAL MEAN REVERSION STRATEGY RESULTS ===")
            self.Debug(f"Total Return: {total_return:.2%}")
            self.Debug(f"Annual Return: {annual_return:.2%}")
            self.Debug(f"Volatility: {volatility:.2%}")
            self.Debug(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            self.Debug(f"Max Drawdown: {max_drawdown:.2%}")
            self.Debug(f"Win Rate: {win_rate:.1%}")
            self.Debug(f"Total Trades: {self.trades_count}")
            self.Debug("=============================================")