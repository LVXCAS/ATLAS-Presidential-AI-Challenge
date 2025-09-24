"""
REAL MOMENTUM STRATEGY - ACTUAL LEAN ALGORITHM
=============================================
Genuine momentum strategy with real market logic
"""

from AlgorithmImports import *

class RealMomentumStrategy(QCAlgorithm):
    """
    Real momentum strategy using actual market data
    Buy stocks with strong 12-1 month momentum, avoid recent losers
    """

    def Initialize(self):
        """Initialize the algorithm"""
        # Set timeframe for backtest
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 9, 18)
        self.SetCash(100000)

        # Universe selection - top tech stocks
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

        # Add securities
        for symbol in self.symbols:
            equity = self.AddEquity(symbol, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Adjusted)

        # Strategy parameters
        self.momentum_period = 252  # 12 months
        self.recent_period = 22     # 1 month to avoid
        self.rebalance_frequency = 22  # Monthly rebalancing

        # Performance tracking
        self.previous_portfolio_value = self.Portfolio.TotalPortfolioValue
        self.daily_returns = []
        self.rebalance_count = 0

        # Risk management
        self.max_position_size = 0.2  # 20% max per position
        self.stop_loss = -0.15  # 15% stop loss

        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )

        # Daily performance tracking
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.BeforeMarketClose("SPY", 0),
            self.TrackPerformance
        )

        self.Debug("Real Momentum Strategy initialized")

    def Rebalance(self):
        """Execute monthly rebalancing based on momentum"""
        self.rebalance_count += 1
        self.Debug(f"Rebalancing #{self.rebalance_count} on {self.Time}")

        # Calculate momentum scores for all symbols
        momentum_scores = {}

        for symbol in self.symbols:
            score = self.CalculateMomentumScore(symbol)
            if score is not None:
                momentum_scores[symbol] = score

        if not momentum_scores:
            self.Debug("No momentum scores calculated")
            return

        # Rank symbols by momentum (highest first)
        ranked_symbols = sorted(momentum_scores.keys(),
                              key=lambda x: momentum_scores[x],
                              reverse=True)

        # Select top 5 momentum stocks
        selected_symbols = ranked_symbols[:5]

        self.Debug(f"Selected symbols: {[s for s in selected_symbols]}")
        self.Debug(f"Momentum scores: {[momentum_scores[s]:.3f for s in selected_symbols]}")

        # Equal weight allocation
        target_weight = 1.0 / len(selected_symbols) if selected_symbols else 0

        # Liquidate positions not in selection
        for symbol in self.symbols:
            if symbol not in selected_symbols and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)
                self.Debug(f"Liquidated {symbol}")

        # Allocate to selected symbols
        for symbol in selected_symbols:
            current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue

            if abs(current_weight - target_weight) > 0.05:  # 5% threshold
                self.SetHoldings(symbol, target_weight)
                self.Debug(f"Set {symbol} to {target_weight:.1%}")

    def CalculateMomentumScore(self, symbol):
        """Calculate 12-1 month momentum score"""
        try:
            # Get historical data
            history = self.History(symbol, self.momentum_period + self.recent_period, Resolution.Daily)

            if history.empty or len(history) < self.momentum_period:
                return None

            # Calculate 12-1 month momentum
            # Price 12 months ago vs price 1 month ago
            price_12m_ago = history.iloc[-(self.momentum_period + self.recent_period)]['close']
            price_1m_ago = history.iloc[-self.recent_period]['close']
            current_price = history.iloc[-1]['close']

            # 12-1 month momentum (classic academic measure)
            momentum_return = (price_1m_ago / price_12m_ago) - 1

            # Recent performance penalty (avoid recent losers)
            recent_return = (current_price / price_1m_ago) - 1

            # Combined momentum score
            momentum_score = momentum_return - (0.5 * max(0, -recent_return))  # Penalty for recent losses

            return momentum_score

        except Exception as e:
            self.Debug(f"Error calculating momentum for {symbol}: {e}")
            return None

    def OnData(self, data):
        """Handle incoming data"""
        # Basic risk management - stop losses
        for holding in self.Portfolio.Values:
            if holding.Invested:
                unrealized_pnl = holding.UnrealizedProfitPercent

                if unrealized_pnl < self.stop_loss:
                    self.Liquidate(holding.Symbol)
                    self.Debug(f"Stop loss triggered for {holding.Symbol}: {unrealized_pnl:.1%}")

    def TrackPerformance(self):
        """Track daily performance metrics"""
        current_value = self.Portfolio.TotalPortfolioValue
        daily_return = (current_value / self.previous_portfolio_value) - 1
        self.daily_returns.append(daily_return)
        self.previous_portfolio_value = current_value

        # Log performance monthly
        if len(self.daily_returns) % 22 == 0:  # Approximately monthly
            if len(self.daily_returns) >= 22:
                recent_returns = self.daily_returns[-22:]
                monthly_return = sum(recent_returns)
                volatility = np.std(recent_returns) * np.sqrt(252)

                self.Debug(f"Monthly return: {monthly_return:.2%}, Annualized vol: {volatility:.1%}")

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
            self.Debug("=== REAL MOMENTUM STRATEGY RESULTS ===")
            self.Debug(f"Total Return: {total_return:.2%}")
            self.Debug(f"Annual Return: {annual_return:.2%}")
            self.Debug(f"Volatility: {volatility:.2%}")
            self.Debug(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            self.Debug(f"Max Drawdown: {max_drawdown:.2%}")
            self.Debug(f"Win Rate: {win_rate:.1%}")
            self.Debug(f"Total Trades: {self.rebalance_count * 5}")  # Approx
            self.Debug("========================================")