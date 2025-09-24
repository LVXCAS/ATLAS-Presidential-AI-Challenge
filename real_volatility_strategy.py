"""
REAL VOLATILITY STRATEGY - ACTUAL LEAN ALGORITHM
===============================================
Genuine volatility strategy with real market logic
"""

from AlgorithmImports import *

class RealVolatilityStrategy(QCAlgorithm):
    """
    Real volatility strategy using actual market data
    Trade based on volatility regimes and volatility mean reversion
    """

    def Initialize(self):
        """Initialize the algorithm"""
        # Set timeframe for backtest
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 9, 18)
        self.SetCash(100000)

        # Universe selection - focus on high volatility names
        self.symbols = ['QQQ', 'AAPL', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMZN', 'GOOGL']

        # Add securities and initialize volatility tracking
        self.volatility_data = {}
        for symbol in self.symbols:
            equity = self.AddEquity(symbol, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Adjusted)

            # Initialize volatility indicators
            self.volatility_data[symbol] = {
                'price_history': RollingWindow[float](252),  # 1 year of prices
                'returns_history': RollingWindow[float](252),  # 1 year of returns
                'volatility_history': RollingWindow[float](63),  # 3 months of volatility
                'atr': self.ATR(symbol, 14),  # Average True Range
                'current_volatility': 0,
                'volatility_percentile': 50,
                'regime': 'NORMAL'
            }

        # Strategy parameters
        self.volatility_lookback = 63  # 3 months
        self.regime_threshold_low = 25  # Low volatility percentile
        self.regime_threshold_high = 75  # High volatility percentile
        self.position_sizing_factor = 0.12  # 12% max position size

        # Rebalancing
        self.rebalance_frequency = 5  # Weekly

        # Performance tracking
        self.previous_portfolio_value = self.Portfolio.TotalPortfolioValue
        self.daily_returns = []
        self.trades_count = 0

        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.WeekStart("QQQ"),
            self.TimeRules.AfterMarketOpen("QQQ", 30),
            self.Rebalance
        )

        # Daily volatility update
        self.Schedule.On(
            self.DateRules.EveryDay("QQQ"),
            self.TimeRules.BeforeMarketClose("QQQ", 0),
            self.UpdateVolatilityMetrics
        )

        self.Debug("Real Volatility Strategy initialized")

    def UpdateVolatilityMetrics(self):
        """Update volatility metrics for all symbols"""
        for symbol in self.symbols:
            try:
                current_price = self.Securities[symbol].Price
                if current_price == 0:
                    continue

                vol_data = self.volatility_data[symbol]

                # Update price history
                vol_data['price_history'].Add(current_price)

                # Calculate returns and volatility if we have enough data
                if vol_data['price_history'].Count >= 2:
                    # Calculate daily return
                    prev_price = vol_data['price_history'][1]
                    daily_return = (current_price / prev_price - 1)
                    vol_data['returns_history'].Add(daily_return)

                    # Calculate realized volatility (30-day)
                    if vol_data['returns_history'].Count >= 30:
                        returns_array = [vol_data['returns_history'][i] for i in range(min(30, vol_data['returns_history'].Count))]
                        realized_vol = np.std(returns_array) * np.sqrt(252)
                        vol_data['current_volatility'] = realized_vol
                        vol_data['volatility_history'].Add(realized_vol)

                        # Calculate volatility percentile
                        if vol_data['volatility_history'].Count >= 63:
                            vol_array = [vol_data['volatility_history'][i] for i in range(vol_data['volatility_history'].Count)]
                            percentile = stats.percentileofscore(vol_array, realized_vol)
                            vol_data['volatility_percentile'] = percentile

                            # Determine volatility regime
                            if percentile < self.regime_threshold_low:
                                vol_data['regime'] = 'LOW_VOL'
                            elif percentile > self.regime_threshold_high:
                                vol_data['regime'] = 'HIGH_VOL'
                            else:
                                vol_data['regime'] = 'NORMAL'

            except Exception as e:
                self.Debug(f"Error updating volatility for {symbol}: {e}")

    def Rebalance(self):
        """Execute weekly rebalancing based on volatility signals"""
        self.Debug(f"Volatility rebalancing on {self.Time}")

        # Classify symbols by volatility regime
        low_vol_symbols = []
        high_vol_symbols = []
        normal_vol_symbols = []

        for symbol in self.symbols:
            vol_data = self.volatility_data[symbol]
            regime = vol_data['regime']

            if regime == 'LOW_VOL':
                low_vol_symbols.append(symbol)
            elif regime == 'HIGH_VOL':
                high_vol_symbols.append(symbol)
            else:
                normal_vol_symbols.append(symbol)

        self.Debug(f"Low vol: {low_vol_symbols}")
        self.Debug(f"High vol: {high_vol_symbols}")
        self.Debug(f"Normal vol: {normal_vol_symbols}")

        # Strategy logic:
        # 1. Buy low volatility stocks (vol mean reversion)
        # 2. Avoid or short high volatility stocks
        # 3. Hold moderate positions in normal volatility

        # Clear existing positions
        for symbol in self.symbols:
            if self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)

        # Allocate based on volatility regime
        total_weight = 0

        # Low volatility - higher allocation (vol mean reversion)
        if low_vol_symbols:
            weight_per_low_vol = min(0.25, 0.6 / len(low_vol_symbols))  # Up to 60% in low vol
            for symbol in low_vol_symbols:
                self.SetHoldings(symbol, weight_per_low_vol)
                total_weight += weight_per_low_vol
                self.trades_count += 1
                self.Debug(f"Long {symbol} (low vol): {weight_per_low_vol:.1%}")

        # Normal volatility - moderate allocation
        if normal_vol_symbols and total_weight < 0.8:
            remaining_weight = min(0.4, 0.8 - total_weight)
            weight_per_normal = remaining_weight / len(normal_vol_symbols)

            if weight_per_normal > 0.05:  # Only if meaningful position
                for symbol in normal_vol_symbols:
                    self.SetHoldings(symbol, weight_per_normal)
                    total_weight += weight_per_normal
                    self.trades_count += 1
                    self.Debug(f"Long {symbol} (normal vol): {weight_per_normal:.1%}")

        # High volatility - avoid for now (could implement short selling)
        # for symbol in high_vol_symbols:
        #     self.Debug(f"Avoiding {symbol} (high vol)")

        self.Debug(f"Total portfolio weight: {total_weight:.1%}")

    def OnData(self, data):
        """Handle incoming data"""
        pass  # Main logic in scheduled rebalancing

    def TrackPerformance(self):
        """Track daily performance metrics"""
        current_value = self.Portfolio.TotalPortfolioValue
        daily_return = (current_value / self.previous_portfolio_value) - 1
        self.daily_returns.append(daily_return)
        self.previous_portfolio_value = current_value

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
            self.Debug("=== REAL VOLATILITY STRATEGY RESULTS ===")
            self.Debug(f"Total Return: {total_return:.2%}")
            self.Debug(f"Annual Return: {annual_return:.2%}")
            self.Debug(f"Volatility: {volatility:.2%}")
            self.Debug(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            self.Debug(f"Max Drawdown: {max_drawdown:.2%}")
            self.Debug(f"Win Rate: {win_rate:.1%}")
            self.Debug(f"Total Trades: {self.trades_count}")
            self.Debug("=========================================")