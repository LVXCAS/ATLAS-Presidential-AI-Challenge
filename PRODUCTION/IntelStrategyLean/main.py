# region imports
from AlgorithmImports import *
# endregion

class IntelStrategyLean(QCAlgorithm):
    """
    Intel-style dual strategy in LEAN format

    Strategy:
    - Cash-secured puts (collect premium, acquire stock at discount)
    - Long calls (upside exposure with limited risk)

    This is your proven Week 1 strategy, now in LEAN professional format.
    """

    def initialize(self):
        """Initialize strategy parameters and universe"""

        # Set dates and capital
        self.set_start_date(2024, 1, 1)  # Backtest from Jan 2024
        self.set_end_date(2024, 12, 31)  # Through Dec 2024
        self.set_cash(126900)  # Your actual capital

        # Intel-style universe (tech stocks with earnings volatility)
        self.symbols = ["INTC", "AMD", "NVDA", "AAPL", "MSFT"]

        # Add equities
        for symbol in self.symbols:
            equity = self.add_equity(symbol, Resolution.MINUTE)
            equity.set_data_normalization_mode(DataNormalizationMode.RAW)

            # Add options for this equity
            option = self.add_option(symbol, Resolution.MINUTE)
            option.set_filter(self.option_filter)

        # Strategy parameters (from your Week 1 constraints)
        self.min_confidence = 4.5  # 90%+ threshold
        self.max_position_size = 0.015  # 1.5% per trade
        self.max_daily_risk = 0.03  # 3% daily risk limit
        self.max_trades_per_week = 2  # Week 1 constraint

        # State tracking
        self.trades_this_week = 0
        self.last_week = 0

        # Schedule weekly reset
        self.schedule.on(
            self.date_rules.week_start("INTC"),
            self.time_rules.at(9, 30),
            self.reset_weekly_counters
        )

        # Schedule daily scans (like your 5-minute scanner)
        self.schedule.on(
            self.date_rules.every_day("INTC"),
            self.time_rules.every(self.time_delta(minutes=5)),
            self.scan_opportunities
        )

    def option_filter(self, universe):
        """Filter options to reasonable strikes and expirations"""
        return (universe
                .strikes(-5, 5)  # 5 strikes OTM to ITM
                .expiration(7, 60))  # 1 week to 2 months

    def reset_weekly_counters(self):
        """Reset trade counter at start of week"""
        self.trades_this_week = 0
        self.debug(f"Week reset: {self.time}")

    def scan_opportunities(self):
        """
        Scan for high-confidence opportunities
        Similar to your continuous_week1_scanner.py
        """

        # Check if we've hit weekly trade limit
        if self.trades_this_week >= self.max_trades_per_week:
            return

        for symbol in self.symbols:
            # Get current price and volatility
            history = self.history(symbol, 20, Resolution.DAILY)
            if history.empty:
                continue

            current_price = self.securities[symbol].price
            if current_price == 0:
                continue

            # Calculate realized volatility
            returns = history['close'].pct_change().dropna()
            if len(returns) < 5:
                continue

            realized_vol = float(returns.std())

            # Get volume
            volume = self.securities[symbol].volume

            # Calculate opportunity score (Intel-style)
            confidence = self.calculate_intel_score(
                current_price,
                realized_vol,
                volume
            )

            # Check if meets threshold
            if confidence >= self.min_confidence:
                self.debug(f"[OPPORTUNITY] {symbol}: {confidence:.2f} confidence")
                self.execute_intel_strategy(symbol, confidence, current_price, realized_vol)

    def calculate_intel_score(self, price: float, volatility: float, volume: float) -> float:
        """
        Calculate Intel-style opportunity score
        Based on your unified_validated_strategy_system.py logic
        """

        score = 0.0

        # Volatility component (higher vol = more premium)
        if volatility > 0.025:  # 2.5%+ daily moves
            score += 2.0
        elif volatility > 0.015:
            score += 1.0

        # Volume component (liquidity check)
        if volume > 50_000_000:  # High volume
            score += 1.5
        elif volume > 20_000_000:
            score += 1.0
        elif volume > 10_000_000:
            score += 0.5

        # Price level (avoid penny stocks)
        if price > 50:
            score += 1.0
        elif price > 20:
            score += 0.5

        # Normalize to 0-5 scale
        return min(score, 5.0)

    def execute_intel_strategy(self, symbol: str, confidence: float, price: float, volatility: float):
        """
        Execute Intel dual strategy:
        1. Cash-secured puts (collect premium)
        2. Long calls (upside exposure)
        """

        # Calculate position size (1.5% max)
        capital = self.portfolio.cash
        position_value = capital * self.max_position_size

        # Get option chain
        option_chain = self.current_slice.option_chains.get(symbol)
        if option_chain is None:
            return

        # Filter to puts and calls around 30 DTE
        puts = [x for x in option_chain if x.right == OptionRight.PUT
                and 25 <= (x.expiry - self.time).days <= 35]
        calls = [x for x in option_chain if x.right == OptionRight.CALL
                 and 25 <= (x.expiry - self.time).days <= 35]

        if not puts or not calls:
            return

        # Find ATM/slightly OTM options
        atm_put = min(puts, key=lambda x: abs(x.strike - price * 0.95))  # 5% OTM
        atm_call = min(calls, key=lambda x: abs(x.strike - price * 1.05))  # 5% OTM

        # Execute puts (sell cash-secured)
        put_contracts = int(position_value / (atm_put.strike * 100))
        if put_contracts > 0:
            self.sell(atm_put.symbol, put_contracts)
            self.debug(f"[TRADE] Sold {put_contracts} puts {atm_put.symbol}")

        # Execute calls (buy long)
        call_value = position_value * 0.5  # 50% in calls
        call_contracts = int(call_value / (atm_call.ask_price * 100))
        if call_contracts > 0:
            self.buy(atm_call.symbol, call_contracts)
            self.debug(f"[TRADE] Bought {call_contracts} calls {atm_call.symbol}")

        # Update counters
        self.trades_this_week += 1

    def on_data(self, data: Slice):
        """
        Main data handler
        LEAN calls this every minute during market hours
        """
        # Store current slice for option chain access
        self.current_slice = data

        # Manage existing positions
        for holding in self.portfolio.values():
            if holding.invested and holding.is_option:
                # Check if we should exit
                days_to_expiry = (holding.security.expiry - self.time).days
                pnl_percent = holding.unrealized_profit_percent

                # Exit rules
                if days_to_expiry <= 5:  # Close near expiry
                    self.liquidate(holding.symbol)
                    self.debug(f"[EXIT] Closed {holding.symbol} near expiry")
                elif pnl_percent > 0.5:  # 50% profit target
                    self.liquidate(holding.symbol)
                    self.debug(f"[EXIT] Took profit on {holding.symbol}: {pnl_percent:.1%}")
                elif pnl_percent < -0.3:  # 30% stop loss
                    self.liquidate(holding.symbol)
                    self.debug(f"[EXIT] Stop loss on {holding.symbol}: {pnl_percent:.1%}")

    def on_end_of_algorithm(self):
        """Called when backtest completes"""
        self.debug(f"Final Portfolio Value: ${self.portfolio.total_portfolio_value:,.2f}")
        self.debug(f"Total Return: {(self.portfolio.total_portfolio_value / 126900 - 1):.2%}")
