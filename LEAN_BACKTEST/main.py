# region imports
from AlgorithmImports import *
# endregion

class ForexMomentumStrategy(QCAlgorithm):
    """
    Multi-timeframe forex strategy
    Port of WORKING_FOREX_OANDA.py for LEAN backtesting
    """

    def Initialize(self):
        # Backtest period - 6 months
        self.SetStartDate(2024, 5, 1)
        self.SetEndDate(2024, 11, 1)

        # Starting capital
        self.SetCash(100000)

        # Set account currency
        self.SetAccountCurrency("USD")

        # Add forex pairs
        self.eurusd = self.AddForex("EURUSD", Resolution.Hour, Market.Oanda)
        self.gbpusd = self.AddForex("GBPUSD", Resolution.Hour, Market.Oanda)
        self.usdjpy = self.AddForex("USDJPY", Resolution.Hour, Market.Oanda)
        self.gbpjpy = self.AddForex("GBPJPY", Resolution.Hour, Market.Oanda)

        # Set leverage
        self.eurusd.SetLeverage(5)
        self.gbpusd.SetLeverage(5)
        self.usdjpy.SetLeverage(5)
        self.gbpjpy.SetLeverage(5)

        # Strategy parameters
        self.min_score = 2.5
        self.risk_per_trade = 0.01  # 1% stop loss
        self.profit_target = 0.02   # 2% take profit
        self.max_positions = 3

        # Tracking
        self.positions = {}
        self.symbols = [self.eurusd.Symbol, self.gbpusd.Symbol,
                       self.usdjpy.Symbol, self.gbpjpy.Symbol]

        # Set up indicators for each symbol
        self.indicators = {}
        for symbol in self.symbols:
            self.indicators[symbol] = {
                'rsi': self.RSI(symbol, 14, Resolution.Hour),
                'macd': self.MACD(symbol, 12, 26, 9, MovingAverageType.Exponential, Resolution.Hour),
                'adx': self.ADX(symbol, 14, Resolution.Hour),
                'atr': self.ATR(symbol, 14, Resolution.Hour),
                'ema_fast': self.EMA(symbol, 10, Resolution.Hour),
                'ema_slow': self.EMA(symbol, 21, Resolution.Hour),
                'ema_trend': self.EMA(symbol, 200, Resolution.Hour)
            }

        # Schedule hourly scans
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self.ScanForSignals
        )

        self.Log("Forex Momentum Strategy Initialized")
        self.Log(f"Pairs: EURUSD, GBPUSD, USDJPY, GBPJPY")
        self.Log(f"Leverage: 5x, Min Score: {self.min_score}")

    def OnData(self, data):
        """Called on every data point"""
        # Check exits for existing positions
        for symbol in list(self.positions.keys()):
            if symbol in self.positions:
                self.CheckExits(symbol, data)

    def ScanForSignals(self):
        """Scan all pairs for entry signals"""
        # Don't exceed max positions
        if len(self.positions) >= self.max_positions:
            return

        for symbol in self.symbols:
            # Skip if already have position
            if symbol in self.positions:
                continue

            # Check if indicators ready
            if not self.IndicatorsReady(symbol):
                continue

            # Calculate signals
            long_signal, short_signal = self.CalculateSignals(symbol)

            # Execute if threshold met
            if long_signal and long_signal['score'] >= self.min_score:
                self.EnterTrade(symbol, 'LONG', long_signal)
            elif short_signal and short_signal['score'] >= self.min_score:
                self.EnterTrade(symbol, 'SHORT', short_signal)

    def IndicatorsReady(self, symbol):
        """Check if all indicators have enough data"""
        ind = self.indicators[symbol]
        return (ind['rsi'].IsReady and
                ind['macd'].IsReady and
                ind['adx'].IsReady and
                ind['ema_fast'].IsReady and
                ind['ema_slow'].IsReady and
                ind['ema_trend'].IsReady)

    def CalculateSignals(self, symbol):
        """Calculate LONG/SHORT signals"""
        ind = self.indicators[symbol]
        current_price = self.Securities[symbol].Price

        # Get indicator values
        rsi = ind['rsi'].Current.Value
        macd_hist = ind['macd'].Current.Value - ind['macd'].Signal.Current.Value
        adx = ind['adx'].Current.Value
        atr = ind['atr'].Current.Value
        volatility = (atr / current_price) * 100

        ema_fast = ind['ema_fast'].Current.Value
        ema_slow = ind['ema_slow'].Current.Value
        ema_trend = ind['ema_trend'].Current.Value

        long_score = 0
        short_score = 0
        long_signals = []
        short_signals = []

        # LONG signals
        if rsi < 40:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")

        if macd_hist > 0:
            long_score += 2.5
            long_signals.append("MACD_BULLISH")

        if ema_fast > ema_slow:
            long_score += 2
            long_signals.append("EMA_BULLISH")

        if current_price > ema_trend:
            long_score += 1
            long_signals.append("UPTREND")

        # SHORT signals
        if rsi > 60:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")

        if macd_hist < 0:
            short_score += 2.5
            short_signals.append("MACD_BEARISH")

        if ema_fast < ema_slow:
            short_score += 2
            short_signals.append("EMA_BEARISH")

        if current_price < ema_trend:
            short_score += 1
            short_signals.append("DOWNTREND")

        # Shared signals
        if adx > 20:
            long_score += 1.5
            short_score += 1.5
            long_signals.append("STRONG_TREND")
            short_signals.append("STRONG_TREND")

        if volatility > 0.3:
            long_score += 1
            short_score += 1
            long_signals.append("VOLATILITY")
            short_signals.append("VOLATILITY")

        # Return signals
        long_signal = None
        short_signal = None

        if long_score >= self.min_score:
            long_signal = {
                'score': long_score,
                'price': current_price,
                'signals': long_signals
            }

        if short_score >= self.min_score:
            short_signal = {
                'score': short_score,
                'price': current_price,
                'signals': short_signals
            }

        return long_signal, short_signal

    def EnterTrade(self, symbol, direction, signal):
        """Enter LONG or SHORT position"""
        entry_price = signal['price']

        # Calculate stop and target
        if direction == 'LONG':
            stop_price = entry_price * (1 - self.risk_per_trade)
            target_price = entry_price * (1 + self.profit_target)
        else:  # SHORT
            stop_price = entry_price * (1 + self.risk_per_trade)
            target_price = entry_price * (1 - self.profit_target)

        # Calculate position size (5% of portfolio with 5x leverage)
        quantity = int((self.Portfolio.TotalPortfolioValue * 0.05) / entry_price)

        if direction == 'SHORT':
            quantity = -quantity

        # Place market order
        self.MarketOrder(symbol, quantity)

        # Store position
        self.positions[symbol] = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_price,
            'take_profit': target_price,
            'quantity': quantity,
            'score': signal['score']
        }

        self.Log(f"[ENTRY] {symbol} {direction} @ {entry_price:.5f} (Score: {signal['score']:.1f})")
        self.Log(f"  Stop: {stop_price:.5f} | Target: {target_price:.5f}")

    def CheckExits(self, symbol, data):
        """Check if stop or target hit"""
        if symbol not in self.positions:
            return

        if not data.ContainsKey(symbol):
            return

        position = self.positions[symbol]
        current_price = self.Securities[symbol].Price

        direction = position['direction']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']

        should_exit = False
        exit_reason = None

        if direction == 'LONG':
            if current_price <= stop_loss:
                should_exit = True
                exit_reason = "STOP_LOSS"
            elif current_price >= take_profit:
                should_exit = True
                exit_reason = "TAKE_PROFIT"
        else:  # SHORT
            if current_price >= stop_loss:
                should_exit = True
                exit_reason = "STOP_LOSS"
            elif current_price <= take_profit:
                should_exit = True
                exit_reason = "TAKE_PROFIT"

        if should_exit:
            # Close position
            self.Liquidate(symbol)

            # Calculate P/L
            entry_price = position['entry_price']
            if direction == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            self.Log(f"[EXIT] {symbol} @ {current_price:.5f} ({exit_reason}) P/L: {pnl_pct:+.2f}%")

            # Remove from tracking
            del self.positions[symbol]

    def OnEndOfAlgorithm(self):
        """Called when backtest completes"""
        self.Log("")
        self.Log("="*60)
        self.Log("BACKTEST COMPLETE")
        self.Log("="*60)
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        total_return = ((self.Portfolio.TotalPortfolioValue - 100000) / 100000) * 100
        self.Log(f"Total Return: {total_return:.2f}%")
        self.Log("="*60)
