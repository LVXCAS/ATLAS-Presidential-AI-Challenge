"""
LEAN-Compatible Forex Strategy
Port of WORKING_FOREX_OANDA.py for QuantConnect backtesting

Upload this to QuantConnect.com for institutional-grade backtest
"""

from AlgorithmImports import *
import numpy as np

class ForexMomentumStrategy(QCAlgorithm):
    """
    Multi-timeframe forex strategy using TA-Lib indicators
    - RSI extremes (overbought/oversold)
    - MACD crossovers
    - EMA trend confirmation
    - ADX trend strength filter
    - 4H trend alignment
    """

    def Initialize(self):
        """
        Initialize algorithm parameters
        """
        # Backtest period
        self.SetStartDate(2024, 5, 1)  # 6 months ago from Nov 2024
        self.SetEndDate(2024, 11, 1)

        # Starting capital
        self.SetCash(100000)

        # Forex pairs to trade
        self.pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'GBPJPY']
        self.symbols = {}

        # Add forex securities
        for pair in self.pairs:
            symbol = self.AddForex(pair, Resolution.Hour, Market.Oanda)
            self.symbols[pair] = symbol.Symbol

            # Set leverage
            symbol.SetLeverage(5)

        # Strategy parameters (EXACT same as WORKING_FOREX_OANDA.py)
        self.min_score = 2.5
        self.risk_per_trade = 0.01  # 1% stop loss
        self.profit_target = 0.02   # 2% take profit
        self.max_positions = 3

        # Indicators for each pair
        self.indicators = {}

        for pair in self.pairs:
            symbol = self.symbols[pair]

            # 1H indicators (primary timeframe)
            self.indicators[pair] = {
                'rsi': self.RSI(symbol, 14, Resolution.Hour),
                'macd': self.MACD(symbol, 12, 26, 9, MovingAverageType.Exponential, Resolution.Hour),
                'adx': self.ADX(symbol, 14, Resolution.Hour),
                'atr': self.ATR(symbol, 14, Resolution.Hour),
                'ema_fast': self.EMA(symbol, 10, Resolution.Hour),
                'ema_slow': self.EMA(symbol, 21, Resolution.Hour),
                'ema_trend': self.EMA(symbol, 200, Resolution.Hour),

                # 4H indicators (higher timeframe)
                'ema_fast_4h': self.EMA(symbol, 10, Resolution.Hour * 4),
                'ema_slow_4h': self.EMA(symbol, 21, Resolution.Hour * 4),
                'ema_trend_4h': self.EMA(symbol, 50, Resolution.Hour * 4)
            }

        # Tracking
        self.positions = {}
        self.trade_count = 0

        # Schedule hourly scans (same as live bot)
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self.ScanForSignals
        )

    def OnData(self, data):
        """
        Event handler for new data
        LEAN calls this automatically
        """
        # Check for stop-loss or take-profit hits
        for pair in list(self.positions.keys()):
            if pair in self.positions:
                self.CheckExits(pair, data)

    def ScanForSignals(self):
        """
        Scan all pairs for LONG/SHORT signals
        Called every hour
        """
        # Don't exceed max positions
        if len(self.positions) >= self.max_positions:
            return

        for pair in self.pairs:
            # Skip if already have position
            if pair in self.positions:
                continue

            # Check if indicators are ready
            if not self.IndicatorsReady(pair):
                continue

            # Calculate signals
            long_signal, short_signal = self.CalculateSignals(pair)

            # Execute if signal meets threshold
            if long_signal and long_signal['score'] >= self.min_score:
                self.EnterTrade(pair, 'LONG', long_signal)

            elif short_signal and short_signal['score'] >= self.min_score:
                self.EnterTrade(pair, 'SHORT', short_signal)

    def IndicatorsReady(self, pair):
        """
        Check if all indicators have enough data
        """
        ind = self.indicators[pair]

        return (ind['rsi'].IsReady and
                ind['macd'].IsReady and
                ind['adx'].IsReady and
                ind['ema_fast'].IsReady and
                ind['ema_slow'].IsReady and
                ind['ema_trend'].IsReady and
                ind['ema_fast_4h'].IsReady and
                ind['ema_slow_4h'].IsReady and
                ind['ema_trend_4h'].IsReady)

    def Get4HTrend(self, pair):
        """
        Get 4H timeframe trend direction
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        ind = self.indicators[pair]

        ema_fast_4h = ind['ema_fast_4h'].Current.Value
        ema_slow_4h = ind['ema_slow_4h'].Current.Value
        ema_trend_4h = ind['ema_trend_4h'].Current.Value

        symbol = self.symbols[pair]
        current_price = self.Securities[symbol].Price

        # Bullish: Fast > Slow and Price > Trend EMA
        if ema_fast_4h > ema_slow_4h and current_price > ema_trend_4h:
            return 'bullish'

        # Bearish: Fast < Slow and Price < Trend EMA
        elif ema_fast_4h < ema_slow_4h and current_price < ema_trend_4h:
            return 'bearish'

        else:
            return 'neutral'

    def CalculateSignals(self, pair):
        """
        Calculate LONG and SHORT signals
        EXACT same logic as WORKING_FOREX_OANDA.py
        """
        ind = self.indicators[pair]
        symbol = self.symbols[pair]

        current_price = self.Securities[symbol].Price

        # Get indicator values
        rsi = ind['rsi'].Current.Value
        macd = ind['macd'].Current.Value
        macd_signal = ind['macd'].Signal.Current.Value
        macd_hist = macd - macd_signal

        adx = ind['adx'].Current.Value
        atr = ind['atr'].Current.Value
        volatility = (atr / current_price) * 100

        ema_fast = ind['ema_fast'].Current.Value
        ema_slow = ind['ema_slow'].Current.Value
        ema_trend = ind['ema_trend'].Current.Value

        # Get previous values for crossovers
        macd_hist_prev = ind['macd'].Current.Value - ind['macd'].Signal.Current.Value
        ema_fast_prev = ind['ema_fast'].Current.Value
        ema_slow_prev = ind['ema_slow'].Current.Value

        long_score = 0
        short_score = 0
        long_signals = []
        short_signals = []

        # === LONG SIGNALS ===

        # 1. RSI oversold
        if rsi < 40:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")

        # 2. MACD bullish cross
        if macd_hist > 0 and macd_hist_prev <= 0:
            long_score += 2.5
            long_signals.append("MACD_BULLISH")

        # 3. EMA bullish crossover
        if ema_fast > ema_slow and ema_fast_prev <= ema_slow_prev:
            long_score += 2
            long_signals.append("EMA_CROSS_BULLISH")

        # 4. Uptrend
        if current_price > ema_trend:
            long_score += 1
            long_signals.append("UPTREND")

        # === SHORT SIGNALS ===

        # 1. RSI overbought
        if rsi > 60:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")

        # 2. MACD bearish cross
        if macd_hist < 0 and macd_hist_prev >= 0:
            short_score += 2.5
            short_signals.append("MACD_BEARISH")

        # 3. EMA bearish crossover
        if ema_fast < ema_slow and ema_fast_prev >= ema_slow_prev:
            short_score += 2
            short_signals.append("EMA_CROSS_BEARISH")

        # 4. Downtrend
        if current_price < ema_trend:
            short_score += 1
            short_signals.append("DOWNTREND")

        # === SHARED SIGNALS ===

        # 5. Trend strength
        if adx > 20:
            long_score += 1.5
            short_score += 1.5
            long_signals.append("STRONG_TREND")
            short_signals.append("STRONG_TREND")

        # 6. Volatility
        if volatility > 0.3:
            long_score += 1
            short_score += 1
            long_signals.append("FX_VOLATILITY")
            short_signals.append("FX_VOLATILITY")

        # === 4H TREND CONFIRMATION ===
        trend_4h = self.Get4HTrend(pair)

        if trend_4h == 'bullish':
            long_score += 2
            long_signals.append("4H_BULLISH_TREND")

            if short_score > 0:
                short_score -= 1.5
                short_signals.append("COUNTER_4H_TREND")

        elif trend_4h == 'bearish':
            short_score += 2
            short_signals.append("4H_BEARISH_TREND")

            if long_score > 0:
                long_score -= 1.5
                long_signals.append("COUNTER_4H_TREND")

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

    def EnterTrade(self, pair, direction, signal):
        """
        Enter LONG or SHORT position with stop-loss and take-profit
        """
        symbol = self.symbols[pair]
        entry_price = signal['price']

        # Calculate stop-loss and take-profit
        if direction == 'LONG':
            stop_price = entry_price * (1 - self.risk_per_trade)
            target_price = entry_price * (1 + self.profit_target)
        else:  # SHORT
            stop_price = entry_price * (1 + self.risk_per_trade)
            target_price = entry_price * (1 - self.profit_target)

        # Calculate position size
        # Risk 1% of portfolio per trade
        risk_amount = self.Portfolio.TotalPortfolioValue * self.risk_per_trade

        # Forex quantity is in base currency units
        # For 1% risk with 5x leverage, we use standard lot sizing
        quantity = int((self.Portfolio.TotalPortfolioValue * 0.05) / entry_price)

        if direction == 'SHORT':
            quantity = -quantity

        # Place market order
        self.MarketOrder(symbol, quantity)

        # Store position tracking
        self.positions[pair] = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_price,
            'take_profit': target_price,
            'quantity': quantity,
            'score': signal['score']
        }

        self.trade_count += 1

        self.Log(f"[ENTRY] {pair} {direction} @ {entry_price:.5f} (Score: {signal['score']:.1f})")
        self.Log(f"  Stop: {stop_price:.5f} | Target: {target_price:.5f}")

    def CheckExits(self, pair, data):
        """
        Check if stop-loss or take-profit hit
        """
        if pair not in self.positions:
            return

        symbol = self.symbols[pair]

        if not data.ContainsKey(symbol):
            return

        position = self.positions[pair]
        current_price = self.Securities[symbol].Price

        direction = position['direction']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']

        should_exit = False
        exit_reason = None

        # Check stop-loss and take-profit
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

            self.Log(f"[EXIT] {pair} @ {current_price:.5f} ({exit_reason}) P/L: {pnl_pct:+.2f}%")

            # Remove from tracking
            del self.positions[pair]

    def OnEndOfAlgorithm(self):
        """
        Called when backtest completes
        """
        self.Log(f"")
        self.Log(f"{'='*60}")
        self.Log(f"BACKTEST COMPLETE")
        self.Log(f"{'='*60}")
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Log(f"Total Return: {((self.Portfolio.TotalPortfolioValue - 100000) / 100000) * 100:.2f}%")
        self.Log(f"{'='*60}")
