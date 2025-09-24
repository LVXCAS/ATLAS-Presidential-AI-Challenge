
from AlgorithmImports import *

class Strategymomentum_test_4(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 01, 01)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # Add equity
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Strategy specific initialization
        
        # Momentum indicators
        self.rsi = self.RSI(self.symbol, 14, Resolution.Daily)
        self.sma_fast = self.SMA(self.symbol, 20, Resolution.Daily)
        self.sma_slow = self.SMA(self.symbol, 50, Resolution.Daily)

        # Parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        

        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return

        
        price = data[self.symbol].Close

        # Simple momentum strategy
        if self.rsi.Current.Value < self.rsi_oversold and not self.Portfolio.Invested:
            self.SetHoldings(self.symbol, 1.0)

        elif self.rsi.Current.Value > self.rsi_overbought and self.Portfolio.Invested:
            self.Liquidate()
        

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            self.trade_count += 1

            # Track performance (simplified)
            if orderEvent.Direction == OrderDirection.Sell:
                # This is a closing trade - calculate P&L
                pass
