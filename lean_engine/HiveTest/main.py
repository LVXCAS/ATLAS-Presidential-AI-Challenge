# region imports
from AlgorithmImports import *
# endregion

class HiveTest(QCAlgorithm):
    """Hive Trading Empire Test Algorithm - LEAN Integration Test"""

    def initialize(self):
        """Initialize the Hive Trading test algorithm"""
        self.debug("🚀 HIVE TRADING EMPIRE - LEAN INTEGRATION TEST")
        self.debug("============================================")
        
        # Set basic parameters for testing
        self.set_start_date(2024, 1, 1)  # Test with recent data
        self.set_end_date(2024, 1, 31)   # One month test
        self.set_cash(100000)            # $100K test capital
        
        # Add SPY for testing
        spy = self.add_equity("SPY", Resolution.DAILY)
        spy.set_data_normalization_mode(DataNormalizationMode.RAW)
        
        self.debug(f"✅ Algorithm initialized with ${self.portfolio.cash} cash")
        self.debug("🎯 Testing Hive Trading system integration with LEAN")
        
        # Track our state
        self.first_trade = True
        self.trade_count = 0

    def on_data(self, data: Slice):
        """Handle incoming market data"""
        if not data.bars.contains_key("SPY"):
            return
            
        spy_bar = data.bars["SPY"]
        
        # Simple buy and hold test strategy
        if not self.portfolio.invested and self.first_trade:
            self.debug(f"📊 SPY Price: ${spy_bar.close:.2f}")
            self.debug("💰 Executing first trade - Buying SPY")
            
            # Buy SPY with all available cash
            self.set_holdings("SPY", 1.0)
            self.first_trade = False
            self.trade_count += 1
            
        elif self.portfolio.invested and self.trade_count == 1:
            # Show portfolio status
            pnl = self.portfolio.total_portfolio_value - self.portfolio.cash
            self.debug(f"💼 Portfolio Value: ${self.portfolio.total_portfolio_value:.2f}")
            self.debug(f"📈 Unrealized P&L: ${pnl:.2f}")
            
            # Increment trade count to avoid spam
            self.trade_count += 1

    def on_order_event(self, order_event):
        """Handle order execution events"""
        if order_event.status == OrderStatus.FILLED:
            self.debug(f"✅ Order Filled: {order_event.symbol} - {order_event.direction} - {order_event.fill_quantity} shares @ ${order_event.fill_price}")
            
    def on_end_of_algorithm(self):
        """Called when algorithm finishes"""
        final_value = self.portfolio.total_portfolio_value
        total_return = (final_value / 100000 - 1) * 100
        
        self.debug("🏁 ALGORITHM COMPLETE")
        self.debug("==================")
        self.debug(f"💰 Final Portfolio Value: ${final_value:.2f}")
        self.debug(f"📊 Total Return: {total_return:.2f}%")
        self.debug("🎉 HIVE TRADING EMPIRE - LEAN INTEGRATION SUCCESS!")
