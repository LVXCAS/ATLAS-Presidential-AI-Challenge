"""
Simple LEAN Test Algorithm
=========================

A minimal algorithm to test LEAN integration.
"""

# Try to import QuantConnect - fallback if not available
try:
    from AlgorithmImports import *
    from QuantConnect import *
    from QuantConnect.Algorithm import QCAlgorithm
    from QuantConnect.Brokerages import BrokerageName
    QUANTCONNECT_AVAILABLE = True
    print("[INFO] QuantConnect imports successful")
except ImportError as e:
    print(f"[WARNING] QuantConnect imports failed: {e}")
    print("[INFO] Creating mock QCAlgorithm for testing")
    QUANTCONNECT_AVAILABLE = False
    
    # Mock QCAlgorithm for testing
    class QCAlgorithm:
        def SetStartDate(self, year, month, day): pass
        def SetEndDate(self, year, month, day): pass  
        def SetCash(self, amount): pass
        def SetBrokerageModel(self, brokerage): pass
        def AddEquity(self, symbol, resolution=None): pass
        def Log(self, message): print(f"[LEAN LOG] {message}")


class HiveSimpleTestAlgorithm(QCAlgorithm):
    """Simple test algorithm for LEAN integration"""
    
    def Initialize(self):
        """Initialize the algorithm"""
        self.Log("Initializing Hive Simple Test Algorithm")
        
        # Set basic parameters
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 31)
        self.SetCash(100000)
        
        if QUANTCONNECT_AVAILABLE:
            self.SetBrokerageModel(BrokerageName.Alpaca)
            self.AddEquity("SPY", Resolution.Daily)
        
        self.Log("Hive Simple Test Algorithm initialized successfully")
        
    def OnData(self, data):
        """Handle incoming data"""
        if QUANTCONNECT_AVAILABLE and data.Bars.ContainsKey("SPY"):
            spy_bar = data.Bars["SPY"]
            self.Log(f"SPY: ${spy_bar.Close}")
            
            # Simple buy and hold strategy for testing
            if not self.Portfolio.Invested:
                self.Log("Buying SPY for testing")
                self.SetHoldings("SPY", 1.0)
        else:
            self.Log("OnData called - mock mode")
            
    def OnOrderEvent(self, orderEvent):
        """Handle order events"""
        self.Log(f"Order event: {orderEvent}")


# Make this the default algorithm
HiveTradingMasterAlgorithm = HiveSimpleTestAlgorithm