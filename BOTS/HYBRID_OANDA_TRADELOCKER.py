"""
HYBRID ADAPTER: OANDA Data + TradeLocker Execution

PROBLEM: TradeLocker returns "Insufficient data"
SOLUTION: Get reliable candle data from OANDA, execute trades on TradeLocker (E8 account)

This adapter:
1. Fetches market data from OANDA (reliable, always works)
2. Executes trades on TradeLocker (your E8 challenge account)
"""

import os
from oandapyV20 import API
import oandapyV20.endpoints.instruments as oanda_instruments
from E8_TRADELOCKER_ADAPTER import E8TradeLockerAdapter

class HybridAdapter:
    """
    Get data from OANDA, trade on TradeLocker.
    Best of both worlds!
    """

    def __init__(self):
        # OANDA for data
        oanda_token = os.getenv('OANDA_API_KEY')
        if not oanda_token:
            raise ValueError("Missing OANDA_API_KEY in .env")

        self.oanda = API(access_token=oanda_token, environment='practice')

        # TradeLocker for trading
        self.tradelocker = E8TradeLockerAdapter()

        print("\n[HYBRID] Using OANDA for data, TradeLocker for execution")

    def get_candles(self, symbol, count=100, granularity='H1'):
        """Get candles from OANDA (reliable data source)"""
        params = {"count": count, "granularity": granularity}
        r = oanda_instruments.InstrumentsCandles(instrument=symbol, params=params)
        self.oanda.request(r)
        return r.response['candles']

    def get_account_summary(self):
        """Get account from TradeLocker (E8 challenge account)"""
        return self.tradelocker.get_account_summary()

    def get_open_positions(self):
        """Get positions from TradeLocker"""
        return self.tradelocker.get_open_positions()

    def place_order(self, symbol, units, side, take_profit, stop_loss):
        """Place order on TradeLocker (E8 account)"""
        return self.tradelocker.place_order(
            symbol=symbol,
            units=units,
            side=side,
            take_profit=take_profit,
            stop_loss=stop_loss
        )
