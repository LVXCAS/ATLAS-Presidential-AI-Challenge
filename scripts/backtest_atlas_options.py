#!/usr/bin/env python3
"""
ATLAS OPTIONS BACKTEST (SIMULATED)
==================================
Backtests options strategies using Black-Scholes simulation on equity data.
No need for expensive historical options data!

Strategies:
- Iron Condor (Neutral)
- Vertical Spreads (Directional)

Logic:
1. Get historical stock data (SPY, QQQ)
2. Calculate indicators (RSI, Bollinger Bands)
3. "Open" virtual option positions using Black-Scholes pricing
4. Simulate Theta decay and Delta moves every bar
5. Close based on profit targets or stops
"""

import backtrader as bt
import math
from scipy.stats import norm
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------
# BLACK-SCHOLES PRICER
# ---------------------------------------------------------------------
class BlackScholesPricer:
    """Calculates theoretical option prices and Greeks"""

    @staticmethod
    def calculate(S, K, T, r, sigma, option_type='call'):
        """
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        """
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S), 0, 0, 0, 0

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1

        # Greeks
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365

        return price, delta, gamma, theta, vega

# ---------------------------------------------------------------------
# SIMULATED OPTION POSITION
# ---------------------------------------------------------------------
class SimulatedOptionPosition:
    """Tracks a simulated multi-leg option position"""
    def __init__(self, entry_date, entry_price, strategy_type, legs):
        self.entry_date = entry_date
        self.entry_underlying_price = entry_price
        self.strategy_type = strategy_type
        self.legs = legs  # List of dicts: {'type': 'call', 'strike': 100, 'action': 'sell'}
        self.is_open = True
        self.pnl = 0.0
        self.max_profit = 0.0
        self.max_loss = 0.0
        self.entry_credit = 0.0
        self.current_value = 0.0
        
        # Calculate initial value
        self._calculate_initial_metrics()

    def _calculate_initial_metrics(self):
        net_credit = 0
        for leg in self.legs:
            if leg['action'] == 'sell':
                net_credit += leg['entry_price']
            else:
                net_credit -= leg['entry_price']
        
        self.entry_credit = net_credit
        self.current_value = -net_credit # Short position value is negative of credit

    def update(self, current_date, current_underlying_price, volatility, risk_free_rate=0.05):
        """Update position value based on current market conditions"""
        if not self.is_open:
            return

        total_value = 0
        days_held = (current_date - self.entry_date).days
        
        # Simplified DTE calculation (assuming 30 days entry)
        initial_dte = 30
        current_dte = max(0, initial_dte - days_held)
        T = current_dte / 365.0

        for leg in self.legs:
            price, _, _, _, _ = BlackScholesPricer.calculate(
                current_underlying_price, 
                leg['strike'], 
                T, 
                risk_free_rate, 
                volatility, 
                leg['type']
            )
            
            if leg['action'] == 'sell':
                total_value -= price # Short option is a liability
            else:
                total_value += price # Long option is an asset

        self.current_value = total_value
        self.pnl = self.current_value - (-self.entry_credit) # Current Value - Initial Liability

# ---------------------------------------------------------------------
# BACKTRADER STRATEGY
# ---------------------------------------------------------------------
class AtlasOptionsStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('risk_free_rate', 0.05),
        ('volatility_window', 30),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.bb = bt.indicators.BollingerBands(self.data.close, period=self.params.bb_period, devfactor=self.params.bb_dev)
        self.positions_list = []
        self.equity_curve = []

    def next(self):
        # 1. Calculate Historical Volatility (Annualized)
        returns = np.diff(np.log(self.data.close.get(size=self.params.volatility_window)))
        if len(returns) < 2:
            return
        volatility = np.std(returns) * np.sqrt(252)

        current_price = self.data.close[0]
        current_date = self.data.datetime.date(0)

        # 2. Update Existing Positions
        for pos in self.positions_list:
            if pos.is_open:
                pos.update(current_date, current_price, volatility, self.params.risk_free_rate)
                
                # Exit Logic: 50% Profit Take
                if pos.pnl > (pos.entry_credit * 0.50):
                    print(f"[{current_date}] CLOSING {pos.strategy_type}: PROFIT TAKE (+${pos.pnl:.2f})")
                    self.broker.add_cash(pos.pnl) # Simulate cash add
                    pos.is_open = False
                
                # Exit Logic: Stop Loss (-100% of credit)
                elif pos.pnl < -(pos.entry_credit * 1.0):
                    print(f"[{current_date}] CLOSING {pos.strategy_type}: STOP LOSS (${pos.pnl:.2f})")
                    self.broker.add_cash(pos.pnl) # Simulate cash deduction
                    pos.is_open = False

        # 3. Entry Logic (Iron Condor)
        # Enter when market is neutral (RSI between 40 and 60) and inside BB
        if not any(p.is_open for p in self.positions_list): # One position at a time for simplicity
            if 40 < self.rsi[0] < 60:
                self._open_iron_condor(current_date, current_price, volatility)

    def _open_iron_condor(self, date, price, vol):
        # 30 DTE, Short 16 Delta (approx), Long 5 Delta
        # Simplified: Short +/- 5%, Long +/- 10%
        
        T = 30 / 365.0
        r = self.params.risk_free_rate

        legs = []
        
        # Put Wing
        short_put_strike = round(price * 0.95)
        long_put_strike = round(price * 0.90)
        sp_price, _, _, _, _ = BlackScholesPricer.calculate(price, short_put_strike, T, r, vol, 'put')
        lp_price, _, _, _, _ = BlackScholesPricer.calculate(price, long_put_strike, T, r, vol, 'put')
        
        legs.append({'type': 'put', 'strike': short_put_strike, 'action': 'sell', 'entry_price': sp_price})
        legs.append({'type': 'put', 'strike': long_put_strike, 'action': 'buy', 'entry_price': lp_price})

        # Call Wing
        short_call_strike = round(price * 1.05)
        long_call_strike = round(price * 1.10)
        sc_price, _, _, _, _ = BlackScholesPricer.calculate(price, short_call_strike, T, r, vol, 'call')
        lc_price, _, _, _, _ = BlackScholesPricer.calculate(price, long_call_strike, T, r, vol, 'call')

        legs.append({'type': 'call', 'strike': short_call_strike, 'action': 'sell', 'entry_price': sc_price})
        legs.append({'type': 'call', 'strike': long_call_strike, 'action': 'buy', 'entry_price': lc_price})

        pos = SimulatedOptionPosition(date, price, 'IRON_CONDOR', legs)
        self.positions_list.append(pos)
        
        print(f"[{date}] OPEN IRON CONDOR: Price=${price:.2f}, Vol={vol:.1%}, Credit=${pos.entry_credit:.2f}")


# ---------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------
def run_backtest():
    cerebro = bt.Cerebro()
    
    # Download data
    print("Downloading SPY data...")
    df = yf.download('SPY', start='2023-01-01', end='2023-12-31', progress=False)
    
    # Fix for recent yfinance versions returning MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(AtlasOptionsStrategy)
    cerebro.broker.setcash(100000.0)

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

if __name__ == '__main__':
    run_backtest()
