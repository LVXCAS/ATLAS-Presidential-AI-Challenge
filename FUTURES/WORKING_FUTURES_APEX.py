"""
FUTURES TRADING BOT - APEX TRADER FUNDING
Separated architecture using SHARED/ libraries
Target: Apex PA $150K account (+$9K profit to pass)
Markets: ES (S&P 500), NQ (Nasdaq), CL (Oil), GC (Gold)
"""
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared libraries
from SHARED.technical_analysis import ta
from SHARED.kelly_criterion import kelly
from SHARED.multi_timeframe import mtf

import numpy as np


class FuturesTraderApex:
    """
    Futures trading bot for Apex Trader Funding
    Uses shared technical analysis, Kelly sizing, and multi-timeframe confirmation
    Trades ES, NQ, CL, GC futures contracts
    """

    def __init__(self):
        print("=" * 70)
        print("FUTURES TRADER - APEX TRADER FUNDING (SEPARATED ARCHITECTURE)")
        print("=" * 70)

        # Futures contracts
        self.contracts = {
            'ES': {  # E-mini S&P 500
                'name': 'E-mini S&P 500',
                'point_value': 50,  # $50 per point
                'tick_size': 0.25,  # Minimum price movement
                'symbol': 'ES',
                'margin': 12000,  # Approx margin per contract
            },
            'NQ': {  # E-mini Nasdaq
                'name': 'E-mini Nasdaq',
                'point_value': 20,  # $20 per point
                'tick_size': 0.25,
                'symbol': 'NQ',
                'margin': 15000,
            },
            'CL': {  # Crude Oil
                'name': 'Crude Oil',
                'point_value': 1000,  # $1,000 per point
                'tick_size': 0.01,
                'symbol': 'CL',
                'margin': 6000,
            },
            'GC': {  # Gold
                'name': 'Gold',
                'point_value': 100,  # $100 per point (per $1 move)
                'tick_size': 0.10,
                'symbol': 'GC',
                'margin': 8000,
            }
        }

        # Risk parameters (Apex-compatible)
        self.account_size = 150000  # Apex PA account size
        self.min_score = 3.0  # Quality filter (slightly higher for futures)
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_positions = 2  # Max 2 futures positions at once

        # Time-based entry filters (futures market hours)
        self.BEST_HOURS = {
            'ES': [9, 10, 11, 12, 13, 14, 15],  # US market hours
            'NQ': [9, 10, 11, 12, 13, 14, 15],
            'CL': [8, 9, 10, 11, 12, 13],  # Energy trading hours
            'GC': [8, 9, 10, 11, 12, 13],  # Metals trading hours
        }

        # Scanning interval
        self.scan_interval = 1800  # 30 minutes (futures move faster)

        print(f"Contracts: {', '.join(self.contracts.keys())}")
        print(f"Account Size: ${self.account_size:,}")
        print(f"Min Score: {self.min_score}/10")
        print(f"Risk Per Trade: {self.risk_per_trade * 100}%")
        print(f"Max Positions: {self.max_positions}")
        print(f"Shared Libraries: TA-Lib, Kelly, MTF")
        print("=" * 70)

    def get_futures_data(self, contract, granularity='1H', count=200):
        """
        Fetch historical futures data

        NOTE: This is a placeholder. In production, you would connect to:
        - Apex Trader Funding's data feed (Rithmic, CQG, or similar)
        - Or use a data provider like IQFeed, CQG, Rithmic

        For now, returns simulated data structure
        """
        # TODO: Replace with actual Apex/Rithmic data feed
        # Example structure:
        return {
            'closes': np.random.randn(count) * 10 + 4500,  # Simulated
            'highs': np.random.randn(count) * 10 + 4510,
            'lows': np.random.randn(count) * 10 + 4490,
            'current_price': 4500.0,  # Simulated
            'volume': np.random.randint(1000, 10000, count)
        }

    def calculate_score(self, contract, data):
        """
        Calculate trading score using SHARED technical analysis
        Returns: dict with score, direction, signals

        Uses IDENTICAL logic to forex bot, just different data source
        """
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

        # Use shared TA library (SAME as forex)
        rsi = ta.calculate_rsi(closes)
        macd = ta.calculate_macd(closes)
        ema_fast = ta.calculate_ema(closes, period=10)
        ema_slow = ta.calculate_ema(closes, period=21)
        adx = ta.calculate_adx(highs, lows, closes)
        atr = ta.calculate_atr(highs, lows, closes)

        # Calculate 4H trend using shared MTF library (SAME as forex)
        data_4h = self.get_futures_data(contract, granularity='4H', count=100)
        trend_4h = mtf.get_higher_timeframe_trend(data_4h['closes'], current_price)

        # Score LONG signals (IDENTICAL to forex logic)
        long_score = 0
        long_signals = []

        if rsi < 30:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")
        elif rsi < 40:
            long_score += 1
            long_signals.append("RSI_LOW")

        if macd['macd'] > macd['signal']:
            long_score += 2
            long_signals.append("MACD_BULL_CROSS")

        if current_price > ema_fast and ema_fast > ema_slow:
            long_score += 2
            long_signals.append("EMA_BULLISH")

        if adx > 25:
            long_score += 1
            long_signals.append("STRONG_TREND")

        if trend_4h == 'bullish':
            long_score += 2
            long_signals.append("4H_BULLISH_TREND")

        # Score SHORT signals (IDENTICAL to forex logic)
        short_score = 0
        short_signals = []

        if rsi > 70:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")

        if macd['macd'] < macd['signal']:
            short_score += 2
            short_signals.append("MACD_BEAR_CROSS")

        if current_price < ema_fast and ema_fast < ema_slow:
            short_score += 2
            short_signals.append("EMA_BEARISH")

        if adx > 25:
            short_score += 1
            short_signals.append("STRONG_TREND")

        if trend_4h == 'bearish':
            short_score += 2
            short_signals.append("4H_BEARISH_TREND")

        # Determine best direction
        if long_score > short_score and long_score >= self.min_score:
            return {
                'score': long_score,
                'direction': 'long',
                'signals': long_signals,
                'rsi': rsi,
                'adx': adx,
                'contract': contract,
                'current_price': current_price
            }
        elif short_score >= self.min_score:
            return {
                'score': short_score,
                'direction': 'short',
                'signals': short_signals,
                'rsi': rsi,
                'adx': adx,
                'contract': contract,
                'current_price': current_price
            }
        else:
            return {'score': 0, 'direction': None, 'contract': contract}

    def calculate_position_size(self, contract_name, score):
        """
        Calculate number of contracts using SHARED Kelly Criterion
        """
        contract_info = self.contracts[contract_name]

        # Use shared Kelly library
        position_data = kelly.calculate_position_size(
            technical_score=score,
            fundamental_score=0,  # No fundamentals for futures yet
            account_balance=self.account_size,
            risk_per_trade=self.risk_per_trade
        )

        # Convert to contracts using shared Kelly function
        num_contracts = kelly.calculate_futures_contracts(
            position_size_dollars=position_data['final_size'],
            point_value=contract_info['point_value'],
            current_price=4500  # Would use actual price in production
        )

        return {
            'num_contracts': num_contracts,
            'kelly_multiplier': position_data['kelly_multiplier'],
            'confidence': position_data['confidence']
        }

    def scan_futures(self):
        """Scan all futures contracts for opportunities"""
        print(f"\n[FUTURES SCAN] {datetime.now().strftime('%H:%M:%S')}")

        opportunities = []
        current_hour = datetime.now().hour

        for contract_name in self.contracts.keys():
            # Time filter (only trade during best hours)
            if current_hour not in self.BEST_HOURS.get(contract_name, []):
                print(f"  [SKIP] {contract_name} - Outside best hours (current: {current_hour})")
                continue

            data = self.get_futures_data(contract_name)
            result = self.calculate_score(contract_name, data)

            if result['score'] >= self.min_score:
                print(f"  [FOUND] {contract_name} {result['direction'].upper()}: {result['score']:.1f}/10")
                print(f"          Signals: {', '.join(result['signals'])}")

                # Calculate position size
                position = self.calculate_position_size(contract_name, result['score'])
                result['num_contracts'] = position['num_contracts']
                result['confidence'] = position['confidence']

                opportunities.append(result)
            else:
                print(f"  [SKIP] {contract_name}: Score {result['score']:.1f}/10")

        return opportunities

    def run(self):
        """Main trading loop"""
        print("\n[STARTING FUTURES SYSTEM]")
        print(f"Scanning every {self.scan_interval/60:.0f} minutes")
        print("Press Ctrl+C to stop\n")

        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")

            opportunities = self.scan_futures()

            if opportunities:
                print(f"\n[FOUND {len(opportunities)} opportunities]")
                for opp in opportunities:
                    print(f"  {opp['contract']} {opp['direction'].upper()}: {opp['num_contracts']} contracts")
                # TODO: Execute trades (connect to Apex platform)
            else:
                print("\n[No opportunities found]")

            print(f"\n[Next scan in {self.scan_interval/60:.0f} minutes...]")
            time.sleep(self.scan_interval)


if __name__ == "__main__":
    trader = FuturesTraderApex()
    trader.run()
