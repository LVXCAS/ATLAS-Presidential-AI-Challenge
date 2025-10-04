#!/usr/bin/env python3
"""Quick market check to see current opportunities"""

import os
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

# Load credentials
load_dotenv('.env.paper')

api = StockHistoricalDataClient(
    api_key=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY')
)

# Check R&D discoveries
symbols = ['INTC', 'AMD', 'NVDA', 'AAPL', 'MSFT']

print("CURRENT MARKET CHECK - WEDNESDAY OCT 1")
print("=" * 60)
print("Threshold: 4.0+ (was 4.5)")
print()

for symbol in symbols:
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=30)
        )
        bars = api.get_stock_bars(request).df

        if bars.empty:
            continue

        current_price = float(bars['close'].iloc[-1])
        volume = float(bars['volume'].iloc[-1])

        # Calculate volatility
        returns = bars['close'].pct_change().dropna()
        volatility = float(returns.std()) if len(returns) > 5 else 0.02

        # Simple scoring (from Week1ExecutionSystem)
        volume_score = min(volume / 50_000_000, 1.0) * 2.0
        vol_score = min(volatility * 50, 2.0)
        price_score = 1.0 if current_price > 20 else 0.5
        total_score = volume_score + vol_score + price_score

        status = "[QUALIFIED]" if total_score >= 4.0 else "           "
        print(f"{status} {symbol}: ${current_price:.2f} - Score: {total_score:.2f} [Vol: {volume:,.0f}, IV: {volatility:.3f}]")

    except Exception as e:
        print(f"    {symbol}: Error - {e}")

print()
print("Scanner is running with these settings")
print("Will execute first qualified opportunity at market open")
