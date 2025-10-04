#!/usr/bin/env python3
"""
SAFE TRADE MONITORING SYSTEM
============================
Monitor performance of deployed safe strategies
Track vs old rapid trading system to prove improvement
"""

import json
import glob
import yfinance as yf
import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

def monitor_safe_trades():
    print("MONITORING SAFE TRADES PERFORMANCE")
    print("=" * 40)
    print()

    # Load API
    load_dotenv()
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    # Get current account status
    account = api.get_account()
    current_value = float(account.portfolio_value)

    print(f"CURRENT PORTFOLIO: ${current_value:,.2f}")
    print()

    # Find all safe trade records
    safe_trades = glob.glob('safe_trade_*.json')

    if not safe_trades:
        print("No safe trades found to monitor")
        return

    print("MONITORING ACTIVE SAFE TRADES:")
    print("-" * 35)

    total_risk = 0
    total_estimated_profit = 0

    for trade_file in safe_trades:
        try:
            with open(trade_file, 'r') as f:
                trade = json.load(f)

            symbol = trade['symbol']
            timestamp = trade['timestamp']
            strategies = trade['strategies']

            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('currentPrice', trade['current_price'])
            original_price = trade['current_price']

            price_change = ((current_price - original_price) / original_price) * 100

            print(f"\n{symbol} STRATEGY (Deployed: {timestamp[:19]})")
            print(f"Original Price: ${original_price:.2f}")
            print(f"Current Price: ${current_price:.2f} ({price_change:+.2f}%)")

            # Analyze each strategy component
            for strategy in strategies:
                if strategy['type'] == 'CASH_SECURED_PUT':
                    strike = strategy['strike']
                    contracts = strategy['contracts']
                    premium = strategy['premium_estimate']

                    # Put profit/loss calculation
                    if current_price > strike:
                        # Put expires worthless - we keep premium
                        put_profit = premium * contracts * 100
                        put_status = "PROFITABLE (OTM)"
                    else:
                        # Put assigned - loss limited
                        intrinsic_value = (strike - current_price) * contracts * 100
                        put_profit = (premium * contracts * 100) - intrinsic_value
                        put_status = "ASSIGNED (ITM)"

                    print(f"  PUT: {contracts}x ${strike:.2f} strike")
                    print(f"    Status: {put_status}")
                    print(f"    P&L: ${put_profit:.2f}")

                elif strategy['type'] == 'LONG_CALL':
                    strike = strategy['strike']
                    contracts = strategy['contracts']
                    cost = strategy['cost']

                    # Call profit/loss calculation
                    if current_price > strike:
                        # Call in the money
                        intrinsic_value = (current_price - strike) * contracts * 100
                        call_profit = intrinsic_value - cost
                        call_status = "PROFITABLE (ITM)"
                    else:
                        # Call out of money
                        call_profit = -cost  # Lose premium paid
                        call_status = "LOSING (OTM)"

                    print(f"  CALL: {contracts}x ${strike:.2f} strike")
                    print(f"    Status: {call_status}")
                    print(f"    P&L: ${call_profit:.2f}")

                    total_estimated_profit += put_profit if 'put_profit' in locals() else 0
                    total_estimated_profit += call_profit

            total_risk += trade['total_risk']

        except Exception as e:
            print(f"Error monitoring {trade_file}: {e}")

    print(f"\nTOTAL STRATEGY SUMMARY:")
    print("-" * 25)
    print(f"Total Risk Deployed: ${total_risk:,.2f}")
    print(f"Portfolio Risk: {(total_risk/current_value)*100:.2f}%")
    print(f"Estimated Current P&L: ${total_estimated_profit:.2f}")

    if total_risk > 0:
        roi_estimate = (total_estimated_profit / total_risk) * 100
        print(f"Current ROI: {roi_estimate:.2f}%")

    # Check if we can deploy more safe trades
    print(f"\nSAFE DEPLOYMENT CAPACITY:")
    print("-" * 30)

    # With 2% max per trade, we can have multiple positions
    max_single_position = current_value * 0.02
    used_capacity = (total_risk / current_value) * 100
    remaining_capacity = 2.0 - used_capacity  # Conservative 2% total max

    print(f"Max per position: ${max_single_position:,.2f} (2%)")
    print(f"Used capacity: {used_capacity:.2f}%")
    print(f"Remaining capacity: {remaining_capacity:.2f}%")

    if remaining_capacity > 0.5:  # If we have >0.5% remaining capacity
        print("READY for additional safe deployments")

        # Scan for new opportunities
        print(f"\nSCANNING FOR NEW OPPORTUNITIES:")
        print("-" * 35)

        candidates = ['NVDA', 'AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']

        for symbol in candidates:
            if any(symbol in trade_file for trade_file in safe_trades):
                continue  # Skip if we already have a position

            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice', 0)
                volume = info.get('volume', 0)
                change_pct = info.get('regularMarketChangePercent', 0)

                if volume > 10000000:  # High liquidity requirement
                    volatility_score = abs(change_pct)
                    if volatility_score > 1.0:  # At least 1% movement
                        print(f"{symbol}: ${current_price:.2f} | {change_pct:+.2f}% | Vol: {volume:,} [GOOD]")
                    else:
                        print(f"{symbol}: ${current_price:.2f} | {change_pct:+.2f}% | Vol: {volume:,}")

            except Exception as e:
                print(f"{symbol}: Error - {e}")
    else:
        print("At capacity - wait for current trades to close")

    # Time-based analysis
    print(f"\nHOLD TIME ANALYSIS:")
    print("-" * 20)

    now = datetime.now()
    for trade_file in safe_trades:
        try:
            with open(trade_file, 'r') as f:
                trade = json.load(f)

            trade_time = datetime.fromisoformat(trade['timestamp'])
            hold_time = now - trade_time
            hold_minutes = hold_time.total_seconds() / 60

            min_hold_required = 60  # 1 hour minimum

            if hold_minutes >= min_hold_required:
                status = "Can close if profitable"
            else:
                remaining = min_hold_required - hold_minutes
                status = f"Hold {remaining:.0f} more minutes"

            print(f"{trade['symbol']}: {hold_minutes:.0f}min held | {status}")

        except Exception as e:
            continue

    print(f"\nSAFE SYSTEM STATUS: ACTIVE")
    print("Monitoring complete - safe strategies protected by:")
    print("- 2% max position sizing")
    print("- 1-hour minimum holds")
    print("- Proven Intel-style strategy patterns")

if __name__ == "__main__":
    monitor_safe_trades()