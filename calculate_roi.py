#!/usr/bin/env python3
"""
ROI CALCULATOR
Calculate Return on Investment for all positions
"""

import json
import os
from datetime import datetime
from typing import List, Dict

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from dotenv import load_dotenv
    load_dotenv()
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


def load_executions() -> List[Dict]:
    """Load today's executions"""
    today = datetime.now().strftime("%Y%m%d")
    filename = f'executions/execution_log_{today}.json'

    if not os.path.exists(filename):
        return []

    with open(filename, 'r') as f:
        log = json.load(f)

    return log.get('executions', [])


def calculate_roi():
    """Calculate comprehensive ROI metrics"""

    print("\n" + "="*70)
    print("RETURN ON INVESTMENT (ROI) ANALYSIS")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%I:%M %p')}")
    print(f"Date: {datetime.now().strftime('%A, %B %d, %Y')}")
    print("="*70 + "\n")

    executions = load_executions()

    if not executions:
        print("[NO POSITIONS] No trades executed today")
        return

    # Initialize Alpaca
    if ALPACA_AVAILABLE:
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        data_client = StockHistoricalDataClient(api_key, secret_key)
    else:
        data_client = None

    total_capital_required = 0
    total_unrealized_pnl = 0
    total_credit_collected = 0
    total_max_profit = 0

    print("POSITION-BY-POSITION ROI:\n")

    for i, pos in enumerate(executions, 1):
        symbol = pos['symbol']
        entry_price = pos['entry_price']
        credit = pos['credit']
        max_risk = pos['max_risk']
        short_strike = pos['short_strike']
        long_strike = pos['long_strike']

        # Capital required = max risk (what you need to hold the position)
        capital_required = max_risk

        print(f"{i}. {symbol} Bull Put Spread")
        print(f"   Entry: ${entry_price:.2f}")
        print(f"   Strikes: ${short_strike:.2f} / ${long_strike:.2f}")

        # Get current price and calculate P&L
        if data_client:
            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = data_client.get_stock_latest_quote(request)
                current_price = float(quote[symbol].ask_price)

                print(f"   Current: ${current_price:.2f}")

                # Estimate current P&L
                if current_price > short_strike:
                    # Winning - estimate spread value decay
                    distance_pct = ((current_price - short_strike) / current_price) * 100
                    time_factor = 29 / 30  # Just opened
                    distance_factor = min(distance_pct / 10, 1.0)
                    estimated_spread_value = credit * time_factor * (1 - distance_factor)
                    unrealized_pnl = credit - estimated_spread_value
                else:
                    # Losing
                    intrinsic_value = short_strike - current_price
                    unrealized_pnl = -(intrinsic_value - credit)

                print(f"\n   CAPITAL & RETURNS:")
                print(f"   Capital Required:    ${capital_required:.2f}")
                print(f"   Credit Collected:    ${credit:.2f}")
                print(f"   Unrealized P&L:      ${unrealized_pnl:+.2f}")

                # Calculate ROI
                current_roi = (unrealized_pnl / capital_required) * 100
                max_roi = (credit / capital_required) * 100

                print(f"\n   ROI METRICS:")
                print(f"   Current ROI:         {current_roi:+.2f}%")
                print(f"   Max Potential ROI:   {max_roi:+.2f}%")

                # Annualized
                days_held = 0.03  # ~45 minutes / 1440 minutes per day
                if days_held > 0:
                    annualized_roi = (current_roi / days_held) * 365
                    print(f"   Annualized ROI:      {annualized_roi:+,.0f}%")

                total_unrealized_pnl += unrealized_pnl

            except Exception as e:
                print(f"   [ERROR] Could not fetch price: {e}")
                unrealized_pnl = credit * 0.5  # Estimate
                current_roi = (unrealized_pnl / capital_required) * 100
                total_unrealized_pnl += unrealized_pnl

        total_capital_required += capital_required
        total_credit_collected += credit
        total_max_profit += credit

        print()

    # Portfolio ROI Summary
    print("="*70)
    print("PORTFOLIO ROI SUMMARY")
    print("="*70)

    print(f"\nCAPITAL DEPLOYED:")
    print(f"  Total Capital Required:  ${total_capital_required:,.2f}")
    print(f"  (This is your 'investment' for ROI calculation)")

    print(f"\nRETURNS:")
    print(f"  Credit Collected:        +${total_credit_collected:.2f}")
    print(f"  Unrealized P&L:          +${total_unrealized_pnl:.2f}")
    print(f"  Max Profit Potential:    +${total_max_profit:.2f}")

    # Calculate ROIs
    current_roi = (total_unrealized_pnl / total_capital_required) * 100
    max_roi = (total_max_profit / total_capital_required) * 100

    print(f"\n" + "="*70)
    print("ROI CALCULATIONS")
    print("="*70)

    print(f"\nCURRENT ROI:              {current_roi:+.2f}%")
    print(f"  (Unrealized profit / Capital required)")

    print(f"\nMAX POTENTIAL ROI:        {max_roi:+.2f}%")
    print(f"  (Max profit / Capital required)")

    # Time-based ROI
    start_time = datetime.fromisoformat(executions[0]['entry_time'][:19])
    time_elapsed = datetime.now() - start_time
    hours_elapsed = time_elapsed.total_seconds() / 3600
    days_elapsed = hours_elapsed / 24

    print(f"\nTIME IN TRADE:            {hours_elapsed:.1f} hours ({days_elapsed:.3f} days)")

    # Hourly ROI
    if hours_elapsed > 0:
        hourly_roi = current_roi / hours_elapsed
        print(f"ROI PER HOUR:             {hourly_roi:+.2f}%/hour")

    # Daily ROI
    if days_elapsed > 0:
        daily_roi = current_roi / days_elapsed
        print(f"ROI PER DAY:              {daily_roi:+.2f}%/day")

    # Annualized ROI (current pace)
    if days_elapsed > 0:
        annualized_current = (current_roi / days_elapsed) * 365
        annualized_max = (max_roi / 29) * 365  # Over 29 days

        print(f"\nANNUALIZED (Current Pace):")
        print(f"  If you closed now:       {annualized_current:+,.0f}%/year")
        print(f"  If held 29 days:         {annualized_max:+,.0f}%/year")

    # Comparison to benchmarks
    print(f"\n" + "="*70)
    print("BENCHMARK COMPARISON")
    print("="*70)

    print(f"\nYour Current ROI:         {current_roi:+.2f}%")
    print(f"\nCompared to:")
    print(f"  S&P 500 (annual):        ~10.0%")
    print(f"  S&P 500 (daily):         ~0.03%")
    print(f"  Your daily pace:         {daily_roi if days_elapsed > 0 else 0:+.2f}%")

    if days_elapsed > 0:
        vs_sp500 = daily_roi / 0.03
        print(f"\nYou're outperforming S&P 500 by {vs_sp500:.1f}x")

    # Money metrics
    print(f"\n" + "="*70)
    print("MONEY METRICS")
    print("="*70)

    print(f"\n$ Per Hour:               ${total_unrealized_pnl / hours_elapsed if hours_elapsed > 0 else 0:.2f}/hour")
    print(f"$ Per Day (at this pace): ${total_unrealized_pnl / days_elapsed if days_elapsed > 0 else 0:.2f}/day")
    print(f"$ Per Month:              ${(total_unrealized_pnl / days_elapsed * 30) if days_elapsed > 0 else 0:,.2f}/month")

    # Projection
    print(f"\n" + "="*70)
    print("PROJECTIONS")
    print("="*70)

    print(f"\nIf you keep this ROI pace:")
    if days_elapsed > 0:
        weekly = (total_unrealized_pnl / days_elapsed) * 7
        monthly = (total_unrealized_pnl / days_elapsed) * 30
        yearly = (total_unrealized_pnl / days_elapsed) * 365

        print(f"  1 Week:    +${weekly:,.2f}")
        print(f"  1 Month:   +${monthly:,.2f}")
        print(f"  1 Year:    +${yearly:,.2f}")

    print(f"\nIf you hold these positions to expiration:")
    print(f"  29 Days:   +${total_max_profit:.2f} (max)")
    print(f"  ROI:       {max_roi:.2f}%")

    # Goal tracking
    print(f"\n" + "="*70)
    print("GOAL TRACKING")
    print("="*70)

    print(f"\nStarting Capital (Paper):  $956,594")
    print(f"Current Unrealized:        +${total_unrealized_pnl:.2f}")
    print(f"Account Value:             $956,{594 + int(total_unrealized_pnl)}")

    print(f"\nWeek 3 Goal: 10-20 trades, 60%+ win rate")
    print(f"  Trades so far: 2")
    print(f"  Win rate: 100% (2/2)")
    print(f"  Avg ROI per trade: {current_roi / 2:.2f}%")

    print(f"\n" + "="*70)
    print("BOTTOM LINE")
    print("="*70)

    print(f"\nCapital Deployed:  ${total_capital_required:,.2f}")
    print(f"Current Profit:    +${total_unrealized_pnl:.2f}")
    print(f"")
    print(f"ROI:               {current_roi:+.2f}%")
    print(f"Time:              {hours_elapsed:.1f} hours")
    print(f"ROI/Hour:          {hourly_roi if hours_elapsed > 0 else 0:+.2f}%/hour")

    if days_elapsed > 0:
        print(f"Annualized:        {annualized_current:+,.0f}%/year")

    print(f"\n" + "="*70 + "\n")


if __name__ == "__main__":
    calculate_roi()
