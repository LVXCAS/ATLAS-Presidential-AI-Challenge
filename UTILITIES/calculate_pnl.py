#!/usr/bin/env python3
"""
P&L CALCULATOR
Calculate current profit/loss on all positions
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


def calculate_pnl():
    """Calculate P&L on all positions"""

    print("\n" + "="*70)
    print("PROFIT & LOSS CALCULATOR")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%I:%M %p')}")
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
        print("[WARNING] Alpaca not available\n")

    total_credit_collected = 0
    total_max_profit = 0
    total_unrealized_pnl = 0
    total_max_risk = 0

    print("POSITION BREAKDOWN:\n")

    for i, pos in enumerate(executions, 1):
        symbol = pos['symbol']
        entry_price = pos['entry_price']
        credit = pos['credit']
        max_risk = pos['max_risk']
        short_strike = pos['short_strike']

        print(f"{i}. {symbol} Bull Put Spread")
        print(f"   Entry Price: ${entry_price:.2f}")
        print(f"   Short Strike: ${short_strike:.2f}")

        # Get current price
        if data_client:
            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = data_client.get_stock_latest_quote(request)
                current_price = float(quote[symbol].ask_price)

                print(f"   Current Price: ${current_price:.2f}")

                # Calculate stock movement
                stock_change = current_price - entry_price
                stock_change_pct = (stock_change / entry_price) * 100

                print(f"   Stock Movement: ${stock_change:+.2f} ({stock_change_pct:+.2f}%)")

                # Calculate distance to short strike
                distance_to_strike = current_price - short_strike
                distance_pct = (distance_to_strike / current_price) * 100

                print(f"   Distance to Strike: ${distance_to_strike:+.2f} ({distance_pct:+.1f}%)")

                # Estimate P&L for Bull Put Spread
                # For a Bull Put Spread, as the stock moves UP or stays flat:
                # - The spread loses value (good for us - we sold it)
                # - Max profit = credit collected (if stock stays above short strike)
                # - Max loss = spread width - credit (if stock drops below long strike)

                if current_price > short_strike:
                    # Winning position
                    # Estimate current spread value based on time and distance
                    days_elapsed = 0  # Just opened today
                    days_to_exp = 29

                    # Very rough estimate: spread value decays as we get closer to expiration
                    # and as stock moves further from strikes
                    time_factor = days_to_exp / 30  # ~0.97 (just opened)
                    distance_factor = min(distance_pct / 10, 1.0)  # More distance = less value

                    # Estimated spread value (rough approximation)
                    estimated_spread_value = credit * time_factor * (1 - distance_factor)

                    unrealized_pnl = credit - estimated_spread_value

                    print(f"   Status: [WIN] WINNING")
                    print(f"   Credit Collected: ${credit:.2f}")
                    print(f"   Estimated Spread Value: ${estimated_spread_value:.2f}")
                    print(f"   Unrealized P&L: +${unrealized_pnl:.2f}")
                    print(f"   Max Profit Remaining: ${credit:.2f} (if stock stays above ${short_strike:.2f})")

                    total_unrealized_pnl += unrealized_pnl

                else:
                    # Losing position
                    print(f"   Status: [LOSS] LOSING")
                    print(f"   Credit Collected: ${credit:.2f}")
                    print(f"   ALERT: Stock below short strike!")

                    # Calculate potential loss
                    intrinsic_value = short_strike - current_price
                    estimated_loss = intrinsic_value - credit

                    print(f"   Estimated Loss: -${estimated_loss:.2f}")
                    total_unrealized_pnl -= estimated_loss

            except Exception as e:
                print(f"   [ERROR] Could not fetch price: {e}")
                # Default to credit collected
                total_unrealized_pnl += credit

        total_credit_collected += credit
        total_max_profit += credit
        total_max_risk += max_risk

        print()

    # Summary
    print("="*70)
    print("TOTAL PORTFOLIO P&L")
    print("="*70)
    print(f"\nCREDIT COLLECTED: +${total_credit_collected:.2f}")
    print(f"  (Money in your account RIGHT NOW)\n")

    if total_unrealized_pnl > 0:
        print(f"UNREALIZED P&L: +${total_unrealized_pnl:.2f}")
        print(f"  (Current profit if you closed positions now)\n")
    else:
        print(f"UNREALIZED P&L: -${abs(total_unrealized_pnl):.2f}")
        print(f"  (Current loss if you closed positions now)\n")

    print(f"MAX PROFIT POTENTIAL: +${total_max_profit:.2f}")
    print(f"  (If all positions expire worthless)\n")

    print(f"MAX RISK: ${total_max_risk:.2f}")
    print(f"  (Worst case scenario)\n")

    profit_pct = (total_unrealized_pnl / total_max_risk * 100) if total_max_risk > 0 else 0
    print(f"RETURN ON RISK: {profit_pct:+.1f}%")
    print(f"  (Current profit as % of risk)\n")

    # Time analysis
    print("="*70)
    print("TIME DECAY (THETA)")
    print("="*70)
    print(f"\nDays to Expiration: 29 days")
    print(f"Time Elapsed Today: ~15 minutes")
    print(f"\nDaily Theta Decay: ~${total_max_profit / 29:.2f}/day")
    print(f"  (You make ~${total_max_profit / 29:.2f} per day from time decay)\n")

    hourly_theta = (total_max_profit / 29) / 24
    print(f"Hourly Theta Decay: ~${hourly_theta:.2f}/hour")
    print(f"  (You make ~${hourly_theta:.2f} per hour from time decay)\n")

    print("="*70)
    print("BOTTOM LINE")
    print("="*70)

    if total_unrealized_pnl > 0:
        print(f"\n[WIN] YOU'RE UP +${total_unrealized_pnl:.2f}")
        print(f"\nCredit Collected: ${total_credit_collected:.2f}")
        print(f"Current Value: ~${total_credit_collected - total_unrealized_pnl:.2f}")
        print(f"Unrealized Profit: +${total_unrealized_pnl:.2f}")
    else:
        print(f"\n[LOSS] YOU'RE DOWN -${abs(total_unrealized_pnl):.2f}")

    print(f"\nIf both positions expire above strikes:")
    print(f"  Final Profit: +${total_max_profit:.2f}")
    print(f"  Return: {(total_max_profit / total_max_risk * 100):.1f}%")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    calculate_pnl()
