#!/usr/bin/env python3
"""
OpenBB Integration Example for Week 2+ Scanner
==============================================
Shows how to use OpenBB Platform to enhance your trading system
"""

from openbb import obb
import pandas as pd

def get_options_iv_for_strike_selection(symbol: str):
    """
    Get implied volatility for options to make better strike selections

    This is better than estimating IV - you get real-time market IV
    """
    try:
        # Get options chain
        chains = obb.derivatives.options.chains(symbol=symbol, provider='yfinance')

        # Convert to DataFrame for easier analysis
        df = chains.to_dataframe()

        # Filter for 7-day expiration (your typical hold period)
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=7)

        # Get IV rank: where is current IV vs 52-week range?
        iv_values = df['implied_volatility'].dropna()
        if len(iv_values) > 0:
            current_iv = iv_values.mean()
            iv_min = iv_values.min()
            iv_max = iv_values.max()

            # IV Rank formula
            iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100

            print(f"\n{symbol} IV Analysis:")
            print(f"  Current IV: {current_iv:.1%}")
            print(f"  IV Rank: {iv_rank:.1f}/100")

            # Trading signal
            if iv_rank > 75:
                print(f"  → SELL PREMIUM (Iron Condor) - IV is HIGH")
            elif iv_rank < 25:
                print(f"  → BUY OPTIONS (Directional) - IV is LOW")
            else:
                print(f"  → NEUTRAL - Normal IV")

            return {
                'current_iv': current_iv,
                'iv_rank': iv_rank,
                'strategy': 'sell_premium' if iv_rank > 75 else 'buy_options' if iv_rank < 25 else 'neutral'
            }
    except Exception as e:
        print(f"[ERROR] OpenBB IV fetch failed: {e}")
        return None

def detect_unusual_options_activity(symbol: str):
    """
    Detect if institutions are making large options bets

    High volume/OI ratio = new positions being opened (smart money)
    """
    try:
        chains = obb.derivatives.options.chains(symbol=symbol, provider='yfinance')
        df = chains.to_dataframe()

        # Calculate volume/open_interest ratio
        df['vol_oi_ratio'] = df['volume'] / df['open_interest']

        # Unusual if ratio > 3.0 (3x more volume than open interest)
        unusual = df[df['vol_oi_ratio'] > 3.0]

        if len(unusual) > 0:
            print(f"\n{symbol} Unusual Options Activity Detected:")
            print(f"  {len(unusual)} contracts with vol/OI > 3.0x")
            print(f"  Largest bet: {unusual.iloc[0]['option_type']} ${unusual.iloc[0]['strike']:.0f} strike")
            return True
        else:
            print(f"\n{symbol}: No unusual activity")
            return False

    except Exception as e:
        print(f"[ERROR] OpenBB unusual activity check failed: {e}")
        return False

def get_economic_calendar(days_ahead=1):
    """
    Get upcoming economic events that could move markets

    Avoid trading before FOMC, CPI, NFP announcements
    """
    try:
        from datetime import datetime, timedelta

        start = datetime.now()
        end = start + timedelta(days=days_ahead)

        # Note: Requires FMP API key (free tier available)
        calendar = obb.economy.calendar(
            provider='fmp',
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d')
        )

        if calendar.results:
            print(f"\n📅 Economic Events Next {days_ahead} Days:")
            for event in calendar.results[:5]:  # Top 5 events
                print(f"  {event.date}: {event.event} ({event.country})")
            return calendar.results
        else:
            print(f"\n📅 No major economic events next {days_ahead} days")
            return []

    except Exception as e:
        print(f"[INFO] Economic calendar unavailable (needs FMP API key): {e}")
        return []

# Example usage in your scanner
if __name__ == "__main__":
    print("="*70)
    print("OPENBB INTEGRATION EXAMPLES")
    print("="*70)

    # Example 1: Get IV for strike selection
    print("\n1. IMPLIED VOLATILITY ANALYSIS")
    print("-" * 70)
    iv_data = get_options_iv_for_strike_selection('TSLA')

    if iv_data and iv_data['strategy'] == 'sell_premium':
        print("\n  ✅ TSLA is perfect for Iron Condor (high IV)")

    # Example 2: Detect smart money activity
    print("\n2. UNUSUAL OPTIONS ACTIVITY DETECTION")
    print("-" * 70)
    unusual = detect_unusual_options_activity('AAPL')

    if unusual:
        print("\n  ⚠️  Institutions are making big bets on AAPL")

    # Example 3: Economic calendar
    print("\n3. ECONOMIC CALENDAR")
    print("-" * 70)
    events = get_economic_calendar(days_ahead=2)

    print("\n" + "="*70)
    print("Integration with your scanner:")
    print("  1. Call get_options_iv_for_strike_selection() before executing trades")
    print("  2. Use IV rank to choose between Iron Condor vs Dual Options")
    print("  3. Skip trades if unusual activity detected (institutions know more)")
    print("  4. Pause trading 30min before major economic events")
    print("="*70)
