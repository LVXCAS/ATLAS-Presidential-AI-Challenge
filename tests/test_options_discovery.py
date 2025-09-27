"""
TEST OPTIONS DISCOVERY SYSTEM
Quick validation of stock screening and options analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def test_single_stock_analysis(ticker="AAPL"):
    """Test options analysis on a single known stock"""

    print(f"Testing options discovery for {ticker}...")

    try:
        # Get basic stock data
        stock = yf.Ticker(ticker)
        info = stock.info

        print(f"Market Cap: ${info.get('marketCap', 0):,}")
        print(f"Current Price: ${info.get('currentPrice', 0):.2f}")
        print(f"Average Volume: {info.get('averageVolume', 0):,}")

        # Get recent price data
        hist = stock.history(period='1mo')
        if len(hist) >= 5:
            current_price = hist['Close'].iloc[-1]
            week_ago_price = hist['Close'].iloc[-5]
            momentum = (current_price / week_ago_price - 1)

            print(f"5-Day Momentum: {momentum:+.2%}")

        # Get options data
        options_dates = stock.options
        print(f"Available option expiration dates: {len(options_dates)}")

        if options_dates:
            # Skip options expiring tomorrow (0 DTE), use next available
            valid_exp = None
            for exp_date in options_dates:
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                dte = (exp_datetime - datetime.now()).days
                if dte >= 7:  # At least 7 days out
                    valid_exp = exp_date
                    break

            if not valid_exp:
                print("No suitable expiration dates found (need at least 7 DTE)")
                return False

            print(f"Using expiration: {valid_exp}")

            chain = stock.option_chain(valid_exp)
            calls = chain.calls
            puts = chain.puts

            print(f"Available calls: {len(calls)}")
            print(f"Available puts: {len(puts)}")

            if not calls.empty:
                # Calculate average implied volatility
                avg_iv = calls['impliedVolatility'].mean()
                print(f"Average Implied Volatility: {avg_iv:.1%}")

                # Find OTM calls for covered call analysis
                otm_calls = calls[calls['strike'] > current_price * 1.02]
                print(f"OTM calls (>2% above current): {len(otm_calls)}")

                if not otm_calls.empty:
                    # Analyze top covered call opportunity
                    best_call = otm_calls.iloc[0]
                    strike = best_call['strike']
                    premium = best_call['lastPrice']

                    # Calculate returns
                    exp_datetime = datetime.strptime(valid_exp, '%Y-%m-%d')
                    dte = (exp_datetime - datetime.now()).days

                    cost_basis = current_price
                    max_profit = (strike - cost_basis) + premium
                    max_return = (max_profit / cost_basis) * (365 / dte) if dte > 0 else 0

                    print(f"\nBEST COVERED CALL OPPORTUNITY:")
                    print(f"Strike: ${strike:.2f}")
                    print(f"Premium: ${premium:.2f}")
                    print(f"Days to expiration: {dte}")
                    print(f"Annualized return: {max_return:.1%}")

                    if max_return > 0.15:  # 15% threshold
                        print("SUCCESS - MEETS CRITERIA - Would be selected for execution")
                    else:
                        print("BELOW 15% threshold")

        return True

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return False

def test_discovery_criteria():
    """Test discovery on multiple tickers"""

    print("\nTESTING DISCOVERY ACROSS MULTIPLE STOCKS")
    print("=" * 50)

    # Test on high-volume stocks that should have options
    test_tickers = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ']

    opportunities_found = 0

    for ticker in test_tickers:
        print(f"\n--- Analyzing {ticker} ---")
        success = test_single_stock_analysis(ticker)
        if success:
            opportunities_found += 1

    print(f"\nSUMMARY: {opportunities_found}/{len(test_tickers)} stocks analyzed successfully")

    return opportunities_found > 0

if __name__ == "__main__":
    print("AUTONOMOUS OPTIONS DISCOVERY - VALIDATION TEST")
    print("=" * 60)

    # Test single stock first
    test_single_stock_analysis("AAPL")

    # Test discovery criteria
    validation_passed = test_discovery_criteria()

    if validation_passed:
        print("\nOPTIONS DISCOVERY SYSTEM VALIDATED!")
        print("Ready to integrate with main autonomous systems")
    else:
        print("\nIssues detected - need to debug discovery logic")