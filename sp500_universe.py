#!/usr/bin/env python3
"""
S&P 500 TICKER UNIVERSE
=======================
Get complete S&P 500 ticker list for Week 2 expanded scanning
"""

import pandas as pd
import json
from datetime import datetime

def get_sp500_tickers():
    """Get S&P 500 ticker list from Wikipedia"""

    print("=" * 70)
    print("FETCHING S&P 500 TICKER UNIVERSE")
    print("=" * 70)

    try:
        # Get S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]

        # Extract tickers
        tickers = sp500_table['Symbol'].tolist()

        # Clean tickers (remove any special characters)
        tickers = [ticker.replace('.', '-') for ticker in tickers]

        print(f"\n[SUCCESS] Retrieved {len(tickers)} S&P 500 tickers")

        # Get sector information too
        sectors = {}
        for idx, row in sp500_table.iterrows():
            ticker = row['Symbol'].replace('.', '-')
            sectors[ticker] = {
                'sector': row['GICS Sector'],
                'industry': row['GICS Sub-Industry'],
                'company': row['Security']
            }

        return tickers, sectors

    except Exception as e:
        print(f"[ERROR] Failed to fetch from Wikipedia: {e}")
        print("\n[FALLBACK] Using manual S&P 500 list...")

        # Fallback: Top 100 most liquid S&P 500 stocks
        fallback_tickers = [
            # Mega Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
            'AMD', 'INTC', 'ADBE', 'CRM', 'NFLX', 'AVGO', 'ORCL', 'CSCO',
            'QCOM', 'TXN', 'INTU', 'NOW', 'AMAT', 'MU', 'LRCX', 'KLAC',

            # Financials
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'SCHW', 'BLK',
            'SPGI', 'CB', 'MMC', 'PGR', 'TFC', 'USB', 'PNC', 'COF',

            # Healthcare
            'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT',
            'DHR', 'CVS', 'BMY', 'AMGN', 'MDT', 'GILD', 'ISRG', 'CI',

            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'DG', 'COST',
            'PG', 'KO', 'PEP', 'PM', 'MO', 'EL', 'CL', 'KHC',

            # Industrials
            'BA', 'CAT', 'GE', 'UNP', 'HON', 'UPS', 'RTX', 'LMT', 'DE',
            'MMM', 'EMR', 'ETN', 'ITW', 'GD', 'CSX', 'NSC', 'FDX',

            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY',

            # Communication
            'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',

            # Utilities & Real Estate
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'PLD', 'AMT', 'CCI',

            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DD', 'DOW'
        ]

        return fallback_tickers, {}


def filter_for_options_trading(tickers, sectors=None):
    """
    Filter S&P 500 for best options trading candidates

    Criteria:
    - High liquidity
    - Active options markets
    - Reasonable volatility
    """

    print("\n" + "=" * 70)
    print("FILTERING FOR OPTIONS TRADING")
    print("=" * 70)

    # High priority sectors for options (high volatility, good premiums)
    priority_sectors = [
        'Information Technology',
        'Communication Services',
        'Consumer Discretionary',
        'Financials',
        'Health Care',
        'Energy'
    ]

    # Always include mega caps (most liquid options)
    must_include = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'AMD', 'INTC', 'JPM', 'BAC', 'GS', 'UNH', 'XOM', 'CVX'
    ]

    filtered = must_include.copy()

    # Add from priority sectors
    if sectors:
        for ticker, info in sectors.items():
            if ticker not in filtered and info['sector'] in priority_sectors:
                filtered.append(ticker)
    else:
        # If no sector info, just use all tickers
        filtered = tickers

    print(f"\n[FILTERED] {len(filtered)} tickers optimized for options trading")

    # Organize by sector
    if sectors:
        by_sector = {}
        for ticker in filtered:
            if ticker in sectors:
                sector = sectors[ticker]['sector']
                if sector not in by_sector:
                    by_sector[sector] = []
                by_sector[sector].append(ticker)

        print("\nBREAKDOWN BY SECTOR:")
        for sector, tickers_list in sorted(by_sector.items()):
            print(f"  {sector}: {len(tickers_list)} stocks")

    return filtered


def save_ticker_universe(tickers, sectors, filepath='sp500_tickers.json'):
    """Save ticker universe to JSON file"""

    data = {
        'generated': datetime.now().isoformat(),
        'total_tickers': len(tickers),
        'tickers': tickers,
        'sectors': sectors
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n[SAVED] Ticker universe saved to {filepath}")
    print(f"Total tickers: {len(tickers)}")


def main():
    """Get and save S&P 500 ticker universe"""

    # Get S&P 500 tickers
    all_tickers, sectors = get_sp500_tickers()

    # Filter for options trading
    options_tickers = filter_for_options_trading(all_tickers, sectors)

    # Save full list
    save_ticker_universe(all_tickers, sectors, 'sp500_full.json')

    # Save filtered list for options
    save_ticker_universe(options_tickers, sectors, 'sp500_options_filtered.json')

    print("\n" + "=" * 70)
    print("WEEK 2 TICKER UNIVERSE READY")
    print("=" * 70)
    print(f"\nFull S&P 500: {len(all_tickers)} tickers")
    print(f"Options-filtered: {len(options_tickers)} tickers")
    print("\nFiles created:")
    print("  1. sp500_full.json - Complete S&P 500")
    print("  2. sp500_options_filtered.json - Best for options trading")

    print("\n[READY] Week 2 scanner can now scan entire S&P 500!")

    # Show some examples
    print("\nEXAMPLE TICKERS:")
    print("Tech:", ', '.join(options_tickers[:10]))
    if len(options_tickers) > 10:
        print("Finance:", ', '.join([t for t in options_tickers if t in ['JPM', 'BAC', 'GS', 'MS', 'C']][:5]))
        print("Healthcare:", ', '.join([t for t in options_tickers if t in ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK']][:5]))


if __name__ == "__main__":
    main()
