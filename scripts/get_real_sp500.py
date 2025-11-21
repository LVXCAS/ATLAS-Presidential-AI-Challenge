#!/usr/bin/env python3
"""
REAL S&P 500 - ALL 503 STOCKS (from Wikipedia)
================================================
Complete and accurate S&P 500 list for Week 2 scanner
"""

import json
from datetime import datetime

def get_real_sp500():
    """
    Complete S&P 500 ticker list (all 503 stocks)
    Extracted from Wikipedia: List of S&P 500 companies
    Updated: October 3, 2025
    """

    sp500_tickers = [
        # All 503 tickers from Wikipedia
        'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A',
        'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL',
        'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK',
        'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT',
        'APP', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK',
        'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX',
        'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BK', 'BA', 'BKNG',
        'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW',
        'CDNS', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE',
        'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX',
        'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX',
        'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'CAG', 'COP', 'ED',
        'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA',
        'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY',
        'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR',
        'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN',
        'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EME', 'EMR', 'ETR',
        'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG',
        'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS',
        'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F',
        'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC',
        'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL',
        'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE',
        'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM',
        'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC',
        'IBKR', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV',
        'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'K', 'KVUE',
        'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR',
        'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN',
        'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MAR',
        'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK',
        'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK',
        'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI',
        'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN',
        'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY',
        'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR',
        'PANW', 'PSKY', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG',
        'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR',
        'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL',
        'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'HOOD',
        'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX',
        'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO',
        'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS',
        'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER',
        'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TTD', 'TSCO', 'TT',
        'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA',
        'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN',
        'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW',
        'WAB', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST',
        'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM',
        'ZBRA', 'ZBH', 'ZTS'
    ]

    print("=" * 70)
    print("REAL S&P 500 - ALL 503 STOCKS (Wikipedia)")
    print("=" * 70)
    print(f"Total tickers: {len(sp500_tickers)}")
    print(f"Source: Wikipedia - List of S&P 500 companies")
    print(f"Updated: October 3, 2025")
    print("=" * 70)

    return sp500_tickers


def save_real_sp500(tickers):
    """Save real S&P 500 list"""

    data = {
        'generated': datetime.now().isoformat(),
        'total_tickers': len(tickers),
        'tickers': tickers,
        'source': 'Wikipedia - List of S&P 500 companies',
        'date_updated': '2025-10-03',
        'note': 'Complete S&P 500 (503 stocks including dual-class shares)'
    }

    # Save complete list
    with open('sp500_complete.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n[SAVED] sp500_complete.json")

    # Also update filtered list
    with open('sp500_options_filtered.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[UPDATED] sp500_options_filtered.json")
    print(f"\nTotal: {len(tickers)} tickers ready for Week 2 scanner!")

    return data


def main():
    """Get and save real S&P 500"""

    print("\n" + "=" * 70)
    print("EXTRACTING REAL S&P 500 FROM WIKIPEDIA")
    print("=" * 70 + "\n")

    # Get real S&P 500
    tickers = get_real_sp500()

    # Save it
    data = save_real_sp500(tickers)

    print("\n" + "=" * 70)
    print("SUCCESS - ALL 503 S&P 500 TICKERS EXTRACTED")
    print("=" * 70)

    # Show examples
    print(f"\nFirst 20: {', '.join(tickers[:20])}")
    print(f"Last 20: {', '.join(tickers[-20:])}")

    print("\n" + "=" * 70)
    print("WEEK 2 SCANNER CAN NOW SCAN ALL 503 S&P 500 STOCKS!")
    print("=" * 70)

    # Count by first letter (approximation)
    print("\nTicker Distribution:")
    from collections import Counter
    first_letters = Counter([t[0] for t in tickers])
    for letter in sorted(first_letters.keys())[:10]:
        print(f"  {letter}: {first_letters[letter]} stocks")


if __name__ == "__main__":
    main()
