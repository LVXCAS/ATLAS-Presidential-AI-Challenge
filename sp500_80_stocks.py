"""
Top 80 S&P 500 Stocks by Market Cap and Options Liquidity
Organized by sector for optimal diversification
"""

def get_sp500_top_80():
    """
    Returns top 80 S&P 500 stocks optimized for:
    - Market capitalization
    - Options liquidity
    - Trading volume
    - Sector diversification
    """

    stocks = [
        # ========================================
        # TECHNOLOGY (20 stocks - 25%)
        # ========================================
        'AAPL',   # Apple - $3T market cap
        'MSFT',   # Microsoft - $2.8T
        'NVDA',   # NVIDIA - $1.5T
        'GOOGL',  # Alphabet Class A - $1.7T
        'GOOG',   # Alphabet Class C - $1.7T
        'AMZN',   # Amazon - $1.6T
        'META',   # Meta - $1.2T
        'TSLA',   # Tesla - $800B
        'AVGO',   # Broadcom - $600B
        'ORCL',   # Oracle - $400B
        'CRM',    # Salesforce - $280B
        'ADBE',   # Adobe - $260B
        'CSCO',   # Cisco - $220B
        'ACN',    # Accenture - $210B
        'AMD',    # AMD - $240B
        'INTC',   # Intel - $180B
        'NOW',    # ServiceNow - $170B
        'QCOM',   # Qualcomm - $180B
        'TXN',    # Texas Instruments - $170B
        'INTU',   # Intuit - $160B

        # ========================================
        # FINANCIALS (15 stocks - 18.75%)
        # ========================================
        'BRK.B',  # Berkshire Hathaway - $900B
        'JPM',    # JPMorgan Chase - $550B
        'V',      # Visa - $540B
        'MA',     # Mastercard - $420B
        'BAC',    # Bank of America - $320B
        'WFC',    # Wells Fargo - $200B
        'MS',     # Morgan Stanley - $160B
        'GS',     # Goldman Sachs - $140B
        'SPGI',   # S&P Global - $140B
        'BLK',    # BlackRock - $130B
        'C',      # Citigroup - $120B
        'AXP',    # American Express - $180B
        'SCHW',   # Charles Schwab - $140B
        'CB',     # Chubb - $110B
        'PGR',    # Progressive - $105B

        # ========================================
        # HEALTHCARE (12 stocks - 15%)
        # ========================================
        'UNH',    # UnitedHealth - $520B
        'LLY',    # Eli Lilly - $700B
        'JNJ',    # Johnson & Johnson - $380B
        'ABBV',   # AbbVie - $320B
        'MRK',    # Merck - $280B
        'TMO',    # Thermo Fisher - $220B
        'ABT',    # Abbott Labs - $200B
        'DHR',    # Danaher - $180B
        'PFE',    # Pfizer - $160B
        'BMY',    # Bristol Myers - $110B
        'AMGN',   # Amgen - $150B
        'GILD',   # Gilead - $110B

        # ========================================
        # CONSUMER DISCRETIONARY (10 stocks - 12.5%)
        # ========================================
        'TSLA',   # Tesla - counted here too (crossover)
        'HD',     # Home Depot - $380B
        'MCD',    # McDonald's - $210B
        'NKE',    # Nike - $160B
        'SBUX',   # Starbucks - $110B
        'LOW',    # Lowe's - $140B
        'TJX',    # TJX Companies - $120B
        'BKNG',   # Booking Holdings - $130B
        'CMG',    # Chipotle - $75B
        'MAR',    # Marriott - $70B

        # ========================================
        # CONSUMER STAPLES (6 stocks - 7.5%)
        # ========================================
        'WMT',    # Walmart - $540B
        'PG',     # Procter & Gamble - $380B
        'COST',   # Costco - $360B
        'KO',     # Coca-Cola - $270B
        'PEP',    # PepsiCo - $240B
        'PM',     # Philip Morris - $150B

        # ========================================
        # ENERGY (5 stocks - 6.25%)
        # ========================================
        'XOM',    # Exxon Mobil - $480B
        'CVX',    # Chevron - $300B
        'COP',    # ConocoPhillips - $140B
        'SLB',    # Schlumberger - $70B
        'EOG',    # EOG Resources - $70B

        # ========================================
        # INDUSTRIALS (6 stocks - 7.5%)
        # ========================================
        'BA',     # Boeing - $140B
        'CAT',    # Caterpillar - $170B
        'GE',     # General Electric - $180B
        'RTX',    # Raytheon - $150B
        'HON',    # Honeywell - $140B
        'UPS',    # UPS - $120B

        # ========================================
        # COMMUNICATION (4 stocks - 5%)
        # ========================================
        'META',   # Meta (already counted)
        'GOOGL',  # Google (already counted)
        'NFLX',   # Netflix - $280B
        'DIS',    # Disney - $200B

        # ========================================
        # UTILITIES (2 stocks - 2.5%)
        # ========================================
        'NEE',    # NextEra Energy - $150B
        'DUK',    # Duke Energy - $80B

        # ========================================
        # ADDITIONAL HIGH-VOLUME (3 stocks)
        # ========================================
        'PYPL',   # PayPal - $70B (Fintech)
        'SQ',     # Block (Square) - $40B (Fintech)
        'UBER',   # Uber - $140B (Tech/Consumer)
    ]

    # Remove duplicates (some stocks appear in multiple sectors)
    unique_stocks = []
    seen = set()
    for stock in stocks:
        if stock not in seen:
            unique_stocks.append(stock)
            seen.add(stock)

    return unique_stocks[:80]  # Ensure exactly 80 stocks


def get_sp500_80_by_sector():
    """Returns stocks organized by sector"""
    return {
        'Technology': [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA',
            'AVGO', 'ORCL', 'CRM', 'ADBE', 'CSCO', 'ACN', 'AMD', 'INTC',
            'NOW', 'QCOM', 'TXN', 'INTU'
        ],
        'Financials': [
            'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS',
            'SPGI', 'BLK', 'C', 'AXP', 'SCHW', 'CB', 'PGR'
        ],
        'Healthcare': [
            'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR',
            'PFE', 'BMY', 'AMGN', 'GILD'
        ],
        'Consumer_Discretionary': [
            'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'MAR'
        ],
        'Consumer_Staples': [
            'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM'
        ],
        'Energy': [
            'XOM', 'CVX', 'COP', 'SLB', 'EOG'
        ],
        'Industrials': [
            'BA', 'CAT', 'GE', 'RTX', 'HON', 'UPS'
        ],
        'Communication': [
            'NFLX', 'DIS'
        ],
        'Utilities': [
            'NEE', 'DUK'
        ]
    }


if __name__ == "__main__":
    stocks = get_sp500_top_80()
    sectors = get_sp500_80_by_sector()

    print("=" * 80)
    print("TOP 80 S&P 500 STOCKS - OPTIMIZED FOR OPTIONS TRADING")
    print("=" * 80)
    print(f"\nTotal stocks: {len(stocks)}")

    print("\nSECTOR BREAKDOWN:")
    for sector, stock_list in sectors.items():
        pct = (len(stock_list) / 80) * 100
        print(f"  {sector:25} {len(stock_list):2} stocks ({pct:5.1f}%)")

    print(f"\n{', '.join(stocks)}")
    print("\n" + "=" * 80)
