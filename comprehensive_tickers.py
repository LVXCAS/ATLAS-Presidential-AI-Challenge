"""
Comprehensive NYSE/NASDAQ Ticker Database
Major stocks, ETFs, and popular tickers across all sectors
"""

import json

def get_comprehensive_nyse_nasdaq_tickers():
    """Get comprehensive list of NYSE and NASDAQ tickers by sector"""
    
    tickers = {
        # Major Indices & ETFs
        'SPY': {'name': 'SPDR S&P 500 ETF', 'sector': 'ETF', 'category': 'INDEX_ETF'},
        'QQQ': {'name': 'Invesco QQQ ETF', 'sector': 'ETF', 'category': 'INDEX_ETF'},
        'IWM': {'name': 'iShares Russell 2000 ETF', 'sector': 'ETF', 'category': 'INDEX_ETF'},
        'DIA': {'name': 'SPDR Dow Jones ETF', 'sector': 'ETF', 'category': 'INDEX_ETF'},
        'VTI': {'name': 'Vanguard Total Stock Market ETF', 'sector': 'ETF', 'category': 'INDEX_ETF'},
        'VEA': {'name': 'Vanguard FTSE Developed Markets ETF', 'sector': 'ETF', 'category': 'INTERNATIONAL_ETF'},
        'VWO': {'name': 'Vanguard FTSE Emerging Markets ETF', 'sector': 'ETF', 'category': 'INTERNATIONAL_ETF'},
        
        # Technology - FAANG+
        'AAPL': {'name': 'Apple Inc', 'sector': 'Technology', 'category': 'FAANG'},
        'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'category': 'FAANG'},
        'GOOGL': {'name': 'Alphabet Inc Class A', 'sector': 'Technology', 'category': 'FAANG'},
        'GOOG': {'name': 'Alphabet Inc Class C', 'sector': 'Technology', 'category': 'FAANG'},
        'AMZN': {'name': 'Amazon.com Inc', 'sector': 'Technology', 'category': 'FAANG'},
        'META': {'name': 'Meta Platforms Inc', 'sector': 'Technology', 'category': 'FAANG'},
        'TSLA': {'name': 'Tesla Inc', 'sector': 'Technology', 'category': 'EV'},
        'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'NFLX': {'name': 'Netflix Inc', 'sector': 'Technology', 'category': 'FAANG'},
        
        # Technology - Major Tech
        'AMD': {'name': 'Advanced Micro Devices Inc', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'INTC': {'name': 'Intel Corporation', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'CRM': {'name': 'Salesforce Inc', 'sector': 'Technology', 'category': 'SOFTWARE'},
        'ORCL': {'name': 'Oracle Corporation', 'sector': 'Technology', 'category': 'SOFTWARE'},
        'ADBE': {'name': 'Adobe Inc', 'sector': 'Technology', 'category': 'SOFTWARE'},
        'NOW': {'name': 'ServiceNow Inc', 'sector': 'Technology', 'category': 'SOFTWARE'},
        'SNOW': {'name': 'Snowflake Inc', 'sector': 'Technology', 'category': 'SOFTWARE'},
        'ZM': {'name': 'Zoom Video Communications', 'sector': 'Technology', 'category': 'SOFTWARE'},
        'CSCO': {'name': 'Cisco Systems Inc', 'sector': 'Technology', 'category': 'NETWORKING'},
        'IBM': {'name': 'International Business Machines', 'sector': 'Technology', 'category': 'SOFTWARE'},
        'QCOM': {'name': 'Qualcomm Inc', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'AVGO': {'name': 'Broadcom Inc', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'TXN': {'name': 'Texas Instruments Inc', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'MU': {'name': 'Micron Technology Inc', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'LRCX': {'name': 'Lam Research Corporation', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'AMAT': {'name': 'Applied Materials Inc', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        'KLAC': {'name': 'KLA Corporation', 'sector': 'Technology', 'category': 'SEMICONDUCTOR'},
        
        # Electric Vehicles & Auto
        'F': {'name': 'Ford Motor Company', 'sector': 'Auto', 'category': 'AUTO'},
        'GM': {'name': 'General Motors Company', 'sector': 'Auto', 'category': 'AUTO'},
        'RIVN': {'name': 'Rivian Automotive Inc', 'sector': 'Auto', 'category': 'EV'},
        'LCID': {'name': 'Lucid Group Inc', 'sector': 'Auto', 'category': 'EV'},
        'NIO': {'name': 'NIO Inc', 'sector': 'Auto', 'category': 'EV'},
        'XPEV': {'name': 'XPeng Inc', 'sector': 'Auto', 'category': 'EV'},
        'LI': {'name': 'Li Auto Inc', 'sector': 'Auto', 'category': 'EV'},
        
        # Financial Services - Banks
        'JPM': {'name': 'JPMorgan Chase & Co', 'sector': 'Financial', 'category': 'BANK'},
        'BAC': {'name': 'Bank of America Corp', 'sector': 'Financial', 'category': 'BANK'},
        'WFC': {'name': 'Wells Fargo & Company', 'sector': 'Financial', 'category': 'BANK'},
        'GS': {'name': 'Goldman Sachs Group Inc', 'sector': 'Financial', 'category': 'INVESTMENT_BANK'},
        'MS': {'name': 'Morgan Stanley', 'sector': 'Financial', 'category': 'INVESTMENT_BANK'},
        'C': {'name': 'Citigroup Inc', 'sector': 'Financial', 'category': 'BANK'},
        'USB': {'name': 'U.S. Bancorp', 'sector': 'Financial', 'category': 'BANK'},
        'PNC': {'name': 'PNC Financial Services', 'sector': 'Financial', 'category': 'BANK'},
        'TFC': {'name': 'Truist Financial Corp', 'sector': 'Financial', 'category': 'BANK'},
        'COF': {'name': 'Capital One Financial Corp', 'sector': 'Financial', 'category': 'BANK'},
        
        # Financial Services - Other
        'BRK.A': {'name': 'Berkshire Hathaway Inc Class A', 'sector': 'Financial', 'category': 'CONGLOMERATE'},
        'BRK.B': {'name': 'Berkshire Hathaway Inc Class B', 'sector': 'Financial', 'category': 'CONGLOMERATE'},
        'V': {'name': 'Visa Inc Class A', 'sector': 'Financial', 'category': 'PAYMENT'},
        'MA': {'name': 'Mastercard Inc Class A', 'sector': 'Financial', 'category': 'PAYMENT'},
        'PYPL': {'name': 'PayPal Holdings Inc', 'sector': 'Financial', 'category': 'PAYMENT'},
        'SQ': {'name': 'Block Inc', 'sector': 'Financial', 'category': 'FINTECH'},
        'AXP': {'name': 'American Express Company', 'sector': 'Financial', 'category': 'PAYMENT'},
        'SPGI': {'name': 'S&P Global Inc', 'sector': 'Financial', 'category': 'FINANCIAL_SERVICES'},
        'BLK': {'name': 'BlackRock Inc', 'sector': 'Financial', 'category': 'ASSET_MANAGEMENT'},
        'CME': {'name': 'CME Group Inc Class A', 'sector': 'Financial', 'category': 'EXCHANGE'},
        
        # Healthcare
        'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'category': 'PHARMA'},
        'PFE': {'name': 'Pfizer Inc', 'sector': 'Healthcare', 'category': 'PHARMA'},
        'UNH': {'name': 'UnitedHealth Group Inc', 'sector': 'Healthcare', 'category': 'HEALTH_INSURANCE'},
        'MRNA': {'name': 'Moderna Inc', 'sector': 'Healthcare', 'category': 'BIOTECH'},
        'BNTX': {'name': 'BioNTech SE', 'sector': 'Healthcare', 'category': 'BIOTECH'},
        'ABBV': {'name': 'AbbVie Inc', 'sector': 'Healthcare', 'category': 'PHARMA'},
        'TMO': {'name': 'Thermo Fisher Scientific', 'sector': 'Healthcare', 'category': 'MEDICAL_DEVICES'},
        'DHR': {'name': 'Danaher Corporation', 'sector': 'Healthcare', 'category': 'MEDICAL_DEVICES'},
        'BMY': {'name': 'Bristol Myers Squibb', 'sector': 'Healthcare', 'category': 'PHARMA'},
        'AMGN': {'name': 'Amgen Inc', 'sector': 'Healthcare', 'category': 'BIOTECH'},
        'GILD': {'name': 'Gilead Sciences Inc', 'sector': 'Healthcare', 'category': 'BIOTECH'},
        'MDT': {'name': 'Medtronic PLC', 'sector': 'Healthcare', 'category': 'MEDICAL_DEVICES'},
        'ABT': {'name': 'Abbott Laboratories', 'sector': 'Healthcare', 'category': 'MEDICAL_DEVICES'},
        'LLY': {'name': 'Eli Lilly and Company', 'sector': 'Healthcare', 'category': 'PHARMA'},
        'MRK': {'name': 'Merck & Co Inc', 'sector': 'Healthcare', 'category': 'PHARMA'},
        
        # Energy
        'XOM': {'name': 'Exxon Mobil Corporation', 'sector': 'Energy', 'category': 'OIL_GAS'},
        'CVX': {'name': 'Chevron Corporation', 'sector': 'Energy', 'category': 'OIL_GAS'},
        'COP': {'name': 'ConocoPhillips', 'sector': 'Energy', 'category': 'OIL_GAS'},
        'SLB': {'name': 'Schlumberger NV', 'sector': 'Energy', 'category': 'OIL_SERVICES'},
        'EOG': {'name': 'EOG Resources Inc', 'sector': 'Energy', 'category': 'OIL_GAS'},
        'PXD': {'name': 'Pioneer Natural Resources', 'sector': 'Energy', 'category': 'OIL_GAS'},
        'OXY': {'name': 'Occidental Petroleum', 'sector': 'Energy', 'category': 'OIL_GAS'},
        'HAL': {'name': 'Halliburton Company', 'sector': 'Energy', 'category': 'OIL_SERVICES'},
        'BKR': {'name': 'Baker Hughes Company', 'sector': 'Energy', 'category': 'OIL_SERVICES'},
        'MPC': {'name': 'Marathon Petroleum Corp', 'sector': 'Energy', 'category': 'REFINING'},
        'VLO': {'name': 'Valero Energy Corporation', 'sector': 'Energy', 'category': 'REFINING'},
        'PSX': {'name': 'Phillips 66', 'sector': 'Energy', 'category': 'REFINING'},
        
        # Consumer Discretionary
        'AMZN': {'name': 'Amazon.com Inc', 'sector': 'Consumer Discretionary', 'category': 'E-COMMERCE'},
        'HD': {'name': 'Home Depot Inc', 'sector': 'Consumer Discretionary', 'category': 'RETAIL'},
        'MCD': {'name': 'McDonalds Corporation', 'sector': 'Consumer Discretionary', 'category': 'RESTAURANTS'},
        'NKE': {'name': 'Nike Inc Class B', 'sector': 'Consumer Discretionary', 'category': 'APPAREL'},
        'SBUX': {'name': 'Starbucks Corporation', 'sector': 'Consumer Discretionary', 'category': 'RESTAURANTS'},
        'DIS': {'name': 'Walt Disney Company', 'sector': 'Consumer Discretionary', 'category': 'ENTERTAINMENT'},
        'LOW': {'name': 'Lowes Companies Inc', 'sector': 'Consumer Discretionary', 'category': 'RETAIL'},
        'TGT': {'name': 'Target Corporation', 'sector': 'Consumer Discretionary', 'category': 'RETAIL'},
        'COST': {'name': 'Costco Wholesale Corp', 'sector': 'Consumer Discretionary', 'category': 'RETAIL'},
        'WMT': {'name': 'Walmart Inc', 'sector': 'Consumer Staples', 'category': 'RETAIL'},
        
        # Consumer Staples
        'KO': {'name': 'Coca-Cola Company', 'sector': 'Consumer Staples', 'category': 'BEVERAGES'},
        'PEP': {'name': 'PepsiCo Inc', 'sector': 'Consumer Staples', 'category': 'BEVERAGES'},
        'PG': {'name': 'Procter & Gamble Company', 'sector': 'Consumer Staples', 'category': 'HOUSEHOLD_PRODUCTS'},
        'PM': {'name': 'Philip Morris International', 'sector': 'Consumer Staples', 'category': 'TOBACCO'},
        'MO': {'name': 'Altria Group Inc', 'sector': 'Consumer Staples', 'category': 'TOBACCO'},
        'MDLZ': {'name': 'Mondelez International', 'sector': 'Consumer Staples', 'category': 'FOOD'},
        'GIS': {'name': 'General Mills Inc', 'sector': 'Consumer Staples', 'category': 'FOOD'},
        'K': {'name': 'Kellogg Company', 'sector': 'Consumer Staples', 'category': 'FOOD'},
        
        # Communications
        'GOOGL': {'name': 'Alphabet Inc Class A', 'sector': 'Communication', 'category': 'INTERNET'},
        'META': {'name': 'Meta Platforms Inc', 'sector': 'Communication', 'category': 'SOCIAL_MEDIA'},
        'NFLX': {'name': 'Netflix Inc', 'sector': 'Communication', 'category': 'STREAMING'},
        'DIS': {'name': 'Walt Disney Company', 'sector': 'Communication', 'category': 'MEDIA'},
        'CMCSA': {'name': 'Comcast Corporation', 'sector': 'Communication', 'category': 'TELECOM'},
        'VZ': {'name': 'Verizon Communications', 'sector': 'Communication', 'category': 'TELECOM'},
        'T': {'name': 'AT&T Inc', 'sector': 'Communication', 'category': 'TELECOM'},
        'TMUS': {'name': 'T-Mobile US Inc', 'sector': 'Communication', 'category': 'TELECOM'},
        
        # Industrials
        'BA': {'name': 'Boeing Company', 'sector': 'Industrials', 'category': 'AEROSPACE'},
        'CAT': {'name': 'Caterpillar Inc', 'sector': 'Industrials', 'category': 'MACHINERY'},
        'GE': {'name': 'General Electric Company', 'sector': 'Industrials', 'category': 'CONGLOMERATE'},
        'MMM': {'name': '3M Company', 'sector': 'Industrials', 'category': 'CONGLOMERATE'},
        'HON': {'name': 'Honeywell International', 'sector': 'Industrials', 'category': 'CONGLOMERATE'},
        'UPS': {'name': 'United Parcel Service', 'sector': 'Industrials', 'category': 'LOGISTICS'},
        'FDX': {'name': 'FedEx Corporation', 'sector': 'Industrials', 'category': 'LOGISTICS'},
        'UNP': {'name': 'Union Pacific Corporation', 'sector': 'Industrials', 'category': 'RAILROAD'},
        'CSX': {'name': 'CSX Corporation', 'sector': 'Industrials', 'category': 'RAILROAD'},
        'NSC': {'name': 'Norfolk Southern Corp', 'sector': 'Industrials', 'category': 'RAILROAD'},
        
        # Utilities
        'NEE': {'name': 'NextEra Energy Inc', 'sector': 'Utilities', 'category': 'ELECTRIC_UTILITY'},
        'DUK': {'name': 'Duke Energy Corporation', 'sector': 'Utilities', 'category': 'ELECTRIC_UTILITY'},
        'SO': {'name': 'Southern Company', 'sector': 'Utilities', 'category': 'ELECTRIC_UTILITY'},
        'D': {'name': 'Dominion Energy Inc', 'sector': 'Utilities', 'category': 'ELECTRIC_UTILITY'},
        'AEP': {'name': 'American Electric Power', 'sector': 'Utilities', 'category': 'ELECTRIC_UTILITY'},
        'EXC': {'name': 'Exelon Corporation', 'sector': 'Utilities', 'category': 'ELECTRIC_UTILITY'},
        
        # Real Estate
        'AMT': {'name': 'American Tower Corporation', 'sector': 'Real Estate', 'category': 'REIT'},
        'PLD': {'name': 'Prologis Inc', 'sector': 'Real Estate', 'category': 'REIT'},
        'CCI': {'name': 'Crown Castle International', 'sector': 'Real Estate', 'category': 'REIT'},
        'EQIX': {'name': 'Equinix Inc', 'sector': 'Real Estate', 'category': 'REIT'},
        'WELL': {'name': 'Welltower Inc', 'sector': 'Real Estate', 'category': 'REIT'},
        'DLR': {'name': 'Digital Realty Trust Inc', 'sector': 'Real Estate', 'category': 'REIT'},
        'O': {'name': 'Realty Income Corporation', 'sector': 'Real Estate', 'category': 'REIT'},
        'SPG': {'name': 'Simon Property Group', 'sector': 'Real Estate', 'category': 'REIT'},
        
        # Materials
        'LIN': {'name': 'Linde PLC', 'sector': 'Materials', 'category': 'CHEMICALS'},
        'APD': {'name': 'Air Products and Chemicals', 'sector': 'Materials', 'category': 'CHEMICALS'},
        'SHW': {'name': 'Sherwin-Williams Company', 'sector': 'Materials', 'category': 'CHEMICALS'},
        'FCX': {'name': 'Freeport-McMoRan Inc', 'sector': 'Materials', 'category': 'MINING'},
        'NEM': {'name': 'Newmont Corporation', 'sector': 'Materials', 'category': 'MINING'},
        'GOLD': {'name': 'Barrick Gold Corporation', 'sector': 'Materials', 'category': 'MINING'},
        
        # Crypto & Fintech
        'COIN': {'name': 'Coinbase Global Inc', 'sector': 'Financial', 'category': 'CRYPTO'},
        'MSTR': {'name': 'MicroStrategy Inc', 'sector': 'Financial', 'category': 'CRYPTO'},
        'RIOT': {'name': 'Riot Platforms Inc', 'sector': 'Financial', 'category': 'CRYPTO'},
        'MARA': {'name': 'Marathon Digital Holdings', 'sector': 'Financial', 'category': 'CRYPTO'},
        'HOOD': {'name': 'Robinhood Markets Inc', 'sector': 'Financial', 'category': 'FINTECH'},
        'SOFI': {'name': 'SoFi Technologies Inc', 'sector': 'Financial', 'category': 'FINTECH'},
        'UPST': {'name': 'Upstart Holdings Inc', 'sector': 'Financial', 'category': 'FINTECH'},
        'AFRM': {'name': 'Affirm Holdings Inc', 'sector': 'Financial', 'category': 'FINTECH'},
        
        # Meme Stocks
        'GME': {'name': 'GameStop Corp', 'sector': 'Consumer Discretionary', 'category': 'MEME'},
        'AMC': {'name': 'AMC Entertainment Holdings', 'sector': 'Consumer Discretionary', 'category': 'MEME'},
        'BB': {'name': 'BlackBerry Limited', 'sector': 'Technology', 'category': 'MEME'},
        'NOK': {'name': 'Nokia Corporation', 'sector': 'Technology', 'category': 'MEME'},
        'PLTR': {'name': 'Palantir Technologies Inc', 'sector': 'Technology', 'category': 'SOFTWARE'},
        'WISH': {'name': 'ContextLogic Inc', 'sector': 'Consumer Discretionary', 'category': 'MEME'},
        'CLOV': {'name': 'Clover Health Investments', 'sector': 'Healthcare', 'category': 'MEME'},
        'SPCE': {'name': 'Virgin Galactic Holdings', 'sector': 'Industrials', 'category': 'SPACE'},
        
        # Sector ETFs
        'XLK': {'name': 'Technology Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLF': {'name': 'Financial Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLE': {'name': 'Energy Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLV': {'name': 'Health Care Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLI': {'name': 'Industrial Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLP': {'name': 'Consumer Staples Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLY': {'name': 'Consumer Discretionary Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLU': {'name': 'Utilities Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLB': {'name': 'Materials Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        'XLRE': {'name': 'Real Estate Select Sector SPDR', 'sector': 'ETF', 'category': 'SECTOR_ETF'},
        
        # ARK ETFs
        'ARKK': {'name': 'ARK Innovation ETF', 'sector': 'ETF', 'category': 'GROWTH_ETF'},
        'ARKQ': {'name': 'ARK Autonomous Technology & Robotics ETF', 'sector': 'ETF', 'category': 'GROWTH_ETF'},
        'ARKW': {'name': 'ARK Next Generation Internet ETF', 'sector': 'ETF', 'category': 'GROWTH_ETF'},
        'ARKG': {'name': 'ARK Genomics Revolution ETF', 'sector': 'ETF', 'category': 'GROWTH_ETF'},
        'ARKF': {'name': 'ARK Fintech Innovation ETF', 'sector': 'ETF', 'category': 'GROWTH_ETF'},
        
        # Leveraged ETFs
        'TQQQ': {'name': '3x Long NASDAQ 100 ETF', 'sector': 'ETF', 'category': 'LEVERAGED_ETF'},
        'SQQQ': {'name': '3x Short NASDAQ 100 ETF', 'sector': 'ETF', 'category': 'LEVERAGED_ETF'},
        'UVXY': {'name': 'VIX Short Term Futures ETF', 'sector': 'ETF', 'category': 'VOLATILITY_ETF'},
        'VXX': {'name': 'VIX Short Term Futures ETN', 'sector': 'ETF', 'category': 'VOLATILITY_ETF'},
        'SPXU': {'name': '3x Short S&P 500 ETF', 'sector': 'ETF', 'category': 'LEVERAGED_ETF'},
        'UPRO': {'name': '3x Long S&P 500 ETF', 'sector': 'ETF', 'category': 'LEVERAGED_ETF'},
        
        # Bond ETFs
        'AGG': {'name': 'iShares Core US Aggregate Bond ETF', 'sector': 'ETF', 'category': 'BOND_ETF'},
        'TLT': {'name': '20+ Year Treasury Bond ETF', 'sector': 'ETF', 'category': 'BOND_ETF'},
        'IEF': {'name': '7-10 Year Treasury Bond ETF', 'sector': 'ETF', 'category': 'BOND_ETF'},
        'SHY': {'name': '1-3 Year Treasury Bond ETF', 'sector': 'ETF', 'category': 'BOND_ETF'},
        'LQD': {'name': 'iShares iBoxx Investment Grade Corporate Bond ETF', 'sector': 'ETF', 'category': 'BOND_ETF'},
        'HYG': {'name': 'iShares iBoxx High Yield Corporate Bond ETF', 'sector': 'ETF', 'category': 'BOND_ETF'},
        'TIPS': {'name': 'iShares TIPS Bond ETF', 'sector': 'ETF', 'category': 'BOND_ETF'},
        
        # Commodity ETFs
        'GLD': {'name': 'SPDR Gold Shares', 'sector': 'ETF', 'category': 'COMMODITY_ETF'},
        'SLV': {'name': 'iShares Silver Trust', 'sector': 'ETF', 'category': 'COMMODITY_ETF'},
        'USO': {'name': 'United States Oil Fund', 'sector': 'ETF', 'category': 'COMMODITY_ETF'},
        'UNG': {'name': 'United States Natural Gas Fund', 'sector': 'ETF', 'category': 'COMMODITY_ETF'},
        'DBA': {'name': 'Invesco DB Agriculture Fund', 'sector': 'ETF', 'category': 'COMMODITY_ETF'},
        
        # International ETFs
        'VNQ': {'name': 'Vanguard Real Estate ETF', 'sector': 'ETF', 'category': 'REAL_ESTATE_ETF'},
        'EFA': {'name': 'iShares MSCI EAFE ETF', 'sector': 'ETF', 'category': 'INTERNATIONAL_ETF'},
        'EEM': {'name': 'iShares MSCI Emerging Markets ETF', 'sector': 'ETF', 'category': 'INTERNATIONAL_ETF'},
        'FXI': {'name': 'iShares China Large-Cap ETF', 'sector': 'ETF', 'category': 'INTERNATIONAL_ETF'},
        'EWJ': {'name': 'iShares MSCI Japan ETF', 'sector': 'ETF', 'category': 'INTERNATIONAL_ETF'},
        'ASHR': {'name': 'Xtrackers Harvest CSI 300 China A-Shares ETF', 'sector': 'ETF', 'category': 'INTERNATIONAL_ETF'},
    }
    
    # Convert to proper format
    formatted_tickers = {}
    for symbol, info in tickers.items():
        formatted_tickers[symbol] = {
            'symbol': symbol,
            'name': info['name'],
            'market': 'ETF' if info['sector'] == 'ETF' else 'stocks',
            'sector': info['sector'],
            'category': info['category'],
            'exchange': 'NYSE/NASDAQ'
        }
    
    return formatted_tickers

if __name__ == "__main__":
    print("=== BUILDING COMPREHENSIVE NYSE/NASDAQ TICKER DATABASE ===")
    
    # Get comprehensive ticker list
    all_tickers = get_comprehensive_nyse_nasdaq_tickers()
    
    # Save to JSON
    with open('comprehensive_ticker_database.json', 'w') as f:
        json.dump(all_tickers, f, indent=2)
    
    print(f"Total tickers: {len(all_tickers)}")
    
    # Category breakdown
    categories = {}
    sectors = {}
    
    for ticker_info in all_tickers.values():
        category = ticker_info['category']
        sector = ticker_info['sector']
        
        categories[category] = categories.get(category, 0) + 1
        sectors[sector] = sectors.get(sector, 0) + 1
    
    print("\nSector breakdown:")
    for sector, count in sorted(sectors.items()):
        print(f"  {sector}: {count}")
    
    print(f"\nDatabase saved to: comprehensive_ticker_database.json")
    print(f"Ready to integrate with backend!")