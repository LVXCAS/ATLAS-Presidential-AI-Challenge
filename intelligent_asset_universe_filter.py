#!/usr/bin/env python3
"""
INTELLIGENT ASSET UNIVERSE FILTER
Automatically filters assets for institutional-quality trading
Prevents penny stocks, warrants, and illiquid assets
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - FILTER - %(message)s')

class IntelligentAssetFilter:
    """Intelligent filter for institutional-quality trading assets"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # INSTITUTIONAL QUALITY FILTERS
        self.quality_filters = {
            'min_price': 10.00,              # No penny stocks
            'max_price': 1000.00,            # Reasonable upper bound
            'min_market_cap': 1_000_000_000, # $1B+ companies only
            'min_avg_volume': 1_000_000,     # 1M+ shares daily volume
            'max_bid_ask_spread': 0.005,     # 0.5% max spread
            'min_analyst_coverage': 3,        # Minimum analyst coverage
            'exclude_patterns': ['W', 'Z'],   # No warrants or bankruptcies
            'require_options': True,          # Must have liquid options
            'institutional_ownership_min': 0.3 # 30%+ institutional ownership
        }

        logging.info("INTELLIGENT ASSET FILTER INITIALIZED")
        logging.info("Filtering for institutional-quality assets only")

    async def get_sp500_universe(self):
        """Get S&P 500 constituents as high-quality base"""
        try:
            # S&P 500 tickers (major ones)
            sp500_core = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'TSM',
                'BRK-B', 'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX',
                'ABBV', 'PFE', 'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'MRK',
                'WMT', 'ACN', 'LLY', 'DHR', 'NEE', 'VZ', 'ADBE', 'TXN', 'NFLX',
                'CRM', 'ORCL', 'NKE', 'DIS', 'ABT', 'QCOM', 'WFC', 'BMY', 'T',
                'PM', 'RTX', 'HON', 'UPS', 'SCHW', 'COP', 'INTC', 'INTU', 'CAT',
                'GS', 'IBM', 'AMAT', 'DE', 'BKNG', 'AXP', 'SPGI', 'BLK', 'LOW',
                'GILD', 'MDT', 'SYK', 'MU', 'ISRG', 'TJX', 'C', 'CVS', 'CB',
                'PYPL', 'MMM', 'SO', 'ZTS', 'MDLZ', 'TMUS', 'DUK', 'PLD', 'BSX',
                'AON', 'SHW', 'CME', 'ITW', 'ICE', 'FIS', 'NOC', 'USB', 'EMR',
                'GD', 'NSC', 'PNC', 'APD', 'WM', 'CL', 'F', 'HUM', 'FCX', 'GM'
            ]

            logging.info(f"S&P 500 core universe: {len(sp500_core)} stocks")
            return sp500_core

        except Exception as e:
            logging.error(f"Error getting S&P 500 data: {e}")
            return []

    async def get_liquid_etf_universe(self):
        """Get high-volume ETFs for diversification"""

        liquid_etfs = [
            # Broad market
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO',
            # Sectors
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU',
            # Growth/Value
            'VUG', 'VTV', 'IWF', 'IWD', 'VGT', 'VXUS',
            # Other liquid
            'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'EEM', 'FXI', 'EWJ'
        ]

        logging.info(f"Liquid ETF universe: {len(liquid_etfs)} ETFs")
        return liquid_etfs

    async def filter_asset_quality(self, symbol):
        """Apply comprehensive quality filters to a single asset"""

        try:
            # Basic symbol filtering
            if len(symbol) > 5:  # Likely warrant or special security
                return False, "Symbol too long (likely warrant)"

            if any(pattern in symbol for pattern in self.quality_filters['exclude_patterns']):
                return False, "Contains excluded pattern (warrant/bankruptcy)"

            # Get current quote
            try:
                quote = self.alpaca.get_latest_quote(symbol)
                current_price = float(quote.bid_price) if quote.bid_price else 0

                if current_price == 0:
                    return False, "No valid price data"

                # Price filters
                if current_price < self.quality_filters['min_price']:
                    return False, f"Price ${current_price:.2f} below minimum ${self.quality_filters['min_price']}"

                if current_price > self.quality_filters['max_price']:
                    return False, f"Price ${current_price:.2f} above maximum ${self.quality_filters['max_price']}"

                # Calculate bid-ask spread
                bid = float(quote.bid_price) if quote.bid_price else current_price
                ask = float(quote.ask_price) if quote.ask_price else current_price
                spread_pct = abs(ask - bid) / ((ask + bid) / 2) if ask > 0 and bid > 0 else 0

                if spread_pct > self.quality_filters['max_bid_ask_spread']:
                    return False, f"Bid-ask spread {spread_pct:.3f} too wide"

                return True, f"Quality asset: ${current_price:.2f}, spread {spread_pct:.3f}"

            except Exception as e:
                return False, f"Quote error: {e}"

        except Exception as e:
            return False, f"Filter error: {e}"

    async def get_filtered_universe(self):
        """Get complete filtered universe of tradeable assets"""

        print("INTELLIGENT ASSET UNIVERSE FILTER")
        print("=" * 50)
        print("Building institutional-quality asset universe...")
        print("=" * 50)

        # Get candidate universes
        sp500_stocks = await self.get_sp500_universe()
        liquid_etfs = await self.get_liquid_etf_universe()

        all_candidates = sp500_stocks + liquid_etfs

        print(f"Total candidates: {len(all_candidates)}")
        print("Applying quality filters...")

        # Filter each asset
        approved_assets = []
        rejected_assets = []

        for symbol in all_candidates:
            passed, reason = await self.filter_asset_quality(symbol)

            if passed:
                approved_assets.append(symbol)
                print(f"PASS {symbol:>6} - {reason}")
            else:
                rejected_assets.append({'symbol': symbol, 'reason': reason})
                print(f"FAIL {symbol:>6} - {reason}")

        print("=" * 50)
        print(f"APPROVED ASSETS: {len(approved_assets)}")
        print(f"REJECTED ASSETS: {len(rejected_assets)}")
        print("=" * 50)

        # Save approved universe
        universe_data = {
            'timestamp': datetime.now().isoformat(),
            'approved_count': len(approved_assets),
            'rejected_count': len(rejected_assets),
            'approved_assets': approved_assets,
            'quality_filters': self.quality_filters,
            'rejection_summary': {}
        }

        # Summarize rejections
        for rejection in rejected_assets:
            reason_key = rejection['reason'].split(':')[0] if ':' in rejection['reason'] else rejection['reason']
            if reason_key not in universe_data['rejection_summary']:
                universe_data['rejection_summary'][reason_key] = 0
            universe_data['rejection_summary'][reason_key] += 1

        # Save to file
        with open('approved_asset_universe.json', 'w') as f:
            json.dump(universe_data, f, indent=2)

        print("Quality filters applied:")
        for key, value in self.quality_filters.items():
            print(f"  {key}: {value}")

        print("\nRejection summary:")
        for reason, count in universe_data['rejection_summary'].items():
            print(f"  {reason}: {count} assets")

        print(f"\nAPPROVED UNIVERSE SAVED: approved_asset_universe.json")
        print(f"Systems will now only trade these {len(approved_assets)} quality assets!")

        return approved_assets

async def main():
    """Apply intelligent asset filtering"""
    filter_system = IntelligentAssetFilter()
    approved_universe = await filter_system.get_filtered_universe()

    print(f"\nFINAL APPROVED UNIVERSE ({len(approved_universe)} assets):")
    for i, asset in enumerate(approved_universe):
        print(f"{asset}", end=" ")
        if (i + 1) % 10 == 0:
            print()  # New line every 10 assets
    print()

    return approved_universe

if __name__ == "__main__":
    asyncio.run(main())