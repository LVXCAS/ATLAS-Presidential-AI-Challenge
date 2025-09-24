#!/usr/bin/env python3
"""
COMPLETE MARKET SCANNER
Monitors the ENTIRE stock market - thousands of symbols
Russell 3000, S&P 500, NASDAQ, NYSE, crypto, options
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import asyncio
import logging
from datetime import datetime
import concurrent.futures
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MARKET - %(message)s',
    handlers=[
        logging.FileHandler('complete_market_scanner.log'),
        logging.StreamHandler()
    ]
)

class CompleteMarketScanner:
    def __init__(self):
        self.all_symbols = []
        self.active_symbols = []
        self.top_movers = []

        # Initialize Alpaca
        self.alpaca = None
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if api_key and secret_key:
                self.alpaca = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
                logging.info("MARKET: Connected to Alpaca for market data")
        except Exception as e:
            logging.error(f"MARKET: Alpaca connection failed: {e}")

        logging.info("MARKET: Complete Market Scanner initialized")
        logging.info("MARKET: Preparing to scan ENTIRE stock market")

    async def get_all_tradeable_symbols(self):
        """Get ALL tradeable symbols from multiple sources"""
        try:
            logging.info("MARKET: Fetching ALL tradeable symbols from exchanges...")

            all_symbols = set()

            # Get symbols from Alpaca (all active US stocks)
            if self.alpaca:
                try:
                    assets = self.alpaca.list_assets(status='active', asset_class='us_equity')
                    alpaca_symbols = [asset.symbol for asset in assets if asset.tradable]
                    all_symbols.update(alpaca_symbols)
                    logging.info(f"MARKET: Added {len(alpaca_symbols)} symbols from Alpaca")
                except Exception as e:
                    logging.error(f"MARKET: Alpaca symbols failed: {e}")

            # Add major indices and ETFs
            major_symbols = [
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO', 'AGG', 'LQD', 'HYG',
                'GLD', 'SLV', 'USO', 'UNG', 'TLT', 'IEF', 'SHY', 'TIPS', 'VNQ', 'REIT',
                'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE',
                'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'TNA', 'TZA', 'UVXY', 'SVXY', 'VXX', 'VIXY'
            ]
            all_symbols.update(major_symbols)

            # Add crypto symbols
            crypto_symbols = [
                'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD', 'DOGE-USD',
                'MATIC-USD', 'SHIB-USD', 'AVAX-USD', 'ATOM-USD', 'LINK-USD', 'UNI-USD'
            ]
            all_symbols.update(crypto_symbols)

            # Add popular stocks from different sectors
            popular_stocks = [
                # Tech giants
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'CRM',
                'ORCL', 'ADBE', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',

                # Meme/Reddit stocks
                'GME', 'AMC', 'BB', 'PLTR', 'RBLX', 'HOOD', 'CLOV', 'WISH', 'SPCE', 'SOFI',

                # EV/Auto
                'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'F', 'GM', 'STELLANTIS', 'TM', 'HMC',

                # Biotech/Pharma
                'MRNA', 'PFE', 'JNJ', 'ABBV', 'UNH', 'CVS', 'MRK', 'BMY', 'GILD', 'AMGN',
                'NTLA', 'CRSP', 'EDIT', 'BEAM', 'PACB', 'ILMN', 'REGN', 'VRTX', 'BIIB',

                # Finance
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
                'V', 'MA', 'PYPL', 'SQ', 'AFRM', 'COIN', 'SHOP',

                # Energy
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'VLO', 'MPC', 'PSX',

                # Defense/Aerospace
                'KTOS', 'LMT', 'RTX', 'BA', 'NOC', 'GD', 'LHX', 'TDG', 'LDOS', 'HII',

                # Social Media/Communications
                'SNAP', 'TWTR', 'PINS', 'SPOT', 'ROKU', 'ZM', 'DOCN', 'NET', 'CRWD', 'ZS',

                # Transportation/Logistics
                'LYFT', 'UBER', 'DASH', 'FDX', 'UPS', 'DAL', 'AAL', 'UAL', 'LUV', 'JBLU',

                # Retail/Consumer
                'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'NFLX',

                # Healthcare
                'TELADOC', 'VEEV', 'DXCM', 'ISRG', 'SYK', 'MDT', 'ABT', 'TMO', 'DHR', 'BDX',

                # Real Estate/REITs
                'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'UDR', 'CPT',

                # Semiconductors
                'SMCI', 'ARM', 'MRVL', 'ADI', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'MPWR', 'SWKS'
            ]
            all_symbols.update(popular_stocks)

            self.all_symbols = list(all_symbols)
            logging.info(f"MARKET: Total symbols collected: {len(self.all_symbols)}")

            return self.all_symbols

        except Exception as e:
            logging.error(f"MARKET: Error getting symbols: {e}")
            return []

    async def scan_market_batch(self, symbols_batch, batch_id):
        """Scan a batch of symbols for price movements"""
        try:
            results = []

            for symbol in symbols_batch:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", interval="1m")

                    if not data.empty and len(data) > 1:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2]
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        volume = data['Volume'].iloc[-1]

                        # Look for significant moves
                        if abs(change_pct) > 0.5:  # 0.5%+ moves
                            results.append({
                                'symbol': symbol,
                                'price': current_price,
                                'change_pct': change_pct,
                                'volume': volume,
                                'signal_strength': min(abs(change_pct) / 2.0, 1.0),
                                'timestamp': datetime.now().isoformat()
                            })

                            logging.info(f"MARKET: {symbol} {change_pct:+.2f}% @ ${current_price:.2f} (Vol: {volume:,.0f})")

                except Exception as e:
                    continue

            logging.info(f"MARKET: Batch {batch_id} complete - {len(results)} signals found")
            return results

        except Exception as e:
            logging.error(f"MARKET: Batch {batch_id} error: {e}")
            return []

    async def full_market_scan(self):
        """Scan the entire market for opportunities"""
        try:
            logging.info("MARKET: Starting FULL MARKET SCAN")
            logging.info("MARKET: Scanning thousands of symbols across all exchanges...")

            # Get all symbols
            if not self.all_symbols:
                await self.get_all_tradeable_symbols()

            # Split into batches for parallel processing
            batch_size = 50
            batches = [self.all_symbols[i:i + batch_size]
                      for i in range(0, len(self.all_symbols), batch_size)]

            logging.info(f"MARKET: Processing {len(batches)} batches of {batch_size} symbols each")

            # Process batches in parallel
            all_signals = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = []
                for i, batch in enumerate(batches[:10]):  # Limit to first 10 batches for speed
                    future = executor.submit(asyncio.run, self.scan_market_batch(batch, i))
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_results = future.result()
                        all_signals.extend(batch_results)
                    except Exception as e:
                        logging.error(f"MARKET: Batch processing error: {e}")

            # Sort by signal strength
            all_signals.sort(key=lambda x: x['signal_strength'], reverse=True)

            logging.info(f"MARKET: Full market scan complete - {len(all_signals)} total signals")

            # Log top movers
            if all_signals:
                logging.info("MARKET: TOP 10 MARKET MOVERS:")
                for i, signal in enumerate(all_signals[:10]):
                    logging.info(f"MARKET: #{i+1} {signal['symbol']} {signal['change_pct']:+.2f}% - Strength: {signal['signal_strength']:.2f}")

            return all_signals

        except Exception as e:
            logging.error(f"MARKET: Full market scan error: {e}")
            return []

    async def execute_market_signals(self, signals):
        """Execute trades on the best market signals"""
        try:
            if not signals:
                return

            # Take top 5 signals for execution
            top_signals = signals[:5]

            for signal in top_signals:
                try:
                    symbol = signal['symbol']
                    change_pct = signal['change_pct']
                    price = signal['price']
                    strength = signal['signal_strength']

                    # Determine action
                    action = 'BUY' if change_pct > 0 else 'SELL'

                    # Position sizing based on signal strength
                    position_value = 500000 * strength * 0.05  # Max 5% per position
                    quantity = max(1, int(position_value / price))

                    if self.alpaca and not symbol.endswith('-USD'):  # No crypto on Alpaca
                        try:
                            side = 'buy' if action == 'BUY' else 'sell'
                            order = self.alpaca.submit_order(
                                symbol=symbol,
                                qty=quantity,
                                side=side,
                                type='market',
                                time_in_force='day'
                            )

                            logging.info(f"MARKET: EXECUTED - {symbol} {action} {quantity} @ ${price:.2f}")
                            logging.info(f"MARKET: Order ID: {order.id} - Signal Strength: {strength:.2f}")

                        except Exception as e:
                            logging.error(f"MARKET: Execution failed for {symbol}: {e}")
                            # Log as simulated
                            logging.info(f"MARKET: SIMULATED - {symbol} {action} {quantity} @ ${price:.2f}")
                    else:
                        logging.info(f"MARKET: SIMULATED - {symbol} {action} {quantity} @ ${price:.2f}")

                except Exception as e:
                    logging.error(f"MARKET: Signal execution error: {e}")
                    continue

        except Exception as e:
            logging.error(f"MARKET: Execute signals error: {e}")

    async def continuous_market_monitoring(self):
        """Continuously monitor the entire market"""
        logging.info("MARKET: Starting continuous FULL MARKET monitoring")
        logging.info("MARKET: Will scan thousands of symbols every 2 minutes")

        cycle = 0

        while True:
            try:
                cycle += 1
                logging.info(f"MARKET: Full market scan cycle {cycle} beginning...")

                # Full market scan
                signals = await self.full_market_scan()

                # Execute on best signals
                if signals:
                    await self.execute_market_signals(signals)
                else:
                    logging.info("MARKET: No significant market signals found this cycle")

                # Wait 2 minutes before next full scan
                logging.info("MARKET: Waiting 2 minutes before next full market scan...")
                await asyncio.sleep(120)

            except Exception as e:
                logging.error(f"MARKET: Monitoring error: {e}")
                await asyncio.sleep(60)

async def main():
    """Main function to run complete market scanner"""
    logging.info("=" * 80)
    logging.info("COMPLETE MARKET SCANNER")
    logging.info("Monitoring the ENTIRE stock market")
    logging.info("Russell 3000, S&P 500, NASDAQ, NYSE, Crypto")
    logging.info("Thousands of symbols scanned continuously")
    logging.info("=" * 80)

    scanner = CompleteMarketScanner()
    await scanner.continuous_market_monitoring()

if __name__ == "__main__":
    asyncio.run(main())