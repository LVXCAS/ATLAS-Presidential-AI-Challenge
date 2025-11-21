#!/usr/bin/env python3
"""
UNIFIED MULTI-ASSET SCANNER
Scans OPTIONS, FOREX, and FUTURES simultaneously

Built for prop firm challenges:
- Options: Bull Put Spreads, Iron Condors
- Forex: EMA Crossover + RSI
- Futures: Market Open Momentum

Usage:
    scanner = UnifiedMultiAssetScanner(asset_types=['options', 'forex'])
    opportunities = scanner.scan_all()
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import sys
import os

# Import existing systems
from multi_source_data_fetcher import MultiSourceDataFetcher
from account_verification_system import AccountVerificationSystem
from market_regime_detector import MarketRegimeDetector

# Import forex/futures components
try:
    from data.oanda_data_fetcher import OandaDataFetcher, MAJOR_PAIRS
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("[WARNING] OANDA not configured yet")

try:
    from strategies.forex.ema_rsi_crossover_optimized import EMACrossoverOptimized
    FOREX_STRATEGIES_AVAILABLE = True
except ImportError:
    FOREX_STRATEGIES_AVAILABLE = False
    print("[WARNING] Forex strategies not available yet")

# Import options strategies (existing)
from strategies.bull_put_spread_engine import BullPutSpreadEngine
from core.adaptive_dual_options_engine import AdaptiveDualOptionsEngine


class UnifiedMultiAssetScanner:
    """
    Multi-asset scanner for OPTIONS, FOREX, FUTURES

    Scans all assets simultaneously
    Routes to appropriate strategies
    Returns ranked opportunities
    """

    def __init__(self, asset_types=['options']):
        """
        Initialize scanner

        Args:
            asset_types: List of assets to scan ['options', 'forex', 'futures']
        """

        print("="*70)
        print("UNIFIED MULTI-ASSET SCANNER")
        print("="*70)

        self.asset_types = asset_types

        # Core systems (always active)
        self.account_verifier = AccountVerificationSystem()
        self.regime_detector = MarketRegimeDetector()

        # Data fetchers
        self.options_data = MultiSourceDataFetcher()

        if 'forex' in asset_types:
            if OANDA_AVAILABLE:
                self.forex_data = OandaDataFetcher(practice=True)
            else:
                print("[WARNING] Forex requested but OANDA not configured")
                self.forex_data = None
        else:
            self.forex_data = None

        # Strategy engines
        self.options_strategies = {
            'bull_put_spread': BullPutSpreadEngine(),
            'dual_options': AdaptiveDualOptionsEngine()
        }

        if 'forex' in asset_types and FOREX_STRATEGIES_AVAILABLE:
            self.forex_strategies = {
                'ema_crossover': EMACrossoverOptimized()
            }
        else:
            self.forex_strategies = {}

        print(f"\n[ASSETS] Enabled: {', '.join(asset_types)}")
        print(f"[OPTIONS] Strategies: {len(self.options_strategies)}")
        print(f"[FOREX] Strategies: {len(self.forex_strategies)}")
        print("="*70 + "\n")

    def scan_options(self, symbols: List[str]) -> List[Dict]:
        """
        Scan options opportunities (existing logic)

        Returns: List of opportunities
        """

        opportunities = []

        print(f"\n[OPTIONS SCAN] Scanning {len(symbols)} symbols...")

        # Get market regime
        regime = self.regime_detector.analyze_market_regime()

        for symbol in symbols:
            try:
                # Get bars
                bars = self.options_data.get_bars(symbol, '1Day', limit=30)
                if not bars or bars.df.empty:
                    continue

                df = bars.df

                # Calculate momentum
                momentum = (df['close'].iloc[-1] / df['close'].iloc[-6]) - 1 if len(df) >= 6 else 0

                # Score opportunity
                score = 5.0

                # Bull Put Spread criteria
                if regime['regime'] == 'NEUTRAL' and abs(momentum) < 0.03:
                    score += 3.0

                if abs(momentum) < 0.02:
                    score += 1.5

                if df['volume'].iloc[-1] > df['volume'].mean() * 1.2:
                    score += 1.0

                if score >= 7.0:
                    opportunities.append({
                        'asset_type': 'options',
                        'symbol': symbol,
                        'strategy': 'BULL_PUT_SPREAD',
                        'score': score,
                        'price': float(df['close'].iloc[-1]),
                        'momentum': momentum,
                        'regime': regime['regime'],
                        'timestamp': datetime.now().isoformat()
                    })

            except Exception as e:
                print(f"[ERROR] Scanning {symbol}: {e}")
                continue

        print(f"[OPTIONS SCAN] Found {len(opportunities)} opportunities")
        return opportunities

    def scan_forex(self, pairs: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan forex opportunities using EMA Crossover

        Args:
            pairs: List of forex pairs (default: major pairs)

        Returns: List of opportunities
        """

        if not self.forex_data or not self.forex_strategies:
            print("[WARNING] Forex scanning not available")
            return []

        opportunities = []
        pairs = pairs or MAJOR_PAIRS

        print(f"\n[FOREX SCAN] Scanning {len(pairs)} pairs...")

        ema_engine = self.forex_strategies['ema_crossover']

        for pair in pairs:
            try:
                # Get 1-hour bars (best for prop challenges)
                df = self.forex_data.get_bars(pair, 'H1', limit=250)

                if df is None or df.empty:
                    continue

                # Reset index to have timestamp as column
                df = df.reset_index()

                # Analyze with EMA Crossover strategy
                opportunity = ema_engine.analyze_opportunity(df, pair)

                if opportunity:
                    # Validate rules
                    if ema_engine.validate_rules(opportunity):
                        opportunity['asset_type'] = 'forex'
                        opportunities.append(opportunity)
                        print(f"  ✅ {pair}: {opportunity['direction']} (Score: {opportunity['score']:.2f})")

            except Exception as e:
                print(f"[ERROR] Scanning {pair}: {e}")
                continue

        print(f"[FOREX SCAN] Found {len(opportunities)} opportunities")
        return opportunities

    def scan_all(self, options_symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan ALL enabled asset types

        Args:
            options_symbols: List of stock symbols for options (default: SP500)

        Returns: Combined list of opportunities, sorted by score
        """

        print("\n" + "="*70)
        print(f"SCANNING ALL ASSETS: {', '.join(self.asset_types)}")
        print("="*70)

        all_opportunities = []

        # Scan options
        if 'options' in self.asset_types:
            if options_symbols is None:
                # Default: Top 20 liquid stocks
                options_symbols = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                    'NVDA', 'META', 'SPY', 'QQQ', 'IWM',
                    'DIS', 'NFLX', 'AMD', 'BABA', 'BA',
                    'JPM', 'V', 'WMT', 'PG', 'JNJ'
                ]

            options_opps = self.scan_options(options_symbols)
            all_opportunities.extend(options_opps)

        # Scan forex
        if 'forex' in self.asset_types:
            forex_opps = self.scan_forex()
            all_opportunities.extend(forex_opps)

        # Sort by score (highest first)
        all_opportunities.sort(key=lambda x: x['score'], reverse=True)

        print("\n" + "="*70)
        print(f"SCAN COMPLETE: {len(all_opportunities)} total opportunities")
        print("="*70)

        return all_opportunities

    def display_opportunities(self, opportunities: List[Dict], top_n: int = 10):
        """Display top opportunities"""

        if not opportunities:
            print("\n[NO OPPORTUNITIES] No signals at this time")
            return

        print(f"\n{'='*70}")
        print(f"TOP {min(top_n, len(opportunities))} OPPORTUNITIES")
        print(f"{'='*70}\n")

        for i, opp in enumerate(opportunities[:top_n], 1):
            asset_type = opp['asset_type'].upper()
            symbol = opp['symbol']
            strategy = opp.get('strategy', 'N/A')
            score = opp['score']

            print(f"{i}. [{asset_type}] {symbol} - {strategy}")
            print(f"   Score: {score:.2f}")

            if asset_type == 'OPTIONS':
                print(f"   Price: ${opp['price']:.2f}")
                print(f"   Momentum: {opp['momentum']:.2%}")
                print(f"   Regime: {opp['regime']}")

            elif asset_type == 'FOREX':
                direction = opp.get('direction', 'N/A')
                entry = opp.get('entry_price', 0)
                stop = opp.get('stop_loss', 0)
                target = opp.get('take_profit', 0)
                rr = opp.get('risk_reward', 0)

                print(f"   Direction: {direction}")
                print(f"   Entry: {entry:.5f}")
                print(f"   Stop: {stop:.5f}")
                print(f"   Target: {target:.5f}")
                print(f"   R/R: {rr:.2f}:1")

            print()

        print(f"{'='*70}\n")


async def main():
    """
    Demo the unified scanner

    Tests both OPTIONS and FOREX scanning
    """

    print("\n" + "="*70)
    print("UNIFIED MULTI-ASSET SCANNER DEMO")
    print("="*70)

    # Test 1: Options only
    print("\n[TEST 1] Options-only scan...")
    scanner_options = UnifiedMultiAssetScanner(asset_types=['options'])
    options_opps = scanner_options.scan_all(options_symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'])
    scanner_options.display_opportunities(options_opps, top_n=5)

    # Test 2: Forex only (if available)
    if OANDA_AVAILABLE and FOREX_STRATEGIES_AVAILABLE:
        print("\n[TEST 2] Forex-only scan...")
        scanner_forex = UnifiedMultiAssetScanner(asset_types=['forex'])
        forex_opps = scanner_forex.scan_forex(['EUR_USD', 'GBP_USD', 'USD_JPY'])
        scanner_forex.display_opportunities(forex_opps, top_n=5)
    else:
        print("\n[TEST 2] Forex scan skipped (setup OANDA first)")

    # Test 3: Both (if forex available)
    if OANDA_AVAILABLE and FOREX_STRATEGIES_AVAILABLE:
        print("\n[TEST 3] Combined scan (Options + Forex)...")
        scanner_all = UnifiedMultiAssetScanner(asset_types=['options', 'forex'])
        all_opps = scanner_all.scan_all(options_symbols=['AAPL', 'MSFT', 'SPY'])
        scanner_all.display_opportunities(all_opps, top_n=10)
    else:
        print("\n[TEST 3] Combined scan skipped (setup OANDA first)")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
