#!/usr/bin/env python3
"""
FUTURES SCANNER
Scan MES/MNQ for EMA crossover signals

Integrates:
- Futures data fetcher (Alpaca)
- EMA crossover strategy
- Opportunity scoring
- Real-time signal detection
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.futures_data_fetcher import MICRO_FUTURES
from data.polygon_futures_fetcher import HybridFuturesFetcher
from strategies.futures_ema_strategy import FuturesEMAStrategy
from typing import List, Dict, Optional
from datetime import datetime


class FuturesScanner:
    """
    Scanner for Micro E-mini Futures

    Features:
    - Scans MES and MNQ
    - Detects EMA crossover signals
    - Scores opportunities
    - Returns actionable trades
    - Uses Polygon API as backup for Alpaca
    """

    def __init__(self, paper_trading: bool = True):
        """
        Initialize futures scanner

        Args:
            paper_trading: Use paper trading data
        """
        print("\n[FUTURES SCANNER] Initializing...")

        # Initialize hybrid data fetcher (Alpaca + Polygon backup)
        self.data_fetcher = HybridFuturesFetcher(paper_trading=paper_trading)

        # Initialize strategy
        self.strategy = FuturesEMAStrategy()

        print("[FUTURES SCANNER] Ready\n")

    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Scan a single futures contract for signals

        Args:
            symbol: Futures symbol (e.g., 'MES', 'MNQ')

        Returns:
            Opportunity dict or None
        """

        print(f"\n[SCANNING] {symbol} - {MICRO_FUTURES[symbol]['name']}")

        # Fetch historical data
        df = self.data_fetcher.get_bars(symbol, timeframe='15Min', limit=500)

        if df is None or df.empty:
            print(f"  [ERROR] No data available for {symbol}")
            return None

        print(f"  [DATA] {len(df)} candles fetched")

        # Analyze for opportunity
        opportunity = self.strategy.analyze_opportunity(df, symbol)

        if opportunity:
            # Validate rules
            valid = self.strategy.validate_rules(opportunity)

            if valid:
                # Get current price
                current_price = self.data_fetcher.get_current_price(symbol)
                if current_price:
                    opportunity['current_price'] = current_price

                # Add contract specifications
                opportunity['contract_specs'] = MICRO_FUTURES[symbol]

                print(f"  [SIGNAL] {opportunity['direction']} - Score: {opportunity['score']:.2f}")
                print(f"  Entry: ${opportunity['entry_price']:.2f}")
                print(f"  Risk per contract: ${opportunity['risk_per_contract']:.2f}")
                return opportunity
            else:
                print(f"  [REJECTED] Signal found but failed validation")
                return None
        else:
            print(f"  [NO SIGNAL] No setup detected")
            return None

    def scan_all_futures(self) -> List[Dict]:
        """
        Scan all supported futures contracts

        Returns:
            List of opportunities, sorted by score
        """

        print("\n" + "="*70)
        print("FUTURES SCANNER - FULL SCAN")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Contracts: {', '.join(MICRO_FUTURES.keys())}")
        print("="*70)

        opportunities = []

        for symbol in MICRO_FUTURES.keys():
            opportunity = self.scan_symbol(symbol)
            if opportunity:
                opportunities.append(opportunity)

        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        print("\n" + "="*70)
        print(f"SCAN COMPLETE - {len(opportunities)} opportunities found")
        print("="*70)

        return opportunities

    def display_opportunities(self, opportunities: List[Dict], top_n: int = 5):
        """
        Display opportunities in formatted output

        Args:
            opportunities: List of opportunity dicts
            top_n: Number of top opportunities to display
        """

        if not opportunities:
            print("\n[NO OPPORTUNITIES] No futures signals detected")
            return

        print("\n" + "="*70)
        print(f"TOP {min(top_n, len(opportunities))} FUTURES OPPORTUNITIES")
        print("="*70)

        for i, opp in enumerate(opportunities[:top_n], 1):
            print(f"\n{i}. {opp['symbol']} - {opp['contract_specs']['name']}")
            print(f"   Direction: {opp['direction']}")
            print(f"   Score: {opp['score']:.2f}/12")
            print(f"   Entry: ${opp['entry_price']:.2f}")
            print(f"   Stop Loss: ${opp['stop_loss']:.2f}")
            print(f"   Take Profit: ${opp['take_profit']:.2f}")
            print(f"   Risk/Reward: {opp['risk_reward']:.2f}:1")
            print(f"   Risk per contract: ${opp['risk_per_contract']:.2f}")
            print(f"   Point Value: ${opp['point_value']:.2f}/point")

            # Technical indicators
            indicators = opp['indicators']
            print(f"\n   Technical Indicators:")
            print(f"   - RSI: {indicators['rsi']:.1f}")
            print(f"   - EMA Fast: ${indicators['ema_fast']:.2f}")
            print(f"   - EMA Slow: ${indicators['ema_slow']:.2f}")
            print(f"   - EMA Trend: ${indicators['ema_trend']:.2f}")
            print(f"   - ATR: ${indicators['atr']:.2f}")

        print("\n" + "="*70)


class AIEnhancedFuturesScanner(FuturesScanner):
    """
    AI-Enhanced Futures Scanner

    Adds AI scoring on top of technical signals
    Compatible with auto-execution engine
    """

    def __init__(self, paper_trading: bool = True):
        super().__init__(paper_trading)

        # AI enhancement parameters
        self.min_score_for_ai = 9.0
        self.confidence_multiplier = 0.85  # Base confidence for futures

    def enhance_opportunity(self, opportunity: Dict) -> Dict:
        """
        Add AI scoring to opportunity

        Args:
            opportunity: Base opportunity dict

        Returns:
            Enhanced opportunity with AI scores
        """

        # Calculate confidence based on technical score
        base_score = opportunity['score']
        confidence = min(base_score / 12.0, 0.95)  # Max 95% confidence

        # Adjust based on indicators
        indicators = opportunity['indicators']
        rsi = indicators['rsi']

        # Strong RSI confirmation adds confidence
        if opportunity['direction'] == 'LONG' and rsi > 60:
            confidence += 0.05
        elif opportunity['direction'] == 'SHORT' and rsi < 40:
            confidence += 0.05

        # Strong trend separation adds confidence
        if indicators['trend_distance'] > 5.0:
            confidence += 0.05

        # Cap at 0.95
        confidence = min(confidence, 0.95)

        # Add AI enhancement fields
        opportunity['final_score'] = base_score
        opportunity['confidence'] = confidence
        opportunity['ai_enhanced'] = True

        return opportunity

    def scan_all_futures(self) -> List[Dict]:
        """
        Scan and enhance with AI scoring

        Returns:
            List of AI-enhanced opportunities
        """

        # Get base opportunities
        opportunities = super().scan_all_futures()

        # Enhance with AI
        enhanced_opportunities = []
        for opp in opportunities:
            if opp['score'] >= self.min_score_for_ai:
                enhanced_opp = self.enhance_opportunity(opp)
                enhanced_opportunities.append(enhanced_opp)

        return enhanced_opportunities


def main():
    """Demo futures scanner"""

    print("\n" + "="*70)
    print("FUTURES SCANNER DEMO")
    print("="*70)

    # Initialize scanner
    scanner = AIEnhancedFuturesScanner(paper_trading=True)

    # Scan all futures
    opportunities = scanner.scan_all_futures()

    # Display results
    scanner.display_opportunities(opportunities)

    # Show tradeable opportunities
    if opportunities:
        print("\n" + "="*70)
        print("READY TO TRADE")
        print("="*70)

        for opp in opportunities:
            print(f"\n{opp['symbol']} {opp['direction']}:")
            print(f"  - Score: {opp['final_score']:.2f}")
            print(f"  - Confidence: {opp['confidence']:.0%}")
            print(f"  - Entry: ${opp['entry_price']:.2f}")
            print(f"  - Risk: ${opp['risk_per_contract']:.2f}/contract")
    else:
        print("\n[WAITING] No high-quality setups at this time")

    print("\n" + "="*70)
    print("Futures scanner demo complete")
    print("="*70)


if __name__ == "__main__":
    main()
