#!/usr/bin/env python3
"""
AI-ENHANCED FOREX SCANNER v3.0
Uses Enhanced EMA Strategy with Multi-Timeframe Confirmation
Target: 60%+ win rate across all pairs

ENHANCEMENTS:
- Volume/Activity Filter
- Multi-Timeframe Confirmation (4H trend)
- Stricter RSI bounds (avoid extremes)
- Dynamic ATR stops
- Fixed USD/JPY pip calculation
"""

from data.oanda_data_fetcher import OandaDataFetcher
from strategies.forex_ema_strategy import ForexEMAStrategy
from ai_strategy_enhancer import AIStrategyEnhancer
import pandas as pd
from typing import List, Dict

class AIEnhancedForexScanner:
    """
    Forex scanner with AI enhancement

    Flow: OANDA Data → EMA Strategy → AI Enhancement → Ranked Opportunities
    """

    def __init__(self):
        self.data_fetcher = OandaDataFetcher(practice=True)
        self.strategy = ForexEMAStrategy()
        self.strategy.set_data_fetcher(self.data_fetcher)  # Enable MTF confirmation
        self.ai_enhancer = AIStrategyEnhancer()

        print("[AI FOREX SCANNER v3.0] Initialized - TARGET: 60%+ WIN RATE")
        print("  - OANDA Data Fetcher: Ready")
        print("  - Enhanced EMA Strategy: Ready (Volume Filter, MTF Confirm, Stricter RSI)")
        print("  - AI Enhancer: Ready")
        print("  - Multi-Timeframe Confirmation: ENABLED")

    def scan_forex_pairs(self, pairs: List[str] = None) -> List[Dict]:
        """
        Scan forex pairs with AI enhancement

        Args:
            pairs: List of forex pairs (default: EUR_USD only)

        Returns:
            AI-enhanced opportunities sorted by final score
        """

        if pairs is None:
            pairs = ['EUR_USD']  # Focus on proven pair

        print(f"\n[AI FOREX SCANNER] Scanning {len(pairs)} pairs...")

        enhanced_opportunities = []

        for pair in pairs:
            try:
                # Get market data
                df = self.data_fetcher.get_bars(pair, 'H1', limit=250)

                if df is None or df.empty:
                    print(f"  [SKIP] {pair}: No data")
                    continue

                # Reset index
                df = df.reset_index()

                # Get strategy signal
                opportunity = self.strategy.analyze_opportunity(df, pair)

                if not opportunity:
                    print(f"  [SKIP] {pair}: No signal")
                    continue

                # Validate rules
                if not self.strategy.validate_rules(opportunity):
                    print(f"  [SKIP] {pair}: Failed validation")
                    continue

                # AI Enhancement
                enhanced = self.ai_enhancer.enhance_forex_opportunity(opportunity, df)

                # Only keep high-scoring opportunities (v3.0 uses stricter filters)
                if enhanced.final_score >= 9.0:  # High quality threshold
                    enhanced_opportunities.append({
                        'symbol': enhanced.symbol,
                        'strategy': enhanced.strategy,
                        'direction': enhanced.direction,
                        'base_score': enhanced.base_score,
                        'ai_score': enhanced.ai_score,
                        'final_score': enhanced.final_score,
                        'confidence': enhanced.confidence,
                        'entry': enhanced.entry_price,
                        'stop': enhanced.stop_loss,
                        'target': enhanced.take_profit,
                        'reasoning': enhanced.reasoning,
                        'asset_type': 'FOREX'
                    })

                    print(f"  [FOUND] {pair}: {enhanced.direction} (Score: {enhanced.final_score:.2f}, Confidence: {enhanced.confidence:.0%})")

            except Exception as e:
                print(f"  [ERROR] {pair}: {e}")
                continue

        # Sort by final score
        enhanced_opportunities.sort(key=lambda x: x['final_score'], reverse=True)

        print(f"\n[AI FOREX SCANNER] Found {len(enhanced_opportunities)} AI-enhanced opportunities")

        return enhanced_opportunities

    def display_opportunities(self, opportunities: List[Dict], top_n: int = 5):
        """Display top AI-enhanced opportunities"""

        if not opportunities:
            print("\n[NO OPPORTUNITIES] No forex signals at this time")
            return

        print(f"\n{'='*70}")
        print(f"TOP {min(top_n, len(opportunities))} AI-ENHANCED FOREX OPPORTUNITIES")
        print(f"{'='*70}\n")

        for i, opp in enumerate(opportunities[:top_n], 1):
            print(f"{i}. {opp['symbol']} - {opp['direction']}")
            print(f"   Strategy: {opp['strategy']}")
            print(f"   Base Score: {opp['base_score']:.2f} | AI Score: {opp['ai_score']:.2f} | Final: {opp['final_score']:.2f}")
            print(f"   Confidence: {opp['confidence']:.0%}")
            print(f"   Entry: {opp['entry']:.5f} | Stop: {opp['stop']:.5f} | Target: {opp['target']:.5f}")

            if opp.get('reasoning'):
                print(f"   AI Reasoning:")
                for reason in opp['reasoning'][:3]:  # Top 3 reasons
                    print(f"     - {reason}")

            print()

        print(f"{'='*70}\n")

    def record_trade_outcome(self, symbol: str, success: bool, return_pct: float):
        """Record trade outcome for AI learning"""
        self.ai_enhancer.record_outcome(symbol, 'FOREX_EMA_ENHANCED', success, return_pct)
        print(f"[AI LEARNING] Recorded outcome for {symbol}: {'WIN' if success else 'LOSS'} ({return_pct:+.1%})")


def main():
    """Monday morning forex scanner"""

    print("\n" + "="*70)
    print("AI-ENHANCED FOREX SCANNER v3.0 - MONDAY READY")
    print("TARGET: 60%+ WIN RATE")
    print("="*70)

    # Initialize
    scanner = AIEnhancedForexScanner()

    # Scan all major pairs (EUR/USD, GBP/USD, USD/JPY)
    opportunities = scanner.scan_forex_pairs(['EUR_USD', 'GBP_USD', 'USD_JPY'])

    # Display
    scanner.display_opportunities(opportunities)

    # Show AI performance
    perf = scanner.ai_enhancer.get_performance_summary()
    print(f"\n[AI PERFORMANCE]")
    print(f"  Historical outcomes: {perf.get('total_outcomes', 0)}")
    if perf.get('win_rate'):
        print(f"  Win rate: {perf['win_rate']:.1%}")
        print(f"  Avg return: {perf['avg_return']:.1%}")

    print("\n" + "="*70)
    print("AI-Enhanced Forex Scanner v3.0 ready for Monday 6:30 AM PT")
    print("Enhanced with: Volume Filter, MTF Confirmation, Stricter RSI, Dynamic Stops")
    print("="*70)


if __name__ == "__main__":
    main()
