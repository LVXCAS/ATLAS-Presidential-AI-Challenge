#!/usr/bin/env python3
"""
AI-ENHANCED OPTIONS SCANNER
Combines Bull Put Spreads with AI scoring
Monday-ready integration
"""

from multi_source_data_fetcher import MultiSourceDataFetcher
from market_regime_detector import MarketRegimeDetector
from ai_strategy_enhancer import AIStrategyEnhancer
from typing import List, Dict

class AIEnhancedOptionsScanner:
    """
    Options scanner with AI enhancement

    Flow: Multi-source Data → Bull Put Spread Logic → AI Enhancement → Ranked Opportunities
    """

    def __init__(self):
        self.data_fetcher = MultiSourceDataFetcher()
        self.regime_detector = MarketRegimeDetector()
        self.ai_enhancer = AIStrategyEnhancer()

        print("[AI OPTIONS SCANNER] Initialized")
        print("  - Multi-source Data: Ready")
        print("  - Market Regime Detector: Ready")
        print("  - AI Enhancer: Ready")

    def scan_options(self, symbols: List[str] = None) -> List[Dict]:
        """
        Scan options opportunities with AI enhancement

        Args:
            symbols: List of stock symbols (default: top liquid stocks)

        Returns:
            AI-enhanced opportunities sorted by final score
        """

        if symbols is None:
            # Default: Top 20 liquid stocks
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'NVDA', 'META', 'SPY', 'QQQ', 'IWM',
                'DIS', 'NFLX', 'AMD', 'BA', 'JPM',
                'V', 'WMT', 'PG', 'JNJ', 'KO'
            ]

        print(f"\n[AI OPTIONS SCANNER] Scanning {len(symbols)} symbols...")

        # Get market regime
        regime = self.regime_detector.analyze_market_regime()
        vix = regime.get('vix', regime.get('vix_level', 20))
        print(f"[MARKET REGIME] {regime['regime']} (S&P: {regime['sp500_momentum']:.1%}, VIX: {vix:.2f})")

        enhanced_opportunities = []

        for symbol in symbols:
            try:
                # Get market data
                bars = self.data_fetcher.get_bars(symbol, '1Day', limit=30)

                if not bars or bars.df.empty:
                    continue

                df = bars.df

                # Calculate momentum
                momentum = (df['close'].iloc[-1] / df['close'].iloc[-6]) - 1 if len(df) >= 6 else 0

                # Bull Put Spread criteria (from week3_production_scanner logic)
                base_score = 5.0

                # Regime bonus
                if regime['regime'] == 'NEUTRAL' and abs(momentum) < 0.03:
                    base_score += 3.0

                # Low volatility bonus
                if abs(momentum) < 0.02:
                    base_score += 1.5

                # Volume confirmation
                if df['volume'].iloc[-1] > df['volume'].mean() * 1.2:
                    base_score += 1.0

                # Only proceed if decent base score
                if base_score < 7.0:
                    continue

                # Create opportunity dict
                opportunity = {
                    'symbol': symbol,
                    'strategy': 'BULL_PUT_SPREAD',
                    'score': base_score,
                    'price': float(df['close'].iloc[-1]),
                    'momentum': momentum,
                    'regime': regime['regime'],
                    'confidence': 0.65  # Base confidence for Bull Put Spreads
                }

                # AI Enhancement
                enhanced = self.ai_enhancer.enhance_options_opportunity(opportunity, df)

                # Only keep high-scoring opportunities
                if enhanced.final_score >= 8.0:  # High quality threshold
                    enhanced_opportunities.append({
                        'symbol': enhanced.symbol,
                        'strategy': enhanced.strategy,
                        'direction': enhanced.direction,
                        'base_score': enhanced.base_score,
                        'ai_score': enhanced.ai_score,
                        'final_score': enhanced.final_score,
                        'confidence': enhanced.confidence,
                        'price': opportunity['price'],
                        'momentum': opportunity['momentum'],
                        'regime': opportunity['regime'],
                        'reasoning': enhanced.reasoning,
                        'asset_type': 'OPTIONS'
                    })

                    print(f"  [FOUND] {symbol}: Bull Put Spread (Score: {enhanced.final_score:.2f}, Confidence: {enhanced.confidence:.0%})")

            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")
                continue

        # Sort by final score
        enhanced_opportunities.sort(key=lambda x: x['final_score'], reverse=True)

        print(f"\n[AI OPTIONS SCANNER] Found {len(enhanced_opportunities)} AI-enhanced opportunities")

        return enhanced_opportunities

    def display_opportunities(self, opportunities: List[Dict], top_n: int = 5):
        """Display top AI-enhanced opportunities"""

        if not opportunities:
            print("\n[NO OPPORTUNITIES] No options signals at this time")
            return

        print(f"\n{'='*70}")
        print(f"TOP {min(top_n, len(opportunities))} AI-ENHANCED OPTIONS OPPORTUNITIES")
        print(f"{'='*70}\n")

        for i, opp in enumerate(opportunities[:top_n], 1):
            print(f"{i}. {opp['symbol']} - {opp['strategy']}")
            print(f"   Price: ${opp['price']:.2f} | Momentum: {opp['momentum']:+.1%} | Regime: {opp['regime']}")
            print(f"   Base Score: {opp['base_score']:.2f} | AI Score: {opp['ai_score']:.2f} | Final: {opp['final_score']:.2f}")
            print(f"   Confidence: {opp['confidence']:.0%}")

            if opp.get('reasoning'):
                print(f"   AI Reasoning:")
                for reason in opp['reasoning'][:3]:  # Top 3 reasons
                    print(f"     - {reason}")

            print()

        print(f"{'='*70}\n")

    def record_trade_outcome(self, symbol: str, success: bool, return_pct: float):
        """Record trade outcome for AI learning"""
        self.ai_enhancer.record_outcome(symbol, 'BULL_PUT_SPREAD', success, return_pct)
        print(f"[AI LEARNING] Recorded outcome for {symbol}: {'WIN' if success else 'LOSS'} ({return_pct:+.1%})")


def main():
    """Monday morning options scanner"""

    print("\n" + "="*70)
    print("AI-ENHANCED OPTIONS SCANNER - MONDAY READY")
    print("="*70)

    # Initialize
    scanner = AIEnhancedOptionsScanner()

    # Scan top stocks
    opportunities = scanner.scan_options(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ', 'NVDA', 'META', 'TSLA'])

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
    print("AI-Enhanced Options Scanner ready for Monday 6:30 AM PT")
    print("="*70)


if __name__ == "__main__":
    main()
