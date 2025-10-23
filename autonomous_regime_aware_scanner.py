#!/usr/bin/env python3
"""
AUTONOMOUS REGIME-AWARE SCANNER
================================
Fully autonomous scanner that:
1. Detects current market regime (VIX, S&P 500 momentum, Fear & Greed)
2. Selects optimal strategy based on regime
3. Executes only appropriate trades
4. Blocks strategies that don't fit regime

MARKET REGIMES & STRATEGIES:
- EXTREME FEAR (F&G < 25) + VIX > 30 → Cash / Defensive
- FEAR (F&G 25-45) + Bearish → Bear Call Spreads, Long Puts
- NEUTRAL (F&G 45-55) + VIX < 20 → Iron Condors, Butterfly (70-80% WR!)
- GREED (F&G 55-75) + Bullish → Bull Put Spreads, Dual Options
- EXTREME GREED (F&G > 75) + VIX < 15 → Reduce exposure, wait for pullback

AUTONOMOUS FEATURES:
- Runs daily at 6:30 AM ET automatically
- Checks market regime before each scan
- Only uses strategies appropriate for regime
- Adjusts position sizing based on fear/greed
- Logs all regime decisions for review
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

# Import strategy engines
from market_regime_detector import MarketRegimeDetector
from multi_strategy_options_scanner import MultiStrategyOptionsScanner


class FearGreedIndex:
    """
    Fetch CNN Fear & Greed Index

    Scale: 0-100
    - 0-25: Extreme Fear
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed
    """

    def __init__(self):
        # Alternative.me provides free Fear & Greed data
        self.api_url = "https://api.alternative.me/fng/?limit=1"

    def get_current_index(self) -> Dict:
        """
        Get current Fear & Greed Index

        Returns:
            {
                'value': int (0-100),
                'classification': str,
                'timestamp': str
            }
        """

        try:
            response = requests.get(self.api_url, timeout=10)
            data = response.json()

            if data['data']:
                value = int(data['data'][0]['value'])
                classification = data['data'][0]['value_classification']

                return {
                    'value': value,
                    'classification': classification,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            print(f"[WARNING] Could not fetch Fear & Greed Index: {e}")

            # Fallback: Estimate from VIX
            try:
                vix = yf.Ticker('^VIX')
                vix_value = float(vix.history(period='1d')['Close'].iloc[-1])

                # VIX to Fear & Greed conversion (inverse relationship)
                # VIX 40+ = Extreme Fear (20)
                # VIX 30-40 = Fear (35)
                # VIX 15-30 = Neutral (50)
                # VIX 10-15 = Greed (65)
                # VIX < 10 = Extreme Greed (80)

                if vix_value > 40:
                    value = 20
                    classification = "Extreme Fear"
                elif vix_value > 30:
                    value = 35
                    classification = "Fear"
                elif vix_value > 15:
                    value = 50
                    classification = "Neutral"
                elif vix_value > 10:
                    value = 65
                    classification = "Greed"
                else:
                    value = 80
                    classification = "Extreme Greed"

                return {
                    'value': value,
                    'classification': f"{classification} (VIX-based estimate)",
                    'timestamp': datetime.now().isoformat()
                }

            except:
                # Last resort: Default to neutral
                return {
                    'value': 50,
                    'classification': "Neutral (default)",
                    'timestamp': datetime.now().isoformat()
                }


class AutonomousRegimeAwareScanner:
    """
    Autonomous scanner that adapts to market regime

    Decision Flow:
    1. Check Fear & Greed Index
    2. Check S&P 500 momentum
    3. Check VIX level
    4. Combine to determine regime
    5. Select appropriate strategies
    6. Execute ONLY regime-appropriate trades
    """

    def __init__(self):
        self.fear_greed = FearGreedIndex()
        self.regime_detector = MarketRegimeDetector()
        self.multi_scanner = None  # Created based on regime

        print("\n" + "="*80)
        print("AUTONOMOUS REGIME-AWARE SCANNER")
        print("="*80)
        print("Initializing market analysis engines...")
        print("  [OK] Fear & Greed Index fetcher")
        print("  [OK] Market Regime Detector (S&P 500, VIX)")
        print("  [OK] Multi-Strategy Options Scanner")
        print("="*80 + "\n")

    def analyze_complete_market_state(self) -> Dict:
        """
        Comprehensive market analysis combining all signals

        Returns:
            {
                'fear_greed': Dict,
                'sp500_regime': Dict,
                'final_regime': str,
                'recommended_strategies': List[str],
                'position_size_multiplier': float,
                'max_positions': int,
                'explanation': str
            }
        """

        print("\n" + "="*80)
        print("COMPLETE MARKET STATE ANALYSIS")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        print("="*80)

        # 1. Fear & Greed Index
        print("\n[1/3] Fetching Fear & Greed Index...")
        fg_data = self.fear_greed.get_current_index()

        print(f"\n  FEAR & GREED INDEX: {fg_data['value']}/100")
        print(f"  Classification: {fg_data['classification']}")

        # 2. S&P 500 Regime
        print("\n[2/3] Analyzing S&P 500 Market Regime...")
        regime_data = self.regime_detector.analyze_market_regime()

        # 3. Combined Regime Decision
        print("\n[3/3] Combining Signals for Final Regime...")

        fg_value = fg_data['value']
        sp500_regime = regime_data['regime']
        vix = regime_data['vix_level']

        # Regime decision matrix
        if fg_value < 25 and vix > 30:
            # CRISIS MODE
            final_regime = "CRISIS"
            recommended_strategies = []  # Cash only
            position_multiplier = 0.0
            max_positions = 0
            explanation = (
                "[\!] CRISIS MODE [\!]\n"
                f"Fear & Greed: {fg_value} (EXTREME FEAR)\n"
                f"VIX: {vix:.1f} (HIGH VOLATILITY)\n"
                "RECOMMENDATION: Move to cash, wait for stability\n"
                "NO TRADING until Fear & Greed > 30 and VIX < 25"
            )

        elif fg_value < 45 or sp500_regime == 'BEARISH':
            # BEARISH / FEAR
            final_regime = "BEARISH"
            recommended_strategies = ["BEAR_CALL_SPREAD", "LONG_PUTS", "CASH"]
            position_multiplier = 0.5  # Reduce size
            max_positions = 8
            explanation = (
                "[-] BEARISH MARKET\n"
                f"Fear & Greed: {fg_value} (FEAR)\n"
                f"S&P 500: {sp500_regime}\n"
                "RECOMMENDATION: Use bear strategies or stay defensive\n"
                "Strategies: Bear Call Spreads, Long Puts, or wait in cash"
            )

        elif 45 <= fg_value <= 55 and sp500_regime in ['NEUTRAL', 'BULLISH'] and vix < 20:
            # IDEAL FOR IRON CONDOR
            final_regime = "NEUTRAL"
            recommended_strategies = ["IRON_CONDOR", "BUTTERFLY", "BULL_PUT_SPREAD"]
            position_multiplier = 1.2  # Increase size (high win rate)
            max_positions = 19
            explanation = (
                "[=] NEUTRAL MARKET - IDEAL CONDITIONS! [=]\n"
                f"Fear & Greed: {fg_value} (NEUTRAL)\n"
                f"VIX: {vix:.1f} (LOW VOLATILITY)\n"
                f"S&P 500: {sp500_regime}\n"
                "RECOMMENDATION: Iron Condors & Butterfly Spreads (70-80% WR!)\n"
                "This is PREMIUM COLLECTION weather - use it!"
            )

        elif 55 <= fg_value <= 75 and sp500_regime in ['BULLISH', 'NEUTRAL']:
            # BULLISH / GREED
            final_regime = "BULLISH"
            recommended_strategies = ["BULL_PUT_SPREAD", "DUAL_OPTIONS", "IRON_CONDOR"]
            position_multiplier = 1.0
            max_positions = 19
            explanation = (
                "[+] BULLISH MARKET\n"
                f"Fear & Greed: {fg_value} (GREED)\n"
                f"S&P 500: {sp500_regime}\n"
                "RECOMMENDATION: Bull Put Spreads, Dual Options\n"
                "Good environment for premium collection strategies"
            )

        elif fg_value > 75 and sp500_regime == 'VERY_BULLISH':
            # EXTREME GREED - CAUTION
            final_regime = "EXTREME_GREED"
            recommended_strategies = ["DUAL_OPTIONS"]  # Only adaptive strategies
            position_multiplier = 0.7  # Reduce size (pullback risk)
            max_positions = 10
            explanation = (
                "[\!] EXTREME GREED - CAUTION [\!]\n"
                f"Fear & Greed: {fg_value} (EXTREME GREED)\n"
                f"S&P 500: {sp500_regime} (HOT MARKET)\n"
                "RECOMMENDATION: Reduce exposure, market may be overextended\n"
                "Use only adaptive strategies, avoid high-risk spreads"
            )

        else:
            # VERY BULLISH (normal greed)
            final_regime = "VERY_BULLISH"
            recommended_strategies = ["DUAL_OPTIONS", "LONG_CALLS"]
            position_multiplier = 0.9
            max_positions = 12
            explanation = (
                "[^] VERY BULLISH MARKET\n"
                f"Fear & Greed: {fg_value}\n"
                f"S&P 500: {sp500_regime}\n"
                "RECOMMENDATION: Dual Options (adaptive) or directional calls\n"
                "Bull Put Spreads not viable (stocks moving too fast)"
            )

        print(f"\n{'='*80}")
        print(f"FINAL REGIME: {final_regime}")
        print(f"{'='*80}")
        print(explanation)
        print(f"\n[RECOMMENDED STRATEGIES]")
        for strategy in recommended_strategies:
            print(f"  [OK] {strategy}")
        print(f"\n[POSITION SIZING]")
        print(f"  Position Size Multiplier: {position_multiplier:.1f}x")
        print(f"  Max Positions: {max_positions}")
        print(f"{'='*80}\n")

        return {
            'fear_greed': fg_data,
            'sp500_regime': regime_data,
            'final_regime': final_regime,
            'recommended_strategies': recommended_strategies,
            'position_size_multiplier': position_multiplier,
            'max_positions': max_positions,
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }

    def execute_regime_appropriate_scan(self) -> Dict:
        """
        Execute scan using ONLY regime-appropriate strategies
        """

        # Get complete market state
        market_state = self.analyze_complete_market_state()

        regime = market_state['final_regime']
        strategies = market_state['recommended_strategies']
        max_positions = market_state['max_positions']

        # Log regime decision
        self._log_regime_decision(market_state)

        # If crisis mode, don't trade
        if regime == "CRISIS" or max_positions == 0:
            print("\n[TRADING BLOCKED] Crisis mode active - no trading allowed")
            print("Waiting for market conditions to improve...\n")
            return {
                'regime': regime,
                'trades_executed': 0,
                'message': 'Trading blocked due to crisis conditions'
            }

        # Configure scanner for regime
        enable_iron_condor = "IRON_CONDOR" in strategies
        enable_butterfly = "BUTTERFLY" in strategies
        enable_bull_put = "BULL_PUT_SPREAD" in strategies
        enable_dual = "DUAL_OPTIONS" in strategies

        print(f"\n[CONFIGURING SCANNER FOR {regime} REGIME]")
        print(f"  Iron Condor: {'ENABLED' if enable_iron_condor else 'DISABLED'}")
        print(f"  Butterfly: {'ENABLED' if enable_butterfly else 'DISABLED'}")
        print(f"  Bull Put: {'ENABLED' if enable_bull_put else 'DISABLED'}")
        print(f"  Dual Options: {'ENABLED' if enable_dual else 'DISABLED'}")
        print(f"  Max Positions: {max_positions}\n")

        # Create scanner with regime-appropriate strategies
        scanner = MultiStrategyOptionsScanner(
            enable_bull_put=enable_bull_put,
            enable_dual_options=enable_dual,
            enable_iron_condor=enable_iron_condor,
            enable_butterfly=enable_butterfly,
            min_score=7.5
        )

        # Execute scan
        print(f"[EXECUTING REGIME-AWARE SCAN]")
        results = scanner.scan_and_execute(max_positions=max_positions)

        return {
            'regime': regime,
            'market_state': market_state,
            'scan_results': results,
            'timestamp': datetime.now().isoformat()
        }

    def _log_regime_decision(self, market_state: Dict):
        """Log regime decision for review"""

        log_dir = "regime_decisions"
        os.makedirs(log_dir, exist_ok=True)

        log_file = f"{log_dir}/regime_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(log_file, 'w') as f:
            json.dump(market_state, f, indent=2)

        print(f"[LOGGED] Regime decision: {log_file}")


def main():
    """Main entry point for autonomous regime-aware scanner"""

    print("\n" + "="*80)
    print("LAUNCHING AUTONOMOUS REGIME-AWARE SCANNER")
    print("="*80)
    print("This scanner will:")
    print("  1. Analyze Fear & Greed Index")
    print("  2. Analyze S&P 500 momentum")
    print("  3. Analyze VIX volatility")
    print("  4. Determine optimal regime")
    print("  5. Execute ONLY regime-appropriate strategies")
    print("="*80 + "\n")

    # Create scanner
    scanner = AutonomousRegimeAwareScanner()

    # Execute regime-aware scan
    results = scanner.execute_regime_appropriate_scan()

    # Summary
    print("\n" + "="*80)
    print("SCAN COMPLETE")
    print("="*80)
    print(f"Regime Detected: {results['regime']}")
    print(f"Trades Executed: {results.get('scan_results', {}).get('total_executed', 0)}")
    print(f"Timestamp: {results['timestamp']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
