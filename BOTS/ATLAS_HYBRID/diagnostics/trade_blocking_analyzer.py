"""
TRADE BLOCKING ANALYZER

Diagnoses why ATLAS isn't taking trades by showing real-time agent voting.

Usage:
    python diagnostics/trade_blocking_analyzer.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from adapters.oanda_adapter import OandaAdapter
from core.coordinator import ATLASCoordinator
from agents.technical_agent import TechnicalAgent
from agents.pattern_recognition_agent import PatternRecognitionAgent
from agents.news_filter_agent import NewsFilterAgent
from agents.e8_compliance_agent import E8ComplianceAgent
from agents.monte_carlo_agent import MonteCarloAgent


def analyze_current_market():
    """Analyze current market and show why trades are blocked."""

    print("=" * 80)
    print("ATLAS TRADE BLOCKING ANALYZER")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}\n")

    # Initialize OANDA
    print("[1/4] Connecting to OANDA...")
    try:
        oanda = OandaAdapter()
        balance_data = oanda.get_account_balance()

        if balance_data:
            print(f"  [OK] Connected - Balance: ${balance_data['balance']:,.2f}")
        else:
            print("  [ERROR] Could not get account balance")
            return
    except Exception as e:
        print(f"  [ERROR] OANDA connection failed: {e}")
        return

    # Initialize agents
    print("\n[2/4] Initializing agents...")

    config = {
        "e8_challenge": {"starting_balance": 200000},
        "score_threshold": 4.5
    }

    coordinator = ATLASCoordinator(config)

    # Add agents
    tech_agent = TechnicalAgent(initial_weight=1.5)
    pattern_agent = PatternRecognitionAgent(initial_weight=1.0)
    news_agent = NewsFilterAgent(initial_weight=2.0)
    e8_agent = E8ComplianceAgent(starting_balance=200000, initial_weight=2.0)
    mc_agent = MonteCarloAgent(initial_weight=2.0, is_veto=False)

    coordinator.add_agent(tech_agent)
    coordinator.add_agent(pattern_agent)
    coordinator.add_agent(news_agent, is_veto=True)
    coordinator.add_agent(e8_agent, is_veto=True)
    coordinator.add_agent(mc_agent, is_veto=False)

    print(f"  [OK] Initialized {len(coordinator.agents)} agents")

    # Analyze each pair
    print("\n[3/4] Analyzing market opportunities...")

    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]

    for pair in pairs:
        print(f"\n{'-' * 80}")
        print(f"PAIR: {pair}")
        print(f"{'-' * 80}")

        # Get market data
        try:
            market_data = oanda.get_market_data(pair)

            if not market_data:
                print(f"  [ERROR] Could not get market data for {pair}")
                continue

            print(f"  Current Price: {market_data['bid']:.5f}")
            print(f"  Spread: {market_data['spread']:.5f}")

            # Get candles for indicators
            candles = oanda.get_candles(pair, 'H1', count=200)

            if not candles or len(candles) < 200:
                print(f"  [ERROR] Insufficient candle data ({len(candles) if candles else 0} candles)")
                continue

            # Calculate indicators
            import numpy as np
            closes = np.array([c['close'] for c in candles])

            # RSI
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:]) or 0.00001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # MACD
            ema12 = closes[-12:].mean()
            ema26 = closes[-26:].mean()
            macd = ema12 - ema26
            macd_signal = closes[-9:].mean()
            macd_hist = macd - macd_signal

            # EMAs
            ema50 = closes[-50:].mean()
            ema200 = closes[-200:].mean()

            # Bollinger Bands
            bb_middle = closes[-20:].mean()
            bb_std = closes[-20:].std()
            bb_upper = bb_middle + (2 * bb_std)
            bb_lower = bb_middle - (2 * bb_std)

            # ADX (simplified)
            adx = 25.0  # Placeholder

            # ATR
            highs = np.array([c['high'] for c in candles[-14:]])
            lows = np.array([c['low'] for c in candles[-14:]])
            atr = np.mean(highs - lows)

            # Build market data for agents
            enriched_data = {
                "pair": pair,
                "price": market_data['bid'],
                "time": datetime.now(),
                "session": "london",  # Placeholder
                "indicators": {
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": macd_signal,
                    "macd_hist": macd_hist,
                    "ema50": ema50,
                    "ema200": ema200,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "bb_middle": bb_middle,
                    "adx": adx,
                    "atr": atr,
                },
                "account_balance": balance_data['balance'],
                "date": datetime.now().date(),
            }

            print(f"\n  INDICATORS:")
            print(f"    RSI: {rsi:.1f}")
            print(f"    MACD: {macd:.6f} (Signal: {macd_signal:.6f}, Hist: {macd_hist:.6f})")
            print(f"    Price vs EMA50: {((closes[-1] - ema50) / ema50 * 100):.2f}%")
            print(f"    Price vs EMA200: {((closes[-1] - ema200) / ema200 * 100):.2f}%")
            print(f"    ADX: {adx:.1f}")

            # Get decision
            decision = coordinator.analyze_opportunity(enriched_data)

            print(f"\n  AGENT VOTES:")
            for vote in decision['agent_votes']:
                vote_symbol = "✓" if vote['vote'] == "BUY" else ("✗" if vote['vote'] == "SELL" else "○")
                print(f"    [{vote_symbol}] {vote['agent']}: {vote['vote']} (confidence: {vote['confidence']:.2f}, weight: {vote['weight']:.2f})")
                if vote.get('reasoning'):
                    for line in vote['reasoning'].split('\n'):
                        if line.strip():
                            print(f"        → {line.strip()}")

            print(f"\n  FINAL DECISION:")
            print(f"    Weighted Score: {decision['weighted_score']:.2f}")
            print(f"    Threshold: {config['score_threshold']}")
            print(f"    Decision: {decision['decision']}")

            if decision['decision'] == "HOLD":
                print(f"\n  [BLOCKED] Score {decision['weighted_score']:.2f} < {config['score_threshold']} threshold")

                # Explain blocking
                if decision['weighted_score'] < 2.0:
                    print(f"    → Agents are mostly neutral or conflicting")
                elif decision['weighted_score'] < 3.5:
                    print(f"    → Setup is weak - not enough confirming signals")
                elif decision['weighted_score'] < 4.5:
                    print(f"    → Setup is moderate - needs stronger confluence")
                    print(f"    → Tip: Lower threshold to 3.5 for exploration phase")

        except Exception as e:
            print(f"  [ERROR] Analysis failed for {pair}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    stats = coordinator.get_statistics()

    print(f"\nTotal Opportunities Analyzed: {stats['total_decisions']}")
    print(f"Trades Executed: {stats['trades_executed']}")
    print(f"Execution Rate: {stats['execution_rate']:.1f}%")

    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 80}")

    if stats['execution_rate'] < 5:
        print(f"\n[LOW EXECUTION RATE]")
        print(f"  Your system is highly selective (score threshold: {config['score_threshold']})")
        print(f"\n  OPTIONS:")
        print(f"  1. WAIT - This is exploration phase, perfect setups are rare")
        print(f"  2. LOWER THRESHOLD - Edit config to 3.5 for more trades")
        print(f"  3. RUN LONGER - Check again in 2-4 hours")
        print(f"\n  Current setting is ultra-conservative (prevents $8k loss scenario)")
        print(f"  Expect 0-2 trades per WEEK with threshold 4.5")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    try:
        analyze_current_market()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Analyzer failed: {e}")
        import traceback
        traceback.print_exc()
