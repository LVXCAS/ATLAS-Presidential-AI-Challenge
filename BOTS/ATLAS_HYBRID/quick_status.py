"""Quick ATLAS Status Check - Run anytime to see current state."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from adapters.oanda_adapter import OandaAdapter
from core.coordinator import ATLASCoordinator
from agents.technical_agent import TechnicalAgent
from agents.pattern_recognition_agent import PatternRecognitionAgent
from agents.news_filter_agent import NewsFilterAgent
from agents.monte_carlo_agent import MonteCarloAgent
from agents.multi_timeframe_agent import MultiTimeframeAgent
from agents.volume_liquidity_agent import VolumeLiquidityAgent
from agents.support_resistance_agent import SupportResistanceAgent
from agents.divergence_agent import DivergenceAgent
import numpy as np
from datetime import datetime
import json

print("=" * 70)
print("ATLAS QUICK STATUS CHECK")
print("=" * 70)
print()

# Load config
with open('config/hybrid_optimized.json', 'r') as f:
    config = json.load(f)

threshold = config['trading_parameters']['score_threshold']
print(f"✓ Threshold: {threshold}")
print()

# Initialize
oanda = OandaAdapter()
coordinator = ATLASCoordinator({"score_threshold": threshold})

# Add all 16 agents (matching run_paper_training.py)
coordinator.add_agent(TechnicalAgent(initial_weight=1.5))
coordinator.add_agent(PatternRecognitionAgent(initial_weight=1.0))
coordinator.add_agent(NewsFilterAgent(initial_weight=2.0), is_veto=True)
coordinator.add_agent(MonteCarloAgent(initial_weight=2.0))
coordinator.add_agent(MultiTimeframeAgent(initial_weight=2.0))
coordinator.add_agent(VolumeLiquidityAgent(initial_weight=1.8))
coordinator.add_agent(SupportResistanceAgent(initial_weight=1.7))
coordinator.add_agent(DivergenceAgent(initial_weight=1.6))

print(f"✓ {len(coordinator.agents)} agents loaded")
print()

print("CURRENT MARKET SCORES:")
print("-" * 70)

for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
    try:
        market_data = oanda.get_market_data(pair)
        candles = oanda.get_candles(pair, 'H1', count=201)

        if not candles or len(candles) < 199:
            print(f"{pair}: No data available")
            continue

        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])

        # Calculate basic indicators for enrichment
        from talib import RSI, MACD, EMA, ADX, ATR

        rsi = RSI(closes, timeperiod=14)[-1]
        macd, signal, hist = MACD(closes)
        ema50 = EMA(closes, timeperiod=50)[-1]
        ema200 = EMA(closes, timeperiod=200)[-1]
        adx = ADX(highs, lows, closes, timeperiod=14)[-1]
        atr = ATR(highs, lows, closes, timeperiod=14)[-1]

        enriched_data = {
            "pair": pair,
            "price": market_data['bid'],
            "time": datetime.now(),
            "session": "ny",
            "indicators": {
                "rsi": rsi,
                "macd": macd[-1],
                "macd_signal": signal[-1],
                "macd_hist": hist[-1],
                "ema50": ema50,
                "ema200": ema200,
                "adx": adx,
                "atr": atr
            },
            "candles": candles,
            "account_balance": 182788,
            "date": datetime.now().date()
        }

        decision = coordinator.analyze_opportunity(enriched_data)
        score = decision.get('score', decision.get('weighted_score', 0))

        # Color-coded output
        if score >= threshold:
            status = "🟢 TRADE SIGNAL"
        elif score >= threshold * 0.7:
            status = "🟡 CLOSE"
        else:
            status = "⚪ HOLD"

        print(f"\n{pair}: {status}")
        print(f"  Score: {score:.2f} / {threshold} ({decision.get('decision', 'HOLD')})")
        print(f"  Price: {market_data['bid']:.5f}")
        print(f"  RSI: {rsi:.1f} | ADX: {adx:.1f} | MACD: {macd[-1]:.5f}")

        # Show top agent votes
        votes = decision.get('agent_votes', [])
        if votes:
            print(f"  Top Votes:")
            for vote in sorted(votes, key=lambda x: x['confidence'], reverse=True)[:3]:
                print(f"    {vote['agent']}: {vote['vote']} (conf={vote['confidence']:.2f})")

    except Exception as e:
        print(f"{pair}: Error - {e}")

print()
print("=" * 70)
print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
