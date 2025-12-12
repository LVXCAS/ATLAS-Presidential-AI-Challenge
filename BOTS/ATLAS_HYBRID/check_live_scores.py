"""Quick check of current market scores."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from adapters.oanda_adapter import OandaAdapter
from core.coordinator import ATLASCoordinator
from agents.technical_agent import TechnicalAgent
from agents.pattern_recognition_agent import PatternRecognitionAgent
from agents.news_filter_agent import NewsFilterAgent
from agents.monte_carlo_agent import MonteCarloAgent
import numpy as np
from datetime import datetime

# Initialize
oanda = OandaAdapter()
coordinator = ATLASCoordinator({"score_threshold": 2.5})

# Add agents
coordinator.add_agent(TechnicalAgent(initial_weight=1.5))
coordinator.add_agent(PatternRecognitionAgent(initial_weight=1.0))
coordinator.add_agent(NewsFilterAgent(initial_weight=2.0), is_veto=True)
coordinator.add_agent(MonteCarloAgent(initial_weight=2.0))

print("Current market scores:\n")

for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
    market_data = oanda.get_market_data(pair)
    candles = oanda.get_candles(pair, 'H1', count=201)

    if not candles or len(candles) < 199:
        print(f"{pair}: No data")
        continue

    closes = np.array([c['close'] for c in candles])

    # Calculate indicators (simplified)
    rsi = 50.0  # Placeholder

    enriched_data = {
        "pair": pair,
        "price": market_data['bid'],
        "time": datetime.now(),
        "session": "ny",
        "indicators": {
            "rsi": rsi,
            "macd": 0.001,
            "macd_hist": 0.0005,
            "ema50": closes[-50:].mean(),
            "ema200": closes[-200:].mean(),
            "adx": 25.0,
            "atr": 0.001
        },
        "account_balance": 182788,
        "date": datetime.now().date()
    }

    decision = coordinator.analyze_opportunity(enriched_data)

    score = decision.get('score', decision.get('weighted_score', 0))
    print(f"{pair}: Score={score:.2f}, Decision={decision.get('decision')}, Threshold=2.5")
    print(f"  Price: {market_data['bid']:.5f}")

    # Show agent votes
    for vote in decision.get('agent_votes', []):
        print(f"  {vote['agent']}: {vote['vote']} (conf={vote['confidence']:.2f})")
    print()
