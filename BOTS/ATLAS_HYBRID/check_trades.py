"""
ATLAS Trade Tracker

Checks coordinator state and displays trade summary.
Use this to monitor trading performance.
"""

import json
from pathlib import Path
from datetime import datetime

def load_coordinator_state():
    """Load coordinator state if it exists"""
    state_file = Path(__file__).parent / "learning" / "state" / "coordinator_state.json"

    if not state_file.exists():
        return None

    with open(state_file, 'r') as f:
        return json.load(f)


def analyze_decision_log(state):
    """Analyze decision log for trade statistics"""
    if not state or 'decision_log' not in state:
        return None

    decisions = state['decision_log']

    if not decisions:
        return None

    # Count decisions
    buy_decisions = sum(1 for d in decisions if d.get('decision') == 'BUY')
    sell_decisions = sum(1 for d in decisions if d.get('decision') == 'SELL')
    hold_decisions = sum(1 for d in decisions if d.get('decision') == 'HOLD')

    # Calculate average scores
    scores = [d.get('score', 0) for d in decisions]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0

    # Find highest conviction setups
    high_conviction = sorted(
        [d for d in decisions if d.get('score', 0) > 3.0],
        key=lambda x: x.get('score', 0),
        reverse=True
    )[:5]

    return {
        'total_decisions': len(decisions),
        'buy_decisions': buy_decisions,
        'sell_decisions': sell_decisions,
        'hold_decisions': hold_decisions,
        'avg_score': avg_score,
        'max_score': max_score,
        'high_conviction': high_conviction
    }


def main():
    print("="*70)
    print(f"ATLAS TRADE SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    state = load_coordinator_state()

    if not state:
        print("\n📊 No trading data yet")
        print("\nThis is normal if ATLAS just started.")
        print("State files are created after first scan completes.\n")
        print("Check back in 1 hour for trading activity.")
        return

    # Display coordinator stats
    print(f"\n📈 System Statistics:")
    print(f"   Total Decisions: {state.get('total_decisions', 0)}")
    print(f"   Trades Executed: {state.get('trades_executed', 0)}")
    print(f"   Trades Blocked: {state.get('trades_blocked', 0)}")

    # Analyze decisions
    analysis = analyze_decision_log(state)

    if analysis:
        print(f"\n📊 Decision Breakdown:")
        print(f"   BUY signals: {analysis['buy_decisions']}")
        print(f"   SELL signals: {analysis['sell_decisions']}")
        print(f"   HOLD (neutral): {analysis['hold_decisions']}")
        print(f"   Average score: {analysis['avg_score']:.2f}")
        print(f"   Highest score: {analysis['max_score']:.2f}")

        # Show high conviction setups
        if analysis['high_conviction']:
            print(f"\n🎯 High Conviction Setups (score > 3.0):")
            for i, setup in enumerate(analysis['high_conviction'], 1):
                print(f"\n   {i}. {setup.get('pair', 'UNKNOWN')} - Score: {setup.get('score', 0):.2f}")
                print(f"      Decision: {setup.get('decision', 'N/A')}")
                print(f"      Time: {setup.get('timestamp', 'N/A')}")

    # Show agent performance (if available)
    state_dir = Path(__file__).parent / "learning" / "state"
    agent_files = list(state_dir.glob("*_agent_state.json")) if state_dir.exists() else []

    if agent_files:
        print(f"\n🤖 Agent Learning Status:")
        print(f"   {len(agent_files)} agent(s) have saved state")
        print("   (Agents are tracking performance and will adjust weights after 50 trades)")

    print("\n" + "="*70)
    print("Next Steps:")
    print("  - ATLAS needs ~50 trades before pattern recognition kicks in")
    print("  - XGBoost ML agent trains after 50+ samples")
    print("  - Check back daily to track progress")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
