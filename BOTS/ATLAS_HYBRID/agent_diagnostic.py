"""
ATLAS Agent Diagnostic Tool
Shows which agents are loaded, working, and their voting patterns

Usage:
    python agent_diagnostic.py
"""
import sys
from pathlib import Path

# Test each agent by importing
print("=" * 80)
print("ATLAS AGENT DIAGNOSTIC - Testing All Agents")
print("=" * 80)

agents_to_test = [
    ('TechnicalAgent', 'agents.technical_agent'),
    ('PatternRecognitionAgent', 'agents.pattern_recognition_agent'),
    ('NewsFilterAgent', 'agents.news_filter_agent'),
    ('MeanReversionAgent', 'agents.mean_reversion_agent'),
    ('XGBoostMLAgent', 'agents.xgboost_ml_agent'),
    ('SentimentAgent', 'agents.sentiment_agent'),
    ('QlibResearchAgent', 'agents.qlib_research_agent'),
    ('GSQuantAgent', 'agents.gs_quant_agent'),
    ('AutoGenRDAgent', 'agents.autogen_rd_agent'),
    ('MonteCarloAgent', 'agents.monte_carlo_agent'),
    ('MarketRegimeAgent', 'agents.market_regime_agent'),
    ('RiskManagementAgent', 'agents.risk_management_agent'),
    ('SessionTimingAgent', 'agents.session_timing_agent'),
    ('CorrelationAgent', 'agents.correlation_agent'),
    ('MultiTimeframeAgent', 'agents.multi_timeframe_agent'),
    ('VolumeLiquidityAgent', 'agents.volume_liquidity_agent'),
    ('SupportResistanceAgent', 'agents.support_resistance_agent'),
    ('DivergenceAgent', 'agents.divergence_agent'),
]

print(f"\n[TESTING {len(agents_to_test)} AGENTS]\n")

working = []
broken = []

for agent_name, module_path in agents_to_test:
    try:
        # Try to import the agent
        module = __import__(module_path, fromlist=[agent_name])
        agent_class = getattr(module, agent_name)

        # Check if it has required methods
        has_analyze = hasattr(agent_class, 'analyze')
        has_init = hasattr(agent_class, '__init__')

        if has_analyze and has_init:
            status = "OK"
            working.append(agent_name)
            icon = "[OK]"
        else:
            status = "MISSING METHODS"
            broken.append((agent_name, "Missing required methods"))
            icon = "[X]"

        print(f"  {icon} {agent_name:<35} {status}")

    except ImportError as e:
        broken.append((agent_name, f"Import error: {str(e)[:40]}"))
        print(f"  [X] {agent_name:<35} IMPORT FAILED: {str(e)[:40]}")
    except Exception as e:
        broken.append((agent_name, f"Error: {str(e)[:40]}"))
        print(f"  [X] {agent_name:<35} ERROR: {str(e)[:40]}")

# Summary
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Working agents: {len(working)}/{len(agents_to_test)}")
print(f"Broken agents:  {len(broken)}/{len(agents_to_test)}")

if broken:
    print(f"\n[BROKEN AGENTS]")
    for agent, reason in broken:
        print(f"  - {agent}: {reason}")

# Load agent states from learning directory
print(f"\n{'='*80}")
print(f"AGENT STATES (Learning Data)")
print(f"{'='*80}\n")

state_dir = Path(__file__).parent / "learning" / "state"
if state_dir.exists():
    import json

    agent_states = []
    for state_file in sorted(state_dir.glob("*agent_state.json")):
        try:
            with open(state_file) as f:
                state = json.load(f)
                agent_name = state_file.stem.replace('agent_state', '').replace('_', '')
                agent_states.append({
                    'name': agent_name,
                    'weight': state.get('weight', 1.0),
                    'wins': state.get('wins', 0),
                    'losses': state.get('losses', 0),
                    'total_votes': state.get('total_votes', 0)
                })
        except:
            pass

    if agent_states:
        print(f"  {'Agent':<30} {'Weight':<10} {'W-L':<12} {'Votes':<10}")
        print(f"  {'-'*70}")

        for agent in sorted(agent_states, key=lambda x: x['weight'], reverse=True):
            wl_str = f"{agent['wins']}-{agent['losses']}" if agent['wins'] or agent['losses'] else "N/A"
            votes_str = str(agent['total_votes']) if agent['total_votes'] else "N/A"
            print(f"  {agent['name']:<30} {agent['weight']:<10.2f} {wl_str:<12} {votes_str:<10}")
    else:
        print("  No agent state files found")
else:
    print("  Learning state directory not found")

print(f"\n{'='*80}")

if len(working) == len(agents_to_test):
    print("ALL AGENTS WORKING!")
elif len(working) >= 14:
    print(f"MOSTLY WORKING - {len(working)}/{len(agents_to_test)} agents operational")
else:
    print(f"ISSUES DETECTED - Only {len(working)}/{len(agents_to_test)} agents working")

print(f"{'='*80}\n")
