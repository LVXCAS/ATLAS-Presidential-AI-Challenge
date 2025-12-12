"""Quick system verification - check config and agents are loaded correctly."""
import json
from pathlib import Path

# Check config file
config_file = Path(__file__).parent / "config" / "hybrid_optimized.json"
with open(config_file, 'r') as f:
    config = json.load(f)

print("="*80)
print("ATLAS SYSTEM VERIFICATION")
print("="*80)
print()

# Check thresholds
base_threshold = config['trading_parameters']['score_threshold']
exploration_threshold = config['paper_training']['phases']['exploration']['score_threshold']

print(f"✓ Base Threshold: {base_threshold}")
print(f"✓ Exploration Threshold: {exploration_threshold}")
print()

# Check agents
agents_config = config.get("agents", {})
enabled_agents = [name for name, cfg in agents_config.items() if cfg.get("enabled", False)]

print(f"✓ Total Agent Configs: {len(agents_config)}")
print(f"✓ Enabled Agents: {len(enabled_agents)}")
print()

# Check for new agents
new_agents = ["MultiTimeframeAgent", "VolumeLiquidityAgent", "SupportResistanceAgent", "DivergenceAgent"]
print("New Agent Status:")
for agent in new_agents:
    status = "✓ ENABLED" if agent in enabled_agents else "✗ NOT FOUND"
    print(f"  {agent}: {status}")
print()

# Show all enabled agents
print("All Enabled Agents:")
for i, agent in enumerate(sorted(enabled_agents), 1):
    weight = agents_config[agent].get("initial_weight", "N/A")
    print(f"  {i:2}. {agent:30} (weight: {weight})")
print()

print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print()

if base_threshold == 1.0 and exploration_threshold == 1.0:
    print("✓✓✓ THRESHOLDS CORRECT - Ready to trade!")
else:
    print(f"⚠⚠⚠ WARNING: Thresholds not 1.0 (base={base_threshold}, exploration={exploration_threshold})")

if all(agent in enabled_agents for agent in new_agents):
    print("✓✓✓ ALL 4 NEW AGENTS ENABLED - System upgraded!")
else:
    missing = [a for a in new_agents if a not in enabled_agents]
    print(f"⚠⚠⚠ WARNING: Missing agents: {missing}")
