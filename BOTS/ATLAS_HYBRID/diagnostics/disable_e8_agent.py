"""
Disable E8ComplianceAgent for OANDA practice validation.

This allows trading on your OANDA practice account which is currently
at $182k (8.6% DD) - this would block trading under E8 rules, but is
fine for validation purposes.

Re-enable this agent when deploying to real E8 challenge.
"""

import json
from pathlib import Path

config_file = Path(__file__).parent.parent / "config" / "hybrid_optimized.json"

# Load config
with open(config_file, 'r') as f:
    config = json.load(f)

# Check current status
current_status = config["agents"]["E8ComplianceAgent"]["enabled"]

print("=" * 80)
print("E8COMPLIANCEAGENT TOGGLE")
print("=" * 80)
print(f"\nCurrent Status: {'ENABLED' if current_status else 'DISABLED'}")

# Toggle
config["agents"]["E8ComplianceAgent"]["enabled"] = not current_status

# Save
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

new_status = config["agents"]["E8ComplianceAgent"]["enabled"]

print(f"New Status:     {'ENABLED' if new_status else 'DISABLED'}")

if new_status:
    print("\n[ENABLED] E8ComplianceAgent will block trades if:")
    print("  - Trailing DD > 6%")
    print("  - Daily DD approaching $3k limit")
    print("  - Losing streak >= 5 trades")
    print("\n  Use this setting for REAL E8 challenges")
else:
    print("\n[DISABLED] E8ComplianceAgent is OFF")
    print("  - No DD limits enforced")
    print("  - Can trade on $182k OANDA account")
    print("  - Use this for OANDA practice validation")
    print("\n  WARNING: Re-enable before deploying to real E8!")

print("\n" + "=" * 80)
print(f"Config updated: {config_file}")
print("=" * 80)
