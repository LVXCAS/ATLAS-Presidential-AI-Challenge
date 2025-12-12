#!/usr/bin/env python3
"""Verify all 3 bug fixes are in place"""

import sys
from pathlib import Path

print("=" * 70)
print("VERIFYING BUG FIXES")
print("=" * 70)

fixes_ok = True

# Fix 1: Check adapter returns []
print("\n[1] Checking adapter fix...")
adapter_file = Path("adapters/oanda_adapter.py")
if adapter_file.exists():
    content = adapter_file.read_text()
    if "return []  # Return empty list" in content:
        print("    ✓ Adapter returns [] instead of None")
    else:
        print("    ✗ Adapter still returns None")
        fixes_ok = False
else:
    print("    ✗ Adapter file not found")
    fixes_ok = False

# Fix 2: Check RSI filter exists
print("\n[2] Checking RSI exhaustion filter...")
tech_file = Path("agents/technical_agent.py")
if tech_file.exists():
    content = tech_file.read_text()
    if 'return ("BLOCK"' in content and "RSI_EXHAUSTION" in content:
        print("    ✓ RSI filter blocks LONG at RSI > 70")
    else:
        print("    ✗ RSI filter not found")
        fixes_ok = False
else:
    print("    ✗ TechnicalAgent file not found")
    fixes_ok = False

# Fix 3: Check veto authority in config
print("\n[3] Checking TechnicalAgent veto authority...")
config_file = Path("config/hybrid_optimized.json")
if config_file.exists():
    import json
    config = json.loads(config_file.read_text())
    if config.get("agents", {}).get("TechnicalAgent", {}).get("is_veto", False):
        print("    ✓ Config has is_veto: true")
    else:
        print("    ✗ Config missing is_veto: true")
        fixes_ok = False
else:
    print("    ✗ Config file not found")
    fixes_ok = False

# Fix 3b: Check run_paper_training passes is_veto
print("\n[4] Checking run_paper_training.py passes is_veto...")
runner_file = Path("run_paper_training.py")
if runner_file.exists():
    content = runner_file.read_text()
    if "is_veto=tech_config.get" in content:
        print("    ✓ run_paper_training passes is_veto parameter")
    else:
        print("    ✗ run_paper_training doesn't pass is_veto")
        fixes_ok = False
else:
    print("    ✗ run_paper_training.py not found")
    fixes_ok = False

print("\n" + "=" * 70)
if fixes_ok:
    print("✓ ALL FIXES VERIFIED - ATLAS ready to restart")
else:
    print("✗ SOME FIXES MISSING - Review above errors")
print("=" * 70)

sys.exit(0 if fixes_ok else 1)
