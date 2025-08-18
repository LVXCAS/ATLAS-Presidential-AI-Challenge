#!/usr/bin/env python3
"""Test import of options volatility agent"""

import sys
import traceback

print("Testing imports...")

try:
    import asyncio
    print("✓ asyncio")
except Exception as e:
    print(f"✗ asyncio: {e}")

try:
    import logging
    print("✓ logging")
except Exception as e:
    print(f"✗ logging: {e}")

try:
    from datetime import datetime, timedelta
    print("✓ datetime")
except Exception as e:
    print(f"✗ datetime: {e}")

try:
    from typing import Dict, List, Optional, Tuple, Any
    print("✓ typing")
except Exception as e:
    print(f"✗ typing: {e}")

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import pandas as pd
    print("✓ pandas")
except Exception as e:
    print(f"✗ pandas: {e}")

try:
    from dataclasses import dataclass, asdict
    print("✓ dataclasses")
except Exception as e:
    print(f"✗ dataclasses: {e}")

try:
    from enum import Enum
    print("✓ enum")
except Exception as e:
    print(f"✗ enum: {e}")

try:
    import yfinance as yf
    print("✓ yfinance")
except Exception as e:
    print(f"✗ yfinance: {e}")

try:
    from scipy import stats
    print("✓ scipy.stats")
except Exception as e:
    print(f"✗ scipy.stats: {e}")

try:
    from scipy.optimize import minimize_scalar
    print("✓ scipy.optimize")
except Exception as e:
    print(f"✗ scipy.optimize: {e}")

print("\nTesting module execution...")

try:
    # Read and execute the file line by line to find the error
    with open('agents/options_volatility_agent.py', 'r') as f:
        content = f.read()
    
    # Try to compile first
    compile(content, 'agents/options_volatility_agent.py', 'exec')
    print("✓ File compiles successfully")
    
    # Try to execute
    exec(content)
    print("✓ File executes successfully")
    
    # Check if classes are defined in local scope
    local_vars = locals()
    classes = [name for name in local_vars if 'Agent' in name and isinstance(local_vars[name], type)]
    print(f"Classes found: {classes}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()