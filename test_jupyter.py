#!/usr/bin/env python3
"""Quick test to verify Jupyter is working"""

import sys
import os

print("=" * 60)
print("JUPYTER NOTEBOOK STATUS CHECK")
print("=" * 60)

# Test 1: Check JupyterLab installation
print("\n[TEST 1] JupyterLab Installation")
try:
    import jupyterlab
    print(f"[PASS] JupyterLab {jupyterlab.__version__} installed")
except ImportError:
    print("[FAIL] JupyterLab not installed")
    sys.exit(1)

# Test 2: Check notebook file exists
print("\n[TEST 2] Notebook File")
notebook_file = "ML_Experimentation.ipynb"
if os.path.exists(notebook_file):
    size = os.path.getsize(notebook_file) / 1024
    print(f"[PASS] {notebook_file} exists ({size:.1f} KB)")
else:
    print(f"[FAIL] {notebook_file} not found")
    sys.exit(1)

# Test 3: Check if notebook is valid JSON
print("\n[TEST 3] Notebook Validity")
try:
    import json
    with open(notebook_file, 'r') as f:
        nb = json.load(f)
    cell_count = len(nb.get('cells', []))
    print(f"[PASS] Valid notebook with {cell_count} cells")
except Exception as e:
    print(f"[FAIL] Invalid notebook: {e}")
    sys.exit(1)

# Test 4: Check required libraries
print("\n[TEST 4] Visualization Libraries")
try:
    import matplotlib
    import seaborn
    print(f"[PASS] Matplotlib {matplotlib.__version__}, Seaborn {seaborn.__version__}")
except ImportError as e:
    print(f"[FAIL] Missing library: {e}")

print("\n" + "=" * 60)
print("JUPYTER STATUS: READY")
print("=" * 60)
print("\nTo start Jupyter:")
print("  Option 1: python -m jupyterlab")
print("  Option 2: Use VSCode Jupyter extension (recommended)")
print("\nNotebook location: ML_Experimentation.ipynb")
print("=" * 60)
