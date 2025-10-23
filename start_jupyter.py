#!/usr/bin/env python3
"""Start Jupyter Notebook Server"""

import subprocess
import sys
import webbrowser
import time

print("=" * 60)
print("STARTING JUPYTER NOTEBOOK SERVER")
print("=" * 60)

try:
    # Try to import notebook
    import notebook.notebookapp
    print("[OK] Jupyter Notebook installed")
except ImportError:
    print("[ERROR] Jupyter Notebook not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "notebook"])
    import notebook.notebookapp

# Start Jupyter server
print("\nStarting Jupyter server...")
print("Working directory: C:/Users/kkdo/PC-HIVE-TRADING")
print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("=" * 60)
print("1. Jupyter will open in your browser automatically")
print("2. Click on 'ML_Experimentation.ipynb' to open the notebook")
print("3. Run cells by pressing Shift+Enter")
print("4. To stop the server, close this terminal or press Ctrl+C twice")
print("=" * 60)
print()

time.sleep(2)

# Start the notebook server (this will open browser automatically)
subprocess.run([
    sys.executable, "-m", "notebook",
    "--notebook-dir=C:/Users/kkdo/PC-HIVE-TRADING",
    "--NotebookApp.open_browser=True"
])
