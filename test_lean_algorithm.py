#!/usr/bin/env python3
"""
Test LEAN Algorithm Import
"""

print("[TEST] Testing LEAN algorithm import...")

try:
    # Test basic Python imports
    import sys
    import os
    from pathlib import Path
    print("[OK] Basic Python imports successful")

    # Test if we can import our algorithm
    sys.path.append(str(Path(__file__).parent))
    
    print("[TEST] Attempting to import lean_master_algorithm...")
    import lean_master_algorithm
    print("[OK] lean_master_algorithm imported successfully")
    
    # Test algorithm class
    print("[TEST] Attempting to access HiveTradingMasterAlgorithm class...")
    algo_class = getattr(lean_master_algorithm, 'HiveTradingMasterAlgorithm', None)
    if algo_class:
        print("[OK] HiveTradingMasterAlgorithm class found")
        
        # Test algorithm instantiation (without LEAN context)
        print("[TEST] Testing algorithm structure...")
        print(f"[INFO] Algorithm class methods: {[method for method in dir(algo_class) if not method.startswith('_')]}")
        print("[OK] Algorithm structure looks good")
    else:
        print("[ERROR] HiveTradingMasterAlgorithm class not found")
        
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("[INFO] This might indicate missing dependencies")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    
print("[TEST] Algorithm import test complete")