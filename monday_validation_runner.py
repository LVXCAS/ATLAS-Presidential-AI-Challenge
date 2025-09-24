#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_validation_system import LiveValidationSystem

if __name__ == "__main__":
    print("MONDAY MARKET VALIDATION - STARTING...")
    
    validator = LiveValidationSystem()
    predictions = validator.run_daily_scan_and_log()
    
    if predictions > 0:
        print(f"SUCCESS: {predictions} predictions logged for validation")
    else:
        print("No high-confidence predictions today - models being selective")
    
    # Generate report if we have data
    validator.generate_performance_report()
