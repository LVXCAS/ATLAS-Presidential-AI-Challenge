#!/usr/bin/env python3
"""
Monitor Training Progress
"""

import os
import time
from datetime import datetime

def monitor_training():
    """Monitor training progress by watching log file"""

    log_file = 'training_output.log'

    if not os.path.exists(log_file):
        print("Training not started yet - log file not found")
        return

    print("="*70)
    print("TRAINING PROGRESS MONITOR")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop monitoring\n")

    last_size = 0
    last_lines = []

    try:
        while True:
            # Check file size
            current_size = os.path.getsize(log_file)

            if current_size > last_size:
                # Read new content
                with open(log_file, 'r') as f:
                    lines = f.readlines()

                # Get new lines
                new_lines = lines[len(last_lines):]

                for line in new_lines:
                    line = line.strip()
                    # Filter interesting lines
                    if any(keyword in line for keyword in [
                        'INFO:', 'Training', 'Loaded', 'complete',
                        'accuracy', 'COMPLETE', 'samples', '%'
                    ]):
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"[{timestamp}] {line}")

                last_lines = lines
                last_size = current_size

            # Check if training is complete
            if os.path.exists('models/training_results_v2.json'):
                print("\n" + "="*70)
                print("TRAINING COMPLETE!")
                print("="*70)
                print("Results saved to: models/training_results_v2.json")
                print("\nRun: python compare_ml_versions.py")
                break

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")
        print(f"Training log: {log_file}")

if __name__ == "__main__":
    monitor_training()
