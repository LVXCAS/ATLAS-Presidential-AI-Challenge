#!/usr/bin/env python3
"""
Automatic Weekly Model Retraining
Schedule this to run every Sunday night to keep models fresh
"""

import os
import sys
from datetime import datetime
import subprocess

def retrain_models():
    """Retrain all ML models with latest data"""

    print("="*70)
    print(f"AUTOMATED MODEL RETRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    # Backup old models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"models/backups/{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)

    print("\n1. Backing up current models...")
    try:
        os.system(f'copy models\\trading_models.pkl {backup_dir}\\trading_models.pkl')
        os.system(f'copy models\\regime_models.pkl {backup_dir}\\regime_models.pkl')
        os.system(f'copy models\\trading_scalers.pkl {backup_dir}\\trading_scalers.pkl')
        print("   ✓ Backup complete")
    except Exception as e:
        print(f"   ✗ Backup failed: {e}")
        return False

    # Retrain models
    print("\n2. Retraining models with fresh data...")
    try:
        result = subprocess.run(
            [sys.executable, 'train_500_stocks.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            print("   ✓ Training complete")
            print(result.stdout[-500:])  # Last 500 chars of output
        else:
            print(f"   ✗ Training failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("   ✗ Training timed out after 1 hour")
        return False
    except Exception as e:
        print(f"   ✗ Training error: {e}")
        return False

    # Verify new models
    print("\n3. Verifying new models...")
    try:
        import pickle
        with open('models/trading_models.pkl', 'rb') as f:
            models = pickle.load(f)

        if len(models) >= 3:
            print(f"   ✓ Verified: {list(models.keys())}")
        else:
            print("   ✗ Model verification failed")
            return False

    except Exception as e:
        print(f"   ✗ Verification error: {e}")
        return False

    # Log success
    print("\n" + "="*70)
    print("✅ RETRAINING COMPLETE")
    print("="*70)
    print(f"Backup location: {backup_dir}")
    print(f"Next scheduled retrain: Next Sunday")

    # Write log
    with open('models/retrain_log.txt', 'a') as f:
        f.write(f"{timestamp}: Retraining successful\n")

    return True

if __name__ == "__main__":
    success = retrain_models()
    sys.exit(0 if success else 1)
