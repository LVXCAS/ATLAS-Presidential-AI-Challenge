"""
ATLAS Status Checker

Checks if ATLAS is running and provides health metrics.
Run this script daily to monitor system health.
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime

def check_atlas_running():
    """Check if ATLAS process is running"""
    try:
        # Check for python processes with ATLAS in command line
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Also check pythonw
        result2 = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq pythonw.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True,
            timeout=10
        )

        combined = result.stdout + result2.stdout

        if 'python' in combined.lower():
            print("[OK] Python processes found")
            # Count how many
            lines = [l for l in combined.split('\n') if 'python' in l.lower()]
            print(f"     Found {len(lines)} Python process(es)")
            return True
        else:
            print("[!!] No Python processes running")
            return False

    except Exception as e:
        print(f"[!!] Error checking processes: {e}")
        return False


def check_log_activity():
    """Check if log file has recent activity"""
    log_dir = Path(__file__).parent / "logs"

    if not log_dir.exists():
        print("[!!] Log directory doesn't exist")
        return False

    # Find most recent log file
    log_files = list(log_dir.glob("*.log"))

    if not log_files:
        print("[!!] No log files found")
        return False

    most_recent = max(log_files, key=lambda p: p.stat().st_mtime)

    # Check file modification time
    mod_time = datetime.fromtimestamp(most_recent.stat().st_mtime)
    time_diff = datetime.now() - mod_time

    print(f"\n[LOG] File: {most_recent.name}")
    print(f"      Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"      Age: {time_diff.seconds // 60} minutes ago")

    if time_diff.seconds < 3600:  # Less than 1 hour
        print("      [OK] Log is recent (active)")
        return True
    else:
        print("      [!!] Log is old (may be stale)")
        return False


def check_state_files():
    """Check if agent state files exist"""
    state_dir = Path(__file__).parent / "learning" / "state"

    if not state_dir.exists():
        print("\n[STATE] State directory doesn't exist (normal for first run)")
        return False

    state_files = list(state_dir.glob("*.json")) + list(state_dir.glob("*.pkl"))

    if state_files:
        print(f"\n[STATE] Found {len(state_files)} state file(s)")
        print("        [OK] Agent learning data preserved")
        return True
    else:
        print("\n[STATE] No state files found (normal for first run)")
        return False


def main():
    print("="*60)
    print(f"ATLAS HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\n[*] Checking system status...\n")

    # Check 1: Process running
    process_ok = check_atlas_running()

    # Check 2: Log activity
    log_ok = check_log_activity()

    # Check 3: State files
    state_ok = check_state_files()

    # Overall status
    print("\n" + "="*60)
    print("OVERALL STATUS")
    print("="*60)

    if process_ok and log_ok:
        print("[SUCCESS] ATLAS is RUNNING and HEALTHY")
        print("\nNext steps:")
        print("  - Check back in 24 hours")
        print("  - Logs are at BOTS/ATLAS_HYBRID/logs/")
        print("  - Use: python check_trades.py")
    elif process_ok and not log_ok:
        print("[WARNING] ATLAS may be running but logs are stale")
        print("\nTroubleshooting:")
        print("  - Process might be hung (restart recommended)")
        print("  - Check BOTS/ATLAS_HYBRID/logs/ for errors")
    else:
        print("[STOPPED] ATLAS is NOT RUNNING")
        print("\nAction required:")
        print("  1. Start ATLAS with: start_atlas.bat")
        print("  2. Or run: pythonw run_paper_training.py")

    print("="*60)


if __name__ == "__main__":
    main()
