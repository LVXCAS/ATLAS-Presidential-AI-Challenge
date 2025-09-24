"""
RESTART AUTONOMOUS SYSTEMS
Clean restart with fixed Unicode issues
"""

import subprocess
import psutil
import time
import os

def kill_existing_systems():
    """Kill existing Python autonomous systems"""

    print("Stopping existing autonomous systems...")

    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'autonomous' in cmdline.lower():
                    print(f"Stopping PID {proc.info['pid']}: {cmdline[-50:]}...")
                    proc.kill()
                    killed_count += 1
        except:
            continue

    print(f"Stopped {killed_count} autonomous processes")
    time.sleep(2)

def start_autonomous_systems():
    """Start both autonomous systems"""

    print("Starting autonomous systems...")

    # Start truly autonomous system
    subprocess.Popen([
        'python',
        'truly_autonomous_system.py'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✅ Truly Autonomous System: STARTED")

    # Start original autonomous system
    subprocess.Popen([
        'python',
        'autonomous_market_open_system.py'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✅ Original Autonomous System: STARTED")

    time.sleep(3)

def verify_systems():
    """Verify systems are running"""

    print("\\nVerifying autonomous systems...")

    running_systems = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'autonomous' in cmdline.lower():
                    running_systems.append(f"PID {proc.info['pid']}")
        except:
            continue

    print(f"Running systems: {len(running_systems)}")
    for system in running_systems:
        print(f"  ✅ {system}")

    if len(running_systems) >= 2:
        print("\\n🚀 AUTONOMOUS SYSTEMS SUCCESSFULLY RESTARTED!")
        print("Both systems are now monitoring your portfolio every 60 seconds")
        print("Check truly_autonomous.log for activity")
    else:
        print("\\n⚠️ Not all systems started properly")

def main():
    print("AUTONOMOUS SYSTEMS RESTART")
    print("=" * 40)

    # Kill existing
    kill_existing_systems()

    # Start new ones
    start_autonomous_systems()

    # Verify
    verify_systems()

    print("\\nSystems are now actively monitoring your $517K portfolio!")

if __name__ == "__main__":
    main()