#!/usr/bin/env python3
"""
SYSTEM WATCHDOG
Monitor trading systems and auto-restart if they crash
Ensures 99%+ uptime for autonomous trading
"""
import os
import sys
import time
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.telegram_notifier import get_notifier
    from utils.trade_database import get_database
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False
    print("[WARNING] Integrations not available - running standalone")

load_dotenv()


class SystemWatchdog:
    """Monitor and auto-restart trading systems"""

    def __init__(self, check_interval: int = 300):
        """
        Args:
            check_interval: How often to check systems (default 300s = 5 min)
        """
        self.check_interval = check_interval
        self.restart_attempts = {}
        self.max_restart_attempts = 3
        self.restart_backoff = 60  # Seconds between restart attempts

        # Systems to monitor
        self.systems = {
            'forex': {
                'name': 'Forex Elite',
                'pid_file': 'forex_elite.pid',
                'start_command': [sys.executable, 'START_FOREX_ELITE.py', '--strategy', 'strict'],
                'log_file': 'forex_elite.log',
                'required': True
            },
            'options': {
                'name': 'Options Scanner',
                'pid_file': 'scanner.pid',
                'start_command': [sys.executable, 'auto_options_scanner.py', '--daily'],
                'log_file': 'scanner_output.log',
                'required': True
            },
            'stop_loss': {
                'name': 'Stop Loss Monitor',
                'pid_file': 'stop_loss.pid',
                'start_command': [sys.executable, 'utils/enhanced_stop_loss_monitor.py'],
                'log_file': 'stop_loss_output.log',
                'required': True
            }
        }

        # Get integrations
        self.notifier = get_notifier() if INTEGRATIONS_AVAILABLE else None
        self.database = get_database() if INTEGRATIONS_AVAILABLE else None

        print("=" * 70)
        print("SYSTEM WATCHDOG - ACTIVE")
        print("=" * 70)
        print(f"Check Interval: {self.check_interval}s ({self.check_interval // 60} minutes)")
        print(f"Monitoring {len(self.systems)} systems:")
        for key, system in self.systems.items():
            print(f"  - {system['name']}")
        print(f"Telegram Alerts: {'Enabled' if self.notifier and self.notifier.enabled else 'Disabled'}")
        print(f"Auto-restart: Enabled (max {self.max_restart_attempts} attempts)")
        print("=" * 70)
        print()

    def is_process_running(self, pid: int) -> bool:
        """Check if process with PID is running"""
        try:
            process = psutil.Process(pid)
            # Check if it's a Python process
            if 'python' in process.name().lower():
                return True
            return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def get_pid(self, pid_file: str) -> Optional[int]:
        """Read PID from file"""
        try:
            if os.path.exists(pid_file):
                with open(pid_file, 'r') as f:
                    return int(f.read().strip())
        except Exception as e:
            print(f"[WARNING] Could not read {pid_file}: {e}")
        return None

    def check_system(self, system_key: str, system: Dict) -> bool:
        """Check if a system is running"""
        pid = self.get_pid(system['pid_file'])

        if pid and self.is_process_running(pid):
            # System is running
            try:
                process = psutil.Process(pid)
                cpu = process.cpu_percent(interval=0.5)
                mem = process.memory_info().rss / 1024 / 1024  # MB

                print(f"  ✓ {system['name']}: RUNNING (PID {pid}, CPU {cpu:.1f}%, RAM {mem:.0f}MB)")
                return True
            except Exception as e:
                print(f"  ? {system['name']}: Running but can't get stats - {e}")
                return True
        else:
            # System is NOT running
            print(f"  ✗ {system['name']}: STOPPED")
            return False

    def start_system(self, system_key: str, system: Dict) -> bool:
        """Start a system"""
        print(f"\n[RESTARTING] {system['name']}...")

        try:
            # Open log file
            log_file = open(system['log_file'], 'a')

            # Start process
            process = subprocess.Popen(
                system['start_command'],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )

            # Write PID
            with open(system['pid_file'], 'w') as f:
                f.write(str(process.pid))

            time.sleep(3)  # Give it time to start

            # Verify it started
            if self.is_process_running(process.pid):
                print(f"  ✓ {system['name']} restarted successfully (PID {process.pid})")

                # Send alert
                if self.notifier and self.notifier.enabled:
                    self.notifier.system_restarted(
                        component=system['name'],
                        reason='Watchdog detected crash'
                    )

                # Log to database
                if self.database:
                    self.database.log_system_event(
                        event_type='RESTART',
                        component=system_key,
                        message=f"{system['name']} auto-restarted",
                        severity='WARNING'
                    )

                return True
            else:
                print(f"  ✗ {system['name']} failed to start")
                return False

        except Exception as e:
            print(f"  ✗ Failed to restart {system['name']}: {e}")
            return False

    def handle_down_system(self, system_key: str, system: Dict):
        """Handle a system that's down"""
        # Track restart attempts
        if system_key not in self.restart_attempts:
            self.restart_attempts[system_key] = {
                'count': 0,
                'last_attempt': None
            }

        attempts = self.restart_attempts[system_key]

        # Check if we've exceeded max attempts
        if attempts['count'] >= self.max_restart_attempts:
            print(f"\n⚠️  {system['name']}: Max restart attempts reached!")
            print(f"   Giving up after {attempts['count']} attempts")
            print(f"   Manual intervention required!")

            # Alert once
            if attempts['count'] == self.max_restart_attempts:
                if self.notifier and self.notifier.enabled:
                    self.notifier.system_error(
                        component=system['name'],
                        error=f"Failed to restart after {self.max_restart_attempts} attempts. Manual intervention needed!"
                    )
                attempts['count'] += 1  # Increment to prevent repeated alerts

            return

        # Try to restart
        print(f"\n🔄 Attempting to restart {system['name']} (attempt {attempts['count'] + 1}/{self.max_restart_attempts})")

        success = self.start_system(system_key, system)

        if success:
            # Reset counter on success
            self.restart_attempts[system_key] = {'count': 0, 'last_attempt': datetime.now()}
        else:
            # Increment counter on failure
            attempts['count'] += 1
            attempts['last_attempt'] = datetime.now()

            if attempts['count'] < self.max_restart_attempts:
                print(f"   Will retry in {self.restart_backoff}s...")
                time.sleep(self.restart_backoff)

    def check_all_systems(self):
        """Check all monitored systems"""
        print(f"\n{'='*70}")
        print(f"HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        all_healthy = True

        for system_key, system in self.systems.items():
            is_running = self.check_system(system_key, system)

            if not is_running and system['required']:
                all_healthy = False
                self.handle_down_system(system_key, system)

        if all_healthy:
            print(f"\n✓ All systems operational")
        else:
            print(f"\n⚠️  Some systems require attention")

        return all_healthy

    def run(self):
        """Run continuous monitoring"""
        print(f"\n[START] Watchdog monitoring active")
        print(f"Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while True:
                iteration += 1
                self.check_all_systems()

                print(f"\n[WAITING] Next check in {self.check_interval}s ({self.check_interval // 60} min)...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] Watchdog stopped by user")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='System Watchdog')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (default: 300 = 5 min)')

    args = parser.parse_args()

    watchdog = SystemWatchdog(check_interval=args.interval)
    watchdog.run()
