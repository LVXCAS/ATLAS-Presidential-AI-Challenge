"""
Unified System Controller - Master Control for All Trading Systems
Manages and monitors all agentic trading systems
"""

import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
import threading
import signal

class UnifiedSystemController:
    def __init__(self):
        self.systems = {
            'forex': {
                'name': 'Forex Elite Trading',
                'script': 'WORKING_FOREX_MONITOR.py',
                'process': None,
                'status': 'stopped',
                'config': 'config/forex_elite_config.json'
            },
            'options': {
                'name': 'Options S&P 500 Scanner',
                'script': 'AGENTIC_OPTIONS_SCANNER_SP500.py',
                'process': None,
                'status': 'stopped'
            },
            'telegram': {
                'name': 'Telegram Control Bot',
                'script': 'TELEGRAM_BOT.py',
                'process': None,
                'status': 'stopped'
            },
            'empire': {
                'name': 'Autonomous Trading Empire',
                'script': 'PRODUCTION/autonomous_trading_empire.py',
                'process': None,
                'status': 'stopped'
            }
        }

        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)

        print("=" * 70)
        print("UNIFIED SYSTEM CONTROLLER")
        print("=" * 70)
        print("Managing all trading systems with no duplicates")
        print("=" * 70)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n[CONTROLLER] Shutting down gracefully...")
        self.running = False
        self.stop_all_systems()
        sys.exit(0)

    def kill_duplicates(self):
        """Kill all duplicate Python processes"""
        print("\n[CLEANING] Removing duplicate processes...")

        try:
            # Get all python processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and len(cmdline) > 1:
                            script = cmdline[1] if len(cmdline) > 1 else ''

                            # List of scripts to kill duplicates
                            duplicate_scripts = [
                                'START_ACTIVE_FOREX_PAPER_TRADING.py',
                                'START_ACTIVE_FUTURES_PAPER_TRADING.py',
                                'futures_live_validation.py',
                                'forex_futures_rd_agent.py',
                                'GPU_TRADING_ORCHESTRATOR.py'
                            ]

                            for dup_script in duplicate_scripts:
                                if dup_script in script:
                                    print(f"  Killing duplicate: PID {proc.info['pid']} - {dup_script}")
                                    proc.kill()
                                    time.sleep(0.5)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"  Error cleaning duplicates: {e}")

        print("  Duplicates cleaned")

    def start_system(self, system_key):
        """Start a specific system"""
        system = self.systems.get(system_key)
        if not system:
            return False

        if system['status'] == 'running':
            print(f"  {system['name']} already running")
            return True

        try:
            # Start the process
            cmd = [sys.executable, system['script']]
            system['process'] = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            system['status'] = 'running'
            print(f"  ✅ Started {system['name']}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to start {system['name']}: {e}")
            system['status'] = 'error'
            return False

    def stop_system(self, system_key):
        """Stop a specific system"""
        system = self.systems.get(system_key)
        if not system:
            return False

        if system['status'] == 'stopped':
            return True

        try:
            if system['process']:
                system['process'].terminate()
                time.sleep(1)
                if system['process'].poll() is None:
                    system['process'].kill()
            system['status'] = 'stopped'
            system['process'] = None
            print(f"  Stopped {system['name']}")
            return True
        except Exception as e:
            print(f"  Error stopping {system['name']}: {e}")
            return False

    def check_system_health(self):
        """Check health of all systems"""
        health_status = {}

        for key, system in self.systems.items():
            if system['process'] and system['status'] == 'running':
                # Check if process is still alive
                if system['process'].poll() is None:
                    health_status[key] = 'healthy'
                else:
                    health_status[key] = 'crashed'
                    system['status'] = 'crashed'
            else:
                health_status[key] = system['status']

        return health_status

    def restart_crashed_systems(self):
        """Automatically restart crashed systems"""
        health = self.check_system_health()

        for key, status in health.items():
            if status == 'crashed':
                print(f"\n[AUTO-RESTART] {self.systems[key]['name']} crashed, restarting...")
                self.stop_system(key)
                time.sleep(2)
                self.start_system(key)

    def display_status(self):
        """Display current status of all systems"""
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 70)
        print(f"UNIFIED SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        health = self.check_system_health()

        for key, system in self.systems.items():
            status = health.get(key, 'unknown')
            status_icon = {
                'healthy': '🟢',
                'running': '🟡',
                'stopped': '🔴',
                'crashed': '❌',
                'error': '⚠️'
            }.get(status, '❓')

            print(f"{status_icon} {system['name']:30} {status.upper()}")

        # Market status
        now = datetime.now()
        if now.weekday() < 5:
            if 9 <= now.hour < 16:
                market_status = "🟢 MARKET OPEN"
            elif 16 <= now.hour < 20:
                market_status = "🟡 AFTER HOURS"
            else:
                market_status = "🔴 MARKET CLOSED"
        else:
            market_status = "🔴 WEEKEND"

        print("=" * 70)
        print(f"Market: {market_status}")
        print("=" * 70)
        print("Commands: [R]estart All | [S]top All | [Q]uit")

    def start_all_systems(self):
        """Start all systems in proper order"""
        print("\n[CONTROLLER] Starting all systems...")

        # Kill duplicates first
        self.kill_duplicates()
        time.sleep(2)

        # Start in order of importance
        order = ['telegram', 'forex', 'options', 'empire']

        for key in order:
            self.start_system(key)
            time.sleep(1)

        print("\n[CONTROLLER] All systems started")

    def stop_all_systems(self):
        """Stop all systems"""
        print("\n[CONTROLLER] Stopping all systems...")

        for key in self.systems.keys():
            self.stop_system(key)

        print("[CONTROLLER] All systems stopped")

    def monitor_loop(self):
        """Main monitoring loop"""
        # Start all systems on launch
        self.start_all_systems()

        last_health_check = time.time()

        while self.running:
            try:
                # Display status
                self.display_status()

                # Health check every 30 seconds
                if time.time() - last_health_check > 30:
                    self.restart_crashed_systems()
                    last_health_check = time.time()

                # Check for user input (non-blocking on Windows)
                if os.name == 'nt':
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()
                        if key == 'r':
                            self.stop_all_systems()
                            time.sleep(2)
                            self.start_all_systems()
                        elif key == 's':
                            self.stop_all_systems()
                        elif key == 'q':
                            self.running = False

                time.sleep(5)  # Update every 5 seconds

            except KeyboardInterrupt:
                break

        # Cleanup on exit
        self.stop_all_systems()

    def run(self):
        """Run the controller"""
        print("\n[CONTROLLER] Initializing Unified System Controller...")
        print("[CONTROLLER] This will manage all trading systems efficiently")
        print("[CONTROLLER] Press Ctrl+C to stop all systems and exit\n")

        self.monitor_loop()

        print("\n[CONTROLLER] Shutdown complete")

if __name__ == "__main__":
    controller = UnifiedSystemController()
    controller.run()