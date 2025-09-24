"""
AUTONOMOUS MASTER SYSTEM
Orchestrates all autonomous trading systems including options discovery
"""

import asyncio
import subprocess
import psutil
import time
import logging
import json
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_master.log'),
        logging.StreamHandler()
    ]
)

class AutonomousMasterSystem:
    """Master orchestrator for all autonomous trading systems"""

    def __init__(self):
        self.systems = {
            'market_open': {
                'script': 'autonomous_market_open_system.py',
                'description': 'Market Open Trading System',
                'active': False,
                'process': None
            },
            'continuous': {
                'script': 'truly_autonomous_system.py',
                'description': 'Continuous Intelligence System',
                'active': False,
                'process': None
            },
            'options_discovery': {
                'script': 'autonomous_options_discovery.py',
                'description': 'Options Discovery System',
                'active': False,
                'process': None
            },
            'intelligent_rebalancer': {
                'script': 'intelligent_rebalancer.py',
                'description': 'Intelligent Portfolio Rebalancer',
                'active': False,
                'process': None
            }
        }

        self.portfolio_target = 515000  # Your current portfolio value
        self.monthly_roi_target = 0.527  # 52.7% monthly target

    def kill_existing_autonomous_systems(self):
        """Clean shutdown of existing autonomous systems"""

        logging.info("Stopping all existing autonomous systems...")

        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if any(script in cmdline for script in [
                        'autonomous_market_open_system.py',
                        'truly_autonomous_system.py',
                        'autonomous_options_discovery.py',
                        'intelligent_rebalancer.py'
                    ]):
                        logging.info(f"Stopping PID {proc.info['pid']}")
                        proc.kill()
                        killed_count += 1
            except:
                continue

        logging.info(f"Stopped {killed_count} existing autonomous processes")
        time.sleep(3)

    def start_system(self, system_key):
        """Start a specific autonomous system"""

        system = self.systems[system_key]
        script = system['script']

        try:
            # Start system in background
            process = subprocess.Popen(
                ['python', script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

            system['process'] = process
            system['active'] = True

            logging.info(f"STARTED: {system['description']} (PID: {process.pid})")
            return True

        except Exception as e:
            logging.error(f"Failed to start {system['description']}: {e}")
            return False

    def start_all_systems(self):
        """Start all autonomous systems"""

        logging.info("Starting complete autonomous trading empire...")

        # Start systems in order
        systems_to_start = ['continuous', 'market_open', 'intelligent_rebalancer', 'options_discovery']

        started_count = 0
        for system_key in systems_to_start:
            if self.start_system(system_key):
                started_count += 1
                time.sleep(2)  # Stagger startup

        logging.info(f"Successfully started {started_count}/{len(systems_to_start)} systems")
        return started_count == len(systems_to_start)

    def check_system_health(self):
        """Check health of all running systems"""

        logging.info("Checking system health...")

        healthy_systems = 0
        total_systems = len(self.systems)

        for system_key, system in self.systems.items():
            if system['active'] and system['process']:
                try:
                    # Check if process is still running
                    if system['process'].poll() is None:
                        healthy_systems += 1
                        logging.info(f"HEALTHY: {system['description']}")
                    else:
                        logging.warning(f"DEAD: {system['description']} - Restarting...")
                        system['active'] = False
                        self.start_system(system_key)
                except:
                    logging.warning(f"ERROR: {system['description']} - Restarting...")
                    system['active'] = False
                    self.start_system(system_key)

        health_percentage = (healthy_systems / total_systems) * 100
        logging.info(f"System Health: {health_percentage:.1f}% ({healthy_systems}/{total_systems})")

        return health_percentage > 75

    def generate_status_report(self):
        """Generate comprehensive status report"""

        status_report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_target,
            'monthly_target': self.monthly_roi_target,
            'systems_status': {},
            'options_discovery_active': self.systems['options_discovery']['active'],
            'health_check': 'PASS' if self.check_system_health() else 'FAIL'
        }

        for system_key, system in self.systems.items():
            status_report['systems_status'][system_key] = {
                'description': system['description'],
                'active': system['active'],
                'pid': system['process'].pid if system['process'] else None
            }

        # Save status report
        with open('autonomous_master_status.json', 'w') as f:
            json.dump(status_report, f, indent=2, default=str)

        logging.info("Status report generated: autonomous_master_status.json")
        return status_report

    async def master_monitoring_loop(self):
        """Main monitoring loop for all systems"""

        logging.info("Master monitoring loop started")

        while True:
            try:
                # Check system health every 5 minutes
                self.check_system_health()

                # Generate status report every 15 minutes
                if datetime.now().minute % 15 == 0:
                    self.generate_status_report()

                # Wait 5 minutes before next check
                await asyncio.sleep(300)

            except Exception as e:
                logging.error(f"Error in master monitoring loop: {e}")
                await asyncio.sleep(60)

    def launch_autonomous_empire(self):
        """Launch complete autonomous trading empire"""

        print("AUTONOMOUS TRADING EMPIRE")
        print("=" * 50)
        print("Launching complete autonomous trading system...")
        print(f"Portfolio: ${self.portfolio_target:,}")
        print(f"Monthly Target: {self.monthly_roi_target:.1%}")
        print("=" * 50)

        # Clean shutdown existing systems
        self.kill_existing_autonomous_systems()

        # Start all systems
        success = self.start_all_systems()

        if success:
            print("\nAUTONOMOUS EMPIRE LAUNCH: SUCCESS")
            print("Systems active:")
            for system_key, system in self.systems.items():
                if system['active']:
                    print(f"  SUCCESS: {system['description']}")

            print(f"\nOptions Discovery: ACTIVE")
            print("Your system will now:")
            print("- Monitor portfolio every 60 seconds")
            print("- Execute trades at market open (6:30 AM PT)")
            print("- Discover new stocks for options trading")
            print("- Intelligently rebalance positions")
            print("- Target 52.7% monthly ROI")
            print("\nCheck autonomous_master.log for activity")

        else:
            print("\nWARNING: Some systems failed to start")
            print("Check logs for details")

        return success

async def main():
    """Start autonomous master system"""

    master = AutonomousMasterSystem()

    # Launch the empire
    launch_success = master.launch_autonomous_empire()

    if launch_success:
        # Start monitoring loop
        await master.master_monitoring_loop()
    else:
        print("Launch failed - check system configuration")

if __name__ == "__main__":
    asyncio.run(main())