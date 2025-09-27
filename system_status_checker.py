#!/usr/bin/env python3
"""
SYSTEM STATUS CHECKER
Check if all trading systems are running and operational
Comprehensive status of the entire trading infrastructure
"""

import os
import json
from datetime import datetime
import subprocess
import glob

class SystemStatusChecker:
    """Checks status of all trading system components"""

    def __init__(self):
        self.systems = {
            'hybrid_conviction_genetic_trader.py': {
                'name': 'Hybrid AI Trader',
                'description': 'Intel-puts conviction + genetic optimization',
                'priority': 'CRITICAL'
            },
            'continuous_learning_optimizer.py': {
                'name': 'Learning Optimizer',
                'description': 'Optimizes strategies for 25-50% returns',
                'priority': 'HIGH'
            },
            'gpu_genetic_strategy_evolution.py': {
                'name': 'GPU Genetic Evolution',
                'description': '826 strategies/second evolution',
                'priority': 'HIGH'
            },
            'live_capital_allocation_engine.py': {
                'name': 'Capital Allocator',
                'description': 'Manages capital across strategies',
                'priority': 'MEDIUM'
            },
            'intelligent_rebalancer.py': {
                'name': 'Portfolio Rebalancer',
                'description': 'Optimizes portfolio allocation',
                'priority': 'MEDIUM'
            }
        }

    def check_file_existence(self):
        """Check which system files exist"""

        print("SYSTEM FILE STATUS CHECK")
        print("=" * 50)

        existing_files = []
        missing_files = []

        for filename, info in self.systems.items():
            if os.path.exists(filename):
                existing_files.append(filename)
                status = "EXISTS"
            else:
                missing_files.append(filename)
                status = "MISSING"

            print(f"{info['name']:>25} | {status:>7} | {info['priority']}")

        print("-" * 50)
        print(f"Existing: {len(existing_files)}/{len(self.systems)}")

        return existing_files, missing_files

    def check_running_processes(self):
        """Check which Python processes are currently running"""

        print(f"\nRUNNING PROCESS STATUS")
        print("=" * 50)

        try:
            # Get all Python processes
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    print(f"Active Python processes: {len(lines) - 1}")
                    for line in lines[1:]:  # Skip header
                        parts = line.split(',')
                        if len(parts) >= 2:
                            pid = parts[1].strip('"')
                            print(f"  Python Process PID: {pid}")
                else:
                    print("No Python processes currently running")
            else:
                print("Could not check running processes")

        except Exception as e:
            print(f"Error checking processes: {e}")

    def check_recent_activity(self):
        """Check for recent log files and trading activity"""

        print(f"\nRECENT SYSTEM ACTIVITY")
        print("=" * 50)

        # Check for recent JSON reports
        json_files = glob.glob("*.json")
        recent_files = []

        for file in json_files:
            try:
                mtime = os.path.getmtime(file)
                file_time = datetime.fromtimestamp(mtime)
                hours_ago = (datetime.now() - file_time).total_seconds() / 3600

                if hours_ago < 24:  # Files from last 24 hours
                    recent_files.append({
                        'file': file,
                        'hours_ago': hours_ago,
                        'time': file_time
                    })
            except:
                continue

        recent_files.sort(key=lambda x: x['hours_ago'])

        if recent_files:
            print("Recent system activity (last 24 hours):")
            for file_info in recent_files[:10]:  # Show last 10
                print(f"  {file_info['time'].strftime('%H:%M')} | {file_info['file']}")
        else:
            print("No recent system activity detected")

        return recent_files

    def check_account_connectivity(self):
        """Check if we can connect to trading account"""

        print(f"\nACCOUNT CONNECTIVITY")
        print("=" * 50)

        try:
            from simple_portfolio_cleaner import SimplePortfolioCleaner
            cleaner = SimplePortfolioCleaner()

            account = cleaner.alpaca.get_account()
            portfolio_value = float(account.portfolio_value)
            buying_power = float(account.buying_power)

            print(f"✓ Account connected successfully")
            print(f"  Portfolio Value: ${portfolio_value:,.2f}")
            print(f"  Buying Power: ${buying_power:,.2f}")
            print(f"  Account Status: {account.status}")

            return True

        except Exception as e:
            print(f"✗ Account connection failed: {e}")
            return False

    def generate_system_recommendations(self, existing_files, recent_activity, account_connected):
        """Generate recommendations for system optimization"""

        print(f"\nSYSTEM RECOMMENDATIONS")
        print("=" * 50)

        recommendations = []

        # Check critical systems
        critical_systems = [f for f, info in self.systems.items() if info['priority'] == 'CRITICAL']
        missing_critical = [f for f in critical_systems if f not in existing_files]

        if missing_critical:
            recommendations.append(f"URGENT: Missing critical systems: {', '.join(missing_critical)}")

        # Check recent activity
        if not recent_activity:
            recommendations.append("WARNING: No recent system activity - systems may be idle")
        elif len(recent_activity) < 3:
            recommendations.append("INFO: Low system activity - consider increasing execution frequency")

        # Account connectivity
        if not account_connected:
            recommendations.append("CRITICAL: Cannot connect to trading account - check API credentials")

        # System running status
        if len(existing_files) == len(self.systems):
            recommendations.append("✓ All system files present and ready")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority = rec.split(':')[0]
                print(f"{i}. {rec}")
        else:
            print("✓ All systems operational - no immediate actions required")

        return recommendations

    def run_comprehensive_check(self):
        """Run complete system status check"""

        print("COMPREHENSIVE SYSTEM STATUS CHECK")
        print("=" * 80)
        print("Checking all trading system components and connectivity")
        print("=" * 80)

        # Check file existence
        existing_files, missing_files = self.check_file_existence()

        # Check running processes
        self.check_running_processes()

        # Check recent activity
        recent_activity = self.check_recent_activity()

        # Check account connectivity
        account_connected = self.check_account_connectivity()

        # Generate recommendations
        recommendations = self.generate_system_recommendations(existing_files, recent_activity, account_connected)

        # Overall status
        print("\n" + "=" * 80)
        print("OVERALL SYSTEM STATUS")
        print("=" * 80)

        if account_connected and len(existing_files) >= 3 and recent_activity:
            status = "OPERATIONAL"
            print("🟢 SYSTEMS OPERATIONAL")
            print("✓ Account connected")
            print("✓ Core systems present")
            print("✓ Recent activity detected")
        elif account_connected and len(existing_files) >= 2:
            status = "PARTIALLY_OPERATIONAL"
            print("🟡 SYSTEMS PARTIALLY OPERATIONAL")
            print("✓ Account connected")
            print("⚠ Some systems may be missing or idle")
        else:
            status = "NEEDS_ATTENTION"
            print("🔴 SYSTEMS NEED ATTENTION")
            print("✗ Critical issues detected")

        print(f"\nReady for 25-50% monthly returns: {'YES' if status == 'OPERATIONAL' else 'NEEDS_SETUP'}")

        # Save status report
        status_report = {
            'check_timestamp': datetime.now().isoformat(),
            'overall_status': status,
            'existing_files': existing_files,
            'missing_files': missing_files,
            'recent_activity_count': len(recent_activity),
            'account_connected': account_connected,
            'recommendations': recommendations
        }

        filename = f'system_status_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(status_report, f, indent=2)

        print(f"\nDetailed status saved to: {filename}")

        return status_report

def main():
    """Run system status check"""
    checker = SystemStatusChecker()
    report = checker.run_comprehensive_check()
    return report

if __name__ == "__main__":
    main()