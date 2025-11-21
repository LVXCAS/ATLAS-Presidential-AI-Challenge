#!/usr/bin/env python3
"""
SYSTEM HEALTH MONITOR
=====================
Monitors all trading systems for health and performance

MONITORS:
- System uptime and status
- P&L (profits and losses)
- Error rates
- Trading frequency
- Position sizes
- Account balance

ALERTS:
- System crashes
- Excessive losses (>5% daily)
- No trading activity (stuck)
- API errors
- Account issues
"""

import os
import json
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class SystemHealthMonitor:
    """Monitor health and performance of all trading systems"""

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # System registry
        self.systems = {}

        # Error log
        self.errors = []

        # Health thresholds
        self.max_daily_loss_pct = 0.05  # 5% max daily loss
        self.max_errors_per_hour = 10
        self.min_trades_per_day = 1

        print("[HEALTH MONITOR] Initialized")

    def register_system(self, system_id: str, system_name: str):
        """Register a trading system for monitoring"""

        self.systems[system_id] = {
            'name': system_name,
            'status': 'RUNNING',
            'start_time': datetime.now(),
            'last_heartbeat': datetime.now(),
            'trades_today': 0,
            'errors_today': 0,
            'last_error': None
        }

        print(f"[HEALTH MONITOR] Registered system: {system_name}")

    def heartbeat(self, system_id: str):
        """Update system heartbeat (call this regularly from each system)"""

        if system_id in self.systems:
            self.systems[system_id]['last_heartbeat'] = datetime.now()
            self.systems[system_id]['status'] = 'RUNNING'

    def log_error(self, system_id: str, error_message: str):
        """Log an error for a system"""

        error = {
            'system': system_id,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }

        self.errors.append(error)

        if system_id in self.systems:
            self.systems[system_id]['errors_today'] += 1
            self.systems[system_id]['last_error'] = error_message

        print(f"[ERROR] {system_id}: {error_message}")

    def check_account_health(self) -> Dict:
        """Check account health and P&L"""

        try:
            account = self.api.get_account()

            equity = float(account.equity)
            buying_power = float(account.buying_power)
            cash = float(account.cash)

            # Calculate day's P&L
            positions = self.api.list_positions()
            total_unrealized_pl = sum(float(p.unrealized_pl) for p in positions)
            total_unrealized_plpc = sum(float(p.unrealized_plpc) for p in positions) / len(positions) if positions else 0

            # Check if account is in good standing
            issues = []

            # Check for pattern day trader restriction
            if account.pattern_day_trader and account.daytrade_count >= 3:
                issues.append("Pattern Day Trader restriction - limited day trades")

            # Check for low buying power
            if buying_power < equity * 0.1:
                issues.append(f"Low buying power: ${buying_power:,.2f} ({buying_power/equity*100:.1f}% of equity)")

            # Check for excessive losses
            if total_unrealized_plpc < -self.max_daily_loss_pct:
                issues.append(f"Excessive losses: {total_unrealized_plpc*100:.1f}% unrealized loss")

            status = 'HEALTHY' if not issues else 'WARNING' if total_unrealized_plpc > -self.max_daily_loss_pct else 'CRITICAL'

            return {
                'status': status,
                'equity': equity,
                'buying_power': buying_power,
                'cash': cash,
                'unrealized_pl': total_unrealized_pl,
                'unrealized_plpc': total_unrealized_plpc,
                'position_count': len(positions),
                'issues': issues
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'issues': ['Cannot fetch account data']
            }

    def check_system_health(self, system_id: str) -> Dict:
        """Check health of a specific system"""

        if system_id not in self.systems:
            return {
                'status': 'UNKNOWN',
                'message': 'System not registered'
            }

        system = self.systems[system_id]

        # Check last heartbeat
        time_since_heartbeat = (datetime.now() - system['last_heartbeat']).total_seconds()

        status = 'HEALTHY'
        issues = []

        # Check if system is responsive
        if time_since_heartbeat > 300:  # 5 minutes
            status = 'UNRESPONSIVE'
            issues.append(f"No heartbeat for {time_since_heartbeat/60:.1f} minutes")

        # Check error rate
        if system['errors_today'] > self.max_errors_per_hour:
            status = 'WARNING' if status == 'HEALTHY' else status
            issues.append(f"High error rate: {system['errors_today']} errors today")

        # Check trading activity
        uptime_hours = (datetime.now() - system['start_time']).total_seconds() / 3600
        if uptime_hours > 2 and system['trades_today'] < self.min_trades_per_day:
            issues.append(f"Low trading activity: {system['trades_today']} trades today")

        return {
            'status': status,
            'name': system['name'],
            'uptime_hours': uptime_hours,
            'trades_today': system['trades_today'],
            'errors_today': system['errors_today'],
            'last_heartbeat': system['last_heartbeat'].isoformat(),
            'issues': issues
        }

    def check_all_systems(self) -> Dict:
        """Check health of all systems"""

        print(f"\n{'='*80}")
        print("SYSTEM HEALTH CHECK")
        print(f"{'='*80}")
        print(f"Time: {datetime.now().strftime('%I:%M:%S %p')}")

        # Check account health
        account_health = self.check_account_health()

        print(f"\n[ACCOUNT]")
        print(f"  Status: {account_health['status']}")
        print(f"  Equity: ${account_health.get('equity', 0):,.2f}")
        print(f"  Unrealized P&L: ${account_health.get('unrealized_pl', 0):,.2f} ({account_health.get('unrealized_plpc', 0)*100:+.2f}%)")
        print(f"  Positions: {account_health.get('position_count', 0)}")

        if account_health.get('issues'):
            print(f"  Issues:")
            for issue in account_health['issues']:
                print(f"    - {issue}")

        # Check each system
        system_statuses = {}

        print(f"\n[SYSTEMS]")
        for system_id, system_data in self.systems.items():
            health = self.check_system_health(system_id)
            system_statuses[system_id] = health

            print(f"  {health['name']}: {health['status']}")
            print(f"    Uptime: {health['uptime_hours']:.1f} hours")
            print(f"    Trades: {health['trades_today']}")
            print(f"    Errors: {health['errors_today']}")

            if health['issues']:
                for issue in health['issues']:
                    print(f"    - {issue}")

        # Overall status
        all_statuses = [account_health['status']] + [s['status'] for s in system_statuses.values()]

        if 'CRITICAL' in all_statuses or 'ERROR' in all_statuses:
            overall_status = 'CRITICAL'
        elif 'WARNING' in all_statuses or 'UNRESPONSIVE' in all_statuses:
            overall_status = 'WARNING'
        else:
            overall_status = 'HEALTHY'

        print(f"\n[OVERALL STATUS]: {overall_status}")
        print(f"{'='*80}\n")

        return {
            'timestamp': datetime.now().isoformat(),
            'status': overall_status,
            'account': account_health,
            'systems': system_statuses,
            'errors': self.errors[-10:]  # Last 10 errors
        }

    def emergency_stop_if_needed(self) -> bool:
        """
        Check if emergency stop is needed

        Returns:
            True if trading should stop, False otherwise
        """

        health = self.check_all_systems()

        # Stop if critical
        if health['status'] == 'CRITICAL':
            print("\n" + "=" * 80)
            print("EMERGENCY STOP TRIGGERED")
            print("=" * 80)
            print("Reason: Critical system health")

            # Try to close all positions
            try:
                self.close_all_positions()
            except Exception as e:
                print(f"[ERROR] Could not close positions: {e}")

            return True

        # Stop if excessive losses
        account = health['account']
        if account.get('unrealized_plpc', 0) < -self.max_daily_loss_pct:
            print("\n" + "=" * 80)
            print("EMERGENCY STOP TRIGGERED")
            print("=" * 80)
            print(f"Reason: Excessive losses ({account['unrealized_plpc']*100:.1f}%)")

            try:
                self.close_all_positions()
            except Exception as e:
                print(f"[ERROR] Could not close positions: {e}")

            return True

        return False

    def close_all_positions(self):
        """Close all positions in emergency"""

        print("\n[EMERGENCY] Closing all positions...")

        try:
            positions = self.api.list_positions()

            for position in positions:
                print(f"  Closing {position.symbol}: {position.qty} shares")
                self.api.close_position(position.symbol)

            print(f"[OK] Closed {len(positions)} positions")

        except Exception as e:
            print(f"[ERROR] Failed to close positions: {e}")
            raise

    def save_health_report(self, filename='system_health_report.json'):
        """Save health report to file"""

        report = self.check_all_systems()

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n[SAVED] Health report: {filename}")


def test_health_monitor():
    """Test the health monitor"""

    print("\n" + "=" * 80)
    print("TESTING SYSTEM HEALTH MONITOR")
    print("=" * 80)

    monitor = SystemHealthMonitor()

    # Register test systems
    monitor.register_system('forex', 'Forex Elite V4')
    monitor.register_system('options', 'Adaptive Dual Options')

    # Simulate heartbeats
    monitor.heartbeat('forex')
    monitor.heartbeat('options')

    # Simulate an error
    monitor.log_error('options', 'Test error: API timeout')

    # Check health
    health = monitor.check_all_systems()

    # Check if emergency stop needed
    needs_stop = monitor.emergency_stop_if_needed()
    print(f"\nEmergency stop needed: {'YES' if needs_stop else 'NO'}")

    # Save report
    monitor.save_health_report()

    print("\n" + "=" * 80)
    print("HEALTH MONITOR TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_health_monitor()
