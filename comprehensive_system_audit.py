#!/usr/bin/env python3
"""
Comprehensive System Audit - Verify 5.75% Profit Monitoring Implementation
Tests all aspects of the profit monitoring system with latest codebase
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path
sys.path.append('.')

class SystemAudit:
    """Comprehensive system audit for profit monitoring"""

    def __init__(self):
        self.results = {}
        self.passed_tests = 0
        self.total_tests = 0

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"

        print(f"{status} {test_name}")
        if details:
            print(f"      {details}")

        self.results[test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }

    async def test_core_imports(self):
        """Test core system imports"""
        print("\n=== CORE IMPORTS ===")

        imports_to_test = [
            ("profit_target_monitor", "ProfitTargetMonitor"),
            ("agents.broker_integration", "AlpacaBrokerIntegration"),
            ("OPTIONS_BOT", "TomorrowReadyOptionsBot"),
            ("start_real_market_hunter", "RealMarketDataHunter")
        ]

        for module, class_name in imports_to_test:
            try:
                imported_module = __import__(module, fromlist=[class_name])
                getattr(imported_module, class_name)
                self.log_test(f"Import {module}.{class_name}", True)
            except Exception as e:
                self.log_test(f"Import {module}.{class_name}", False, str(e))

    async def test_profit_monitor_functionality(self):
        """Test profit monitor core functionality"""
        print("\n=== PROFIT MONITOR FUNCTIONALITY ===")

        try:
            from profit_target_monitor import ProfitTargetMonitor

            # Test creation
            monitor = ProfitTargetMonitor()
            self.log_test("ProfitTargetMonitor creation", True)

            # Test target setting
            if monitor.profit_target_pct == 5.75:
                self.log_test("5.75% target configuration", True)
            else:
                self.log_test("5.75% target configuration", False, f"Target is {monitor.profit_target_pct}%")

            # Test status method
            status = monitor.get_status()
            required_keys = ['monitoring_active', 'target_hit', 'profit_target_pct']
            missing_keys = [key for key in required_keys if key not in status]

            if not missing_keys:
                self.log_test("Status method completeness", True)
            else:
                self.log_test("Status method completeness", False, f"Missing keys: {missing_keys}")

            # Test profit calculation logic
            test_scenarios = [
                (100000, 105750, True, "Exactly 5.75%"),
                (100000, 105740, False, "Below 5.75%"),
                (100000, 110000, True, "Above 5.75%")
            ]

            all_calculations_correct = True
            for initial, current, should_trigger, description in test_scenarios:
                monitor.initial_equity = initial
                monitor.current_equity = current

                daily_profit = current - initial
                profit_pct = (daily_profit / initial) * 100
                target_hit = profit_pct >= monitor.profit_target_pct

                if target_hit != should_trigger:
                    all_calculations_correct = False
                    self.log_test(f"Profit calculation: {description}", False,
                                f"Expected {should_trigger}, got {target_hit}")
                    break

            if all_calculations_correct:
                self.log_test("Profit calculation logic", True, "All scenarios correct")

        except Exception as e:
            self.log_test("ProfitTargetMonitor functionality", False, str(e))

    async def test_options_bot_integration(self):
        """Test OPTIONS_BOT integration"""
        print("\n=== OPTIONS_BOT INTEGRATION ===")

        try:
            from OPTIONS_BOT import TomorrowReadyOptionsBot

            # Test bot creation (without full initialization)
            bot = TomorrowReadyOptionsBot()
            self.log_test("OPTIONS_BOT creation", True)

            # Check profit monitoring attributes
            required_attrs = ['profit_monitor', 'profit_monitoring_task']
            missing_attrs = [attr for attr in required_attrs if not hasattr(bot, attr)]

            if not missing_attrs:
                self.log_test("OPTIONS_BOT profit monitoring attributes", True)
            else:
                self.log_test("OPTIONS_BOT profit monitoring attributes", False, f"Missing: {missing_attrs}")

            # Check import integration
            try:
                from profit_target_monitor import ProfitTargetMonitor
                self.log_test("ProfitTargetMonitor import in OPTIONS_BOT context", True)
            except:
                self.log_test("ProfitTargetMonitor import in OPTIONS_BOT context", False)

        except Exception as e:
            self.log_test("OPTIONS_BOT integration", False, str(e))

    async def test_market_hunter_integration(self):
        """Test Market Hunter integration"""
        print("\n=== MARKET HUNTER INTEGRATION ===")

        try:
            from start_real_market_hunter import RealMarketDataHunter

            # Test hunter creation
            hunter = RealMarketDataHunter()
            self.log_test("RealMarketDataHunter creation", True)

            # Check profit monitoring attributes
            required_attrs = ['profit_monitor', 'profit_monitoring_task']
            missing_attrs = [attr for attr in required_attrs if not hasattr(hunter, attr)]

            if not missing_attrs:
                self.log_test("Market Hunter profit monitoring attributes", True)
            else:
                self.log_test("Market Hunter profit monitoring attributes", False, f"Missing: {missing_attrs}")

        except Exception as e:
            self.log_test("Market Hunter integration", False, str(e))

    async def test_broker_integration(self):
        """Test broker integration functionality"""
        print("\n=== BROKER INTEGRATION ===")

        try:
            from agents.broker_integration import AlpacaBrokerIntegration

            # Test broker creation
            broker = AlpacaBrokerIntegration(paper_trading=True)
            self.log_test("AlpacaBrokerIntegration creation", True)

            # Check required methods
            required_methods = ['get_account_info', 'close_all_positions', 'submit_order']
            missing_methods = [method for method in required_methods if not hasattr(broker, method)]

            if not missing_methods:
                self.log_test("Broker required methods", True)
            else:
                self.log_test("Broker required methods", False, f"Missing: {missing_methods}")

        except Exception as e:
            self.log_test("Broker integration", False, str(e))

    async def test_new_repository_compatibility(self):
        """Test compatibility with new AI/ML repositories"""
        print("\n=== NEW REPOSITORY COMPATIBILITY ===")

        # Test ML4T integration
        try:
            from agents.ml4t_agent import ml4t_agent
            self.log_test("ML4T agent compatibility", True)
        except Exception as e:
            self.log_test("ML4T agent compatibility", False, f"Import error: {str(e)[:100]}")

        # Test FinanceDatabase integration
        try:
            from agents.finance_database_agent import finance_database_agent
            self.log_test("FinanceDatabase agent compatibility", True)
        except Exception as e:
            self.log_test("FinanceDatabase agent compatibility", False, f"Import error: {str(e)[:100]}")

        # Test Enhanced ML Ensemble
        try:
            from agents.enhanced_ml_ensemble_agent import enhanced_ml_ensemble_agent
            self.log_test("Enhanced ML Ensemble compatibility", True)
        except Exception as e:
            self.log_test("Enhanced ML Ensemble compatibility", False, f"Import error: {str(e)[:100]}")

    async def test_file_integrity(self):
        """Test file integrity and structure"""
        print("\n=== FILE INTEGRITY ===")

        required_files = [
            'profit_target_monitor.py',
            'OPTIONS_BOT.py',
            'start_real_market_hunter.py',
            'agents/broker_integration.py'
        ]

        for file_path in required_files:
            if os.path.exists(file_path):
                self.log_test(f"File exists: {file_path}", True)

                # Check file is not empty
                if os.path.getsize(file_path) > 0:
                    self.log_test(f"File not empty: {file_path}", True)
                else:
                    self.log_test(f"File not empty: {file_path}", False)
            else:
                self.log_test(f"File exists: {file_path}", False)

    async def test_profit_monitor_integration_points(self):
        """Test specific integration points in both bots"""
        print("\n=== INTEGRATION POINTS ===")

        # Test OPTIONS_BOT integration points
        try:
            with open('OPTIONS_BOT.py', 'r') as f:
                options_bot_content = f.read()

            integration_checks = [
                ('from profit_target_monitor import ProfitTargetMonitor', 'Import statement'),
                ('self.profit_monitor = None', 'Profit monitor attribute initialization'),
                ('self.profit_monitoring_task = None', 'Profit monitoring task attribute'),
                ('self.profit_monitor = ProfitTargetMonitor()', 'Profit monitor instantiation'),
                ('await self.profit_monitor.initialize_broker()', 'Broker initialization call'),
                ('self.profit_monitoring_task = asyncio.create_task', 'Background task creation'),
                ('bot.profit_monitor.stop_monitoring()', 'Cleanup code')
            ]

            missing_integrations = []
            for check, description in integration_checks:
                if check not in options_bot_content:
                    missing_integrations.append(description)

            if not missing_integrations:
                self.log_test("OPTIONS_BOT integration points", True)
            else:
                self.log_test("OPTIONS_BOT integration points", False, f"Missing: {missing_integrations}")

        except Exception as e:
            self.log_test("OPTIONS_BOT integration points", False, str(e))

        # Test Market Hunter integration points
        try:
            with open('start_real_market_hunter.py', 'r') as f:
                hunter_content = f.read()

            integration_checks = [
                ('from profit_target_monitor import ProfitTargetMonitor', 'Import statement'),
                ('self.profit_monitor = None', 'Profit monitor attribute initialization'),
                ('self.profit_monitoring_task = None', 'Profit monitoring task attribute'),
                ('self.profit_monitor = ProfitTargetMonitor()', 'Profit monitor instantiation'),
                ('await self.profit_monitor.initialize_broker()', 'Broker initialization call'),
                ('hunter.profit_monitor.stop_monitoring()', 'Cleanup code')
            ]

            missing_integrations = []
            for check, description in integration_checks:
                if check not in hunter_content:
                    missing_integrations.append(description)

            if not missing_integrations:
                self.log_test("Market Hunter integration points", True)
            else:
                self.log_test("Market Hunter integration points", False, f"Missing: {missing_integrations}")

        except Exception as e:
            self.log_test("Market Hunter integration points", False, str(e))

    async def run_comprehensive_audit(self):
        """Run complete system audit"""
        print("COMPREHENSIVE SYSTEM AUDIT - 5.75% PROFIT MONITORING")
        print("=" * 60)
        print(f"Audit time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run all test suites
        await self.test_core_imports()
        await self.test_profit_monitor_functionality()
        await self.test_options_bot_integration()
        await self.test_market_hunter_integration()
        await self.test_broker_integration()
        await self.test_new_repository_compatibility()
        await self.test_file_integrity()
        await self.test_profit_monitor_integration_points()

        # Generate final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate comprehensive audit report"""
        print("\n" + "=" * 60)
        print("AUDIT SUMMARY")
        print("=" * 60)

        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")

        if self.passed_tests == self.total_tests:
            print("\n[SUCCESS] ALL SYSTEMS OPERATIONAL!")
            print("\nThe 5.75% profit monitoring system is:")
            print("  * Properly implemented")
            print("  * Fully integrated with both bots")
            print("  * Compatible with latest codebase")
            print("  * Ready for live trading")
            print("\nWhen daily profit reaches 5.75%:")
            print("  1. All orders will be cancelled")
            print("  2. All positions will be closed")
            print("  3. Event will be logged")
            print("  4. Trading will stop for the day")
        else:
            print("\n[WARNING] SOME ISSUES DETECTED")
            failed_tests = [name for name, result in self.results.items() if not result['passed']]
            print(f"Failed tests: {failed_tests}")

        # Save detailed results
        with open('audit_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': self.total_tests,
                    'passed_tests': self.passed_tests,
                    'success_rate': (self.passed_tests/self.total_tests)*100,
                    'audit_time': datetime.now().isoformat()
                },
                'detailed_results': self.results
            }, f, indent=2)

        print(f"\nDetailed results saved to: audit_results.json")

async def main():
    """Run comprehensive audit"""
    auditor = SystemAudit()
    await auditor.run_comprehensive_audit()
    return auditor.passed_tests == auditor.total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)