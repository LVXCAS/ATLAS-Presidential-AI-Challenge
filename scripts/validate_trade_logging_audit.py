#!/usr/bin/env python3
"""
Trade Logging and Audit Trail Validation Script - Task 8.2

This script validates the trade logging and audit trail implementation to ensure it meets all requirements:
- Comprehensive trade logging with complete metadata
- Audit trail for all system decisions and actions
- Trade reconciliation and reporting
- Data backup and recovery procedures

Requirements: Requirement 14 (Regulatory Compliance and Reporting)
Task: 8.2 Trade Logging and Audit Trail
"""

import sys
import os
import logging
from pathlib import Path
import time
import traceback
import asyncio
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.trade_logging_audit_agent import (
    TradeLoggingAuditAgent, ActionType, EntityType, LogLevel, TradeLog, AuditEvent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Class to track validation results"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.error_message = None
        self.execution_time = 0.0
        self.details = {}
    
    def set_passed(self, details: dict = None):
        self.passed = True
        if details:
            self.details = details
    
    def set_failed(self, error_message: str, details: dict = None):
        self.passed = False
        self.error_message = error_message
        if details:
            self.details = details


def test_trade_logging() -> ValidationResult:
    """Test comprehensive trade logging functionality"""
    result = ValidationResult("Trade Logging")
    start_time = time.time()
    
    try:
        # Initialize agent
        agent = TradeLoggingAuditAgent()
        
        # Test trade logging
        test_trade = {
            'order_id': 'TEST_ORD_001',
            'symbol': 'TEST',
            'side': 'BUY',
            'quantity': 100,
            'price': 100.0,
            'timestamp': time.time(),
            'strategy': 'test_strategy',
            'agent_id': 'test_agent',
            'signal_strength': 0.8,
            'market_conditions': {'test': 'data'},
            'execution_quality': {'test': 'data'},
            'risk_metrics': {'test': 'data'},
            'compliance_flags': [],
            'metadata': {'test': 'data'}
        }
        
        # Log trade
        trade_id = agent.log_trade(test_trade)
        if not trade_id:
            raise ValueError("Trade logging failed - no trade ID returned")
        
        # Verify trade was logged
        trade_history = agent.get_trade_history(limit=10)
        if not trade_history:
            raise ValueError("No trade history returned after logging trade")
        
        # Find our test trade
        test_trade_found = False
        for trade in trade_history:
            if trade.trade_id == trade_id:
                test_trade_found = True
                # Verify trade data
                if trade.symbol != test_trade['symbol']:
                    raise ValueError(f"Trade symbol mismatch: expected {test_trade['symbol']}, got {trade.symbol}")
                if trade.quantity != test_trade['quantity']:
                    raise ValueError(f"Trade quantity mismatch: expected {test_trade['quantity']}, got {trade.quantity}")
                if trade.price != test_trade['price']:
                    raise ValueError(f"Trade price mismatch: expected {test_trade['price']}, got {trade.price}")
                break
        
        if not test_trade_found:
            raise ValueError("Test trade not found in trade history")
        
        # Test trade report generation
        end_date = time.time()
        start_date = end_date - 86400  # 24 hours ago
        report = agent.generate_trade_report(start_date, end_date)
        
        if 'total_trades' not in report:
            raise ValueError("Trade report missing required fields")
        
        result.set_passed({
            'trade_logged': True,
            'trade_id': trade_id,
            'trade_history_retrieved': True,
            'trade_data_verified': True,
            'report_generated': True,
            'total_trades': report['total_trades']
        })
        
        # Clean up
        agent.trade_logger.close()
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_audit_trail() -> ValidationResult:
    """Test audit trail functionality"""
    result = ValidationResult("Audit Trail")
    start_time = time.time()
    
    try:
        # Initialize agent
        agent = TradeLoggingAuditAgent()
        
        # Test audit event logging
        test_event_id = agent.log_audit_event(
            action_type=ActionType.ORDER_SUBMITTED,
            entity_type=EntityType.ORDER,
            entity_id='TEST_ORDER_001',
            description='Test audit event for validation',
            details={'test': 'data'},
            log_level=LogLevel.INFO,
            agent_id='test_agent'
        )
        
        if not test_event_id:
            raise ValueError("Audit event logging failed - no event ID returned")
        
        # Verify audit event was logged
        audit_trail = agent.get_audit_trail(limit=10)
        if not audit_trail:
            raise ValueError("No audit trail returned after logging event")
        
        # Find our test event
        test_event_found = False
        for event in audit_trail:
            if event.event_id == test_event_id:
                test_event_found = True
                # Verify event data
                if event.action_type != ActionType.ORDER_SUBMITTED:
                    raise ValueError(f"Event action type mismatch: expected {ActionType.ORDER_SUBMITTED}, got {event.action_type}")
                if event.entity_type != EntityType.ORDER:
                    raise ValueError(f"Event entity type mismatch: expected {EntityType.ORDER}, got {event.entity_type}")
                if event.description != 'Test audit event for validation':
                    raise ValueError("Event description mismatch")
                break
        
        if not test_event_found:
            raise ValueError("Test audit event not found in audit trail")
        
        # Test filtering by action type
        order_events = agent.get_audit_trail(action_type=ActionType.ORDER_SUBMITTED, limit=10)
        if not order_events:
            raise ValueError("No order events found when filtering by action type")
        
        # Test filtering by entity type
        order_entities = agent.get_audit_trail(entity_type=EntityType.ORDER, limit=10)
        if not order_entities:
            raise ValueError("No order entities found when filtering by entity type")
        
        result.set_passed({
            'audit_event_logged': True,
            'event_id': test_event_id,
            'audit_trail_retrieved': True,
            'event_data_verified': True,
            'filtering_works': True,
            'total_events': len(audit_trail)
        })
        
        # Clean up
        agent.trade_logger.close()
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_trade_reconciliation() -> ValidationResult:
    """Test trade reconciliation functionality"""
    result = ValidationResult("Trade Reconciliation")
    start_time = time.time()
    
    try:
        # Initialize agent
        agent = TradeLoggingAuditAgent()
        
        # Test data
        broker_positions = {
            'AAPL': {'quantity': 100, 'market_value': 15000, 'unrealized_pl': 500},
            'TSLA': {'quantity': 50, 'market_value': 10000, 'unrealized_pl': -250}
        }
        
        system_positions = {
            'AAPL': {'quantity': 100, 'market_value': 15000, 'unrealized_pl': 500},
            'TSLA': {'quantity': 50, 'market_value': 10000, 'unrealized_pl': -250}
        }
        
        # Test reconciliation
        reconciliation = agent.reconcile_positions(broker_positions, system_positions)
        
        if not reconciliation:
            raise ValueError("Reconciliation failed - no report returned")
        
        # Verify reconciliation report structure
        required_fields = ['report_id', 'timestamp', 'broker_positions', 'system_positions', 
                          'discrepancies', 'reconciliation_status', 'total_market_value', 'total_unrealized_pnl']
        
        for field in required_fields:
            if not hasattr(reconciliation, field):
                raise ValueError(f"Reconciliation report missing required field: {field}")
        
        # Verify reconciliation status
        if reconciliation.reconciliation_status != "RECONCILED":
            raise ValueError(f"Expected reconciliation status 'RECONCILED', got '{reconciliation.reconciliation_status}'")
        
        # Verify no discrepancies for matching data
        if len(reconciliation.discrepancies) != 0:
            raise ValueError(f"Expected 0 discrepancies for matching data, got {len(reconciliation.discrepancies)}")
        
        # Test with discrepancies
        discrepant_system_positions = system_positions.copy()
        discrepant_system_positions['AAPL']['quantity'] = 95  # Introduce discrepancy
        
        reconciliation_with_discrepancy = agent.reconcile_positions(broker_positions, discrepant_system_positions)
        
        if len(reconciliation_with_discrepancy.discrepancies) == 0:
            raise ValueError("Expected discrepancies to be detected")
        
        result.set_passed({
            'reconciliation_works': True,
            'report_structure_valid': True,
            'matching_data_reconciled': True,
            'discrepancies_detected': True,
            'total_positions': len(broker_positions)
        })
        
        # Clean up
        agent.trade_logger.close()
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_backup_recovery() -> ValidationResult:
    """Test backup and recovery functionality"""
    result = ValidationResult("Backup and Recovery")
    start_time = time.time()
    
    try:
        # Initialize agent
        agent = TradeLoggingAuditAgent()
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Test backup data for validation\n" + "X" * 1000)
            temp_file_path = temp_file.name
        
        try:
            # Test backup creation
            backup_id = agent.create_backup(temp_file_path, "validation_test")
            
            if not backup_id:
                raise ValueError("Backup creation failed - no backup ID returned")
            
            # Verify backup exists
            backup_info = agent.get_backup_info(backup_id)
            if not backup_info:
                raise ValueError("Backup info not found after creation")
            
            # Verify backup metadata
            if backup_info.status != "COMPLETED":
                raise ValueError(f"Expected backup status 'COMPLETED', got '{backup_info.status}'")
            
            if backup_info.size_bytes <= 0:
                raise ValueError(f"Invalid backup size: {backup_info.size_bytes}")
            
            # Test backup listing
            backups = agent.list_backups()
            if len(backups) == 0:
                raise ValueError("No backups found after creation")
            
            # Test backup restoration
            restore_path = temp_file_path + ".restored"
            if agent.restore_backup(backup_id, restore_path):
                # Verify restoration
                if not Path(restore_path).exists():
                    raise ValueError("Restored file does not exist")
                
                # Clean up restored file
                Path(restore_path).unlink()
            else:
                raise ValueError("Backup restoration failed")
            
            # Test backup deletion
            if agent.delete_backup(backup_id):
                # Verify deletion
                backup_info_after = agent.get_backup_info(backup_id)
                if backup_info_after:
                    raise ValueError("Backup still exists after deletion")
            else:
                raise ValueError("Backup deletion failed")
            
            result.set_passed({
                'backup_created': True,
                'backup_id': backup_id,
                'backup_info_retrieved': True,
                'backup_listed': True,
                'backup_restored': True,
                'backup_deleted': True,
                'total_backups_tested': len(backups)
            })
            
        finally:
            # Clean up test file
            if Path(temp_file_path).exists():
                Path(temp_file_path).unlink()
        
        # Clean up
        agent.trade_logger.close()
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_integration() -> ValidationResult:
    """Test integration of all components"""
    result = ValidationResult("Integration Test")
    start_time = time.time()
    
    try:
        # Initialize agent
        agent = TradeLoggingAuditAgent()
        
        # Test complete workflow
        # 1. Log trade
        test_trade = {
            'order_id': 'INTEGRATION_TEST_001',
            'symbol': 'INTEGRATION',
            'side': 'BUY',
            'quantity': 200,
            'price': 200.0,
            'timestamp': time.time(),
            'strategy': 'integration_test',
            'agent_id': 'integration_agent',
            'signal_strength': 0.9,
            'market_conditions': {'test': 'integration'},
            'execution_quality': {'test': 'integration'},
            'risk_metrics': {'test': 'integration'},
            'compliance_flags': [],
            'metadata': {'test': 'integration'}
        }
        
        trade_id = agent.log_trade(test_trade)
        
        # 2. Log audit events
        event_id = agent.log_audit_event(
            action_type=ActionType.SIGNAL_EXECUTED,
            entity_type=EntityType.SIGNAL,
            entity_id=f"SIGNAL_{trade_id}",
            description='Integration test signal execution',
            details={'trade_id': trade_id},
            log_level=LogLevel.INFO
        )
        
        # 3. Test reconciliation
        broker_positions = {'INTEGRATION': {'quantity': 200, 'market_value': 40000, 'unrealized_pl': 0}}
        system_positions = {'INTEGRATION': {'quantity': 200, 'market_value': 40000, 'unrealized_pl': 0}}
        
        reconciliation = agent.reconcile_positions(broker_positions, system_positions)
        
        # 4. Generate report
        end_date = time.time()
        start_date = end_date - 3600  # 1 hour ago
        report = agent.generate_trade_report(start_date, end_date)
        
        # 5. Test backup
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Integration test backup data\n")
            temp_file_path = temp_file.name
        
        try:
            backup_id = agent.create_backup(temp_file_path, "integration_test")
            
            # 6. Get system status
            status = agent.get_system_status()
            
            # Verify integration
            if not trade_id:
                raise ValueError("Trade logging failed in integration test")
            
            if not event_id:
                raise ValueError("Audit event logging failed in integration test")
            
            if reconciliation.reconciliation_status != "RECONCILED":
                raise ValueError("Reconciliation failed in integration test")
            
            if 'total_trades' not in report:
                raise ValueError("Trade report generation failed in integration test")
            
            if not backup_id:
                raise ValueError("Backup creation failed in integration test")
            
            if status['status'] != 'running':
                raise ValueError("System status check failed in integration test")
            
            result.set_passed({
                'trade_logging_integrated': True,
                'audit_trail_integrated': True,
                'reconciliation_integrated': True,
                'reporting_integrated': True,
                'backup_integrated': True,
                'system_status_working': True,
                'workflow_completed': True
            })
            
        finally:
            # Clean up
            if Path(temp_file_path).exists():
                Path(temp_file_path).unlink()
            
            # Delete test backup
            if 'backup_id' in locals():
                agent.delete_backup(backup_id)
        
        # Clean up
        agent.trade_logger.close()
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def run_validation_suite() -> list[ValidationResult]:
    """Run the complete validation suite"""
    print("=" * 80)
    print("TRADE LOGGING AND AUDIT TRAIL VALIDATION SUITE - Task 8.2")
    print("=" * 80)
    
    validation_tests = [
        test_trade_logging,
        test_audit_trail,
        test_trade_reconciliation,
        test_backup_recovery,
        test_integration
    ]
    
    results = []
    
    for test_func in validation_tests:
        print(f"\n[INFO] Running: {test_func.__name__}")
        print("-" * 60)
        
        try:
            result = test_func()
            results.append(result)
            
            if result.passed:
                print(f"[OK] PASSED: {result.test_name}")
                print(f"   Execution time: {result.execution_time:.2f}s")
                if result.details:
                    print(f"   Details: {result.details}")
            else:
                print(f"[X] FAILED: {result.test_name}")
                print(f"   Error: {result.error_message}")
                print(f"   Execution time: {result.execution_time:.2f}s")
                
        except Exception as e:
            print(f"[INFO] CRASHED: {test_func.__name__}")
            print(f"   Error: {e}")
            result = ValidationResult(test_func.__name__)
            result.set_failed(f"Test crashed: {e}")
            result.execution_time = 0.0
            results.append(result)
    
    return results


def generate_validation_report(results: list[ValidationResult]) -> str:
    """Generate comprehensive validation report"""
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests
    
    print(f"\n[CHART] Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n[TIMER]  Performance Summary:")
    total_time = sum(r.execution_time for r in results)
    avg_time = total_time / total_tests if total_tests > 0 else 0
    print(f"   Total Execution Time: {total_time:.2f}s")
    print(f"   Average Test Time: {avg_time:.2f}s")
    
    print(f"\n[INFO] Detailed Results:")
    for result in results:
        status = "[OK] PASSED" if result.passed else "[X] FAILED"
        print(f"   {status}: {result.test_name} ({result.execution_time:.2f}s)")
        
        if not result.passed and result.error_message:
            print(f"      Error: {result.error_message}")
    
    # Overall assessment
    print(f"\n[TARGET] Overall Assessment:")
    if failed_tests == 0:
        print("   [PARTY] ALL TESTS PASSED! Trade Logging and Audit Trail is fully functional.")
        print("   [OK] Task 8.2 requirements have been met successfully.")
        print("   [LAUNCH] System is ready for production deployment.")
    elif failed_tests <= 2:
        print("   [WARN]  MOST TESTS PASSED. Minor issues detected.")
        print("   [TOOL] Some functionality may need attention before production.")
    else:
        print("   [X] MULTIPLE TEST FAILURES. Significant issues detected.")
        print("   [TOOLS]  System needs major fixes before proceeding.")
    
    return f"Validation completed with {passed_tests}/{total_tests} tests passed"


def main():
    """Main validation execution"""
    try:
        # Run validation suite
        results = run_validation_suite()
        
        # Generate report
        report = generate_validation_report(results)
        
        # Save validation results
        validation_file = "trade_logging_audit_validation_results.json"
        try:
            import json
            
            validation_data = {
                'timestamp': time.time(),
                'task': '8.2 Trade Logging and Audit Trail',
                'summary': {
                    'total_tests': len(results),
                    'passed_tests': sum(1 for r in results if r.passed),
                    'failed_tests': len(results) - sum(1 for r in results if r.passed),
                    'success_rate': sum(1 for r in results if r.passed) / len(results) * 100
                },
                'test_results': [
                    {
                        'test_name': r.test_name,
                        'passed': r.passed,
                        'error_message': r.error_message,
                        'execution_time': r.execution_time,
                        'details': r.details
                    } for r in results
                ]
            }
            
            with open(validation_file, 'w') as f:
                json.dump(validation_data, f, indent=2, default=str)
            
            print(f"\n[INFO] Validation results saved to: {validation_file}")
            
        except Exception as e:
            print(f"[WARN]  Warning: Could not save validation results: {e}")
        
        print(f"\n{report}")
        return failed_tests == 0
        
    except Exception as e:
        print(f"\n[INFO] Validation suite crashed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 