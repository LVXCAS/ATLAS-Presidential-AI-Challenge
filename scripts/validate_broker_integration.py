"""
Broker Integration Validation Script

This script validates the Alpaca broker integration implementation by testing:
- Connection and authentication
- Order submission and lifecycle management
- Position reconciliation
- Error handling and recovery
- Trade reporting functionality

Usage:
    python scripts/validate_broker_integration.py
"""

import asyncio
import sys
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional

from agents.broker_integration import (
    AlpacaBrokerIntegration,
    OrderRequest,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce
)
from config.logging_config import get_logger

logger = get_logger(__name__)


class BrokerIntegrationValidator:
    """
    Comprehensive validator for broker integration functionality
    """
    
    def __init__(self):
        """Initialize the validator"""
        self.broker = AlpacaBrokerIntegration(paper_trading=True)
        self.test_results: Dict[str, bool] = {}
        self.test_orders: List[str] = []
        self.validation_errors: List[str] = []
    
    async def run_validation(self) -> bool:
        """
        Run complete validation suite
        
        Returns:
            bool: True if all validations pass, False otherwise
        """
        print("[SEARCH] Starting Broker Integration Validation")
        print("=" * 60)
        
        validation_tests = [
            ("Connection Health Check", self.validate_connection),
            ("Account Access", self.validate_account_access),
            ("Market Order Submission", self.validate_market_order),
            ("Limit Order Submission", self.validate_limit_order),
            ("Order Status Monitoring", self.validate_order_status),
            ("Order Cancellation", self.validate_order_cancellation),
            ("Position Management", self.validate_position_management),
            ("Error Handling", self.validate_error_handling),
            ("Position Reconciliation", self.validate_position_reconciliation),
            ("Trade Reporting", self.validate_trade_reporting),
            ("Partial Fill Handling", self.validate_partial_fills),
            ("Order Lifecycle", self.validate_order_lifecycle)
        ]
        
        passed_tests = 0
        total_tests = len(validation_tests)
        
        for test_name, test_func in validation_tests:
            print(f"\n[INFO] Testing: {test_name}")
            print("-" * 40)
            
            try:
                result = await test_func()
                self.test_results[test_name] = result
                
                if result:
                    print(f"[OK] {test_name}: PASSED")
                    passed_tests += 1
                else:
                    print(f"[X] {test_name}: FAILED")
                    
            except Exception as e:
                print(f"[INFO] {test_name}: ERROR - {e}")
                self.test_results[test_name] = False
                self.validation_errors.append(f"{test_name}: {e}")
                logger.error(f"Validation test {test_name} failed with error: {e}")
        
        # Cleanup
        await self.cleanup_test_orders()
        
        # Print summary
        print("\n" + "=" * 60)
        print("[CHART] VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.validation_errors:
            print(f"\n[X] Errors encountered:")
            for error in self.validation_errors:
                print(f"  - {error}")
        
        success = passed_tests == total_tests
        
        if success:
            print("\n[PARTY] All broker integration validations PASSED!")
        else:
            print(f"\n[WARN]  {total_tests - passed_tests} validation(s) FAILED")
        
        return success
    
    async def validate_connection(self) -> bool:
        """Validate broker connection and health"""
        try:
            health_status = await self.broker.health_check()
            
            required_checks = [
                'connection_status',
                'account_accessible',
                'orders_accessible',
                'positions_accessible'
            ]
            
            for check in required_checks:
                if check not in health_status:
                    print(f"[X] Missing health check: {check}")
                    return False
            
            if health_status['connection_status'] != 'healthy':
                print(f"[X] Connection not healthy: {health_status['connection_status']}")
                return False
            
            if not health_status['account_accessible']:
                print(f"[X] Account not accessible")
                return False
            
            print(f"[OK] Connection healthy, paper trading: {health_status['paper_trading']}")
            return True
            
        except Exception as e:
            print(f"[X] Connection validation failed: {e}")
            return False
    
    async def validate_account_access(self) -> bool:
        """Validate account information access"""
        try:
            account_info = await self.broker.get_account_info()
            
            if not account_info:
                print("[X] Failed to retrieve account information")
                return False
            
            required_fields = [
                'account_number', 'status', 'buying_power', 
                'portfolio_value', 'cash'
            ]
            
            for field in required_fields:
                if field not in account_info:
                    print(f"[X] Missing account field: {field}")
                    return False
            
            print(f"[OK] Account accessible: {account_info['account_number']}")
            print(f"   Status: {account_info['status']}")
            print(f"   Buying Power: ${account_info['buying_power']:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"[X] Account access validation failed: {e}")
            return False
    
    async def validate_market_order(self) -> bool:
        """Validate market order submission"""
        try:
            order_request = OrderRequest(
                symbol='AAPL',
                qty=1,
                side=OrderSide.BUY,
                type=OrderType.MARKET
            )
            
            order_response = await self.broker.submit_order(order_request)
            self.test_orders.append(order_response.id)
            
            # Validate response structure
            if not order_response.id:
                print("[X] Order response missing ID")
                return False
            
            if order_response.symbol != 'AAPL':
                print(f"[X] Wrong symbol in response: {order_response.symbol}")
                return False
            
            if order_response.qty != Decimal('1'):
                print(f"[X] Wrong quantity in response: {order_response.qty}")
                return False
            
            if order_response.side != OrderSide.BUY:
                print(f"[X] Wrong side in response: {order_response.side}")
                return False
            
            if order_response.type != OrderType.MARKET:
                print(f"[X] Wrong type in response: {order_response.type}")
                return False
            
            print(f"[OK] Market order submitted: {order_response.id}")
            print(f"   Status: {order_response.status.value}")
            
            return True
            
        except Exception as e:
            print(f"[X] Market order validation failed: {e}")
            return False
    
    async def validate_limit_order(self) -> bool:
        """Validate limit order submission"""
        try:
            order_request = OrderRequest(
                symbol='GOOGL',
                qty=1,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                limit_price=Decimal('100.00'),  # Low price, unlikely to fill
                time_in_force=TimeInForce.DAY
            )
            
            order_response = await self.broker.submit_order(order_request)
            self.test_orders.append(order_response.id)
            
            # Validate limit order specific fields
            if order_response.limit_price != Decimal('100.00'):
                print(f"[X] Wrong limit price: {order_response.limit_price}")
                return False
            
            if order_response.time_in_force != TimeInForce.DAY:
                print(f"[X] Wrong time in force: {order_response.time_in_force}")
                return False
            
            print(f"[OK] Limit order submitted: {order_response.id}")
            print(f"   Limit Price: ${order_response.limit_price}")
            
            return True
            
        except Exception as e:
            print(f"[X] Limit order validation failed: {e}")
            return False
    
    async def validate_order_status(self) -> bool:
        """Validate order status monitoring"""
        try:
            if not self.test_orders:
                print("[X] No test orders available for status check")
                return False
            
            order_id = self.test_orders[0]
            order_status = await self.broker.get_order_status(order_id)
            
            if not order_status:
                print(f"[X] Failed to retrieve order status for {order_id}")
                return False
            
            if order_status.id != order_id:
                print(f"[X] Order ID mismatch: expected {order_id}, got {order_status.id}")
                return False
            
            # Validate status is a valid enum value
            valid_statuses = [status.value for status in OrderStatus]
            if order_status.status.value not in valid_statuses:
                print(f"[X] Invalid order status: {order_status.status}")
                return False
            
            print(f"[OK] Order status retrieved: {order_status.status.value}")
            print(f"   Filled: {order_status.filled_qty}/{order_status.qty}")
            
            return True
            
        except Exception as e:
            print(f"[X] Order status validation failed: {e}")
            return False
    
    async def validate_order_cancellation(self) -> bool:
        """Validate order cancellation"""
        try:
            # Submit a limit order that won't fill immediately
            order_request = OrderRequest(
                symbol='MSFT',
                qty=1,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                limit_price=Decimal('50.00'),  # Very low price
                time_in_force=TimeInForce.DAY
            )
            
            order_response = await self.broker.submit_order(order_request)
            order_id = order_response.id
            
            # Wait a moment for order to be accepted
            await asyncio.sleep(1)
            
            # Cancel the order
            cancel_result = await self.broker.cancel_order(order_id)
            
            if not cancel_result:
                print(f"[X] Failed to cancel order {order_id}")
                return False
            
            # Check that order status was updated
            await asyncio.sleep(1)  # Wait for status update
            updated_status = await self.broker.get_order_status(order_id)
            
            if updated_status and updated_status.status not in [OrderStatus.CANCELED, OrderStatus.PENDING_CANCEL]:
                print(f"[X] Order not canceled, status: {updated_status.status}")
                return False
            
            print(f"[OK] Order canceled successfully: {order_id}")
            
            return True
            
        except Exception as e:
            print(f"[X] Order cancellation validation failed: {e}")
            return False
    
    async def validate_position_management(self) -> bool:
        """Validate position management functionality"""
        try:
            # Get all positions
            positions = await self.broker.get_positions()
            
            print(f"[OK] Retrieved {len(positions)} positions")
            
            # Test getting a specific position (may not exist)
            test_position = await self.broker.get_position('AAPL')
            
            if test_position:
                print(f"[OK] Retrieved AAPL position: {test_position.qty} shares")
                
                # Validate position structure
                required_fields = ['symbol', 'qty', 'avg_entry_price', 'market_value']
                for field in required_fields:
                    if not hasattr(test_position, field):
                        print(f"[X] Position missing field: {field}")
                        return False
            else:
                print("[OK] No AAPL position found (expected for new account)")
            
            return True
            
        except Exception as e:
            print(f"[X] Position management validation failed: {e}")
            return False
    
    async def validate_error_handling(self) -> bool:
        """Validate error handling scenarios"""
        try:
            # Test invalid symbol
            try:
                invalid_order = OrderRequest(
                    symbol='INVALID_SYMBOL_12345',
                    qty=1,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET
                )
                
                await self.broker.submit_order(invalid_order)
                print("[X] Expected error for invalid symbol but order succeeded")
                return False
                
            except Exception:
                print("[OK] Correctly handled invalid symbol error")
            
            # Test invalid order ID
            fake_order_status = await self.broker.get_order_status('fake-order-id-12345')
            if fake_order_status is not None:
                print("[X] Expected None for fake order ID")
                return False
            
            print("[OK] Correctly handled invalid order ID")
            
            # Test non-existent position
            fake_position = await self.broker.get_position('NONEXISTENT_SYMBOL')
            if fake_position is not None:
                print("[X] Expected None for non-existent position")
                return False
            
            print("[OK] Correctly handled non-existent position")
            
            return True
            
        except Exception as e:
            print(f"[X] Error handling validation failed: {e}")
            return False
    
    async def validate_position_reconciliation(self) -> bool:
        """Validate position reconciliation functionality"""
        try:
            reconciliation_report = await self.broker.reconcile_positions()
            
            # Validate report structure
            required_fields = [
                'timestamp', 'broker_positions_count', 'positions',
                'discrepancies', 'total_market_value', 'total_unrealized_pl'
            ]
            
            for field in required_fields:
                if field not in reconciliation_report:
                    print(f"[X] Reconciliation report missing field: {field}")
                    return False
            
            # Validate timestamp is recent
            report_time = reconciliation_report['timestamp']
            if isinstance(report_time, str):
                # If it's a string, try to parse it
                report_time = datetime.fromisoformat(report_time.replace('Z', '+00:00'))
            
            time_diff = datetime.now(timezone.utc) - report_time
            if time_diff.total_seconds() > 60:  # More than 1 minute old
                print(f"[X] Reconciliation report timestamp too old: {report_time}")
                return False
            
            print(f"[OK] Position reconciliation completed")
            print(f"   Positions: {reconciliation_report['broker_positions_count']}")
            print(f"   Market Value: ${reconciliation_report['total_market_value']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"[X] Position reconciliation validation failed: {e}")
            return False
    
    async def validate_trade_reporting(self) -> bool:
        """Validate trade reporting functionality"""
        try:
            # Generate report for last 24 hours
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(hours=24)
            
            trade_report = await self.broker.generate_trade_report(
                start_date=start_date,
                end_date=end_date
            )
            
            # Validate report structure
            required_fields = ['timestamp', 'period', 'summary', 'orders']
            
            for field in required_fields:
                if field not in trade_report:
                    print(f"[X] Trade report missing field: {field}")
                    return False
            
            # Validate summary structure
            summary_fields = ['total_trades', 'buy_orders', 'sell_orders', 'unique_symbols']
            
            for field in summary_fields:
                if field not in trade_report['summary']:
                    print(f"[X] Trade report summary missing field: {field}")
                    return False
            
            print(f"[OK] Trade report generated")
            print(f"   Period: {start_date.date()} to {end_date.date()}")
            print(f"   Total Trades: {trade_report['summary']['total_trades']}")
            
            return True
            
        except Exception as e:
            print(f"[X] Trade reporting validation failed: {e}")
            return False
    
    async def validate_partial_fills(self) -> bool:
        """Validate partial fill handling"""
        try:
            # This is difficult to test reliably in paper trading
            # We'll validate the data structures and logic instead
            
            # Get recent orders and check for any partial fills
            recent_orders = await self.broker.get_all_orders(limit=10)
            
            partial_fill_found = False
            for order in recent_orders:
                if order.status == OrderStatus.PARTIALLY_FILLED:
                    partial_fill_found = True
                    
                    # Validate partial fill data
                    if order.filled_qty >= order.qty:
                        print(f"[X] Partial fill has filled_qty >= qty: {order.filled_qty}/{order.qty}")
                        return False
                    
                    if order.filled_qty <= 0:
                        print(f"[X] Partial fill has invalid filled_qty: {order.filled_qty}")
                        return False
                    
                    print(f"[OK] Found valid partial fill: {order.filled_qty}/{order.qty}")
                    break
            
            if not partial_fill_found:
                print("[OK] No partial fills found (expected in paper trading)")
            
            return True
            
        except Exception as e:
            print(f"[X] Partial fill validation failed: {e}")
            return False
    
    async def validate_order_lifecycle(self) -> bool:
        """Validate complete order lifecycle"""
        try:
            # Submit order
            order_request = OrderRequest(
                symbol='TSLA',
                qty=1,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                limit_price=Decimal('100.00'),  # Low price
                time_in_force=TimeInForce.DAY
            )
            
            # 1. Submit
            order_response = await self.broker.submit_order(order_request)
            order_id = order_response.id
            
            if order_response.status not in [OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW]:
                print(f"[X] Unexpected initial status: {order_response.status}")
                return False
            
            # 2. Monitor
            await asyncio.sleep(1)
            status_1 = await self.broker.get_order_status(order_id)
            
            if not status_1:
                print(f"[X] Failed to get order status")
                return False
            
            # 3. Cancel
            cancel_result = await self.broker.cancel_order(order_id)
            
            if not cancel_result:
                print(f"[X] Failed to cancel order")
                return False
            
            # 4. Verify cancellation
            await asyncio.sleep(1)
            final_status = await self.broker.get_order_status(order_id)
            
            if final_status and final_status.status not in [OrderStatus.CANCELED, OrderStatus.PENDING_CANCEL]:
                print(f"[X] Order not properly canceled: {final_status.status}")
                return False
            
            print(f"[OK] Complete order lifecycle validated")
            print(f"   Submit -> Monitor -> Cancel -> Verify")
            
            return True
            
        except Exception as e:
            print(f"[X] Order lifecycle validation failed: {e}")
            return False
    
    async def cleanup_test_orders(self):
        """Clean up any remaining test orders"""
        try:
            print(f"\n[INFO] Cleaning up {len(self.test_orders)} test orders...")
            
            # Get all open orders
            open_orders = await self.broker.get_all_orders(status='open', limit=100)
            
            # Cancel test orders that are still open
            for order in open_orders:
                if order.id in self.test_orders:
                    await self.broker.cancel_order(order.id)
                    print(f"   Canceled test order: {order.id}")
            
            print("[OK] Cleanup completed")
            
        except Exception as e:
            print(f"[WARN]  Cleanup failed: {e}")


async def main():
    """Main validation function"""
    validator = BrokerIntegrationValidator()
    
    try:
        success = await validator.run_validation()
        
        if success:
            print("\n[PARTY] Broker integration validation PASSED!")
            sys.exit(0)
        else:
            print("\n[X] Broker integration validation FAILED!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[INFO] Validation failed with error: {e}")
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    asyncio.run(main())