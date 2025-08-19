"""
Broker Integration Demo

This demo showcases the Alpaca broker integration capabilities including:
- Order lifecycle management (submit, monitor, cancel)
- Position reconciliation and trade reporting
- Error handling for API failures and rejections
- Real-time order status monitoring

Usage:
    python examples/broker_integration_demo.py
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from agents.broker_integration import (
    AlpacaBrokerIntegration,
    OrderRequest,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    create_market_order,
    create_limit_order,
    create_stop_loss_order
)
from config.logging_config import get_logger

logger = get_logger(__name__)


class BrokerIntegrationDemo:
    """
    Comprehensive demo of broker integration functionality
    """
    
    def __init__(self):
        """Initialize the demo"""
        self.broker = AlpacaBrokerIntegration(paper_trading=True)
        self.demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.submitted_orders: List[str] = []
    
    async def run_demo(self):
        """Run the complete broker integration demo"""
        print("🚀 Starting Broker Integration Demo")
        print("=" * 60)
        
        try:
            # 1. Health Check
            await self.demo_health_check()
            
            # 2. Account Information
            await self.demo_account_info()
            
            # 3. Current Positions
            await self.demo_current_positions()
            
            # 4. Order Lifecycle Management
            await self.demo_order_lifecycle()
            
            # 5. Position Reconciliation
            await self.demo_position_reconciliation()
            
            # 6. Trade Reporting
            await self.demo_trade_reporting()
            
            # 7. Error Handling
            await self.demo_error_handling()
            
            print("\n✅ Broker Integration Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"\n❌ Demo failed: {e}")
    
    async def demo_health_check(self):
        """Demonstrate broker health check"""
        print("\n📊 1. Broker Health Check")
        print("-" * 30)
        
        health_status = await self.broker.health_check()
        
        print(f"Connection Status: {health_status['connection_status']}")
        print(f"Account Accessible: {health_status['account_accessible']}")
        print(f"Orders Accessible: {health_status['orders_accessible']}")
        print(f"Positions Accessible: {health_status['positions_accessible']}")
        print(f"Paper Trading: {health_status['paper_trading']}")
        
        if health_status['errors']:
            print(f"Errors: {health_status['errors']}")
        
        if health_status['connection_status'] != 'healthy':
            print("⚠️  Warning: Broker connection is not healthy")
            return False
        
        print("✅ Broker connection is healthy")
        return True
    
    async def demo_account_info(self):
        """Demonstrate account information retrieval"""
        print("\n💰 2. Account Information")
        print("-" * 30)
        
        account_info = await self.broker.get_account_info()
        
        if account_info:
            print(f"Account Number: {account_info['account_number']}")
            print(f"Status: {account_info['status']}")
            print(f"Buying Power: ${account_info['buying_power']:,.2f}")
            print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
            print(f"Cash: ${account_info['cash']:,.2f}")
            print(f"Pattern Day Trader: {account_info['pattern_day_trader']}")
            print(f"Day Trade Count: {account_info['daytrade_count']}")
        else:
            print("❌ Failed to retrieve account information")
    
    async def demo_current_positions(self):
        """Demonstrate current positions retrieval"""
        print("\n📈 3. Current Positions")
        print("-" * 30)
        
        positions = await self.broker.get_positions()
        
        if positions:
            print(f"Total Positions: {len(positions)}")
            print()
            
            total_market_value = Decimal('0')
            total_unrealized_pl = Decimal('0')
            
            for position in positions:
                print(f"Symbol: {position.symbol}")
                print(f"  Quantity: {position.qty}")
                print(f"  Avg Entry Price: ${position.avg_entry_price:.2f}")
                print(f"  Current Price: ${position.current_price:.2f}" if position.current_price else "  Current Price: N/A")
                print(f"  Market Value: ${position.market_value:.2f}")
                print(f"  Unrealized P&L: ${position.unrealized_pl:.2f}")
                print(f"  Side: {position.side}")
                print()
                
                total_market_value += position.market_value
                total_unrealized_pl += position.unrealized_pl
            
            print(f"Total Market Value: ${total_market_value:.2f}")
            print(f"Total Unrealized P&L: ${total_unrealized_pl:.2f}")
        else:
            print("No current positions found")
    
    async def demo_order_lifecycle(self):
        """Demonstrate complete order lifecycle management"""
        print("\n📋 4. Order Lifecycle Management")
        print("-" * 30)
        
        # Test different order types
        await self.demo_market_order()
        await self.demo_limit_order()
        await self.demo_stop_loss_order()
        await self.demo_order_monitoring()
        await self.demo_order_cancellation()
    
    async def demo_market_order(self):
        """Demonstrate market order submission"""
        print("\n🎯 Market Order Example")
        
        try:
            # Submit a small market order
            order_response = await create_market_order(
                self.broker,
                'AAPL',
                1,  # Small quantity for demo
                OrderSide.BUY
            )
            
            self.submitted_orders.append(order_response.id)
            
            print(f"✅ Market order submitted successfully")
            print(f"  Order ID: {order_response.id}")
            print(f"  Symbol: {order_response.symbol}")
            print(f"  Side: {order_response.side.value}")
            print(f"  Quantity: {order_response.qty}")
            print(f"  Status: {order_response.status.value}")
            print(f"  Created: {order_response.created_at}")
            
        except Exception as e:
            print(f"❌ Market order failed: {e}")
    
    async def demo_limit_order(self):
        """Demonstrate limit order submission"""
        print("\n💲 Limit Order Example")
        
        try:
            # Submit a limit order below current market price
            order_response = await create_limit_order(
                self.broker,
                'GOOGL',
                1,  # Small quantity for demo
                OrderSide.BUY,
                100.00,  # Low limit price (likely won't fill)
                TimeInForce.DAY
            )
            
            self.submitted_orders.append(order_response.id)
            
            print(f"✅ Limit order submitted successfully")
            print(f"  Order ID: {order_response.id}")
            print(f"  Symbol: {order_response.symbol}")
            print(f"  Side: {order_response.side.value}")
            print(f"  Quantity: {order_response.qty}")
            print(f"  Limit Price: ${order_response.limit_price}")
            print(f"  Status: {order_response.status.value}")
            print(f"  Time in Force: {order_response.time_in_force.value}")
            
        except Exception as e:
            print(f"❌ Limit order failed: {e}")
    
    async def demo_stop_loss_order(self):
        """Demonstrate stop loss order submission"""
        print("\n🛑 Stop Loss Order Example")
        
        try:
            # Submit a stop loss order
            order_response = await create_stop_loss_order(
                self.broker,
                'MSFT',
                1,  # Small quantity for demo
                OrderSide.SELL,
                50.00  # Low stop price (likely won't trigger)
            )
            
            self.submitted_orders.append(order_response.id)
            
            print(f"✅ Stop loss order submitted successfully")
            print(f"  Order ID: {order_response.id}")
            print(f"  Symbol: {order_response.symbol}")
            print(f"  Side: {order_response.side.value}")
            print(f"  Quantity: {order_response.qty}")
            print(f"  Stop Price: ${order_response.stop_price}")
            print(f"  Status: {order_response.status.value}")
            print(f"  Time in Force: {order_response.time_in_force.value}")
            
        except Exception as e:
            print(f"❌ Stop loss order failed: {e}")
    
    async def demo_order_monitoring(self):
        """Demonstrate order status monitoring"""
        print("\n👀 Order Status Monitoring")
        
        if not self.submitted_orders:
            print("No orders to monitor")
            return
        
        for order_id in self.submitted_orders:
            try:
                order_status = await self.broker.get_order_status(order_id)
                
                if order_status:
                    print(f"Order {order_id[:8]}...")
                    print(f"  Status: {order_status.status.value}")
                    print(f"  Filled Qty: {order_status.filled_qty}/{order_status.qty}")
                    
                    if order_status.filled_at:
                        print(f"  Filled At: {order_status.filled_at}")
                    
                    if order_status.status == OrderStatus.REJECTED:
                        print(f"  ❌ Order was rejected")
                    elif order_status.status == OrderStatus.FILLED:
                        print(f"  ✅ Order was filled")
                    elif order_status.status == OrderStatus.PARTIALLY_FILLED:
                        print(f"  🔄 Order is partially filled")
                    else:
                        print(f"  ⏳ Order is pending")
                else:
                    print(f"❌ Could not retrieve status for order {order_id}")
                
                print()
                
            except Exception as e:
                print(f"❌ Error monitoring order {order_id}: {e}")
    
    async def demo_order_cancellation(self):
        """Demonstrate order cancellation"""
        print("\n❌ Order Cancellation Example")
        
        # Get all open orders
        open_orders = await self.broker.get_all_orders(status='open', limit=10)
        
        if open_orders:
            # Cancel the first open order
            order_to_cancel = open_orders[0]
            
            print(f"Attempting to cancel order: {order_to_cancel.id}")
            print(f"  Symbol: {order_to_cancel.symbol}")
            print(f"  Side: {order_to_cancel.side.value}")
            print(f"  Status: {order_to_cancel.status.value}")
            
            success = await self.broker.cancel_order(order_to_cancel.id)
            
            if success:
                print(f"✅ Order canceled successfully")
                
                # Check updated status
                await asyncio.sleep(1)  # Wait a moment
                updated_status = await self.broker.get_order_status(order_to_cancel.id)
                if updated_status:
                    print(f"  Updated Status: {updated_status.status.value}")
            else:
                print(f"❌ Failed to cancel order")
        else:
            print("No open orders to cancel")
    
    async def demo_position_reconciliation(self):
        """Demonstrate position reconciliation"""
        print("\n🔄 5. Position Reconciliation")
        print("-" * 30)
        
        reconciliation_report = await self.broker.reconcile_positions()
        
        print(f"Reconciliation Timestamp: {reconciliation_report['timestamp']}")
        print(f"Broker Positions Count: {reconciliation_report['broker_positions_count']}")
        print(f"Total Market Value: ${reconciliation_report['total_market_value']:.2f}")
        print(f"Total Unrealized P&L: ${reconciliation_report['total_unrealized_pl']:.2f}")
        
        if reconciliation_report['discrepancies']:
            print(f"\n⚠️  Found {len(reconciliation_report['discrepancies'])} discrepancies:")
            for discrepancy in reconciliation_report['discrepancies']:
                print(f"  - {discrepancy}")
        else:
            print("\n✅ No discrepancies found")
        
        if reconciliation_report['positions']:
            print(f"\nPosition Details:")
            for position in reconciliation_report['positions']:
                print(f"  {position['symbol']}: {position['qty']} shares @ ${position['avg_entry_price']:.2f}")
    
    async def demo_trade_reporting(self):
        """Demonstrate trade reporting"""
        print("\n📊 6. Trade Reporting")
        print("-" * 30)
        
        # Generate report for last 7 days
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        trade_report = await self.broker.generate_trade_report(
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"Trade Report Period: {start_date.date()} to {end_date.date()}")
        print(f"Total Trades: {trade_report['summary']['total_trades']}")
        print(f"Buy Orders: {trade_report['summary']['buy_orders']}")
        print(f"Sell Orders: {trade_report['summary']['sell_orders']}")
        print(f"Total Volume: ${trade_report['summary']['total_volume']:,.2f}")
        print(f"Unique Symbols: {trade_report['summary']['unique_symbols']}")
        
        if trade_report['by_symbol']:
            print(f"\nTrades by Symbol:")
            for symbol, stats in trade_report['by_symbol'].items():
                print(f"  {symbol}: {stats['total_trades']} trades ({stats['buy_orders']} buy, {stats['sell_orders']} sell)")
        
        if trade_report['orders']:
            print(f"\nRecent Orders (showing first 5):")
            for order in trade_report['orders'][:5]:
                print(f"  {order['created_at'].strftime('%Y-%m-%d %H:%M')} - {order['symbol']} {order['side']} {order['qty']} @ ${order['limit_price'] or 'market'}")
    
    async def demo_error_handling(self):
        """Demonstrate error handling scenarios"""
        print("\n⚠️  7. Error Handling Examples")
        print("-" * 30)
        
        # Test invalid symbol
        print("Testing invalid symbol order...")
        try:
            invalid_order = OrderRequest(
                symbol='INVALID_SYMBOL_12345',
                qty=1,
                side=OrderSide.BUY,
                type=OrderType.MARKET
            )
            
            await self.broker.submit_order(invalid_order)
            print("❌ Expected error but order succeeded")
            
        except Exception as e:
            print(f"✅ Correctly handled invalid symbol error: {e}")
        
        # Test invalid quantity
        print("\nTesting invalid quantity order...")
        try:
            invalid_order = OrderRequest(
                symbol='AAPL',
                qty=0,  # Invalid quantity
                side=OrderSide.BUY,
                type=OrderType.MARKET
            )
            
            await self.broker.submit_order(invalid_order)
            print("❌ Expected error but order succeeded")
            
        except Exception as e:
            print(f"✅ Correctly handled invalid quantity error: {e}")
        
        # Test getting non-existent order
        print("\nTesting non-existent order status...")
        fake_order_id = "non-existent-order-12345"
        order_status = await self.broker.get_order_status(fake_order_id)
        
        if order_status is None:
            print(f"✅ Correctly handled non-existent order")
        else:
            print(f"❌ Expected None but got order status")
        
        # Test getting non-existent position
        print("\nTesting non-existent position...")
        position = await self.broker.get_position('NONEXISTENT')
        
        if position is None:
            print(f"✅ Correctly handled non-existent position")
        else:
            print(f"❌ Expected None but got position")
        
        # Show error log
        if self.broker.error_log:
            print(f"\nError Log ({len(self.broker.error_log)} entries):")
            for i, error in enumerate(self.broker.error_log[-3:], 1):  # Show last 3 errors
                print(f"  {i}. {error.timestamp.strftime('%H:%M:%S')} - {error.error_message}")
                if error.symbol:
                    print(f"     Symbol: {error.symbol}")
                print(f"     Retryable: {error.is_retryable}")
        else:
            print("\nNo errors logged during demo")
    
    async def cleanup_demo_orders(self):
        """Clean up any remaining demo orders"""
        print("\n🧹 Cleaning up demo orders...")
        
        # Cancel any remaining open orders from this demo
        open_orders = await self.broker.get_all_orders(status='open', limit=50)
        
        demo_orders = [order for order in open_orders if order.id in self.submitted_orders]
        
        if demo_orders:
            print(f"Canceling {len(demo_orders)} demo orders...")
            
            for order in demo_orders:
                success = await self.broker.cancel_order(order.id)
                if success:
                    print(f"  ✅ Canceled {order.symbol} {order.side.value} order")
                else:
                    print(f"  ❌ Failed to cancel {order.symbol} {order.side.value} order")
        else:
            print("No demo orders to clean up")

    async def run_all_demos(self):
        """Run all broker integration demos"""
        try:
            await self.run_demo()
        finally:
            # Always try to clean up
            await self.cleanup_demo_orders()


async def main():
    """Main demo function"""
    demo = BrokerIntegrationDemo()
    
    try:
        await demo.run_demo()
    finally:
        # Always try to clean up
        await demo.cleanup_demo_orders()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demo
    asyncio.run(main())