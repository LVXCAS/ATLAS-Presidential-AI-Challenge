"""
Execution Engine Agent Demo

This demo showcases the capabilities of the Execution Engine Agent:
- Smart order routing across multiple venues
- Market impact estimation and slippage minimization
- Multiple execution algorithms (TWAP, VWAP, Implementation Shortfall)
- Order size optimization based on liquidity
- Real-time execution monitoring and optimization
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any

from agents.execution_engine_agent import (
    ExecutionEngineAgent, ExecutionOrder, ExecutionAlgorithm, 
    OrderSide, SmartOrderRouter, MarketImpactModel, LiquidityProfile
)
from agents.broker_integration import AlpacaBrokerIntegration
from agents.market_data_ingestor import MarketDataIngestorAgent
from config.logging_config import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class ExecutionEngineDemo:
    """Demo class for Execution Engine Agent"""
    
    def __init__(self):
        """Initialize demo components"""
        # Initialize components (using paper trading)
        self.broker = AlpacaBrokerIntegration(paper_trading=True)
        self.market_data_ingestor = MarketDataIngestorAgent()
        self.execution_engine = ExecutionEngineAgent(
            broker_integration=self.broker,
            market_data_ingestor=self.market_data_ingestor
        )
        
        logger.info("Execution Engine Demo initialized")
    
    async def demo_smart_order_routing(self):
        """Demonstrate smart order routing capabilities"""
        print("\n" + "="*60)
        print("SMART ORDER ROUTING DEMO")
        print("="*60)
        
        # Create test order for routing
        test_order = ExecutionOrder(
            symbol="AAPL",
            total_quantity=Decimal('5000'),
            side=OrderSide.BUY,
            urgency=0.5,
            allow_dark_pools=True,
            client_order_id="routing_demo_001"
        )
        
        # Mock market data
        market_data = {
            'current_price': 150.0,
            'spread': 0.02,
            'volume': 2000000,
            'avg_daily_volume': 50000000
        }
        
        print(f"Order Details:")
        print(f"- Symbol: {test_order.symbol}")
        print(f"- Quantity: {test_order.total_quantity:,}")
        print(f"- Side: {test_order.side.value}")
        print(f"- Urgency: {test_order.urgency}")
        print(f"- Allow Dark Pools: {test_order.allow_dark_pools}")
        
        # Demonstrate smart routing
        router = SmartOrderRouter()
        slices = router.route_order(test_order, market_data)
        
        print(f"\nSmart Routing Results:")
        print(f"- Total Slices: {len(slices)}")
        
        total_routed = Decimal('0')
        for i, slice_order in enumerate(slices, 1):
            print(f"\nSlice {i}:")
            print(f"  - Venue: {slice_order.venue.name} ({slice_order.venue.venue_type.value})")
            print(f"  - Quantity: {slice_order.quantity:,}")
            print(f"  - Order Type: {slice_order.order_type}")
            print(f"  - Limit Price: ${slice_order.limit_price:.2f}" if slice_order.limit_price else "  - Market Order")
            print(f"  - Priority Score: {slice_order.priority_score:.3f}")
            print(f"  - Venue Fees: {slice_order.venue.fee_structure}")
            total_routed += slice_order.quantity
        
        print(f"\nRouting Summary:")
        print(f"- Total Quantity Routed: {total_routed:,}")
        print(f"- Routing Efficiency: {(total_routed / test_order.total_quantity) * 100:.1f}%")
        
        # Analyze venue distribution
        venue_distribution = {}
        for slice_order in slices:
            venue_name = slice_order.venue.name
            if venue_name not in venue_distribution:
                venue_distribution[venue_name] = Decimal('0')
            venue_distribution[venue_name] += slice_order.quantity
        
        print(f"\nVenue Distribution:")
        for venue, quantity in venue_distribution.items():
            percentage = (quantity / total_routed) * 100
            print(f"- {venue}: {quantity:,} shares ({percentage:.1f}%)")
    
    async def demo_market_impact_estimation(self):
        """Demonstrate market impact estimation"""
        print("\n" + "="*60)
        print("MARKET IMPACT ESTIMATION DEMO")
        print("="*60)
        
        # Create market impact models for different scenarios
        scenarios = [
            {
                'name': 'High Liquidity Stock (AAPL)',
                'model': MarketImpactModel(
                    symbol='AAPL',
                    avg_daily_volume=Decimal('50000000'),
                    volatility=Decimal('0.02'),
                    bid_ask_spread=Decimal('0.01'),
                    liquidity_profile=LiquidityProfile.HIGH
                )
            },
            {
                'name': 'Medium Liquidity Stock (XYZ)',
                'model': MarketImpactModel(
                    symbol='XYZ',
                    avg_daily_volume=Decimal('5000000'),
                    volatility=Decimal('0.03'),
                    bid_ask_spread=Decimal('0.05'),
                    liquidity_profile=LiquidityProfile.MEDIUM
                )
            },
            {
                'name': 'Low Liquidity Stock (ABC)',
                'model': MarketImpactModel(
                    symbol='ABC',
                    avg_daily_volume=Decimal('500000'),
                    volatility=Decimal('0.05'),
                    bid_ask_spread=Decimal('0.10'),
                    liquidity_profile=LiquidityProfile.LOW
                )
            }
        ]
        
        order_sizes = [Decimal('1000'), Decimal('5000'), Decimal('10000'), Decimal('50000')]
        urgency_levels = [0.2, 0.5, 0.8]
        
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            print(f"- Daily Volume: {scenario['model'].avg_daily_volume:,}")
            print(f"- Volatility: {scenario['model'].volatility:.1%}")
            print(f"- Bid-Ask Spread: {scenario['model'].bid_ask_spread:.1%}")
            print(f"- Liquidity Profile: {scenario['model'].liquidity_profile.value}")
            
            print(f"\nMarket Impact Estimates (basis points):")
            print(f"{'Order Size':<12} {'Low Urgency':<12} {'Med Urgency':<12} {'High Urgency':<12}")
            print("-" * 50)
            
            for order_size in order_sizes:
                impacts = []
                for urgency in urgency_levels:
                    impact = scenario['model'].estimate_impact(order_size, urgency)
                    impact_bps = float(impact) * 10000  # Convert to basis points
                    impacts.append(impact_bps)
                
                print(f"{order_size:>10,} {impacts[0]:>10.1f} {impacts[1]:>10.1f} {impacts[2]:>10.1f}")
    
    async def demo_execution_algorithms(self):
        """Demonstrate different execution algorithms"""
        print("\n" + "="*60)
        print("EXECUTION ALGORITHMS DEMO")
        print("="*60)
        
        # Test different algorithms
        algorithms = [
            ExecutionAlgorithm.TWAP,
            ExecutionAlgorithm.VWAP,
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
            ExecutionAlgorithm.SMART_ROUTING
        ]
        
        base_order = ExecutionOrder(
            symbol="MSFT",
            total_quantity=Decimal('2000'),
            side=OrderSide.BUY,
            time_horizon_minutes=60,
            urgency=0.5,
            max_participation_rate=0.1
        )
        
        for algorithm in algorithms:
            print(f"\n{algorithm.value.upper()} Algorithm:")
            print("-" * 30)
            
            # Create order with specific algorithm
            test_order = ExecutionOrder(
                symbol=base_order.symbol,
                total_quantity=base_order.total_quantity,
                side=base_order.side,
                algorithm=algorithm,
                time_horizon_minutes=base_order.time_horizon_minutes,
                urgency=base_order.urgency,
                client_order_id=f"algo_demo_{algorithm.value}"
            )
            
            print(f"Order Configuration:")
            print(f"- Symbol: {test_order.symbol}")
            print(f"- Quantity: {test_order.total_quantity:,}")
            print(f"- Algorithm: {test_order.algorithm.value}")
            print(f"- Time Horizon: {test_order.time_horizon_minutes} minutes")
            print(f"- Urgency: {test_order.urgency}")
            
            # Simulate execution planning (without actual execution)
            try:
                # Get market data
                market_data = await self.execution_engine._get_market_data(test_order.symbol)
                
                # Create market impact model
                market_impact_model = await self.execution_engine._create_market_impact_model(
                    test_order.symbol, market_data
                )
                
                # Create execution plan
                if algorithm == ExecutionAlgorithm.TWAP:
                    slices = await self.execution_engine.algorithm_engine.execute_twap(
                        test_order, test_order.time_horizon_minutes
                    )
                elif algorithm == ExecutionAlgorithm.VWAP:
                    volume_profile = await self.execution_engine._get_volume_profile(test_order.symbol)
                    slices = await self.execution_engine.algorithm_engine.execute_vwap(
                        test_order, volume_profile
                    )
                elif algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
                    slices = await self.execution_engine.algorithm_engine.execute_implementation_shortfall(
                        test_order, market_impact_model
                    )
                else:  # SMART_ROUTING
                    slices = self.execution_engine.smart_router.route_order(test_order, market_data)
                
                print(f"\nExecution Plan:")
                print(f"- Total Slices: {len(slices)}")
                
                if slices:
                    avg_slice_size = sum(s.quantity for s in slices) / len(slices)
                    print(f"- Average Slice Size: {avg_slice_size:,.0f}")
                    print(f"- Slice Size Range: {min(s.quantity for s in slices):,.0f} - {max(s.quantity for s in slices):,.0f}")
                    
                    # Show first few slices as examples
                    print(f"\nSample Slices:")
                    for i, slice_order in enumerate(slices[:3], 1):
                        print(f"  Slice {i}: {slice_order.quantity:,} shares")
                        if hasattr(slice_order, 'expected_fill_time') and slice_order.expected_fill_time:
                            print(f"    Expected Time: {slice_order.expected_fill_time.strftime('%H:%M:%S')}")
                        if hasattr(slice_order, 'venue'):
                            print(f"    Venue: {slice_order.venue.name}")
                
                # Estimate total market impact
                total_impact = market_impact_model.estimate_impact(
                    test_order.total_quantity, test_order.urgency
                )
                print(f"- Estimated Market Impact: {float(total_impact) * 10000:.1f} bps")
                
            except Exception as e:
                print(f"Error creating execution plan: {e}")
    
    async def demo_live_execution_simulation(self):
        """Demonstrate live execution simulation"""
        print("\n" + "="*60)
        print("LIVE EXECUTION SIMULATION DEMO")
        print("="*60)
        
        # Create a realistic execution order
        execution_order = ExecutionOrder(
            symbol="GOOGL",
            total_quantity=Decimal('1000'),
            side=OrderSide.BUY,
            algorithm=ExecutionAlgorithm.SMART_ROUTING,
            time_horizon_minutes=30,
            urgency=0.6,
            max_participation_rate=0.08,
            allow_dark_pools=True,
            max_slippage_bps=25,
            client_order_id="live_demo_001"
        )
        
        print(f"Executing Order:")
        print(f"- Symbol: {execution_order.symbol}")
        print(f"- Quantity: {execution_order.total_quantity:,}")
        print(f"- Side: {execution_order.side.value}")
        print(f"- Algorithm: {execution_order.algorithm.value}")
        print(f"- Max Slippage: {execution_order.max_slippage_bps} bps")
        
        print(f"\nStarting execution workflow...")
        
        try:
            # Execute the order (this will run the full LangGraph workflow)
            final_state = await self.execution_engine.execute_order(execution_order)
            
            print(f"\nExecution Results:")
            print(f"- Status: {'Completed' if final_state.next_action == 'end' else 'In Progress'}")
            print(f"- Executed Quantity: {final_state.execution_order.executed_quantity:,}")
            print(f"- Remaining Quantity: {final_state.execution_order.remaining_quantity:,}")
            
            if final_state.execution_order.avg_execution_price:
                print(f"- Average Execution Price: ${final_state.execution_order.avg_execution_price:.2f}")
            
            # Show execution metrics
            metrics = final_state.execution_metrics
            if metrics:
                print(f"\nExecution Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if 'bps' in key:
                            print(f"- {key.replace('_', ' ').title()}: {value:.2f}")
                        elif 'rate' in key or 'percentage' in key:
                            print(f"- {key.replace('_', ' ').title()}: {value:.2%}")
                        else:
                            print(f"- {key.replace('_', ' ').title()}: {value}")
                    else:
                        print(f"- {key.replace('_', ' ').title()}: {value}")
            
            # Show slice execution summary
            print(f"\nSlice Execution Summary:")
            print(f"- Completed Slices: {len(final_state.completed_slices)}")
            print(f"- Failed Slices: {len(final_state.failed_slices)}")
            print(f"- Child Orders: {len(final_state.execution_order.child_orders)}")
            
            if final_state.error_log:
                print(f"\nExecution Warnings/Errors:")
                for error in final_state.error_log[-5:]:  # Show last 5 errors
                    print(f"- {error}")
            
        except Exception as e:
            print(f"Execution failed: {e}")
            logger.error(f"Demo execution failed: {e}")
    
    async def demo_performance_metrics(self):
        """Demonstrate performance metrics tracking"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS DEMO")
        print("="*60)
        
        # Get current performance metrics
        metrics = self.execution_engine.get_performance_metrics()
        
        print(f"Execution Engine Performance:")
        print(f"- Total Orders Processed: {metrics['total_orders']}")
        print(f"- Successful Executions: {metrics['successful_executions']}")
        print(f"- Success Rate: {metrics['success_rate']:.2%}")
        print(f"- Average Slippage: {metrics['avg_slippage_bps']:.2f} bps")
        print(f"- Average Market Impact: {metrics['avg_market_impact_bps']:.2f} bps")
        print(f"- Active Executions: {metrics['active_executions']}")
        print(f"- Completed Executions: {metrics['completed_executions']}")
        
        # Show venue performance if available
        router = self.execution_engine.smart_router
        print(f"\nVenue Information:")
        for venue_id, venue in router.venues.items():
            print(f"- {venue.name}:")
            print(f"  Type: {venue.venue_type.value}")
            print(f"  Liquidity Score: {venue.liquidity_score:.2f}")
            print(f"  Market Share: {venue.market_share:.1%}")
            print(f"  Avg Latency: {venue.avg_latency_ms:.1f}ms")
            print(f"  Maker Fee: {venue.fee_structure.get('maker', 'N/A')}")
            print(f"  Taker Fee: {venue.fee_structure.get('taker', 'N/A')}")
    
    async def run_all_demos(self):
        """Run all demo scenarios"""
        print("EXECUTION ENGINE AGENT COMPREHENSIVE DEMO")
        print("=" * 80)
        print("This demo showcases the advanced capabilities of the Execution Engine Agent")
        print("including smart order routing, market impact estimation, and execution algorithms.")
        
        try:
            # Run all demo scenarios
            await self.demo_smart_order_routing()
            await self.demo_market_impact_estimation()
            await self.demo_execution_algorithms()
            await self.demo_live_execution_simulation()
            await self.demo_performance_metrics()
            
            print("\n" + "="*80)
            print("DEMO COMPLETED SUCCESSFULLY")
            print("="*80)
            print("The Execution Engine Agent demonstrates:")
            print("✓ Smart order routing across multiple venues")
            print("✓ Market impact estimation and slippage minimization")
            print("✓ Multiple execution algorithms (TWAP, VWAP, Implementation Shortfall)")
            print("✓ Order size optimization based on liquidity")
            print("✓ Real-time execution monitoring and optimization")
            print("✓ Comprehensive performance metrics tracking")
            
        except Exception as e:
            print(f"\nDemo failed with error: {e}")
            logger.error(f"Demo execution failed: {e}")


async def main():
    """Main demo function"""
    demo = ExecutionEngineDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())