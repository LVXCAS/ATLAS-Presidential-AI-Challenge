"""
Validation Script for Execution Engine Agent

This script validates the implementation of the Execution Engine Agent:
- LangGraph workflow functionality
- Smart order routing capabilities
- Market impact estimation accuracy
- Execution algorithm implementations
- Order size optimization
- Slippage minimization features
"""

import asyncio
import sys
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any

from agents.execution_engine_agent import (
    ExecutionEngineAgent, ExecutionOrder, ExecutionAlgorithm, OrderSide,
    SmartOrderRouter, MarketImpactModel, LiquidityProfile, VenueType
)
from agents.broker_integration import AlpacaBrokerIntegration
from agents.market_data_ingestor import MarketDataIngestorAgent
from config.logging_config import get_logger

logger = get_logger(__name__)


class ExecutionEngineValidator:
    """Validator for Execution Engine Agent implementation"""
    
    def __init__(self):
        """Initialize validator"""
        self.test_results = []
        self.broker = AlpacaBrokerIntegration(paper_trading=True)
        self.market_data_ingestor = MarketDataIngestorAgent()
        self.execution_engine = ExecutionEngineAgent(
            broker_integration=self.broker,
            market_data_ingestor=self.market_data_ingestor
        )
    
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        self.test_results.append({
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now(timezone.utc)
        })
        print(f"[{status}] {test_name}: {details}")
    
    async def validate_market_impact_model(self) -> bool:
        """Validate market impact estimation"""
        print("\n=== Validating Market Impact Model ===")
        
        try:
            # Test 1: Basic impact calculation
            model = MarketImpactModel(
                symbol="AAPL",
                avg_daily_volume=Decimal('50000000'),
                volatility=Decimal('0.02'),
                bid_ask_spread=Decimal('0.01'),
                liquidity_profile=LiquidityProfile.HIGH
            )
            
            impact = model.estimate_impact(Decimal('10000'), urgency=0.5)
            
            if impact > 0 and impact < Decimal('0.1'):  # Impact should be positive but reasonable
                self.log_test_result("Market Impact Basic Calculation", True, f"Impact: {float(impact)*10000:.2f} bps")
            else:
                self.log_test_result("Market Impact Basic Calculation", False, f"Unrealistic impact: {impact}")
                return False
            
            # Test 2: Urgency effect
            low_urgency_impact = model.estimate_impact(Decimal('10000'), urgency=0.2)
            high_urgency_impact = model.estimate_impact(Decimal('10000'), urgency=0.8)
            
            if high_urgency_impact > low_urgency_impact:
                self.log_test_result("Market Impact Urgency Effect", True, 
                                   f"High urgency: {float(high_urgency_impact)*10000:.2f} bps > Low urgency: {float(low_urgency_impact)*10000:.2f} bps")
            else:
                self.log_test_result("Market Impact Urgency Effect", False, "High urgency should have higher impact")
                return False
            
            # Test 3: Order size effect
            small_impact = model.estimate_impact(Decimal('1000'), urgency=0.5)
            large_impact = model.estimate_impact(Decimal('100000'), urgency=0.5)
            
            if large_impact > small_impact:
                self.log_test_result("Market Impact Size Effect", True, 
                                   f"Large order: {float(large_impact)*10000:.2f} bps > Small order: {float(small_impact)*10000:.2f} bps")
            else:
                self.log_test_result("Market Impact Size Effect", False, "Large orders should have higher impact")
                return False
            
            # Test 4: Liquidity profile effect
            high_liquidity = MarketImpactModel(
                symbol="TEST", avg_daily_volume=Decimal('10000000'),
                volatility=Decimal('0.02'), bid_ask_spread=Decimal('0.01'),
                liquidity_profile=LiquidityProfile.HIGH
            )
            
            low_liquidity = MarketImpactModel(
                symbol="TEST", avg_daily_volume=Decimal('10000000'),
                volatility=Decimal('0.02'), bid_ask_spread=Decimal('0.01'),
                liquidity_profile=LiquidityProfile.LOW
            )
            
            high_liq_impact = high_liquidity.estimate_impact(Decimal('5000'))
            low_liq_impact = low_liquidity.estimate_impact(Decimal('5000'))
            
            if low_liq_impact > high_liq_impact:
                self.log_test_result("Market Impact Liquidity Effect", True, 
                                   f"Low liquidity: {float(low_liq_impact)*10000:.2f} bps > High liquidity: {float(high_liq_impact)*10000:.2f} bps")
            else:
                self.log_test_result("Market Impact Liquidity Effect", False, "Low liquidity should have higher impact")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Market Impact Model Validation", False, f"Exception: {e}")
            return False
    
    async def validate_smart_order_router(self) -> bool:
        """Validate smart order routing"""
        print("\n=== Validating Smart Order Router ===")
        
        try:
            router = SmartOrderRouter()
            
            # Test 1: Basic routing
            test_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('5000'),
                side=OrderSide.BUY,
                urgency=0.5,
                allow_dark_pools=True
            )
            
            market_data = {
                'current_price': 150.0,
                'spread': 0.02,
                'volume': 2000000
            }
            
            slices = router.route_order(test_order, market_data)
            
            if len(slices) > 0:
                total_routed = sum(s.quantity for s in slices)
                if total_routed <= test_order.total_quantity:
                    self.log_test_result("Smart Router Basic Routing", True, 
                                       f"Routed {len(slices)} slices, total: {total_routed}")
                else:
                    self.log_test_result("Smart Router Basic Routing", False, "Over-routed quantity")
                    return False
            else:
                self.log_test_result("Smart Router Basic Routing", False, "No slices generated")
                return False
            
            # Test 2: Venue scoring
            scores = router._score_venues(test_order, market_data)
            
            if len(scores) > 0 and all(0 <= score <= 2 for score in scores.values()):
                self.log_test_result("Smart Router Venue Scoring", True, f"Scored {len(scores)} venues")
            else:
                self.log_test_result("Smart Router Venue Scoring", False, "Invalid venue scores")
                return False
            
            # Test 3: Dark pool routing for large orders
            large_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('20000'),
                side=OrderSide.BUY,
                urgency=0.3,
                allow_dark_pools=True
            )
            
            large_slices = router.route_order(large_order, market_data)
            dark_pool_slices = [s for s in large_slices if s.venue.venue_type == VenueType.DARK_POOL]
            
            if len(dark_pool_slices) > 0:
                self.log_test_result("Smart Router Dark Pool Routing", True, 
                                   f"Routed {len(dark_pool_slices)} slices to dark pools")
            else:
                self.log_test_result("Smart Router Dark Pool Routing", False, "No dark pool routing for large order")
                return False
            
            # Test 4: Small order routing (no dark pools)
            small_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('100'),
                side=OrderSide.BUY,
                urgency=0.8,
                allow_dark_pools=False
            )
            
            small_slices = router.route_order(small_order, market_data)
            small_dark_slices = [s for s in small_slices if s.venue.venue_type == VenueType.DARK_POOL]
            
            if len(small_dark_slices) == 0:
                self.log_test_result("Smart Router Small Order Routing", True, "No dark pool routing for small order")
            else:
                self.log_test_result("Smart Router Small Order Routing", False, "Unexpected dark pool routing")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Smart Order Router Validation", False, f"Exception: {e}")
            return False
    
    async def validate_execution_algorithms(self) -> bool:
        """Validate execution algorithms"""
        print("\n=== Validating Execution Algorithms ===")
        
        try:
            engine = self.execution_engine.algorithm_engine
            
            test_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('2000'),
                side=OrderSide.BUY,
                time_horizon_minutes=60
            )
            
            # Test 1: TWAP Algorithm
            twap_slices = await engine.execute_twap(test_order, 30)
            
            if len(twap_slices) > 0:
                total_twap = sum(s.quantity for s in twap_slices)
                if abs(total_twap - test_order.total_quantity) < Decimal('1'):
                    self.log_test_result("TWAP Algorithm", True, 
                                       f"Generated {len(twap_slices)} slices, total: {total_twap}")
                else:
                    self.log_test_result("TWAP Algorithm", False, f"Quantity mismatch: {total_twap} vs {test_order.total_quantity}")
                    return False
            else:
                self.log_test_result("TWAP Algorithm", False, "No TWAP slices generated")
                return False
            
            # Test 2: VWAP Algorithm
            volume_profile = [
                {'timestamp': datetime.now(timezone.utc), 'volume': 1000000, 'interval': 0},
                {'timestamp': datetime.now(timezone.utc), 'volume': 1500000, 'interval': 1},
                {'timestamp': datetime.now(timezone.utc), 'volume': 800000, 'interval': 2}
            ]
            
            vwap_slices = await engine.execute_vwap(test_order, volume_profile)
            
            if len(vwap_slices) > 0:
                total_vwap = sum(s.quantity for s in vwap_slices)
                if abs(total_vwap - test_order.total_quantity) < Decimal('10'):  # Allow small rounding differences
                    self.log_test_result("VWAP Algorithm", True, 
                                       f"Generated {len(vwap_slices)} slices, total: {total_vwap}")
                else:
                    self.log_test_result("VWAP Algorithm", False, f"Quantity mismatch: {total_vwap} vs {test_order.total_quantity}")
                    return False
            else:
                self.log_test_result("VWAP Algorithm", False, "No VWAP slices generated")
                return False
            
            # Test 3: Implementation Shortfall Algorithm
            market_impact_model = MarketImpactModel(
                symbol="AAPL",
                avg_daily_volume=Decimal('50000000'),
                volatility=Decimal('0.02'),
                bid_ask_spread=Decimal('0.01')
            )
            
            is_slices = await engine.execute_implementation_shortfall(test_order, market_impact_model)
            
            if len(is_slices) > 0 and len(is_slices) <= 20:  # Should respect max slices limit
                total_is = sum(s.quantity for s in is_slices)
                if total_is <= test_order.total_quantity:
                    self.log_test_result("Implementation Shortfall Algorithm", True, 
                                       f"Generated {len(is_slices)} slices, total: {total_is}")
                else:
                    self.log_test_result("Implementation Shortfall Algorithm", False, "Over-allocated quantity")
                    return False
            else:
                self.log_test_result("Implementation Shortfall Algorithm", False, 
                                   f"Invalid slice count: {len(is_slices) if is_slices else 0}")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Execution Algorithms Validation", False, f"Exception: {e}")
            return False
    
    async def validate_langgraph_workflow(self) -> bool:
        """Validate LangGraph workflow functionality"""
        print("\n=== Validating LangGraph Workflow ===")
        
        try:
            # Test 1: Workflow creation
            workflow = self.execution_engine.workflow
            
            if workflow is not None:
                self.log_test_result("LangGraph Workflow Creation", True, "Workflow compiled successfully")
            else:
                self.log_test_result("LangGraph Workflow Creation", False, "Workflow is None")
                return False
            
            # Test 2: Individual workflow steps
            test_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('1000'),
                side=OrderSide.BUY,
                client_order_id="validation_test"
            )
            
            # Mock market data
            market_data = {
                'current_price': 150.0,
                'avg_daily_volume': 50000000,
                'volume': 1000000,
                'spread': 0.01,
                'volatility': 0.02
            }
            
            market_impact_model = MarketImpactModel(
                symbol="AAPL",
                avg_daily_volume=Decimal('50000000'),
                volatility=Decimal('0.02'),
                bid_ask_spread=Decimal('0.01')
            )
            
            from agents.execution_engine_agent import ExecutionState
            
            initial_state = ExecutionState(
                execution_order=test_order,
                market_data=market_data,
                venue_data=self.execution_engine.smart_router.venues,
                market_impact_model=market_impact_model,
                execution_slices=[],
                completed_slices=[],
                failed_slices=[],
                current_slice=None,
                execution_metrics={},
                error_log=[],
                next_action="analyze_order"
            )
            
            # Test analyze_order step
            analyzed_state = await self.execution_engine._analyze_order(initial_state)
            
            if 'order_classification' in analyzed_state.execution_metrics:
                self.log_test_result("LangGraph Analyze Order Step", True, 
                                   f"Classification: {analyzed_state.execution_metrics['order_classification']}")
            else:
                self.log_test_result("LangGraph Analyze Order Step", False, "Missing order classification")
                return False
            
            # Test market impact estimation step
            impact_state = await self.execution_engine._estimate_market_impact(analyzed_state)
            
            if 'selected_impact_bps' in impact_state.execution_metrics:
                self.log_test_result("LangGraph Market Impact Step", True, 
                                   f"Impact: {impact_state.execution_metrics['selected_impact_bps']:.2f} bps")
            else:
                self.log_test_result("LangGraph Market Impact Step", False, "Missing impact estimation")
                return False
            
            # Test algorithm selection step
            algo_state = await self.execution_engine._select_algorithm(impact_state)
            
            if 'selected_algorithm' in algo_state.execution_metrics:
                self.log_test_result("LangGraph Algorithm Selection Step", True, 
                                   f"Algorithm: {algo_state.execution_metrics['selected_algorithm']}")
            else:
                self.log_test_result("LangGraph Algorithm Selection Step", False, "Missing algorithm selection")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("LangGraph Workflow Validation", False, f"Exception: {e}")
            return False
    
    async def validate_order_size_optimization(self) -> bool:
        """Validate order size optimization"""
        print("\n=== Validating Order Size Optimization ===")
        
        try:
            # Test 1: Participation rate limits
            large_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('100000'),  # Large order
                side=OrderSide.BUY,
                max_participation_rate=0.05,  # 5% max participation
                urgency=0.5
            )
            
            market_data = {
                'current_price': 150.0,
                'avg_daily_volume': 50000000,  # 50M daily volume
                'volume': 2000000,
                'spread': 0.01
            }
            
            # Calculate expected max slice size
            expected_max_slice = Decimal('50000000') * Decimal('0.05') / 390  # Daily volume * participation / minutes in trading day
            
            slices = self.execution_engine.smart_router.route_order(large_order, market_data)
            
            if slices:
                max_slice_size = max(s.quantity for s in slices)
                if max_slice_size <= expected_max_slice * 2:  # Allow some flexibility
                    self.log_test_result("Order Size Optimization - Participation Limits", True, 
                                       f"Max slice: {max_slice_size:,}, Expected max: {expected_max_slice:,.0f}")
                else:
                    self.log_test_result("Order Size Optimization - Participation Limits", False, 
                                       f"Slice too large: {max_slice_size:,}")
                    return False
            else:
                self.log_test_result("Order Size Optimization - Participation Limits", False, "No slices generated")
                return False
            
            # Test 2: Venue size limits
            router = self.execution_engine.smart_router
            
            # Check that venue allocations respect limits
            venue_allocations = {}
            for slice_order in slices:
                venue_id = slice_order.venue.venue_id
                if venue_id not in venue_allocations:
                    venue_allocations[venue_id] = Decimal('0')
                venue_allocations[venue_id] += slice_order.quantity
            
            max_venue_participation = router.routing_rules['max_venue_participation']
            max_venue_allocation = large_order.total_quantity * Decimal(str(max_venue_participation))
            
            venue_limit_respected = all(
                allocation <= max_venue_allocation * Decimal('1.1')  # Allow 10% tolerance
                for allocation in venue_allocations.values()
            )
            
            if venue_limit_respected:
                self.log_test_result("Order Size Optimization - Venue Limits", True, 
                                   f"Max venue allocation: {max(venue_allocations.values()):,}")
            else:
                self.log_test_result("Order Size Optimization - Venue Limits", False, 
                                   "Venue allocation limits exceeded")
                return False
            
            # Test 3: Liquidity-based sizing
            low_liquidity_order = ExecutionOrder(
                symbol="SMALLCAP",
                total_quantity=Decimal('10000'),
                side=OrderSide.BUY,
                urgency=0.3
            )
            
            low_liquidity_data = {
                'current_price': 50.0,
                'avg_daily_volume': 500000,  # Low volume stock
                'volume': 50000,
                'spread': 0.05
            }
            
            low_liq_slices = self.execution_engine.smart_router.route_order(
                low_liquidity_order, low_liquidity_data
            )
            
            if low_liq_slices:
                avg_slice_size = sum(s.quantity for s in low_liq_slices) / len(low_liq_slices)
                # For low liquidity stocks, slices should be smaller
                if avg_slice_size < Decimal('2000'):  # Reasonable for low liquidity
                    self.log_test_result("Order Size Optimization - Liquidity Based", True, 
                                       f"Avg slice size for low liquidity: {avg_slice_size:,.0f}")
                else:
                    self.log_test_result("Order Size Optimization - Liquidity Based", False, 
                                       f"Slice too large for low liquidity: {avg_slice_size:,.0f}")
                    return False
            else:
                self.log_test_result("Order Size Optimization - Liquidity Based", False, 
                                   "No slices for low liquidity stock")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Order Size Optimization Validation", False, f"Exception: {e}")
            return False
    
    async def validate_slippage_minimization(self) -> bool:
        """Validate slippage minimization features"""
        print("\n=== Validating Slippage Minimization ===")
        
        try:
            # Test 1: Limit price calculation
            router = self.execution_engine.smart_router
            
            test_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('1000'),
                side=OrderSide.BUY,
                urgency=0.5,
                max_slippage_bps=25
            )
            
            market_data = {
                'current_price': 150.0,
                'spread': 0.02,
                'volume': 1000000
            }
            
            # Test buy order limit price
            venue = list(router.venues.values())[0]  # Get first venue
            buy_limit_price = router._calculate_limit_price(venue, test_order, market_data)
            
            current_price = Decimal('150.0')
            spread = Decimal('0.02')
            
            if buy_limit_price:
                # Buy limit should be between bid and a reasonable premium
                bid_price = current_price - (spread / 2)
                max_acceptable = current_price + (spread / 2)
                
                if bid_price <= buy_limit_price <= max_acceptable:
                    self.log_test_result("Slippage Minimization - Buy Limit Price", True, 
                                       f"Limit: ${buy_limit_price:.2f}, Range: ${bid_price:.2f}-${max_acceptable:.2f}")
                else:
                    self.log_test_result("Slippage Minimization - Buy Limit Price", False, 
                                       f"Limit price out of range: ${buy_limit_price:.2f}")
                    return False
            else:
                self.log_test_result("Slippage Minimization - Buy Limit Price", False, "No limit price calculated")
                return False
            
            # Test sell order limit price
            sell_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('1000'),
                side=OrderSide.SELL,
                urgency=0.5
            )
            
            sell_limit_price = router._calculate_limit_price(venue, sell_order, market_data)
            
            if sell_limit_price:
                # Sell limit should be between ask and a reasonable discount
                ask_price = current_price + (spread / 2)
                min_acceptable = current_price - (spread / 2)
                
                if min_acceptable <= sell_limit_price <= ask_price:
                    self.log_test_result("Slippage Minimization - Sell Limit Price", True, 
                                       f"Limit: ${sell_limit_price:.2f}, Range: ${min_acceptable:.2f}-${ask_price:.2f}")
                else:
                    self.log_test_result("Slippage Minimization - Sell Limit Price", False, 
                                       f"Limit price out of range: ${sell_limit_price:.2f}")
                    return False
            else:
                self.log_test_result("Slippage Minimization - Sell Limit Price", False, "No sell limit price calculated")
                return False
            
            # Test 2: Urgency-based pricing
            urgent_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('1000'),
                side=OrderSide.BUY,
                urgency=0.9  # High urgency
            )
            
            patient_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('1000'),
                side=OrderSide.BUY,
                urgency=0.1  # Low urgency
            )
            
            urgent_limit = router._calculate_limit_price(venue, urgent_order, market_data)
            patient_limit = router._calculate_limit_price(venue, patient_order, market_data)
            
            if urgent_limit and patient_limit:
                # Urgent orders should have more aggressive (higher for buy) limit prices
                if urgent_limit > patient_limit:
                    self.log_test_result("Slippage Minimization - Urgency Pricing", True, 
                                       f"Urgent: ${urgent_limit:.2f} > Patient: ${patient_limit:.2f}")
                else:
                    self.log_test_result("Slippage Minimization - Urgency Pricing", False, 
                                       "Urgent orders should have more aggressive pricing")
                    return False
            else:
                self.log_test_result("Slippage Minimization - Urgency Pricing", False, "Missing limit prices")
                return False
            
            # Test 3: Dark pool preference for large orders
            large_order = ExecutionOrder(
                symbol="AAPL",
                total_quantity=Decimal('50000'),  # Large order
                side=OrderSide.BUY,
                urgency=0.3,  # Low urgency should favor dark pools
                allow_dark_pools=True
            )
            
            large_slices = router.route_order(large_order, market_data)
            dark_pool_slices = [s for s in large_slices if s.venue.venue_type == VenueType.DARK_POOL]
            
            if len(dark_pool_slices) > 0:
                dark_pool_quantity = sum(s.quantity for s in dark_pool_slices)
                total_quantity = sum(s.quantity for s in large_slices)
                dark_pool_percentage = float(dark_pool_quantity / total_quantity)
                
                if dark_pool_percentage > 0.2:  # At least 20% to dark pools
                    self.log_test_result("Slippage Minimization - Dark Pool Usage", True, 
                                       f"Dark pool allocation: {dark_pool_percentage:.1%}")
                else:
                    self.log_test_result("Slippage Minimization - Dark Pool Usage", False, 
                                       f"Insufficient dark pool usage: {dark_pool_percentage:.1%}")
                    return False
            else:
                self.log_test_result("Slippage Minimization - Dark Pool Usage", False, 
                                   "No dark pool routing for large order")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Slippage Minimization Validation", False, f"Exception: {e}")
            return False
    
    async def validate_performance_metrics(self) -> bool:
        """Validate performance metrics tracking"""
        print("\n=== Validating Performance Metrics ===")
        
        try:
            # Test 1: Initial metrics
            initial_metrics = self.execution_engine.get_performance_metrics()
            
            required_metrics = [
                'total_orders', 'successful_executions', 'avg_slippage_bps',
                'avg_market_impact_bps', 'active_executions', 'completed_executions',
                'success_rate'
            ]
            
            missing_metrics = [m for m in required_metrics if m not in initial_metrics]
            
            if not missing_metrics:
                self.log_test_result("Performance Metrics - Structure", True, 
                                   f"All {len(required_metrics)} metrics present")
            else:
                self.log_test_result("Performance Metrics - Structure", False, 
                                   f"Missing metrics: {missing_metrics}")
                return False
            
            # Test 2: Metrics data types
            valid_types = True
            for metric, value in initial_metrics.items():
                if not isinstance(value, (int, float)):
                    valid_types = False
                    break
            
            if valid_types:
                self.log_test_result("Performance Metrics - Data Types", True, "All metrics have valid types")
            else:
                self.log_test_result("Performance Metrics - Data Types", False, "Invalid metric data types")
                return False
            
            # Test 3: Execution status tracking
            test_execution_id = "test_execution_123"
            
            # Should return None for non-existent execution
            status = await self.execution_engine.get_execution_status(test_execution_id)
            
            if status is None:
                self.log_test_result("Performance Metrics - Status Tracking", True, 
                                   "Correctly returns None for non-existent execution")
            else:
                self.log_test_result("Performance Metrics - Status Tracking", False, 
                                   "Should return None for non-existent execution")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Performance Metrics Validation", False, f"Exception: {e}")
            return False
    
    async def run_all_validations(self) -> bool:
        """Run all validation tests"""
        print("EXECUTION ENGINE AGENT VALIDATION")
        print("=" * 60)
        
        validations = [
            ("Market Impact Model", self.validate_market_impact_model),
            ("Smart Order Router", self.validate_smart_order_router),
            ("Execution Algorithms", self.validate_execution_algorithms),
            ("LangGraph Workflow", self.validate_langgraph_workflow),
            ("Order Size Optimization", self.validate_order_size_optimization),
            ("Slippage Minimization", self.validate_slippage_minimization),
            ("Performance Metrics", self.validate_performance_metrics)
        ]
        
        all_passed = True
        
        for validation_name, validation_func in validations:
            try:
                result = await validation_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.log_test_result(f"{validation_name} Validation", False, f"Exception: {e}")
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        passed_tests = [r for r in self.test_results if r['status'] == 'PASS']
        failed_tests = [r for r in self.test_results if r['status'] == 'FAIL']
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {len(passed_tests)}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success Rate: {len(passed_tests)/len(self.test_results)*100:.1f}%")
        
        if failed_tests:
            print(f"\nFailed Tests:")
            for test in failed_tests:
                print(f"- {test['test']}: {test['details']}")
        
        if all_passed:
            print(f"\n✅ ALL VALIDATIONS PASSED")
            print("The Execution Engine Agent implementation is valid and ready for use.")
        else:
            print(f"\n❌ SOME VALIDATIONS FAILED")
            print("Please review and fix the failed tests before deployment.")
        
        return all_passed


async def main():
    """Main validation function"""
    validator = ExecutionEngineValidator()
    
    try:
        success = await validator.run_all_validations()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Validation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())