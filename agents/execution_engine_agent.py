"""
Execution Engine Agent - LangGraph-powered Smart Order Execution

This module implements a sophisticated execution engine agent using LangGraph for:
- Smart order routing and slippage minimization
- Market impact estimation and timing optimization
- Order size optimization based on liquidity
- Multi-venue execution with latency arbitrage
- TWAP, VWAP, Implementation Shortfall algorithms
"""

import sys
import os
from pathlib import Path

# Add project root to Python path to ensure local config is imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import asyncio
import logging
import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from langgraph.graph import StateGraph, END

from agents.broker_integration import (
    AlpacaBrokerIntegration, OrderRequest, OrderResponse, OrderSide, 
    OrderType, TimeInForce, BrokerError
)
from agents.market_data_ingestor import MarketDataIngestorAgent
from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ExecutionAlgorithm(str, Enum):
    """Execution algorithm types"""
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ARRIVAL_PRICE = "arrival_price"
    SMART_ROUTING = "smart_routing"


class VenueType(str, Enum):
    """Trading venue types"""
    PRIMARY_EXCHANGE = "primary_exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    ALTERNATIVE_VENUE = "alternative_venue"


class LiquidityProfile(str, Enum):
    """Liquidity profile classifications"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class VenueInfo:
    """Trading venue information"""
    venue_id: str
    venue_type: VenueType
    name: str
    fee_structure: Dict[str, Decimal]
    avg_latency_ms: float
    liquidity_score: float
    market_share: float
    supports_dark_pool: bool = False
    min_order_size: Optional[Decimal] = None
    max_order_size: Optional[Decimal] = None


@dataclass
class MarketImpactModel:
    """Market impact estimation model"""
    symbol: str
    avg_daily_volume: Decimal
    volatility: Decimal
    bid_ask_spread: Decimal
    market_cap: Optional[Decimal] = None
    liquidity_profile: LiquidityProfile = LiquidityProfile.MEDIUM
    
    def estimate_impact(self, order_size: Decimal, urgency: float = 0.5) -> Decimal:
        """
        Estimate market impact for an order
        
        Args:
            order_size: Size of the order
            urgency: Urgency factor (0-1, higher = more urgent)
            
        Returns:
            Estimated market impact as percentage of price
        """
        # Volume participation rate
        participation_rate = order_size / self.avg_daily_volume
        
        # Base impact using square root law
        base_impact = self.volatility * Decimal(str(math.sqrt(float(participation_rate))))
        
        # Adjust for bid-ask spread
        spread_impact = self.bid_ask_spread * Decimal('0.5')
        
        # Urgency adjustment
        urgency_multiplier = Decimal('1.0') + (Decimal(str(urgency)) * Decimal('0.5'))
        
        # Liquidity adjustment
        liquidity_multipliers = {
            LiquidityProfile.HIGH: 0.7,
            LiquidityProfile.MEDIUM: 1.0,
            LiquidityProfile.LOW: 1.5,
            LiquidityProfile.VERY_LOW: 2.5
        }
        
        liquidity_multiplier = liquidity_multipliers[self.liquidity_profile]
        
        total_impact = (base_impact + spread_impact) * urgency_multiplier * Decimal(str(liquidity_multiplier))
        
        return min(total_impact, Decimal('0.1'))  # Cap at 10%


@dataclass
class ExecutionOrder:
    """Execution order with optimization parameters"""
    symbol: str
    total_quantity: Decimal
    side: OrderSide
    target_price: Optional[Decimal] = None
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART_ROUTING
    time_horizon_minutes: int = 60
    urgency: float = 0.5  # 0-1 scale
    max_participation_rate: float = 0.1  # Max % of volume
    allow_dark_pools: bool = True
    max_slippage_bps: int = 50  # Max slippage in basis points
    client_order_id: Optional[str] = None
    
    # Execution state
    remaining_quantity: Decimal = field(init=False)
    executed_quantity: Decimal = field(default=Decimal('0'))
    child_orders: List[OrderResponse] = field(default_factory=list)
    avg_execution_price: Optional[Decimal] = None
    total_slippage: Decimal = field(default=Decimal('0'))
    total_market_impact: Decimal = field(default=Decimal('0'))
    
    def __post_init__(self):
        self.remaining_quantity = self.total_quantity


@dataclass
class ExecutionSlice:
    """Individual execution slice"""
    slice_id: str
    symbol: str
    quantity: Decimal
    side: OrderSide
    venue: VenueInfo
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.IOC
    expected_fill_time: Optional[datetime] = None
    priority_score: float = 0.0


@dataclass
class ExecutionState:
    """LangGraph execution state"""
    execution_order: ExecutionOrder
    market_data: Dict[str, Any]
    venue_data: Dict[str, VenueInfo]
    market_impact_model: MarketImpactModel
    execution_slices: List[ExecutionSlice]
    completed_slices: List[ExecutionSlice]
    failed_slices: List[ExecutionSlice]
    current_slice: Optional[ExecutionSlice]
    execution_metrics: Dict[str, Any]
    error_log: List[str]
    next_action: str


class SmartOrderRouter:
    """Smart order routing engine"""
    
    def __init__(self):
        self.venues = self._initialize_venues()
        self.routing_rules = self._initialize_routing_rules()
    
    def _initialize_venues(self) -> Dict[str, VenueInfo]:
        """Initialize trading venues"""
        venues = {
            'NASDAQ': VenueInfo(
                venue_id='NASDAQ',
                venue_type=VenueType.PRIMARY_EXCHANGE,
                name='NASDAQ',
                fee_structure={'maker': Decimal('0.0030'), 'taker': Decimal('0.0030')},
                avg_latency_ms=2.5,
                liquidity_score=0.95,
                market_share=0.25
            ),
            'NYSE': VenueInfo(
                venue_id='NYSE',
                venue_type=VenueType.PRIMARY_EXCHANGE,
                name='New York Stock Exchange',
                fee_structure={'maker': Decimal('0.0025'), 'taker': Decimal('0.0025')},
                avg_latency_ms=3.0,
                liquidity_score=0.90,
                market_share=0.20
            ),
            'ARCA': VenueInfo(
                venue_id='ARCA',
                venue_type=VenueType.ECN,
                name='NYSE Arca',
                fee_structure={'maker': Decimal('0.0020'), 'taker': Decimal('0.0030')},
                avg_latency_ms=2.0,
                liquidity_score=0.85,
                market_share=0.15
            ),
            'DARK_POOL_1': VenueInfo(
                venue_id='DARK_POOL_1',
                venue_type=VenueType.DARK_POOL,
                name='Dark Pool Alpha',
                fee_structure={'maker': Decimal('0.0015'), 'taker': Decimal('0.0015')},
                avg_latency_ms=5.0,
                liquidity_score=0.70,
                market_share=0.08,
                supports_dark_pool=True
            ),
            'IEX': VenueInfo(
                venue_id='IEX',
                venue_type=VenueType.ALTERNATIVE_VENUE,
                name='Investors Exchange',
                fee_structure={'maker': Decimal('0.0000'), 'taker': Decimal('0.0009')},
                avg_latency_ms=4.0,
                liquidity_score=0.75,
                market_share=0.05
            )
        }
        return venues
    
    def _initialize_routing_rules(self) -> Dict[str, Any]:
        """Initialize smart routing rules"""
        return {
            'min_size_for_dark_pool': Decimal('1000'),
            'max_venue_participation': 0.4,
            'latency_weight': 0.3,
            'cost_weight': 0.4,
            'liquidity_weight': 0.3,
            'dark_pool_preference_threshold': 0.7
        }
    
    def route_order(self, execution_order: ExecutionOrder, 
                   market_data: Dict[str, Any]) -> List[ExecutionSlice]:
        """
        Route order across multiple venues for optimal execution
        
        Args:
            execution_order: Order to route
            market_data: Current market data
            
        Returns:
            List of execution slices across venues
        """
        slices = []
        remaining_qty = execution_order.remaining_quantity
        
        # Get venue scores
        venue_scores = self._score_venues(execution_order, market_data)
        
        # Sort venues by score
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate quantity across venues
        for venue_id, score in sorted_venues:
            if remaining_qty <= 0:
                break
                
            venue = self.venues[venue_id]
            
            # Calculate allocation for this venue
            allocation = self._calculate_venue_allocation(
                venue, execution_order, remaining_qty, score
            )
            
            if allocation > 0:
                slice_order = ExecutionSlice(
                    slice_id=f"{execution_order.client_order_id}_{venue_id}_{len(slices)}",
                    symbol=execution_order.symbol,
                    quantity=allocation,
                    side=execution_order.side,
                    venue=venue,
                    order_type=self._determine_order_type(venue, execution_order),
                    limit_price=self._calculate_limit_price(venue, execution_order, market_data),
                    priority_score=score
                )
                
                slices.append(slice_order)
                remaining_qty -= allocation
        
        return slices
    
    def _score_venues(self, execution_order: ExecutionOrder, 
                     market_data: Dict[str, Any]) -> Dict[str, float]:
        """Score venues for order routing"""
        scores = {}
        
        for venue_id, venue in self.venues.items():
            # Skip dark pools for small orders unless specifically requested
            if (venue.venue_type == VenueType.DARK_POOL and 
                execution_order.total_quantity < self.routing_rules['min_size_for_dark_pool'] and
                not execution_order.allow_dark_pools):
                continue
            
            # Calculate composite score
            latency_score = 1.0 / (1.0 + venue.avg_latency_ms / 10.0)
            cost_score = 1.0 - float(venue.fee_structure.get('taker', Decimal('0.003')))
            liquidity_score = venue.liquidity_score
            
            # Weighted composite score
            composite_score = (
                latency_score * self.routing_rules['latency_weight'] +
                cost_score * self.routing_rules['cost_weight'] +
                liquidity_score * self.routing_rules['liquidity_weight']
            )
            
            # Adjust for dark pool preference
            if (venue.venue_type == VenueType.DARK_POOL and 
                execution_order.urgency < self.routing_rules['dark_pool_preference_threshold']):
                composite_score *= 1.2  # Boost dark pool score for non-urgent orders
            
            scores[venue_id] = composite_score
        
        return scores
    
    def _calculate_venue_allocation(self, venue: VenueInfo, execution_order: ExecutionOrder,
                                  remaining_qty: Decimal, venue_score: float) -> Decimal:
        """Calculate quantity allocation for a venue"""
        # Base allocation based on venue score and market share
        base_allocation = remaining_qty * Decimal(str(venue.market_share * venue_score))
        
        # Apply venue participation limits
        max_venue_qty = remaining_qty * Decimal(str(self.routing_rules['max_venue_participation']))
        
        # Apply venue-specific limits
        if venue.max_order_size:
            max_venue_qty = min(max_venue_qty, venue.max_order_size)
        
        allocation = min(base_allocation, max_venue_qty)
        
        # Ensure minimum order size
        if venue.min_order_size and allocation < venue.min_order_size:
            allocation = Decimal('0')
        
        return allocation
    
    def _determine_order_type(self, venue: VenueInfo, execution_order: ExecutionOrder) -> OrderType:
        """Determine optimal order type for venue"""
        if venue.venue_type == VenueType.DARK_POOL:
            return OrderType.LIMIT  # Dark pools typically use limit orders
        elif execution_order.urgency > 0.8:
            return OrderType.MARKET  # High urgency uses market orders
        else:
            return OrderType.LIMIT  # Default to limit orders
    
    def _calculate_limit_price(self, venue: VenueInfo, execution_order: ExecutionOrder,
                             market_data: Dict[str, Any]) -> Optional[Decimal]:
        """Calculate optimal limit price for venue"""
        if not market_data.get('current_price'):
            return None
        
        current_price = Decimal(str(market_data['current_price']))
        spread = Decimal(str(market_data.get('spread', 0.01)))
        
        # Adjust price based on side and urgency
        if execution_order.side == OrderSide.BUY:
            # For buy orders, start at bid and adjust up based on urgency
            base_price = current_price - (spread / 2)
            urgency_adjustment = spread * Decimal(str(execution_order.urgency))
            limit_price = base_price + urgency_adjustment
        else:
            # For sell orders, start at ask and adjust down based on urgency
            base_price = current_price + (spread / 2)
            urgency_adjustment = spread * Decimal(str(execution_order.urgency))
            limit_price = base_price - urgency_adjustment
        
        return limit_price


class ExecutionAlgorithmEngine:
    """Execution algorithm implementations"""
    
    def __init__(self, market_data_ingestor: MarketDataIngestorAgent):
        self.market_data_ingestor = market_data_ingestor
    
    async def execute_twap(self, execution_order: ExecutionOrder, 
                          time_horizon_minutes: int) -> List[ExecutionSlice]:
        """
        Time-Weighted Average Price execution
        
        Args:
            execution_order: Order to execute
            time_horizon_minutes: Time horizon for execution
            
        Returns:
            List of execution slices over time
        """
        slices = []
        total_slices = max(1, time_horizon_minutes // 5)  # 5-minute intervals
        slice_quantity = execution_order.remaining_quantity / total_slices
        
        for i in range(total_slices):
            execution_time = datetime.now(timezone.utc) + timedelta(minutes=i * 5)
            
            slice_order = ExecutionSlice(
                slice_id=f"{execution_order.client_order_id}_twap_{i}",
                symbol=execution_order.symbol,
                quantity=slice_quantity,
                side=execution_order.side,
                venue=self._select_primary_venue(),
                order_type=OrderType.LIMIT,
                expected_fill_time=execution_time,
                priority_score=1.0 - (i / total_slices)  # Earlier slices have higher priority
            )
            
            slices.append(slice_order)
        
        return slices
    
    async def execute_vwap(self, execution_order: ExecutionOrder,
                          historical_volume_profile: List[Dict]) -> List[ExecutionSlice]:
        """
        Volume-Weighted Average Price execution
        
        Args:
            execution_order: Order to execute
            historical_volume_profile: Historical volume profile
            
        Returns:
            List of execution slices based on volume profile
        """
        slices = []
        total_volume = sum(period['volume'] for period in historical_volume_profile)
        
        for i, period in enumerate(historical_volume_profile):
            volume_weight = period['volume'] / total_volume
            slice_quantity = execution_order.remaining_quantity * Decimal(str(volume_weight))
            
            if slice_quantity > 0:
                slice_order = ExecutionSlice(
                    slice_id=f"{execution_order.client_order_id}_vwap_{i}",
                    symbol=execution_order.symbol,
                    quantity=slice_quantity,
                    side=execution_order.side,
                    venue=self._select_primary_venue(),
                    order_type=OrderType.LIMIT,
                    expected_fill_time=period.get('timestamp'),
                    priority_score=volume_weight
                )
                
                slices.append(slice_order)
        
        return slices
    
    async def execute_implementation_shortfall(self, execution_order: ExecutionOrder,
                                             market_impact_model: MarketImpactModel) -> List[ExecutionSlice]:
        """
        Implementation Shortfall execution algorithm
        
        Args:
            execution_order: Order to execute
            market_impact_model: Market impact model
            
        Returns:
            Optimized execution slices
        """
        slices = []
        
        # Calculate optimal execution rate based on market impact vs timing risk
        optimal_rate = self._calculate_optimal_execution_rate(
            execution_order, market_impact_model
        )
        
        # Create execution schedule
        remaining_qty = execution_order.remaining_quantity
        slice_count = 0
        
        while remaining_qty > 0 and slice_count < 20:  # Max 20 slices
            slice_qty = min(remaining_qty, optimal_rate)
            
            slice_order = ExecutionSlice(
                slice_id=f"{execution_order.client_order_id}_is_{slice_count}",
                symbol=execution_order.symbol,
                quantity=slice_qty,
                side=execution_order.side,
                venue=self._select_optimal_venue(slice_qty),
                order_type=OrderType.LIMIT,
                expected_fill_time=datetime.now(timezone.utc) + timedelta(minutes=slice_count * 2),
                priority_score=1.0 - (slice_count * 0.05)
            )
            
            slices.append(slice_order)
            remaining_qty -= slice_qty
            slice_count += 1
        
        return slices
    
    def _calculate_optimal_execution_rate(self, execution_order: ExecutionOrder,
                                        market_impact_model: MarketImpactModel) -> Decimal:
        """Calculate optimal execution rate for implementation shortfall"""
        # Simplified implementation - in practice would use more sophisticated optimization
        daily_volume = market_impact_model.avg_daily_volume
        max_participation = Decimal(str(execution_order.max_participation_rate))
        
        # Calculate rate based on participation limits and urgency
        base_rate = daily_volume * max_participation / 390  # 390 minutes in trading day
        urgency_multiplier = Decimal(str(1.0 + execution_order.urgency))
        
        optimal_rate = base_rate * urgency_multiplier
        
        # Ensure we don't exceed remaining quantity
        return min(optimal_rate, execution_order.remaining_quantity)
    
    def _select_primary_venue(self) -> VenueInfo:
        """Select primary venue for execution"""
        # Return NASDAQ as default primary venue
        return VenueInfo(
            venue_id='NASDAQ',
            venue_type=VenueType.PRIMARY_EXCHANGE,
            name='NASDAQ',
            fee_structure={'maker': Decimal('0.0030'), 'taker': Decimal('0.0030')},
            avg_latency_ms=2.5,
            liquidity_score=0.95,
            market_share=0.25
        )
    
    def _select_optimal_venue(self, quantity: Decimal) -> VenueInfo:
        """Select optimal venue based on order size"""
        if quantity >= Decimal('1000'):
            # Large orders go to dark pools
            return VenueInfo(
                venue_id='DARK_POOL_1',
                venue_type=VenueType.DARK_POOL,
                name='Dark Pool Alpha',
                fee_structure={'maker': Decimal('0.0015'), 'taker': Decimal('0.0015')},
                avg_latency_ms=5.0,
                liquidity_score=0.70,
                market_share=0.08,
                supports_dark_pool=True
            )
        else:
            # Small orders go to primary exchange
            return self._select_primary_venue()


class ExecutionEngineAgent:
    """
    LangGraph-powered Execution Engine Agent
    
    Implements sophisticated order execution with:
    - Smart order routing across multiple venues
    - Market impact estimation and minimization
    - Multiple execution algorithms (TWAP, VWAP, Implementation Shortfall)
    - Real-time optimization and adaptation
    """
    
    def __init__(self, 
                 broker_integration: Optional[AlpacaBrokerIntegration] = None,
                 market_data_ingestor: Optional[MarketDataIngestorAgent] = None):
        """
        Initialize Execution Engine Agent
        
        Args:
            broker_integration: Broker integration instance
            market_data_ingestor: Market data ingestor instance
        """
        self.broker = broker_integration or AlpacaBrokerIntegration()
        self.market_data_ingestor = market_data_ingestor or MarketDataIngestorAgent()
        
        # Initialize components
        self.smart_router = SmartOrderRouter()
        self.algorithm_engine = ExecutionAlgorithmEngine(self.market_data_ingestor)
        
        # Execution tracking
        self.active_executions: Dict[str, ExecutionState] = {}
        self.completed_executions: Dict[str, ExecutionState] = {}
        
        # Performance metrics
        self.execution_metrics = {
            'total_orders': 0,
            'successful_executions': 0,
            'avg_slippage_bps': 0.0,
            'avg_market_impact_bps': 0.0,
            'avg_execution_time_minutes': 0.0
        }
        
        # Initialize LangGraph workflow
        self.workflow = self._create_execution_workflow()
        
        logger.info("Execution Engine Agent initialized")
    
    def _create_execution_workflow(self) -> StateGraph:
        """Create LangGraph workflow for execution"""
        workflow = StateGraph(ExecutionState)
        
        # Add nodes
        workflow.add_node("analyze_order", self._analyze_order)
        workflow.add_node("estimate_market_impact", self._estimate_market_impact)
        workflow.add_node("select_algorithm", self._select_algorithm)
        workflow.add_node("create_execution_plan", self._create_execution_plan)
        workflow.add_node("route_orders", self._route_orders)
        workflow.add_node("execute_slice", self._execute_slice)
        workflow.add_node("monitor_execution", self._monitor_execution)
        workflow.add_node("optimize_execution", self._optimize_execution)
        workflow.add_node("complete_execution", self._complete_execution)
        
        # Define workflow edges
        workflow.add_edge("analyze_order", "estimate_market_impact")
        workflow.add_edge("estimate_market_impact", "select_algorithm")
        workflow.add_edge("select_algorithm", "create_execution_plan")
        workflow.add_edge("create_execution_plan", "route_orders")
        workflow.add_edge("route_orders", "execute_slice")
        
        # Conditional edges for execution loop
        workflow.add_conditional_edges(
            "execute_slice",
            self._should_continue_execution,
            {
                "continue": "monitor_execution",
                "optimize": "optimize_execution",
                "complete": "complete_execution"
            }
        )
        
        workflow.add_edge("monitor_execution", "execute_slice")
        workflow.add_edge("optimize_execution", "execute_slice")
        workflow.add_edge("complete_execution", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_order")
        
        return workflow.compile()
    
    async def execute_order(self, execution_order: ExecutionOrder) -> ExecutionState:
        """
        Execute order using LangGraph workflow
        
        Args:
            execution_order: Order to execute
            
        Returns:
            Final execution state
        """
        logger.info(f"Starting execution for order: {execution_order.symbol} {execution_order.side} {execution_order.total_quantity}")
        
        # Get market data
        market_data = await self._get_market_data(execution_order.symbol)
        
        # Create market impact model
        market_impact_model = await self._create_market_impact_model(
            execution_order.symbol, market_data
        )
        
        # Initialize execution state
        initial_state = ExecutionState(
            execution_order=execution_order,
            market_data=market_data,
            venue_data=self.smart_router.venues,
            market_impact_model=market_impact_model,
            execution_slices=[],
            completed_slices=[],
            failed_slices=[],
            current_slice=None,
            execution_metrics={},
            error_log=[],
            next_action="analyze_order"
        )
        
        # Track execution
        self.active_executions[execution_order.client_order_id or "default"] = initial_state
        
        try:
            # Run LangGraph workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Move to completed executions
            execution_id = execution_order.client_order_id or "default"
            self.completed_executions[execution_id] = final_state
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            # Update metrics
            self._update_execution_metrics(final_state)
            
            logger.info(f"Execution completed for order: {execution_order.symbol}")
            return final_state
            
        except Exception as e:
            logger.error(f"Execution failed for order {execution_order.symbol}: {e}")
            initial_state.error_log.append(str(e))
            return initial_state
    
    async def _analyze_order(self, state: ExecutionState) -> ExecutionState:
        """Analyze order characteristics"""
        order = state.execution_order
        market_data = state.market_data
        
        # Analyze order size relative to average volume
        avg_daily_volume = Decimal(str(market_data.get('avg_daily_volume', 1000000)))
        participation_rate = order.total_quantity / avg_daily_volume
        
        # Classify order urgency and size
        if participation_rate > Decimal('0.1'):
            order_classification = "large"
        elif participation_rate > Decimal('0.05'):
            order_classification = "medium"
        else:
            order_classification = "small"
        
        # Update execution metrics
        state.execution_metrics.update({
            'order_classification': order_classification,
            'participation_rate': float(participation_rate),
            'analysis_timestamp': datetime.now(timezone.utc)
        })
        
        state.next_action = "estimate_market_impact"
        return state
    
    async def _estimate_market_impact(self, state: ExecutionState) -> ExecutionState:
        """Estimate market impact for the order"""
        order = state.execution_order
        impact_model = state.market_impact_model
        
        # Estimate impact for different execution speeds
        impact_estimates = {}
        for urgency in [0.2, 0.5, 0.8]:
            impact = impact_model.estimate_impact(order.total_quantity, urgency)
            impact_estimates[f'urgency_{urgency}'] = float(impact)
        
        # Select optimal urgency based on order requirements
        if order.urgency > 0.7:
            selected_impact = impact_estimates['urgency_0.8']
        elif order.urgency > 0.4:
            selected_impact = impact_estimates['urgency_0.5']
        else:
            selected_impact = impact_estimates['urgency_0.2']
        
        state.execution_metrics.update({
            'impact_estimates': impact_estimates,
            'selected_impact_bps': selected_impact * 10000,  # Convert to basis points
            'impact_estimation_timestamp': datetime.now(timezone.utc)
        })
        
        state.next_action = "select_algorithm"
        return state
    
    async def _select_algorithm(self, state: ExecutionState) -> ExecutionState:
        """Select optimal execution algorithm"""
        order = state.execution_order
        metrics = state.execution_metrics
        
        # Algorithm selection logic
        if order.algorithm != ExecutionAlgorithm.SMART_ROUTING:
            selected_algorithm = order.algorithm
        else:
            # Smart algorithm selection
            participation_rate = metrics.get('participation_rate', 0)
            impact_bps = metrics.get('selected_impact_bps', 0)
            
            if order.urgency > 0.8:
                selected_algorithm = ExecutionAlgorithm.MARKET
            elif participation_rate > 0.1 or impact_bps > 100:
                selected_algorithm = ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL
            elif order.time_horizon_minutes > 60:
                selected_algorithm = ExecutionAlgorithm.VWAP
            else:
                selected_algorithm = ExecutionAlgorithm.TWAP
        
        state.execution_metrics['selected_algorithm'] = selected_algorithm.value
        state.next_action = "create_execution_plan"
        return state
    
    async def _create_execution_plan(self, state: ExecutionState) -> ExecutionState:
        """Create detailed execution plan"""
        order = state.execution_order
        algorithm = ExecutionAlgorithm(state.execution_metrics['selected_algorithm'])
        
        # Create execution slices based on selected algorithm
        if algorithm == ExecutionAlgorithm.TWAP:
            slices = await self.algorithm_engine.execute_twap(
                order, order.time_horizon_minutes
            )
        elif algorithm == ExecutionAlgorithm.VWAP:
            # Get historical volume profile
            volume_profile = await self._get_volume_profile(order.symbol)
            slices = await self.algorithm_engine.execute_vwap(order, volume_profile)
        elif algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
            slices = await self.algorithm_engine.execute_implementation_shortfall(
                order, state.market_impact_model
            )
        else:
            # Default to smart routing
            slices = self.smart_router.route_order(order, state.market_data)
        
        state.execution_slices = slices
        state.execution_metrics['total_slices'] = len(slices)
        state.next_action = "route_orders"
        return state
    
    async def _route_orders(self, state: ExecutionState) -> ExecutionState:
        """Route orders to optimal venues"""
        # Apply smart routing to each slice if not already done
        if state.execution_metrics.get('selected_algorithm') != ExecutionAlgorithm.SMART_ROUTING.value:
            # Re-route slices for optimal venue selection
            optimized_slices = []
            for slice_order in state.execution_slices:
                # Create temporary execution order for routing
                temp_order = ExecutionOrder(
                    symbol=slice_order.symbol,
                    total_quantity=slice_order.quantity,
                    side=slice_order.side,
                    urgency=state.execution_order.urgency,
                    allow_dark_pools=state.execution_order.allow_dark_pools
                )
                
                # Route this slice
                routed_slices = self.smart_router.route_order(temp_order, state.market_data)
                optimized_slices.extend(routed_slices)
            
            state.execution_slices = optimized_slices
        
        # Sort slices by priority
        state.execution_slices.sort(key=lambda x: x.priority_score, reverse=True)
        
        state.next_action = "execute_slice"
        return state
    
    async def _execute_slice(self, state: ExecutionState) -> ExecutionState:
        """Execute next slice in the plan"""
        if not state.execution_slices:
            state.next_action = "complete"
            return state
        
        # Get next slice to execute
        current_slice = state.execution_slices.pop(0)
        state.current_slice = current_slice
        
        try:
            # Create order request
            order_request = OrderRequest(
                symbol=current_slice.symbol,
                qty=current_slice.quantity,
                side=current_slice.side,
                type=current_slice.order_type,
                time_in_force=current_slice.time_in_force,
                limit_price=current_slice.limit_price,
                client_order_id=current_slice.slice_id
            )
            
            # Submit order
            order_response = await self.broker.submit_order(order_request)
            
            # Update execution order
            state.execution_order.child_orders.append(order_response)
            
            # Track slice execution
            current_slice.expected_fill_time = datetime.now(timezone.utc)
            
            logger.info(f"Executed slice: {current_slice.slice_id} for {current_slice.quantity} shares")
            
            state.next_action = "continue"
            
        except Exception as e:
            logger.error(f"Failed to execute slice {current_slice.slice_id}: {e}")
            state.failed_slices.append(current_slice)
            state.error_log.append(f"Slice execution failed: {e}")
            state.next_action = "optimize"
        
        return state
    
    async def _monitor_execution(self, state: ExecutionState) -> ExecutionState:
        """Monitor execution progress"""
        if not state.current_slice:
            state.next_action = "continue"
            return state
        
        try:
            # Check order status
            order_id = None
            for order in state.execution_order.child_orders:
                if order.client_order_id == state.current_slice.slice_id:
                    order_id = order.id
                    break
            
            if order_id:
                order_status = await self.broker.get_order_status(order_id)
                
                if order_status and order_status.status.value in ['filled', 'canceled', 'rejected']:
                    # Slice completed
                    if order_status.status.value == 'filled':
                        state.completed_slices.append(state.current_slice)
                        
                        # Update execution order
                        state.execution_order.executed_quantity += order_status.filled_qty
                        state.execution_order.remaining_quantity -= order_status.filled_qty
                        
                        # Calculate execution price
                        if order_status.limit_price:
                            if state.execution_order.avg_execution_price is None:
                                state.execution_order.avg_execution_price = order_status.limit_price
                            else:
                                # Weighted average
                                total_executed = state.execution_order.executed_quantity
                                prev_total = total_executed - order_status.filled_qty
                                
                                if total_executed > 0:
                                    state.execution_order.avg_execution_price = (
                                        (state.execution_order.avg_execution_price * prev_total + 
                                         order_status.limit_price * order_status.filled_qty) / total_executed
                                    )
                    else:
                        # Slice failed
                        state.failed_slices.append(state.current_slice)
                        state.error_log.append(f"Slice {state.current_slice.slice_id} {order_status.status.value}")
                    
                    state.current_slice = None
            
        except Exception as e:
            logger.error(f"Error monitoring execution: {e}")
            state.error_log.append(f"Monitoring error: {e}")
        
        state.next_action = "continue"
        return state
    
    async def _optimize_execution(self, state: ExecutionState) -> ExecutionState:
        """Optimize execution based on current performance"""
        # Analyze failed slices and adjust strategy
        if state.failed_slices:
            logger.info(f"Optimizing execution after {len(state.failed_slices)} failed slices")
            
            # Adjust remaining slices
            for slice_order in state.execution_slices:
                # Increase urgency for remaining slices
                if slice_order.order_type == OrderType.LIMIT:
                    # Make limit prices more aggressive
                    if state.market_data.get('current_price'):
                        current_price = Decimal(str(state.market_data['current_price']))
                        spread = Decimal(str(state.market_data.get('spread', 0.01)))
                        
                        if slice_order.side == OrderSide.BUY:
                            slice_order.limit_price = current_price + (spread * Decimal('0.5'))
                        else:
                            slice_order.limit_price = current_price - (spread * Decimal('0.5'))
                
                # Reduce slice size to improve fill probability
                slice_order.quantity *= Decimal('0.8')
        
        state.next_action = "continue"
        return state
    
    async def _complete_execution(self, state: ExecutionState) -> ExecutionState:
        """Complete execution and calculate final metrics"""
        order = state.execution_order
        
        # Calculate final metrics
        if order.avg_execution_price and state.market_data.get('current_price'):
            benchmark_price = Decimal(str(state.market_data['current_price']))
            
            if order.side == OrderSide.BUY:
                slippage = order.avg_execution_price - benchmark_price
            else:
                slippage = benchmark_price - order.avg_execution_price
            
            order.total_slippage = slippage
            slippage_bps = (slippage / benchmark_price) * 10000 if benchmark_price > 0 else Decimal('0')
        else:
            slippage_bps = Decimal('0')
        
        # Update final metrics
        state.execution_metrics.update({
            'completion_timestamp': datetime.now(timezone.utc),
            'executed_quantity': float(order.executed_quantity),
            'remaining_quantity': float(order.remaining_quantity),
            'avg_execution_price': float(order.avg_execution_price) if order.avg_execution_price else None,
            'total_slippage_bps': float(slippage_bps),
            'successful_slices': len(state.completed_slices),
            'failed_slices': len(state.failed_slices),
            'execution_rate': float(order.executed_quantity / order.total_quantity) if order.total_quantity > 0 else 0
        })
        
        logger.info(f"Execution completed: {order.symbol} - {state.execution_metrics['execution_rate']:.2%} filled")
        
        state.next_action = "end"
        return state
    
    def _should_continue_execution(self, state: ExecutionState) -> str:
        """Determine next action in execution workflow"""
        if state.next_action == "complete":
            return "complete"
        elif state.next_action == "optimize":
            return "optimize"
        elif state.execution_slices or state.current_slice:
            return "continue"
        else:
            return "complete"
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        try:
            # For now, return mock data since the market data ingestor doesn't have get_latest_data method
            # In production, this would integrate with the actual market data ingestor
            market_data = {
                'close': 100.0,
                'volume': 1000000,
                'symbol': symbol
            }
            
            return {
                'current_price': market_data.get('close', 100.0),
                'volume': market_data.get('volume', 1000000),
                'avg_daily_volume': market_data.get('volume', 1000000) * 20,  # Estimate
                'spread': 0.01,  # Default spread
                'volatility': 0.02  # Default volatility
            }
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {
                'current_price': 100.0,
                'volume': 1000000,
                'avg_daily_volume': 20000000,
                'spread': 0.01,
                'volatility': 0.02
            }
    
    async def _create_market_impact_model(self, symbol: str, 
                                        market_data: Dict[str, Any]) -> MarketImpactModel:
        """Create market impact model for symbol"""
        return MarketImpactModel(
            symbol=symbol,
            avg_daily_volume=Decimal(str(market_data.get('avg_daily_volume', 20000000))),
            volatility=Decimal(str(market_data.get('volatility', 0.02))),
            bid_ask_spread=Decimal(str(market_data.get('spread', 0.01))),
            liquidity_profile=LiquidityProfile.MEDIUM
        )
    
    async def _get_volume_profile(self, symbol: str) -> List[Dict]:
        """Get historical volume profile for VWAP execution"""
        # Simplified volume profile - in practice would use historical data
        trading_hours = 6.5 * 60  # 390 minutes
        intervals = 39  # 10-minute intervals
        
        volume_profile = []
        for i in range(intervals):
            # Simulate U-shaped volume pattern
            hour_factor = abs(i - intervals/2) / (intervals/2)
            volume_weight = 0.5 + (hour_factor * 0.5)
            
            volume_profile.append({
                'timestamp': datetime.now(timezone.utc) + timedelta(minutes=i * 10),
                'volume': int(1000000 * volume_weight),
                'interval': i
            })
        
        return volume_profile
    
    def _update_execution_metrics(self, final_state: ExecutionState):
        """Update global execution metrics"""
        self.execution_metrics['total_orders'] += 1
        
        if final_state.execution_metrics.get('execution_rate', 0) > 0.9:
            self.execution_metrics['successful_executions'] += 1
        
        # Update running averages
        slippage_bps = final_state.execution_metrics.get('total_slippage_bps', 0)
        impact_bps = final_state.execution_metrics.get('selected_impact_bps', 0)
        
        total_orders = self.execution_metrics['total_orders']
        
        # Running average calculation
        self.execution_metrics['avg_slippage_bps'] = (
            (self.execution_metrics['avg_slippage_bps'] * (total_orders - 1) + slippage_bps) / total_orders
        )
        
        self.execution_metrics['avg_market_impact_bps'] = (
            (self.execution_metrics['avg_market_impact_bps'] * (total_orders - 1) + impact_bps) / total_orders
        )
    
    async def get_execution_status(self, execution_id: str) -> Optional[ExecutionState]:
        """Get execution status by ID"""
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        elif execution_id in self.completed_executions:
            return self.completed_executions[execution_id]
        else:
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        return {
            **self.execution_metrics,
            'active_executions': len(self.active_executions),
            'completed_executions': len(self.completed_executions),
            'success_rate': (
                self.execution_metrics['successful_executions'] / 
                max(1, self.execution_metrics['total_orders'])
            )
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_execution_engine():
        """Test the execution engine agent"""
        # Initialize components
        broker = AlpacaBrokerIntegration(paper_trading=True)
        market_data_ingestor = MarketDataIngestorAgent()
        execution_engine = ExecutionEngineAgent(broker, market_data_ingestor)
        
        # Create test execution order
        test_order = ExecutionOrder(
            symbol="AAPL",
            total_quantity=Decimal('1000'),
            side=OrderSide.BUY,
            algorithm=ExecutionAlgorithm.SMART_ROUTING,
            time_horizon_minutes=30,
            urgency=0.5,
            max_participation_rate=0.1,
            allow_dark_pools=True,
            max_slippage_bps=50,
            client_order_id="test_execution_001"
        )
        
        # Execute order
        final_state = await execution_engine.execute_order(test_order)
        
        # Print results
        print(f"Execution completed:")
        print(f"- Executed: {final_state.execution_order.executed_quantity}")
        print(f"- Remaining: {final_state.execution_order.remaining_quantity}")
        print(f"- Avg Price: {final_state.execution_order.avg_execution_price}")
        print(f"- Slippage: {final_state.execution_metrics.get('total_slippage_bps', 0):.2f} bps")
        print(f"- Success Rate: {final_state.execution_metrics.get('execution_rate', 0):.2%}")
        
        # Get performance metrics
        metrics = execution_engine.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"- {key}: {value}")
    
    # Run test
    asyncio.run(test_execution_engine())