"""
Smart Pricing Agent - Intelligent Entry and Exit Price Determination
Maximizes profit through optimal order pricing and timing
"""

import sys
import os
from pathlib import Path

# Add project root to Python path to ensure local config is imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import math

from agents.broker_integration import OrderSide, OrderType
from agents.options_broker import OptionsOrderType
from agents.quantlib_pricing import quantlib_pricer
from config.logging_config import get_logger

logger = get_logger(__name__)

class PricingStrategy(Enum):
    """Different pricing strategies for order execution"""
    AGGRESSIVE = "aggressive"        # Market orders for speed
    SMART_MID = "smart_mid"         # Target mid-price with patience
    SCALPER = "scalper"             # Try to get better than mid-price
    TIME_WEIGHTED = "time_weighted" # Adjust based on time to expiry
    GREEKS_BASED = "greeks_based"   # Use Greeks for optimal pricing

@dataclass
class PricingContext:
    """Context for pricing decisions"""
    symbol: str
    underlying_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    spread_pct: float
    time_to_expiry_days: int
    volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class SmartPriceTarget:
    """Intelligent price target with reasoning"""
    target_price: float
    order_type: str  # 'MARKET', 'LIMIT', 'SMART_LIMIT'
    limit_price: Optional[float]
    time_in_force: str  # 'DAY', 'GTC', 'IOC', 'FOK'
    expected_fill_probability: float
    reasoning: str
    max_wait_seconds: int = 300  # 5 minutes default

class SmartPricingAgent:
    """Advanced pricing agent for optimal trade execution"""
    
    def __init__(self):
        # Pricing parameters
        self.max_spread_tolerance = 0.15  # 15% max spread for trading
        self.mid_price_improvement_threshold = 0.02  # Try to improve by 2% minimum
        self.volume_threshold = 50  # Minimum volume for liquid options
        self.open_interest_threshold = 100  # Minimum OI for liquid options
        
        # Time-based pricing adjustments
        self.close_to_expiry_days = 7  # Aggressive pricing when < 7 days
        self.far_expiry_days = 30      # Patient pricing when > 30 days
        
        # Greeks-based pricing factors
        self.high_gamma_threshold = 0.05   # High gamma = more price sensitive
        self.high_theta_threshold = 0.10   # High theta = time decay risk
        self.high_vega_threshold = 0.30    # High vega = volatility sensitive
        
    def analyze_pricing_context(self, contract_data: Dict) -> PricingContext:
        """Analyze contract data to create pricing context"""
        
        bid = float(contract_data.get('bid', 0))
        ask = float(contract_data.get('ask', 0))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        spread_pct = (ask - bid) / mid * 100 if mid > 0 else 100
        
        # Calculate time to expiry
        expiry = contract_data.get('expiration')
        if isinstance(expiry, datetime):
            days_to_expiry = (expiry - datetime.now()).days
        else:
            days_to_expiry = 15  # Default assumption
        
        return PricingContext(
            symbol=contract_data.get('symbol', ''),
            underlying_price=contract_data.get('underlying_price', 0),
            bid=bid,
            ask=ask,
            volume=int(contract_data.get('volume', 0)),
            open_interest=int(contract_data.get('open_interest', 0)),
            spread_pct=spread_pct,
            time_to_expiry_days=days_to_expiry,
            volatility=float(contract_data.get('implied_volatility', 0.25)),
            delta=float(contract_data.get('delta', 0.5)),
            gamma=float(contract_data.get('gamma', 0.02)),
            theta=float(contract_data.get('theta', -0.05)),
            vega=float(contract_data.get('vega', 0.2))
        )
    
    def determine_optimal_entry_price(self, context: PricingContext, 
                                    side: OrderSide, 
                                    strategy_confidence: float = 0.5) -> SmartPriceTarget:
        """Determine optimal entry price based on market conditions and Greeks"""
        
        # Check liquidity first
        if context.volume < self.volume_threshold or context.open_interest < self.open_interest_threshold:
            return self._create_market_order(context, side, "Low liquidity - use market order")
        
        # Check spread width
        if context.spread_pct > self.max_spread_tolerance * 100:
            return self._create_market_order(context, side, f"Wide spread ({context.spread_pct:.1f}%) - use market order")
        
        # Determine pricing strategy based on context
        pricing_strategy = self._select_pricing_strategy(context, strategy_confidence)
        
        if pricing_strategy == PricingStrategy.AGGRESSIVE:
            return self._create_aggressive_price(context, side)
        elif pricing_strategy == PricingStrategy.SMART_MID:
            return self._create_smart_mid_price(context, side)
        elif pricing_strategy == PricingStrategy.SCALPER:
            return self._create_scalper_price(context, side)
        elif pricing_strategy == PricingStrategy.TIME_WEIGHTED:
            return self._create_time_weighted_price(context, side)
        elif pricing_strategy == PricingStrategy.GREEKS_BASED:
            return self._create_greeks_based_price(context, side, strategy_confidence)
        else:
            return self._create_smart_mid_price(context, side)  # Default
    
    def determine_optimal_exit_price(self, context: PricingContext,
                                   side: OrderSide,
                                   position_pnl_pct: float,
                                   time_held_hours: float) -> SmartPriceTarget:
        """Determine optimal exit price based on position performance and Greeks"""
        
        # Profit-taking logic
        if position_pnl_pct > 50:  # Large profit - be aggressive to lock it in
            return self._create_aggressive_price(context, side, "Large profit - lock in gains")
        
        # Stop-loss logic
        if position_pnl_pct < -30:  # Large loss - be aggressive to limit damage
            return self._create_aggressive_price(context, side, "Stop loss - limit damage")
        
        # Time decay considerations for options
        if context.time_to_expiry_days <= 3:  # Close to expiry
            return self._create_aggressive_price(context, side, "Close to expiry - avoid time decay")
        
        # Theta decay acceleration (last week)
        if context.time_to_expiry_days <= 7 and abs(context.theta) > self.high_theta_threshold:
            theta_urgency = min(1.0, abs(context.theta) / 0.20)  # Scale 0-1
            if theta_urgency > 0.7:
                return self._create_aggressive_price(context, side, "High theta decay - exit quickly")
        
        # Volatility-based exits
        if context.time_to_expiry_days > 7:
            # Use smart pricing for longer-dated options
            return self._create_greeks_based_price(context, side, 0.3)  # Lower confidence for exits
        
        # Default to smart mid-price
        return self._create_smart_mid_price(context, side)
    
    def _select_pricing_strategy(self, context: PricingContext, confidence: float) -> PricingStrategy:
        """Select optimal pricing strategy based on context"""
        
        # High confidence trades - be more aggressive
        if confidence > 0.8:
            return PricingStrategy.AGGRESSIVE
        
        # Close to expiry - time is critical
        if context.time_to_expiry_days <= self.close_to_expiry_days:
            return PricingStrategy.TIME_WEIGHTED
        
        # High Greeks sensitivity - use Greeks-based pricing
        if (abs(context.gamma) > self.high_gamma_threshold or 
            abs(context.theta) > self.high_theta_threshold or
            abs(context.vega) > self.high_vega_threshold):
            return PricingStrategy.GREEKS_BASED
        
        # Good liquidity and tight spread - try to scalp
        if (context.volume > 200 and context.spread_pct < 5.0 and
            context.time_to_expiry_days > self.far_expiry_days):
            return PricingStrategy.SCALPER
        
        # Default to smart mid-price
        return PricingStrategy.SMART_MID
    
    def _create_market_order(self, context: PricingContext, side: OrderSide, reasoning: str) -> SmartPriceTarget:
        """Create market order for immediate execution"""
        
        target_price = context.ask if side == OrderSide.BUY else context.bid
        
        return SmartPriceTarget(
            target_price=target_price,
            order_type='MARKET',
            limit_price=None,
            time_in_force='IOC',  # Immediate or Cancel
            expected_fill_probability=0.95,
            reasoning=reasoning,
            max_wait_seconds=10
        )
    
    def _create_aggressive_price(self, context: PricingContext, side: OrderSide, reasoning: str = "") -> SmartPriceTarget:
        """Create aggressive limit order just inside the spread"""
        
        spread = context.ask - context.bid
        aggressive_improvement = spread * 0.25  # Move 25% into the spread
        
        if side == OrderSide.BUY:
            target_price = context.ask - aggressive_improvement
            limit_price = min(target_price, context.ask)  # Don't exceed ask
        else:
            target_price = context.bid + aggressive_improvement
            limit_price = max(target_price, context.bid)  # Don't go below bid
        
        return SmartPriceTarget(
            target_price=target_price,
            order_type='LIMIT',
            limit_price=limit_price,
            time_in_force='DAY',
            expected_fill_probability=0.75,
            reasoning=f"Aggressive pricing: {reasoning}",
            max_wait_seconds=60
        )
    
    def _create_smart_mid_price(self, context: PricingContext, side: OrderSide) -> SmartPriceTarget:
        """Create smart limit order targeting mid-price with slight bias"""
        
        mid_price = (context.bid + context.ask) / 2
        spread = context.ask - context.bid
        
        # Slight bias toward favorable side
        bias_factor = 0.1  # 10% bias
        if side == OrderSide.BUY:
            target_price = mid_price + (spread * bias_factor)  # Slightly higher than mid
        else:
            target_price = mid_price - (spread * bias_factor)  # Slightly lower than mid
        
        # Ensure we don't cross the spread
        if side == OrderSide.BUY:
            limit_price = min(target_price, context.ask - 0.01)
        else:
            limit_price = max(target_price, context.bid + 0.01)
        
        return SmartPriceTarget(
            target_price=target_price,
            order_type='LIMIT',
            limit_price=limit_price,
            time_in_force='DAY',
            expected_fill_probability=0.60,
            reasoning="Smart mid-price targeting",
            max_wait_seconds=180
        )
    
    def _create_scalper_price(self, context: PricingContext, side: OrderSide) -> SmartPriceTarget:
        """Create patient limit order trying to get better than mid-price"""
        
        mid_price = (context.bid + context.ask) / 2
        spread = context.ask - context.bid
        
        # Try to get better than mid-price
        improvement_factor = 0.3  # Try to improve by 30% of spread
        if side == OrderSide.BUY:
            target_price = mid_price - (spread * improvement_factor)
            limit_price = max(target_price, context.bid + 0.01)  # Don't go below bid
        else:
            target_price = mid_price + (spread * improvement_factor)
            limit_price = min(target_price, context.ask - 0.01)  # Don't go above ask
        
        return SmartPriceTarget(
            target_price=target_price,
            order_type='LIMIT', 
            limit_price=limit_price,
            time_in_force='DAY',
            expected_fill_probability=0.40,  # Lower probability but better price
            reasoning="Scalping for better price",
            max_wait_seconds=300
        )
    
    def _create_time_weighted_price(self, context: PricingContext, side: OrderSide) -> SmartPriceTarget:
        """Create time-sensitive pricing based on expiration urgency"""
        
        urgency_factor = max(0.1, min(1.0, (7 - context.time_to_expiry_days) / 7))
        
        if urgency_factor > 0.8:  # Very urgent
            return self._create_aggressive_price(context, side, "Time urgency - close to expiry")
        elif urgency_factor > 0.5:  # Moderately urgent
            return self._create_smart_mid_price(context, side)
        else:  # Patient
            return self._create_scalper_price(context, side)
    
    def _create_greeks_based_price(self, context: PricingContext, side: OrderSide, confidence: float) -> SmartPriceTarget:
        """Create pricing based on Greeks sensitivity and edge calculation"""
        
        # Calculate Greeks-based edge
        edge_score = self._calculate_greeks_edge(context, confidence)
        
        mid_price = (context.bid + context.ask) / 2
        spread = context.ask - context.bid
        
        # Use edge score to determine aggressiveness
        if edge_score > 0.7:  # High edge - be more aggressive
            improvement = spread * 0.2  # Take 20% of spread
        elif edge_score > 0.4:  # Moderate edge
            improvement = spread * 0.35  # Take 35% of spread
        else:  # Low edge - be patient
            improvement = spread * 0.45  # Take 45% of spread
        
        if side == OrderSide.BUY:
            target_price = context.ask - improvement
            limit_price = max(target_price, context.bid + 0.01)
        else:
            target_price = context.bid + improvement
            limit_price = min(target_price, context.ask - 0.01)
        
        return SmartPriceTarget(
            target_price=target_price,
            order_type='LIMIT',
            limit_price=limit_price,
            time_in_force='DAY',
            expected_fill_probability=0.5 + (edge_score * 0.3),
            reasoning=f"Greeks-based pricing (edge: {edge_score:.2f})",
            max_wait_seconds=int(180 + (edge_score * 120))  # More time for higher edge
        )
    
    def _calculate_greeks_edge(self, context: PricingContext, confidence: float) -> float:
        """Calculate trading edge based on Greeks analysis"""
        
        edge_factors = []
        
        # Delta edge (directional exposure vs confidence)
        delta_edge = min(1.0, abs(context.delta) * confidence * 2)
        edge_factors.append(delta_edge)
        
        # Gamma edge (price acceleration potential)
        if context.gamma > self.high_gamma_threshold:
            gamma_edge = min(1.0, context.gamma / 0.10)  # Scale to 0-1
            edge_factors.append(gamma_edge)
        
        # Theta edge (time decay considerations)
        if context.time_to_expiry_days > 14:  # Only for longer-dated
            theta_edge = 1.0 - min(1.0, abs(context.theta) / 0.15)
            edge_factors.append(theta_edge)
        
        # Vega edge (volatility expansion potential)
        if context.volatility < 0.30:  # Lower IV = potential expansion
            vega_edge = 1.0 - (context.volatility / 0.30)
            edge_factors.append(vega_edge)
        
        # Time to expiry edge
        if 7 <= context.time_to_expiry_days <= 45:  # Sweet spot
            time_edge = 1.0 - abs(context.time_to_expiry_days - 25) / 25
            edge_factors.append(time_edge)
        
        # Combine edges (average with confidence weighting)
        if edge_factors:
            raw_edge = sum(edge_factors) / len(edge_factors)
            return min(1.0, raw_edge * confidence)
        else:
            return confidence * 0.5  # Default conservative edge
    
    def create_dynamic_stop_loss(self, context: PricingContext, 
                                entry_price: float, 
                                position_side: OrderSide,
                                base_stop_pct: float = 0.20) -> Tuple[float, str]:
        """Create dynamic stop loss based on Greeks and volatility"""
        
        # Adjust stop based on volatility
        vol_adjustment = context.volatility / 0.25  # Base 25% vol
        vol_adjusted_stop = base_stop_pct * vol_adjustment
        
        # Adjust based on time decay risk
        if context.time_to_expiry_days <= 7:
            theta_adjustment = min(2.0, abs(context.theta) / 0.10)
            vol_adjusted_stop *= theta_adjustment
        
        # Adjust based on gamma risk
        if context.gamma > self.high_gamma_threshold:
            gamma_adjustment = 1.0 + (context.gamma / 0.10)
            vol_adjusted_stop *= gamma_adjustment
        
        # Calculate stop price
        final_stop_pct = min(0.50, max(0.10, vol_adjusted_stop))  # 10-50% range
        
        if position_side == OrderSide.BUY:
            stop_price = entry_price * (1 - final_stop_pct)
        else:
            stop_price = entry_price * (1 + final_stop_pct)
        
        reasoning = f"Dynamic stop: {final_stop_pct*100:.1f}% (vol={context.volatility:.0%}, theta={context.theta:.3f}, gamma={context.gamma:.3f})"
        
        return stop_price, reasoning
    
    def create_dynamic_take_profit(self, context: PricingContext,
                                 entry_price: float,
                                 position_side: OrderSide,
                                 base_target_pct: float = 0.50) -> Tuple[float, str]:
        """Create dynamic take profit based on Greeks and time"""
        
        # Adjust target based on time to expiry
        if context.time_to_expiry_days <= 7:
            # Aggressive profit taking near expiry
            time_adjustment = 0.5
        elif context.time_to_expiry_days >= 30:
            # Patient profit taking for longer-dated
            time_adjustment = 1.5
        else:
            time_adjustment = 1.0
        
        # Adjust based on delta (higher delta = more aggressive targets)
        delta_adjustment = 1.0 + abs(context.delta)
        
        # Adjust based on volatility (higher vol = higher targets)
        vol_adjustment = context.volatility / 0.25
        
        # Calculate target price
        adjusted_target_pct = base_target_pct * time_adjustment * delta_adjustment * vol_adjustment
        final_target_pct = min(2.0, max(0.20, adjusted_target_pct))  # 20-200% range
        
        if position_side == OrderSide.BUY:
            target_price = entry_price * (1 + final_target_pct)
        else:
            target_price = entry_price * (1 - final_target_pct)
        
        reasoning = f"Dynamic target: {final_target_pct*100:.1f}% (time={context.time_to_expiry_days}d, delta={context.delta:.2f}, vol={context.volatility:.0%})"
        
        return target_price, reasoning

# Global instance
smart_pricing_agent = SmartPricingAgent()