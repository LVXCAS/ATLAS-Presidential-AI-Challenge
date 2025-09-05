"""
Exit Strategy Agent - Intelligent Profit/Loss Decision Making
Uses machine learning-inspired logic to determine optimal exit points
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import yfinance as yf
from dataclasses import dataclass

from config.logging_config import get_logger

logger = get_logger(__name__)

class ExitSignal(str, Enum):
    """Exit signal types"""
    HOLD = "hold"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    REDUCE_POSITION = "reduce_position"
    ROLL_POSITION = "roll_position"

class ExitReason(str, Enum):
    """Detailed exit reasons"""
    PROFIT_TARGET = "profit_target_hit"
    STOP_LOSS_HIT = "stop_loss_hit"
    TIME_DECAY = "time_decay_risk"
    VOLATILITY_CRUSH = "volatility_crush"
    MOMENTUM_REVERSAL = "momentum_reversal"
    MARKET_REGIME_CHANGE = "market_regime_change"
    TECHNICAL_BREAKDOWN = "technical_breakdown"
    RISK_MANAGEMENT = "risk_management"
    DELTA_THRESHOLD = "delta_threshold"
    THETA_BURN = "theta_burn_excessive"

@dataclass
class ExitDecision:
    """Exit decision with reasoning"""
    signal: ExitSignal
    reason: ExitReason
    confidence: float  # 0-1
    urgency: float  # 0-1, how urgent the exit is
    target_exit_pct: float  # What % of position to exit
    expected_pnl_impact: float  # Expected P&L impact of decision
    reasoning: str  # Human-readable explanation
    supporting_factors: List[str]  # Factors supporting this decision

class ExitStrategyAgent:
    """Intelligent agent for making profit/loss decisions"""
    
    def __init__(self):
        # Learning parameters (would be ML model weights in production)
        self.profit_taking_sensitivity = 0.7  # Higher = take profits earlier
        self.loss_cutting_aggressiveness = 0.8  # Higher = cut losses faster
        self.time_decay_awareness = 0.9  # How much to weight time decay
        self.volatility_sensitivity = 0.6  # Reaction to vol changes
        
        # Market regime memory
        self.recent_market_performance = []
        self.volatility_history = []
        
        # Exit thresholds (adaptive)
        self.profit_thresholds = {
            'conservative': 0.25,  # 25% of max profit
            'moderate': 0.50,      # 50% of max profit  
            'aggressive': 0.75     # 75% of max profit
        }
        
        self.loss_thresholds = {
            'tight': 0.15,      # 15% of max loss
            'moderate': 0.25,   # 25% of max loss
            'loose': 0.40       # 40% of max loss
        }
    
    def analyze_position_exit(self, position_data: Dict, market_data: Dict, 
                            current_pnl: float) -> ExitDecision:
        """
        Main decision engine - analyzes all factors to make exit decision
        """
        try:
            # Extract position information
            entry_time = position_data['entry_time']
            strategy = position_data['opportunity']['strategy']
            max_profit = position_data['opportunity']['max_profit']
            max_loss = position_data['opportunity']['max_loss']
            days_to_expiry = self._get_days_to_expiry(position_data)
            
            # Calculate metrics
            profit_pct = current_pnl / (max_profit * 100) if max_profit > 0 else 0
            loss_pct = abs(current_pnl) / (max_loss * 100) if current_pnl < 0 and max_loss > 0 else 0
            days_in_trade = (datetime.now() - entry_time).days
            
            # Multi-factor analysis
            factors = self._analyze_exit_factors(
                position_data, market_data, current_pnl, 
                profit_pct, loss_pct, days_in_trade, days_to_expiry
            )
            
            # Weighted decision making
            decision = self._make_weighted_decision(factors, position_data, current_pnl)
            
            symbol = position_data.get('opportunity', {}).get('symbol', 'UNKNOWN')
            logger.info(f"Exit analysis for {symbol}: {decision.signal} ({decision.confidence:.1%} confidence)")
            
            return decision
            
        except Exception as e:
            logger.error(f"Exit analysis error: {e}")
            # Default conservative decision
            return ExitDecision(
                signal=ExitSignal.HOLD,
                reason=ExitReason.RISK_MANAGEMENT,
                confidence=0.5,
                urgency=0.0,
                target_exit_pct=0.0,
                expected_pnl_impact=0.0,
                reasoning="Error in analysis - holding position",
                supporting_factors=["analysis_error"]
            )
    
    def _analyze_exit_factors(self, position_data: Dict, market_data: Dict, 
                            current_pnl: float, profit_pct: float, loss_pct: float,
                            days_in_trade: int, days_to_expiry: int) -> Dict[str, float]:
        """Analyze all factors that influence exit decisions"""
        
        factors = {}
        
        # 1. Profit/Loss Factor
        if current_pnl > 0:
            # Profit scenario - diminishing returns curve
            if profit_pct >= 0.75:
                factors['profit_pressure'] = 0.9  # Strong take profit signal
            elif profit_pct >= 0.50:
                factors['profit_pressure'] = 0.6  # Moderate take profit
            elif profit_pct >= 0.25:
                factors['profit_pressure'] = 0.3  # Weak take profit
            else:
                factors['profit_pressure'] = 0.0
        else:
            # Loss scenario - accelerating loss cutting
            if loss_pct >= 0.50:
                factors['loss_pressure'] = 0.95  # Emergency exit
            elif loss_pct >= 0.30:
                factors['loss_pressure'] = 0.7   # Strong stop loss
            elif loss_pct >= 0.15:
                factors['loss_pressure'] = 0.4   # Moderate concern
            else:
                factors['loss_pressure'] = 0.0
        
        # 2. Time Decay Factor (especially critical for options)
        time_decay_pressure = self._calculate_time_decay_pressure(days_in_trade, days_to_expiry)
        factors['time_decay'] = time_decay_pressure
        
        # 3. Volatility Factor
        vol_factor = self._analyze_volatility_impact(market_data, position_data)
        factors['volatility_impact'] = vol_factor
        
        # 4. Momentum Factor
        momentum_factor = self._analyze_momentum_change(market_data, position_data)
        factors['momentum_change'] = momentum_factor
        
        # 5. Market Regime Factor
        regime_factor = self._analyze_market_regime_change(market_data, position_data)
        factors['regime_change'] = regime_factor
        
        # 6. Greeks Factor (for options)
        greeks_factor = self._analyze_greeks_deterioration(position_data)
        factors['greeks_deterioration'] = greeks_factor
        
        # 7. Technical Factor
        technical_factor = self._analyze_technical_levels(market_data, position_data)
        factors['technical_signals'] = technical_factor
        
        return factors
    
    def _calculate_time_decay_pressure(self, days_in_trade: int, days_to_expiry: int) -> float:
        """Calculate pressure to exit due to time decay"""
        if days_to_expiry <= 0:
            return 1.0  # Maximum pressure - expiration day
        
        # Exponential pressure as expiration approaches
        if days_to_expiry <= 7:
            return 0.9  # Final week - very high pressure
        elif days_to_expiry <= 14:
            return 0.6  # Two weeks - high pressure
        elif days_to_expiry <= 30:
            return 0.3  # One month - moderate pressure
        else:
            return 0.1  # More than month - low pressure
    
    def _analyze_volatility_impact(self, market_data: Dict, position_data: Dict) -> float:
        """Analyze if volatility changes favor exit"""
        try:
            current_vol = market_data.get('realized_vol', 25.0)
            
            # For options strategies, vol crush is a major factor
            strategy = position_data['opportunity']['strategy']
            
            if 'CALL' in str(strategy) or 'PUT' in str(strategy):
                # Long options hurt by vol crush
                if current_vol < 15:  # Low vol environment
                    return 0.7  # Pressure to exit
                elif current_vol > 35:  # High vol environment
                    return -0.2  # Incentive to hold
            else:
                # Spreads less affected by vol, but still factor
                if current_vol < 12:  # Very low vol
                    return 0.4
                elif current_vol > 40:  # Very high vol
                    return 0.3
            
            return 0.0
            
        except:
            return 0.0
    
    def _analyze_momentum_change(self, market_data: Dict, position_data: Dict) -> float:
        """Analyze if momentum supports or opposes position"""
        try:
            momentum = market_data.get('price_momentum', 0.0)
            strategy = position_data['opportunity']['strategy']
            
            # Check if momentum aligns with strategy
            is_bullish_strategy = 'BULL' in str(strategy) or 'CALL' in str(strategy)
            is_bearish_strategy = 'BEAR' in str(strategy) or 'PUT' in str(strategy)
            
            if is_bullish_strategy:
                if momentum < -0.03:  # Strong negative momentum
                    return 0.8  # Exit pressure
                elif momentum < -0.01:  # Weak negative momentum
                    return 0.4
            elif is_bearish_strategy:
                if momentum > 0.03:  # Strong positive momentum
                    return 0.8  # Exit pressure
                elif momentum > 0.01:  # Weak positive momentum
                    return 0.4
            
            return 0.0
            
        except:
            return 0.0
    
    def _analyze_market_regime_change(self, market_data: Dict, position_data: Dict) -> float:
        """Analyze if market regime changed since entry"""
        try:
            entry_regime = position_data.get('market_regime_at_entry', 'NEUTRAL')
            # Would compare with current regime in full implementation
            # For now, assume no major regime change
            return 0.0
        except:
            return 0.0
    
    def _analyze_greeks_deterioration(self, position_data: Dict) -> float:
        """Analyze Greeks deterioration (simplified)"""
        try:
            # In full implementation, would calculate current Greeks vs entry Greeks
            # For now, use time-based approximation
            days_in_trade = (datetime.now() - position_data['entry_time']).days
            
            if days_in_trade > 14:
                return 0.3  # Some deterioration expected
            elif days_in_trade > 21:
                return 0.5  # Notable deterioration
            
            return 0.0
        except:
            return 0.0
    
    def _analyze_technical_levels(self, market_data: Dict, position_data: Dict) -> float:
        """Analyze technical support/resistance levels"""
        try:
            price_position = market_data.get('price_position', 0.5)
            strategy = position_data['opportunity']['strategy']
            
            # Check if price at critical technical levels
            is_bullish_strategy = 'BULL' in str(strategy) or 'CALL' in str(strategy)
            is_bearish_strategy = 'BEAR' in str(strategy) or 'PUT' in str(strategy)
            
            if is_bullish_strategy and price_position < 0.2:
                return 0.6  # Near support, pressure to exit bullish position
            elif is_bearish_strategy and price_position > 0.8:
                return 0.6  # Near resistance, pressure to exit bearish position
            
            return 0.0
        except:
            return 0.0
    
    def _make_weighted_decision(self, factors: Dict[str, float], position_data: Dict, 
                              current_pnl: float) -> ExitDecision:
        """Make final weighted decision based on all factors"""
        
        # Factor weights (tunable parameters)
        weights = {
            'profit_pressure': 0.25,
            'loss_pressure': 0.30,
            'time_decay': 0.15,
            'volatility_impact': 0.10,
            'momentum_change': 0.10,
            'regime_change': 0.05,
            'greeks_deterioration': 0.03,
            'technical_signals': 0.02
        }
        
        # Calculate weighted score
        exit_score = 0.0
        supporting_factors = []
        
        for factor, value in factors.items():
            if factor in weights and value > 0.3:  # Only count significant factors
                weighted_value = value * weights[factor]
                exit_score += weighted_value
                supporting_factors.append(f"{factor}: {value:.1%}")
        
        # Determine decision based on score
        if exit_score >= 0.7:
            if current_pnl > 0:
                return ExitDecision(
                    signal=ExitSignal.TAKE_PROFIT,
                    reason=ExitReason.PROFIT_TARGET,
                    confidence=min(0.95, exit_score),
                    urgency=0.8,
                    target_exit_pct=100.0,
                    expected_pnl_impact=current_pnl * 0.95,  # Lock in most gains
                    reasoning=f"Strong profit-taking signal (score: {exit_score:.1%})",
                    supporting_factors=supporting_factors
                )
            else:
                return ExitDecision(
                    signal=ExitSignal.STOP_LOSS,
                    reason=ExitReason.STOP_LOSS_HIT,
                    confidence=min(0.95, exit_score),
                    urgency=0.9,
                    target_exit_pct=100.0,
                    expected_pnl_impact=current_pnl * 1.1,  # Prevent further loss
                    reasoning=f"Strong stop-loss signal (score: {exit_score:.1%})",
                    supporting_factors=supporting_factors
                )
        
        elif exit_score >= 0.5:
            return ExitDecision(
                signal=ExitSignal.REDUCE_POSITION,
                reason=ExitReason.RISK_MANAGEMENT,
                confidence=exit_score,
                urgency=0.5,
                target_exit_pct=50.0,
                expected_pnl_impact=current_pnl * 0.5,
                reasoning=f"Moderate exit signal - reducing position (score: {exit_score:.1%})",
                supporting_factors=supporting_factors
            )
        
        elif exit_score >= 0.3:
            # Elevated monitoring mode
            return ExitDecision(
                signal=ExitSignal.HOLD,
                reason=ExitReason.RISK_MANAGEMENT,
                confidence=1.0 - exit_score,
                urgency=0.2,
                target_exit_pct=0.0,
                expected_pnl_impact=0.0,
                reasoning=f"Elevated monitoring - watching for changes (score: {exit_score:.1%})",
                supporting_factors=supporting_factors
            )
        
        else:
            # Normal hold
            return ExitDecision(
                signal=ExitSignal.HOLD,
                reason=ExitReason.RISK_MANAGEMENT,
                confidence=0.8,
                urgency=0.0,
                target_exit_pct=0.0,
                expected_pnl_impact=0.0,
                reasoning="All factors support holding position",
                supporting_factors=["low_exit_pressure"]
            )
    
    def _get_days_to_expiry(self, position_data: Dict) -> int:
        """Get days to expiration for the position"""
        try:
            opportunity = position_data['opportunity']
            if 'days_to_expiry' in opportunity:
                entry_time = position_data['entry_time']
                days_since_entry = (datetime.now() - entry_time).days
                return max(0, opportunity['days_to_expiry'] - days_since_entry)
            return 30  # Default assumption
        except:
            return 30
    
    def update_learning_parameters(self, exit_results: List[Dict]):
        """Update agent parameters based on exit results (simplified ML)"""
        try:
            if len(exit_results) < 5:
                return  # Need minimum data
            
            # Analyze recent exit performance
            profitable_exits = [r for r in exit_results if r['final_pnl'] > 0]
            unprofitable_exits = [r for r in exit_results if r['final_pnl'] <= 0]
            
            profit_rate = len(profitable_exits) / len(exit_results)
            
            # Adjust parameters based on performance
            if profit_rate < 0.4:  # Poor performance
                self.loss_cutting_aggressiveness = min(0.9, self.loss_cutting_aggressiveness + 0.05)
                self.profit_taking_sensitivity = max(0.5, self.profit_taking_sensitivity - 0.05)
            elif profit_rate > 0.7:  # Good performance
                self.profit_taking_sensitivity = min(0.8, self.profit_taking_sensitivity + 0.02)
                self.loss_cutting_aggressiveness = max(0.6, self.loss_cutting_aggressiveness - 0.02)
            
            logger.info(f"Updated learning parameters: profit_sens={self.profit_taking_sensitivity:.2f}, loss_aggr={self.loss_cutting_aggressiveness:.2f}")
            
        except Exception as e:
            logger.error(f"Parameter update error: {e}")

# Singleton instance
exit_strategy_agent = ExitStrategyAgent()