"""
AUTONOMOUS DECISION FRAMEWORK
============================

Core autonomous decision-making module for the Hive Trading System.
Implements the central decision logic that coordinates all trading agents
and manages autonomous strategy execution.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"

class DecisionConfidence(Enum):
    """Decision confidence levels"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

class TradingAction(Enum):
    """Available trading actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_POSITION = "close_position"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"

@dataclass
class TradingDecision:
    """Represents a trading decision from the autonomous framework"""
    symbol: str
    action: TradingAction
    confidence: DecisionConfidence
    position_size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AutonomousDecisionFramework:
    """
    Core autonomous decision-making engine for the Hive Trading System.

    This framework coordinates all trading agents and makes final trading decisions
    based on multiple inputs including market data, sentiment, and risk parameters.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.market_regime = MarketRegime.SIDEWAYS
        self.active_positions = {}
        self.decision_history = []
        self.performance_metrics = {}

        # Initialize components
        self._initialize_risk_parameters()
        self._initialize_market_analyzers()

        logger.info("Autonomous Decision Framework initialized")

    def _default_config(self) -> Dict:
        """Default configuration for the decision framework"""
        return {
            "max_position_size": 0.10,  # 10% max position size
            "stop_loss_percentage": 0.08,  # 8% stop loss
            "take_profit_percentage": 0.20,  # 20% take profit
            "max_open_positions": 5,
            "risk_tolerance": "moderate",
            "enable_paper_trading": True,
            "decision_threshold": 0.6,  # Minimum confidence for action
        }

    def _initialize_risk_parameters(self):
        """Initialize risk management parameters"""
        self.risk_params = {
            "max_daily_loss": 0.02,  # 2% max daily loss
            "max_portfolio_risk": 0.15,  # 15% max portfolio risk
            "correlation_limit": 0.7,  # Max correlation between positions
            "volatility_threshold": 0.30,  # High volatility threshold
        }

    def _initialize_market_analyzers(self):
        """Initialize market analysis components"""
        self.analyzers = {
            "technical": self._technical_analyzer,
            "sentiment": self._sentiment_analyzer,
            "momentum": self._momentum_analyzer,
            "risk": self._risk_analyzer,
        }

    async def make_decision(self, market_data: Dict, agent_signals: List[Dict]) -> Optional[TradingDecision]:
        """
        Core decision-making method that processes all inputs and returns a trading decision.
        """
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')

            # Step 1: Analyze current market regime
            self.market_regime = await self._detect_market_regime(market_data)

            # Step 2: Process agent signals
            consensus_signal = await self._process_agent_signals(agent_signals)

            # Step 3: Apply risk management filters
            risk_adjusted_signal = await self._apply_risk_filters(consensus_signal, market_data)

            # Step 4: Make final decision
            decision = await self._generate_final_decision(risk_adjusted_signal, market_data)

            # Step 5: Log and store decision
            if decision:
                self.decision_history.append(decision)
                logger.info(f"Decision made for {symbol}: {decision.action.value} with {decision.confidence.value} confidence")

            return decision

        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            return None

    async def _detect_market_regime(self, market_data: Dict) -> MarketRegime:
        """Detect current market regime based on market data"""
        try:
            volatility = market_data.get('volatility', 0.2)
            price_change = market_data.get('price_change_24h', 0)

            if volatility > self.risk_params['volatility_threshold']:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.15:
                return MarketRegime.LOW_VOLATILITY
            elif price_change > 0.05:
                return MarketRegime.BULL
            elif price_change < -0.05:
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS

    async def _process_agent_signals(self, agent_signals: List[Dict]) -> Dict:
        """Process and combine signals from multiple trading agents"""
        if not agent_signals:
            return {"action": TradingAction.HOLD, "confidence": DecisionConfidence.VERY_LOW}

        # Weight and combine signals
        signal_weights = {
            "momentum_agent": 0.3,
            "mean_reversion_agent": 0.2,
            "sentiment_agent": 0.2,
            "risk_agent": 0.3,
        }

        total_score = 0
        total_weight = 0
        action_votes = {}

        for signal in agent_signals:
            agent_name = signal.get('agent', 'unknown')
            weight = signal_weights.get(agent_name, 0.1)
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0.5)

            score = confidence * weight
            total_score += score
            total_weight += weight

            if action not in action_votes:
                action_votes[action] = 0
            action_votes[action] += weight

        # Determine consensus action
        consensus_action = max(action_votes, key=action_votes.get) if action_votes else 'hold'
        consensus_confidence = total_score / total_weight if total_weight > 0 else 0.5

        return {
            "action": TradingAction(consensus_action),
            "confidence": self._score_to_confidence(consensus_confidence),
            "supporting_agents": len(agent_signals)
        }

    async def _apply_risk_filters(self, signal: Dict, market_data: Dict) -> Dict:
        """Apply risk management filters to the trading signal"""
        # Check position limits
        if len(self.active_positions) >= self.config['max_open_positions']:
            if signal['action'] in [TradingAction.BUY]:
                signal['action'] = TradingAction.HOLD
                signal['confidence'] = DecisionConfidence.VERY_LOW

        return signal

    async def _generate_final_decision(self, signal: Dict, market_data: Dict) -> Optional[TradingDecision]:
        """Generate final trading decision based on processed signal"""
        action = signal.get('action', TradingAction.HOLD)
        confidence = signal.get('confidence', DecisionConfidence.MEDIUM)

        # Only act if confidence is above threshold
        if confidence.value < self.config['decision_threshold']:
            return None

        if action == TradingAction.HOLD:
            return None

        symbol = market_data.get('symbol', 'UNKNOWN')
        current_price = market_data.get('price', 0)
        position_size = self._calculate_position_size(confidence, market_data)

        return TradingDecision(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            entry_price=current_price,
            reasoning=f"Market regime: {self.market_regime.value}"
        )

    def _calculate_position_size(self, confidence: DecisionConfidence, market_data: Dict) -> float:
        """Calculate appropriate position size based on confidence and risk"""
        base_size = self.config['max_position_size']
        confidence_multiplier = confidence.value
        return base_size * confidence_multiplier

    def _score_to_confidence(self, score: float) -> DecisionConfidence:
        """Convert numerical score to confidence level"""
        if score >= 0.9:
            return DecisionConfidence.VERY_HIGH
        elif score >= 0.8:
            return DecisionConfidence.HIGH
        elif score >= 0.6:
            return DecisionConfidence.MEDIUM
        elif score >= 0.4:
            return DecisionConfidence.LOW
        else:
            return DecisionConfidence.VERY_LOW

    def _technical_analyzer(self, data: Dict) -> Dict:
        return {"score": 0.6, "signal": "neutral"}

    def _sentiment_analyzer(self, data: Dict) -> Dict:
        return {"score": 0.5, "signal": "neutral"}

    def _momentum_analyzer(self, data: Dict) -> Dict:
        return {"score": 0.7, "signal": "positive"}

    def _risk_analyzer(self, data: Dict) -> Dict:
        return {"score": 0.8, "signal": "low_risk"}

    def get_status(self) -> Dict:
        """Get current framework status"""
        return {
            "market_regime": self.market_regime.value,
            "active_positions": len(self.active_positions),
            "decisions_made": len(self.decision_history),
            "framework_status": "operational"
        }

    async def make_autonomous_decision(self, context: 'DecisionContext') -> 'AutonomousDecision':
        """Legacy method for backward compatibility"""
        # Convert DecisionContext to our internal format
        market_data = context.market_data
        agent_signals = context.agent_signals

        decision = await self.make_decision(market_data, agent_signals)

        if decision:
            return AutonomousDecision(
                action=decision.action.value,
                confidence=decision.confidence.value,
                reasoning=decision.reasoning,
                position_size=decision.position_size,
                success=True
            )
        else:
            return AutonomousDecision(
                action="hold",
                confidence=0.0,
                reasoning="No decision made - insufficient confidence or market data",
                success=False
            )

# Legacy aliases for backward compatibility
AutonomousDecisionEngine = AutonomousDecisionFramework

class DecisionType(Enum):
    """Decision types for compatibility"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRATEGY_SELECTION = "strategy_selection"
    POSITION_SIZING = "position_sizing"
    RISK_MANAGEMENT = "risk_management"

@dataclass
class AutonomousDecision:
    """Simple decision object for backward compatibility"""
    action: str
    confidence: float
    reasoning: str
    position_size: float = 0.0
    success: bool = True

class DecisionContext:
    """Decision context class for compatibility"""
    def __init__(self, market_data: Dict = None, agent_signals: List = None,
                 decision_type: DecisionType = None, historical_performance: Dict = None,
                 current_positions: Dict = None, risk_metrics: Dict = None, **kwargs):
        self.market_data = market_data or {}
        self.agent_signals = agent_signals or []
        self.decision_type = decision_type
        self.historical_performance = historical_performance or {}
        self.current_positions = current_positions or {}
        self.risk_metrics = risk_metrics or {}
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_autonomous_framework(config: Optional[Dict] = None) -> AutonomousDecisionFramework:
    """Create and return an autonomous decision framework instance"""
    return AutonomousDecisionFramework(config)