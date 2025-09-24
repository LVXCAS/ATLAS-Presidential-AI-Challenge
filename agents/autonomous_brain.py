import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, TypedDict, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pickle
from pathlib import Path
import uuid
import math
from collections import defaultdict, deque

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available. Install with: pip install langgraph")

from event_bus import TradingEventBus, Event, Priority
from data.market_scanner import MarketScanner, TradingOpportunity, OpportunityType, ScanFilter
from core.portfolio import Portfolio, Position


class TradingAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_POSITION = "close_position"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9


class TradingState(TypedDict):
    """State shared across all nodes in the trading graph"""
    # Market data and opportunities
    market_scan_results: Dict[str, List[TradingOpportunity]]
    current_opportunities: List[TradingOpportunity]
    market_conditions: Dict[str, Any]
    
    # Analysis results
    technical_analysis: Dict[str, Dict[str, float]]
    fundamental_analysis: Dict[str, Dict[str, Any]]
    risk_analysis: Dict[str, float]
    sentiment_analysis: Dict[str, float]
    
    # Decision making
    trading_signals: List[Dict[str, Any]]
    position_recommendations: List[Dict[str, Any]]
    risk_score: float
    
    # Execution
    pending_orders: List[Dict[str, Any]]
    executed_trades: List[Dict[str, Any]]
    execution_errors: List[str]
    
    # Learning and feedback
    performance_metrics: Dict[str, float]
    learning_feedback: Dict[str, Any]
    model_updates: List[str]
    
    # System state
    iteration_count: int
    last_update: str
    node_history: List[str]
    confidence_score: float
    portfolio_snapshot: Dict[str, Any]


@dataclass
class TradingDecision:
    symbol: str
    action: TradingAction
    quantity: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float
    time_horizon: str
    metadata: Dict[str, Any]


@dataclass
class LearningRecord:
    decision_id: str
    symbol: str
    action: TradingAction
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    predicted_outcome: float
    actual_outcome: Optional[float]
    confidence: float
    timestamp: datetime
    market_conditions: Dict[str, Any]
    technical_indicators: Dict[str, float]
    success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'symbol': self.symbol,
            'action': self.action.value,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'predicted_outcome': self.predicted_outcome,
            'actual_outcome': self.actual_outcome,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'market_conditions': self.market_conditions,
            'technical_indicators': self.technical_indicators,
            'success': self.success
        }


class MarketAnalyzer:
    """Advanced market analysis with multiple techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MarketAnalyzer")
    
    async def analyze_technical_indicators(self, opportunities: List[TradingOpportunity]) -> Dict[str, Dict[str, float]]:
        """Perform technical analysis on opportunities"""
        analysis_results = {}
        
        for opp in opportunities:
            symbol = opp.symbol
            technical = opp.technical_indicators
            
            # Enhanced technical scoring
            rsi_score = self._calculate_rsi_score(technical.get('rsi', 50))
            momentum_score = self._calculate_momentum_score(opp.change_percent)
            volume_score = min(1.0, opp.volume / 1_000_000)  # Volume in millions
            
            # Bollinger Band analysis
            bb_score = 0.5
            if technical.get('bb_upper') and technical.get('bb_lower'):
                bb_position = (opp.price - technical['bb_lower']) / (technical['bb_upper'] - technical['bb_lower'])
                bb_score = 1.0 - abs(0.5 - bb_position)  # Higher score for middle band
            
            # Moving average convergence
            ma_score = 0.5
            if technical.get('sma_20') and technical.get('sma_50'):
                if technical['sma_20'] > technical['sma_50']:
                    ma_score = 0.8
                else:
                    ma_score = 0.2
            
            analysis_results[symbol] = {
                'rsi_score': rsi_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'bollinger_score': bb_score,
                'moving_average_score': ma_score,
                'composite_technical_score': (rsi_score + momentum_score + volume_score + bb_score + ma_score) / 5,
                'volatility_score': min(1.0, abs(opp.change_percent) / 10)
            }
        
        return analysis_results
    
    def _calculate_rsi_score(self, rsi: float) -> float:
        """Convert RSI to actionable score"""
        if rsi < 30:  # Oversold - potential buy
            return 0.8
        elif rsi > 70:  # Overbought - potential sell
            return 0.2
        else:  # Neutral
            return 0.5
    
    def _calculate_momentum_score(self, change_percent: float) -> float:
        """Convert price change to momentum score"""
        return min(1.0, max(0.0, (change_percent + 10) / 20))
    
    async def analyze_market_conditions(self, state: TradingState) -> Dict[str, Any]:
        """Analyze overall market conditions"""
        opportunities = state.get('current_opportunities', [])
        
        if not opportunities:
            return {
                'market_trend': 'neutral',
                'volatility_level': 'medium',
                'opportunity_count': 0,
                'average_confidence': 0.5
            }
        
        # Calculate market metrics
        avg_change = np.mean([opp.change_percent for opp in opportunities])
        volatility = np.std([opp.change_percent for opp in opportunities])
        avg_confidence = np.mean([opp.confidence for opp in opportunities])
        
        market_trend = 'bullish' if avg_change > 1 else 'bearish' if avg_change < -1 else 'neutral'
        volatility_level = 'high' if volatility > 5 else 'low' if volatility < 2 else 'medium'
        
        return {
            'market_trend': market_trend,
            'volatility_level': volatility_level,
            'average_change_percent': avg_change,
            'volatility': volatility,
            'opportunity_count': len(opportunities),
            'average_confidence': avg_confidence,
            'momentum_stocks': len([opp for opp in opportunities if opp.change_percent > 3]),
            'reversal_candidates': len([opp for opp in opportunities if abs(opp.change_percent) > 5])
        }


class RiskManager:
    """Sophisticated risk management system"""
    
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_risk: float = 0.01):
        self.logger = logging.getLogger(f"{__name__}.RiskManager")
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
    
    async def calculate_risk_metrics(self, state: TradingState) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        portfolio_snapshot = state.get('portfolio_snapshot', {})
        opportunities = state.get('current_opportunities', [])
        
        portfolio_value = portfolio_snapshot.get('portfolio_value', 100000)
        
        # Portfolio risk metrics
        var_95 = portfolio_snapshot.get('risk_metrics', {}).get('var_95', 0)
        max_drawdown = portfolio_snapshot.get('risk_metrics', {}).get('max_drawdown', 0)
        volatility = portfolio_snapshot.get('risk_metrics', {}).get('volatility', 0)
        leverage = portfolio_snapshot.get('risk_metrics', {}).get('leverage_ratio', 1)
        
        # Opportunity risk assessment
        opportunity_risk = 0
        if opportunities:
            risk_scores = []
            for opp in opportunities:
                position_risk = abs(opp.price - opp.stop_loss) / opp.price if opp.stop_loss else 0.05
                risk_scores.append(position_risk)
            opportunity_risk = np.mean(risk_scores)
        
        # Overall risk score
        risk_factors = [
            min(1.0, abs(var_95) / (portfolio_value * 0.05)),  # VaR risk
            min(1.0, max_drawdown * 2),  # Drawdown risk
            min(1.0, volatility),  # Volatility risk
            min(1.0, (leverage - 1) * 0.5),  # Leverage risk
            min(1.0, opportunity_risk * 10)  # Current opportunity risk
        ]
        
        overall_risk_score = np.mean(risk_factors)
        
        return {
            'portfolio_var_risk': risk_factors[0],
            'drawdown_risk': risk_factors[1],
            'volatility_risk': risk_factors[2],
            'leverage_risk': risk_factors[3],
            'opportunity_risk': risk_factors[4],
            'overall_risk_score': overall_risk_score,
            'risk_capacity': max(0, 1 - overall_risk_score),
            'max_position_size': self._calculate_max_position_size(portfolio_value, overall_risk_score)
        }
    
    def _calculate_max_position_size(self, portfolio_value: float, risk_score: float) -> float:
        """Calculate maximum position size based on risk"""
        base_position_size = portfolio_value * self.max_position_risk
        risk_adjustment = 1 - risk_score
        return base_position_size * max(0.1, risk_adjustment)


class DecisionEngine:
    """Advanced decision making with multiple algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DecisionEngine")
    
    async def generate_trading_decisions(self, state: TradingState) -> List[TradingDecision]:
        """Generate trading decisions based on analysis"""
        opportunities = state.get('current_opportunities', [])
        technical_analysis = state.get('technical_analysis', {})
        risk_analysis = state.get('risk_analysis', {})
        market_conditions = state.get('market_conditions', {})
        
        decisions = []
        max_position_size = risk_analysis.get('max_position_size', 1000)
        risk_capacity = risk_analysis.get('risk_capacity', 0.5)
        
        for opp in opportunities:
            if opp.confidence < 0.6:  # Skip low confidence opportunities
                continue
            
            technical_score = technical_analysis.get(opp.symbol, {}).get('composite_technical_score', 0.5)
            
            # Decision logic
            action = self._determine_action(opp, technical_score, market_conditions)
            
            if action == TradingAction.HOLD:
                continue
            
            # Position sizing
            confidence_factor = (opp.confidence + technical_score) / 2
            risk_factor = 1 - opp.risk_reward_ratio if opp.risk_reward_ratio > 0 else 0.5
            
            position_size = min(
                max_position_size * confidence_factor * risk_capacity,
                max_position_size * 0.5  # Never more than 50% of max
            )
            
            quantity = max(1, int(position_size / opp.price))
            
            # Generate decision
            decision = TradingDecision(
                symbol=opp.symbol,
                action=action,
                quantity=quantity,
                target_price=opp.target_price,
                stop_loss=opp.stop_loss,
                confidence=confidence_factor,
                reasoning=self._generate_reasoning(opp, technical_score, market_conditions),
                risk_score=risk_factor,
                expected_return=opp.risk_reward_ratio * 0.02,  # Conservative estimate
                time_horizon=opp.timeframe,
                metadata={
                    'opportunity_type': opp.opportunity_type.value,
                    'technical_score': technical_score,
                    'market_trend': market_conditions.get('market_trend', 'neutral')
                }
            )
            
            decisions.append(decision)
        
        # Sort by confidence and expected return
        decisions.sort(key=lambda x: (x.confidence, x.expected_return), reverse=True)
        
        # Limit to top 5 decisions to avoid over-trading
        return decisions[:5]
    
    def _determine_action(self, opp: TradingOpportunity, technical_score: float, market_conditions: Dict) -> TradingAction:
        """Determine the appropriate trading action"""
        market_trend = market_conditions.get('market_trend', 'neutral')
        
        # Bullish market conditions
        if market_trend == 'bullish' and technical_score > 0.6:
            if opp.opportunity_type in [OpportunityType.BREAKOUT, OpportunityType.MOMENTUM, OpportunityType.VOLUME_SPIKE]:
                return TradingAction.BUY
        
        # Bearish market conditions
        elif market_trend == 'bearish' and technical_score < 0.4:
            if opp.opportunity_type in [OpportunityType.RSI_OVERBOUGHT, OpportunityType.GAP_DOWN]:
                return TradingAction.SELL
        
        # Neutral market - be more selective
        elif market_trend == 'neutral':
            if opp.confidence > 0.8 and technical_score > 0.7:
                if opp.opportunity_type in [OpportunityType.BREAKOUT, OpportunityType.RSI_OVERSOLD]:
                    return TradingAction.BUY
        
        return TradingAction.HOLD
    
    def _generate_reasoning(self, opp: TradingOpportunity, technical_score: float, market_conditions: Dict) -> str:
        """Generate human-readable reasoning for the decision"""
        return (f"{opp.opportunity_type.value.replace('_', ' ').title()} opportunity in {opp.symbol} "
               f"with {opp.confidence:.1%} confidence. Technical score: {technical_score:.2f}. "
               f"Market trend: {market_conditions.get('market_trend', 'neutral')}. "
               f"R/R ratio: {opp.risk_reward_ratio:.2f}")


class LearningSystem:
    """Self-learning system with multiple ML approaches"""
    
    def __init__(self, learning_file: str = "trading_memory.pkl"):
        self.logger = logging.getLogger(f"{__name__}.LearningSystem")
        self.learning_file = learning_file
        self.decision_history: List[LearningRecord] = []
        self.performance_cache = deque(maxlen=1000)  # Last 1000 decisions
        self.model_weights = {
            'technical_weight': 0.4,
            'fundamental_weight': 0.2,
            'market_sentiment_weight': 0.2,
            'momentum_weight': 0.2
        }
        self._load_memory()
    
    def _load_memory(self):
        """Load historical learning data"""
        try:
            if Path(self.learning_file).exists():
                with open(self.learning_file, 'rb') as f:
                    data = pickle.load(f)
                    self.decision_history = data.get('decisions', [])
                    self.model_weights = data.get('weights', self.model_weights)
                self.logger.info(f"Loaded {len(self.decision_history)} historical decisions")
        except Exception as e:
            self.logger.warning(f"Could not load learning memory: {e}")
    
    def save_memory(self):
        """Save learning data to disk"""
        try:
            data = {
                'decisions': self.decision_history[-5000:],  # Keep last 5000
                'weights': self.model_weights,
                'last_update': datetime.now().isoformat()
            }
            with open(self.learning_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Could not save learning memory: {e}")
    
    async def record_decision(self, decision: TradingDecision, market_conditions: Dict, technical_indicators: Dict):
        """Record a trading decision for learning"""
        record = LearningRecord(
            decision_id=str(uuid.uuid4()),
            symbol=decision.symbol,
            action=decision.action,
            entry_price=decision.target_price or 0,
            exit_price=None,
            quantity=decision.quantity,
            predicted_outcome=decision.expected_return,
            actual_outcome=None,
            confidence=decision.confidence,
            timestamp=datetime.now(),
            market_conditions=market_conditions,
            technical_indicators=technical_indicators
        )
        
        self.decision_history.append(record)
        
        # Periodic cleanup and learning
        if len(self.decision_history) % 100 == 0:
            await self._update_model_weights()
            self.save_memory()
    
    async def update_decision_outcome(self, decision_id: str, exit_price: float, actual_return: float):
        """Update the outcome of a previous decision"""
        for record in reversed(self.decision_history):
            if record.decision_id == decision_id:
                record.exit_price = exit_price
                record.actual_outcome = actual_return
                record.success = actual_return > 0
                
                # Add to performance cache
                performance_score = actual_return / max(0.01, abs(record.predicted_outcome))
                self.performance_cache.append(performance_score)
                
                break
    
    async def _update_model_weights(self):
        """Update model weights based on performance"""
        if len(self.performance_cache) < 50:
            return
        
        # Calculate success rates for different factors
        successful_decisions = [r for r in self.decision_history[-500:] if r.success is True]
        failed_decisions = [r for r in self.decision_history[-500:] if r.success is False]
        
        if not successful_decisions or not failed_decisions:
            return
        
        # Analyze what factors led to success vs failure
        successful_confidence = np.mean([r.confidence for r in successful_decisions])
        failed_confidence = np.mean([r.confidence for r in failed_decisions])
        
        # Adjust weights based on performance
        if successful_confidence > failed_confidence:
            self.model_weights['technical_weight'] = min(0.6, self.model_weights['technical_weight'] + 0.01)
        else:
            self.model_weights['technical_weight'] = max(0.2, self.model_weights['technical_weight'] - 0.01)
        
        self.logger.info(f"Updated model weights: {self.model_weights}")
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if not self.performance_cache:
            return {'win_rate': 0.5, 'avg_return': 0.0, 'confidence': 0.5}
        
        recent_performance = list(self.performance_cache)[-100:]  # Last 100 decisions
        
        win_rate = len([p for p in recent_performance if p > 1]) / len(recent_performance)
        avg_return = np.mean(recent_performance) - 1  # Convert to return percentage
        confidence = min(1.0, win_rate + (avg_return * 0.5))
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'confidence': confidence,
            'total_decisions': len(self.decision_history),
            'recent_performance': np.mean(recent_performance)
        }


class AutonomousTradingBrain:
    """Main autonomous trading system using LangGraph"""
    
    def __init__(self, 
                 event_bus: TradingEventBus,
                 portfolio: Portfolio,
                 market_scanner: MarketScanner,
                 learning_enabled: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        self.portfolio = portfolio
        self.market_scanner = market_scanner
        self.learning_enabled = learning_enabled
        
        # Initialize components
        self.market_analyzer = MarketAnalyzer()
        self.risk_manager = RiskManager()
        self.decision_engine = DecisionEngine()
        self.learning_system = LearningSystem() if learning_enabled else None
        
        # State management
        self.current_state: TradingState = self._initialize_state()
        self.running = False
        
        # Build the StateGraph
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_state_graph()
        else:
            self.graph = None
            self.logger.error("LangGraph not available. Please install: pip install langgraph")
    
    def _initialize_state(self) -> TradingState:
        """Initialize the trading state"""
        return TradingState(
            market_scan_results={},
            current_opportunities=[],
            market_conditions={},
            technical_analysis={},
            fundamental_analysis={},
            risk_analysis={},
            sentiment_analysis={},
            trading_signals=[],
            position_recommendations=[],
            risk_score=0.5,
            pending_orders=[],
            executed_trades=[],
            execution_errors=[],
            performance_metrics={},
            learning_feedback={},
            model_updates=[],
            iteration_count=0,
            last_update=datetime.now().isoformat(),
            node_history=[],
            confidence_score=0.5,
            portfolio_snapshot={}
        )
    
    def _build_state_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for trading decisions"""
        
        # Create the graph
        workflow = StateGraph(TradingState)
        
        # Add nodes
        workflow.add_node("scan", self._scan_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("decide", self._decide_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("learn", self._learn_node)
        
        # Define the flow
        workflow.set_entry_point("scan")
        
        # Define edges (conditional routing)
        workflow.add_edge("scan", "analyze")
        workflow.add_conditional_edges(
            "analyze",
            self._should_proceed_to_decide,
            {
                "decide": "decide",
                "scan": "scan"  # Go back to scan if analysis is insufficient
            }
        )
        workflow.add_conditional_edges(
            "decide",
            self._should_execute_trades,
            {
                "execute": "execute",
                "learn": "learn"  # Skip execution if no decisions
            }
        )
        workflow.add_edge("execute", "learn")
        workflow.add_conditional_edges(
            "learn",
            self._should_continue_cycle,
            {
                "scan": "scan",
                END: END
            }
        )
        
        # Compile the graph
        return workflow.compile()
    
    async def _scan_node(self, state: TradingState) -> TradingState:
        """Node 1: Market scanning and data collection"""
        self.logger.info("Executing SCAN node")
        
        try:
            # Update portfolio snapshot
            state['portfolio_snapshot'] = self.portfolio.get_portfolio_summary()
            
            # Perform market scan
            scan_filter = ScanFilter(
                min_price=5.0,
                max_price=500.0,
                min_volume=500_000,
                min_market_cap=100_000_000
            )
            
            scan_results = await self.market_scanner.scan_market(scan_filter, max_symbols=200)
            state['market_scan_results'] = scan_results
            
            # Extract top opportunities
            all_opportunities = []
            for symbol, opportunities in scan_results.items():
                all_opportunities.extend(opportunities[:2])  # Top 2 per symbol
            
            # Sort by confidence and limit
            all_opportunities.sort(key=lambda x: x.confidence, reverse=True)
            state['current_opportunities'] = all_opportunities[:20]  # Top 20 overall
            
            state['node_history'].append("scan")
            state['last_update'] = datetime.now().isoformat()
            
            self.logger.info(f"SCAN complete: Found {len(state['current_opportunities'])} opportunities")
            
        except Exception as e:
            self.logger.error(f"Error in SCAN node: {e}")
            state['execution_errors'].append(f"SCAN error: {str(e)}")
        
        return state
    
    async def _analyze_node(self, state: TradingState) -> TradingState:
        """Node 2: Deep analysis of opportunities"""
        self.logger.info("Executing ANALYZE node")
        
        try:
            opportunities = state.get('current_opportunities', [])
            
            if not opportunities:
                self.logger.warning("No opportunities to analyze")
                return state
            
            # Technical analysis
            state['technical_analysis'] = await self.market_analyzer.analyze_technical_indicators(opportunities)
            
            # Market conditions analysis
            state['market_conditions'] = await self.market_analyzer.analyze_market_conditions(state)
            
            # Risk analysis
            state['risk_analysis'] = await self.risk_manager.calculate_risk_metrics(state)
            
            # Update confidence score
            confidence_scores = [opp.confidence for opp in opportunities]
            state['confidence_score'] = np.mean(confidence_scores) if confidence_scores else 0.5
            
            state['node_history'].append("analyze")
            state['last_update'] = datetime.now().isoformat()
            
            self.logger.info(f"ANALYZE complete: Confidence score: {state['confidence_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in ANALYZE node: {e}")
            state['execution_errors'].append(f"ANALYZE error: {str(e)}")
        
        return state
    
    async def _decide_node(self, state: TradingState) -> TradingState:
        """Node 3: Trading decision generation"""
        self.logger.info("Executing DECIDE node")
        
        try:
            # Generate trading decisions
            decisions = await self.decision_engine.generate_trading_decisions(state)
            
            # Convert decisions to signals
            trading_signals = []
            for decision in decisions:
                signal = {
                    'symbol': decision.symbol,
                    'action': decision.action.value,
                    'quantity': decision.quantity,
                    'target_price': decision.target_price,
                    'stop_loss': decision.stop_loss,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning,
                    'timestamp': datetime.now().isoformat(),
                    'decision_id': str(uuid.uuid4())
                }
                trading_signals.append(signal)
            
            state['trading_signals'] = trading_signals
            state['position_recommendations'] = [
                {
                    'symbol': decision.symbol,
                    'recommended_position_size': decision.quantity * decision.target_price if decision.target_price else 0,
                    'risk_score': decision.risk_score,
                    'expected_return': decision.expected_return
                }
                for decision in decisions
            ]
            
            state['node_history'].append("decide")
            state['last_update'] = datetime.now().isoformat()
            
            self.logger.info(f"DECIDE complete: Generated {len(trading_signals)} trading signals")
            
        except Exception as e:
            self.logger.error(f"Error in DECIDE node: {e}")
            state['execution_errors'].append(f"DECIDE error: {str(e)}")
        
        return state
    
    async def _execute_node(self, state: TradingState) -> TradingState:
        """Node 4: Trade execution"""
        self.logger.info("Executing EXECUTE node")
        
        try:
            trading_signals = state.get('trading_signals', [])
            executed_trades = []
            
            for signal in trading_signals:
                if signal['confidence'] < 0.7:  # Only execute high confidence trades
                    continue
                
                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                target_price = signal.get('target_price', 0)
                
                # Simulate trade execution (in production, connect to broker API)
                try:
                    if action == 'buy':
                        success = await self.portfolio.add_trade(symbol, quantity, target_price)
                    elif action == 'sell':
                        success = await self.portfolio.add_trade(symbol, -quantity, target_price)
                    else:
                        success = False
                    
                    if success:
                        executed_trade = {
                            'decision_id': signal['decision_id'],
                            'symbol': symbol,
                            'action': action,
                            'quantity': quantity,
                            'price': target_price,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'executed'
                        }
                        executed_trades.append(executed_trade)
                        
                        # Publish to event bus
                        await self.event_bus.publish(
                            "autonomous_trade_executed",
                            executed_trade,
                            priority=Priority.HIGH
                        )
                        
                        self.logger.info(f"Executed: {action} {quantity} {symbol} @ {target_price}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to execute trade {symbol}: {e}")
                    state['execution_errors'].append(f"Execution failed for {symbol}: {str(e)}")
            
            state['executed_trades'] = executed_trades
            state['node_history'].append("execute")
            state['last_update'] = datetime.now().isoformat()
            
            self.logger.info(f"EXECUTE complete: Executed {len(executed_trades)} trades")
            
        except Exception as e:
            self.logger.error(f"Error in EXECUTE node: {e}")
            state['execution_errors'].append(f"EXECUTE error: {str(e)}")
        
        return state
    
    async def _learn_node(self, state: TradingState) -> TradingState:
        """Node 5: Learning and feedback"""
        self.logger.info("Executing LEARN node")
        
        try:
            if not self.learning_system:
                self.logger.info("Learning disabled, skipping LEARN node")
                return state
            
            # Record decisions for learning
            trading_signals = state.get('trading_signals', [])
            market_conditions = state.get('market_conditions', {})
            
            for signal in trading_signals:
                decision = TradingDecision(
                    symbol=signal['symbol'],
                    action=TradingAction(signal['action']),
                    quantity=signal['quantity'],
                    target_price=signal.get('target_price'),
                    stop_loss=signal.get('stop_loss'),
                    confidence=signal['confidence'],
                    reasoning=signal['reasoning'],
                    risk_score=0.5,  # Simplified
                    expected_return=0.02,  # Simplified
                    time_horizon="short",
                    metadata={}
                )
                
                technical_indicators = state.get('technical_analysis', {}).get(signal['symbol'], {})
                await self.learning_system.record_decision(decision, market_conditions, technical_indicators)
            
            # Get performance metrics
            state['performance_metrics'] = await self.learning_system.get_performance_metrics()
            
            # Update learning feedback
            state['learning_feedback'] = {
                'total_decisions': len(self.learning_system.decision_history),
                'model_weights': self.learning_system.model_weights,
                'recent_performance': state['performance_metrics'].get('recent_performance', 1.0)
            }
            
            state['iteration_count'] += 1
            state['node_history'].append("learn")
            state['last_update'] = datetime.now().isoformat()
            
            self.logger.info(f"LEARN complete: Performance metrics updated")
            
        except Exception as e:
            self.logger.error(f"Error in LEARN node: {e}")
            state['execution_errors'].append(f"LEARN error: {str(e)}")
        
        return state
    
    # Conditional edge functions
    def _should_proceed_to_decide(self, state: TradingState) -> str:
        """Decide whether to proceed to decision making or rescan"""
        opportunities = state.get('current_opportunities', [])
        confidence = state.get('confidence_score', 0)
        
        if len(opportunities) >= 3 and confidence >= 0.6:
            return "decide"
        else:
            return "scan"
    
    def _should_execute_trades(self, state: TradingState) -> str:
        """Decide whether to execute trades or skip to learning"""
        signals = state.get('trading_signals', [])
        high_confidence_signals = [s for s in signals if s.get('confidence', 0) >= 0.7]
        
        if len(high_confidence_signals) > 0:
            return "execute"
        else:
            return "learn"
    
    def _should_continue_cycle(self, state: TradingState) -> str:
        """Decide whether to continue the trading cycle"""
        iteration_count = state.get('iteration_count', 0)
        errors = state.get('execution_errors', [])
        
        # Stop after 10 iterations or if too many errors
        if iteration_count >= 10 or len(errors) > 5:
            return END
        else:
            return "scan"
    
    async def start_autonomous_trading(self, max_iterations: int = 100):
        """Start the autonomous trading system"""
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("Cannot start autonomous trading without LangGraph")
            return
        
        self.running = True
        self.logger.info("Starting autonomous trading brain")
        
        try:
            iteration = 0
            while self.running and iteration < max_iterations:
                self.logger.info(f"Starting trading cycle {iteration + 1}")
                
                # Execute the state graph
                result = await self.graph.ainvoke(self.current_state)
                self.current_state = result
                
                # Publish status update
                await self.event_bus.publish(
                    "autonomous_brain_cycle_complete",
                    {
                        'iteration': iteration + 1,
                        'opportunities_found': len(self.current_state.get('current_opportunities', [])),
                        'signals_generated': len(self.current_state.get('trading_signals', [])),
                        'trades_executed': len(self.current_state.get('executed_trades', [])),
                        'confidence_score': self.current_state.get('confidence_score', 0),
                        'performance_metrics': self.current_state.get('performance_metrics', {})
                    },
                    priority=Priority.NORMAL
                )
                
                iteration += 1
                
                # Wait between cycles (configurable)
                await asyncio.sleep(300)  # 5 minutes between cycles
                
        except Exception as e:
            self.logger.error(f"Error in autonomous trading: {e}")
        finally:
            self.running = False
            if self.learning_system:
                self.learning_system.save_memory()
    
    async def stop_autonomous_trading(self):
        """Stop the autonomous trading system"""
        self.running = False
        self.logger.info("Stopping autonomous trading brain")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state for monitoring"""
        return {
            'running': self.running,
            'iteration_count': self.current_state.get('iteration_count', 0),
            'last_update': self.current_state.get('last_update'),
            'current_opportunities': len(self.current_state.get('current_opportunities', [])),
            'pending_signals': len(self.current_state.get('trading_signals', [])),
            'confidence_score': self.current_state.get('confidence_score', 0),
            'recent_performance': self.current_state.get('performance_metrics', {})
        }


# Example usage
async def main():
    """Example usage of the autonomous trading brain"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    event_bus = TradingEventBus()
    await event_bus.start()
    
    portfolio = Portfolio(initial_cash=100000, event_bus=event_bus)
    market_scanner = MarketScanner(event_bus=event_bus)
    
    # Create autonomous brain
    brain = AutonomousTradingBrain(
        event_bus=event_bus,
        portfolio=portfolio,
        market_scanner=market_scanner,
        learning_enabled=True
    )
    
    try:
        # Start autonomous trading
        await brain.start_autonomous_trading(max_iterations=5)
        
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())