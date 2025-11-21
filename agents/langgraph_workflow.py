"""
LangGraph Workflow Implementation for Trading System

This module implements the core LangGraph StateGraph for coordinating all trading agents.
It defines the system state structure, agent transitions, communication protocols,
and conditional routing based on market conditions.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, TypedDict, Literal, Annotated
from enum import Enum
import json
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Import communication protocols
from .communication_protocols import AgentRole

# Import all trading agents
from .market_data_ingestor import MarketDataIngestorAgent
from .news_sentiment_agent import NewsSentimentAgent
from .momentum_trading_agent import MomentumTradingAgent
from .mean_reversion_agent import MeanReversionTradingAgent
from .options_volatility_agent import OptionsVolatilityAgent
from .portfolio_allocator_agent import PortfolioAllocatorAgent
from .risk_manager_agent import RiskManagerAgent
from .execution_engine_agent import ExecutionEngineAgent
# from .satellite_trading_agent import SatelliteTradingAgent

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime types for conditional routing"""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NEWS_DRIVEN = "news_driven"
    CRISIS = "crisis"
    NORMAL = "normal"


class WorkflowPhase(str, Enum):
    """Workflow execution phases"""
    DATA_INGESTION = "data_ingestion"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SIGNAL_GENERATION = "signal_generation"
    SIGNAL_FUSION = "signal_fusion"
    RISK_ASSESSMENT = "risk_assessment"
    ORDER_EXECUTION = "order_execution"
    MONITORING = "monitoring"
    LEARNING = "learning"


class SatelliteMetrics(BaseModel):
    """Satellite data metrics structure"""
    crop_health_index: Optional[float] = None
    soil_moisture: Optional[float] = None
    vegetation_density: Optional[float] = None
    storage_fill_rate: Optional[float] = None
    facility_activity: Optional[float] = None
    tanker_traffic: Optional[int] = None
    parking_lot_occupancy: Optional[float] = None
    delivery_truck_count: Optional[int] = None
    foot_traffic_estimate: Optional[int] = None


class MarketData(BaseModel):
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    exchange: str = "NASDAQ"
    satellite_metrics: Optional[SatelliteMetrics] = None


class Signal(BaseModel):
    """Trading signal structure"""
    symbol: str
    signal_type: str
    value: float
    confidence: float
    top_3_reasons: List[str]
    timestamp: datetime
    model_version: str
    agent_name: str
    fibonacci_levels: Optional[Dict[str, float]] = None
    sentiment_score: Optional[float] = None
    satellite_data_source: Optional[str] = None  # e.g., 'agriculture', 'oil_storage', 'retail'
    satellite_indicators: Optional[Dict[str, float]] = None  # Key satellite metrics that influenced the signal


class Order(BaseModel):
    """Order structure"""
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float
    order_type: Literal["MARKET", "LIMIT", "STOP"]
    price: Optional[float] = None
    time_in_force: str = "DAY"
    strategy: str
    signal_id: Optional[str] = None


class RiskMetrics(BaseModel):
    """Risk metrics structure"""
    portfolio_value: float
    daily_pnl: float
    var_95: float
    max_drawdown: float
    position_count: int
    leverage: float
    risk_score: float
    alerts: List[str] = Field(default_factory=list)


class Alert(BaseModel):
    """System alert structure"""
    level: Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
    message: str
    timestamp: datetime
    component: str
    data: Optional[Dict[str, Any]] = None


class TradingSystemState(TypedDict):
    """
    Central state structure for the LangGraph workflow.
    All agents read from and write to this shared state.
    
    Uses Annotated types for keys that can receive multiple updates.
    """
    # Market data - can be updated by multiple sources
    market_data: Annotated[Dict[str, MarketData], operator.add]
    historical_data: Annotated[Dict[str, List[MarketData]], operator.add]
    
    # Sentiment and news - can be updated by multiple sources
    news_articles: Annotated[List[Dict[str, Any]], operator.add]
    sentiment_scores: Annotated[Dict[str, float], operator.add]
    market_events: Annotated[List[Dict[str, Any]], operator.add]
    
    # Trading signals - can be updated by multiple agents
    raw_signals: Annotated[Dict[str, List[Signal]], operator.add]
    fused_signals: Dict[str, Signal]
    signal_conflicts: Annotated[List[Dict[str, Any]], operator.add]
    
    # Portfolio and risk
    portfolio_state: Dict[str, Any]
    positions: Annotated[Dict[str, Dict[str, Any]], operator.add]
    risk_metrics: RiskMetrics
    risk_limits: Dict[str, float]
    
    # Orders and execution
    pending_orders: Annotated[List[Order], operator.add]
    executed_orders: Annotated[List[Dict[str, Any]], operator.add]
    execution_reports: Annotated[List[Dict[str, Any]], operator.add]
    
    # System state
    market_regime: MarketRegime
    workflow_phase: WorkflowPhase
    system_alerts: Annotated[List[Alert], operator.add]
    performance_metrics: Annotated[Dict[str, float], operator.add]
    
    # Configuration
    symbols_universe: List[str]
    active_strategies: List[str]
    model_versions: Dict[str, str]
    
    # Debugging and monitoring - can be updated by multiple agents
    agent_states: Annotated[Dict[str, Dict[str, Any]], operator.add]
    execution_log: Annotated[List[Dict[str, Any]], operator.add]
    error_log: Annotated[List[Dict[str, Any]], operator.add]


class LangGraphTradingWorkflow:
    """
    Main LangGraph workflow orchestrator for the trading system.
    
    This class sets up the StateGraph, defines agent nodes and edges,
    implements conditional routing, and provides monitoring capabilities.
    """
    
    def __init__(self):
        self.graph = None
        self.checkpointer = MemorySaver()
        self.agents = {}
        self.workflow_config = {}
        self.monitoring_enabled = True
        
        # Initialize agents
        self._initialize_agents()
        
        # Build the workflow graph
        self._build_workflow_graph()
        
        logger.info("LangGraph Trading Workflow initialized successfully")
    
    def _initialize_agents(self) -> None:
        """Initialize all trading agents"""
        try:
            self.agents = {
                "market_data_ingestor": MarketDataIngestorAgent(),
                "sentiment_agent": NewsSentimentAgent(),
                "momentum_agent": MomentumTradingAgent(),
                "mean_reversion_agent": MeanReversionTradingAgent(),
                "options_agent": OptionsVolatilityAgent(),
                "portfolio_allocator": PortfolioAllocatorAgent(),
                "risk_manager": RiskManagerAgent(),
                "execution_engine": ExecutionEngineAgent()
            }
            logger.info(f"Initialized {len(self.agents)} trading agents")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _build_workflow_graph(self) -> None:
        """Build the LangGraph StateGraph with all nodes and edges"""
        
        # Create the state graph
        workflow = StateGraph(TradingSystemState)
        
        # Add agent nodes
        workflow.add_node("market_data_ingestor", self._market_data_node)
        workflow.add_node("sentiment_analyzer", self._sentiment_analysis_node)
        workflow.add_node("momentum_trader", self._momentum_trading_node)
        workflow.add_node("mean_reversion_trader", self._mean_reversion_node)
        workflow.add_node("options_trader", self._options_trading_node)
        workflow.add_node("portfolio_allocator", self._portfolio_allocation_node)
        workflow.add_node("risk_manager", self._risk_management_node)
        workflow.add_node("execution_engine", self._execution_node)
        workflow.add_node("monitor_and_learn", self._monitoring_node)
        
        # Define the workflow edges
        workflow.set_entry_point("market_data_ingestor")
        
        # Sequential flow for data ingestion and sentiment
        workflow.add_edge("market_data_ingestor", "sentiment_analyzer")
        
        # Parallel signal generation from sentiment analysis
        workflow.add_edge("sentiment_analyzer", "momentum_trader")
        workflow.add_edge("sentiment_analyzer", "mean_reversion_trader")
        workflow.add_edge("sentiment_analyzer", "options_trader")
        
        # Conditional routing from strategy agents to portfolio allocator
        workflow.add_conditional_edges(
            "momentum_trader",
            self._route_to_portfolio_or_continue,
            {
                "portfolio_allocator": "portfolio_allocator",
                "continue": "mean_reversion_trader"
            }
        )
        
        workflow.add_conditional_edges(
            "mean_reversion_trader", 
            self._route_to_portfolio_or_continue,
            {
                "portfolio_allocator": "portfolio_allocator",
                "continue": "options_trader"
            }
        )
        
        workflow.add_conditional_edges(
            "options_trader",
            self._route_to_portfolio_or_continue,
            {
                "portfolio_allocator": "portfolio_allocator",
                "continue": "portfolio_allocator"
            }
        )
        
        # Risk management conditional routing
        workflow.add_conditional_edges(
            "portfolio_allocator",
            self._route_risk_or_execution,
            {
                "risk_manager": "risk_manager",
                "execution_engine": "execution_engine",
                "halt": END
            }
        )
        
        # Risk manager routing
        workflow.add_conditional_edges(
            "risk_manager",
            self._route_after_risk_check,
            {
                "execution_engine": "execution_engine",
                "halt": END,
                "emergency_stop": END
            }
        )
        
        # Execution to monitoring
        workflow.add_edge("execution_engine", "monitor_and_learn")
        workflow.add_edge("monitor_and_learn", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("LangGraph workflow graph compiled successfully")
    
    # Agent Node Implementations
    
    async def _market_data_node(self, state: TradingSystemState) -> Dict[str, Any]:
        """Market data ingestion node"""
        try:
            # Get market data for all symbols
            market_data = {}
            for symbol in state.get("symbols_universe", ["AAPL", "GOOGL", "MSFT"]):
                try:
                    data = await self.agents["market_data_ingestor"].get_latest_data(symbol)
                    if data:
                        market_data[symbol] = MarketData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            open=data.get("open", 0.0),
                            high=data.get("high", 0.0),
                            low=data.get("low", 0.0),
                            close=data.get("close", 0.0),
                            volume=data.get("volume", 0),
                            vwap=data.get("vwap")
                        )
                except Exception as e:
                    logger.error(f"Failed to get data for {symbol}: {e}")
                    continue
            
            # Detect market regime
            market_regime = self._detect_market_regime(market_data)
            
            # Return partial state update
            return {
                "workflow_phase": WorkflowPhase.DATA_INGESTION,
                "market_data": market_data,
                "market_regime": market_regime,
                "agent_states": {"market_data_ingestor": {"status": "completed", "symbols_processed": len(market_data)}},
                "execution_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "component": "market_data_ingestor",
                    "message": f"Processed {len(market_data)} symbols",
                    "workflow_phase": "data_ingestion"
                }]
            }
            
        except Exception as e:
            logger.error(f"Market data node error: {e}")
            return {
                "agent_states": {"market_data_ingestor": {"status": "error", "error": str(e)}},
                "error_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "component": "market_data_ingestor",
                    "error": str(e),
                    "workflow_phase": "data_ingestion"
                }]
            }
    
    async def _sentiment_analysis_node(self, state: TradingSystemState) -> Dict[str, Any]:
        """Sentiment analysis node"""
        try:
            # Analyze sentiment for each symbol
            sentiment_scores = {}
            for symbol in state.get("market_data", {}).keys():
                try:
                    sentiment_data = await self.agents["sentiment_agent"].analyze_sentiment(symbol)
                    sentiment_scores[symbol] = sentiment_data.get("sentiment_score", 0.0)
                except Exception as e:
                    logger.error(f"Failed sentiment analysis for {symbol}: {e}")
                    sentiment_scores[symbol] = 0.0
            
            return {
                "workflow_phase": WorkflowPhase.SENTIMENT_ANALYSIS,
                "sentiment_scores": sentiment_scores,
                "agent_states": {"sentiment_agent": {"status": "completed", "symbols_analyzed": len(sentiment_scores)}},
                "execution_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "component": "sentiment_agent",
                    "message": f"Analyzed sentiment for {len(sentiment_scores)} symbols",
                    "workflow_phase": "sentiment_analysis"
                }]
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis node error: {e}")
            return {
                "agent_states": {"sentiment_agent": {"status": "error", "error": str(e)}},
                "error_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "component": "sentiment_agent",
                    "error": str(e),
                    "workflow_phase": "sentiment_analysis"
                }]
            }
    
    async def _momentum_trading_node(self, state: TradingSystemState) -> Dict[str, Any]:
        """Momentum trading strategy node"""
        try:
            momentum_signals = []
            for symbol, market_data in state.get("market_data", {}).items():
                try:
                    # Get sentiment score for this symbol
                    sentiment_score = state.get("sentiment_scores", {}).get(symbol, 0.0)
                    
                    # Generate momentum signal
                    signal_data = await self.agents["momentum_agent"].generate_signal(
                        symbol, market_data.dict() if hasattr(market_data, 'dict') else market_data, sentiment_score
                    )
                    
                    if signal_data and signal_data.get("confidence", 0) > 0.5:
                        signal = Signal(
                            symbol=symbol,
                            signal_type="momentum",
                            value=signal_data["signal_strength"],
                            confidence=signal_data["confidence"],
                            top_3_reasons=signal_data.get("top_3_reasons", []),
                            timestamp=datetime.now(),
                            model_version="1.0.0",
                            agent_name="momentum_agent",
                            fibonacci_levels=signal_data.get("fibonacci_levels"),
                            sentiment_score=sentiment_score
                        )
                        momentum_signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Failed momentum signal for {symbol}: {e}")
                    continue
            
            return {
                "workflow_phase": WorkflowPhase.SIGNAL_GENERATION,
                "raw_signals": {"momentum": momentum_signals},
                "agent_states": {"momentum_agent": {"status": "completed", "signals_generated": len(momentum_signals)}},
                "execution_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "component": "momentum_agent",
                    "message": f"Generated {len(momentum_signals)} momentum signals",
                    "workflow_phase": "signal_generation"
                }]
            }
            
        except Exception as e:
            logger.error(f"Momentum trading node error: {e}")
            return {
                "agent_states": {"momentum_agent": {"status": "error", "error": str(e)}},
                "error_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "component": "momentum_agent",
                    "error": str(e),
                    "workflow_phase": "signal_generation"
                }]
            }
    
    async def _mean_reversion_node(self, state: TradingSystemState) -> TradingSystemState:
        """Mean reversion trading strategy node"""
        try:
            state["agent_states"]["mean_reversion_agent"] = {"status": "running"}
            
            mean_reversion_signals = []
            for symbol, market_data in state["market_data"].items():
                try:
                    sentiment_score = state["sentiment_scores"].get(symbol, 0.0)
                    
                    signal_data = await self.agents["mean_reversion_agent"].generate_signal(
                        symbol, market_data.dict(), sentiment_score
                    )
                    
                    if signal_data and signal_data.get("confidence", 0) > 0.5:
                        signal = Signal(
                            symbol=symbol,
                            signal_type="mean_reversion",
                            value=signal_data["signal_strength"],
                            confidence=signal_data["confidence"],
                            top_3_reasons=signal_data.get("top_3_reasons", []),
                            timestamp=datetime.now(),
                            model_version="1.0.0",
                            agent_name="mean_reversion_agent",
                            fibonacci_levels=signal_data.get("fibonacci_levels"),
                            sentiment_score=sentiment_score
                        )
                        mean_reversion_signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Failed mean reversion signal for {symbol}: {e}")
                    continue
            
            state["raw_signals"]["mean_reversion"] = mean_reversion_signals
            state["agent_states"]["mean_reversion_agent"] = {"status": "completed", "signals_generated": len(mean_reversion_signals)}
            
            self._log_execution(state, "mean_reversion_agent", f"Generated {len(mean_reversion_signals)} mean reversion signals")
            
        except Exception as e:
            self._log_error(state, "mean_reversion_agent", str(e))
            state["agent_states"]["mean_reversion_agent"] = {"status": "error", "error": str(e)}
        
        return state
    
    async def _options_trading_node(self, state: TradingSystemState) -> TradingSystemState:
        """Options trading strategy node"""
        try:
            state["agent_states"]["options_agent"] = {"status": "running"}
            
            options_signals = []
            for symbol, market_data in state["market_data"].items():
                try:
                    signal_data = await self.agents["options_agent"].generate_signal(
                        symbol, market_data.dict()
                    )
                    
                    if signal_data and signal_data.get("confidence", 0) > 0.5:
                        signal = Signal(
                            symbol=symbol,
                            signal_type="options_volatility",
                            value=signal_data["signal_strength"],
                            confidence=signal_data["confidence"],
                            top_3_reasons=signal_data.get("top_3_reasons", []),
                            timestamp=datetime.now(),
                            model_version="1.0.0",
                            agent_name="options_agent"
                        )
                        options_signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Failed options signal for {symbol}: {e}")
                    continue
            
            state["raw_signals"]["options"] = options_signals
            state["agent_states"]["options_agent"] = {"status": "completed", "signals_generated": len(options_signals)}
            
            self._log_execution(state, "options_agent", f"Generated {len(options_signals)} options signals")
            
        except Exception as e:
            self._log_error(state, "options_agent", str(e))
            state["agent_states"]["options_agent"] = {"status": "error", "error": str(e)}
        
        return state
    
    async def _portfolio_allocation_node(self, state: TradingSystemState) -> TradingSystemState:
        """Portfolio allocation and signal fusion node"""
        try:
            state["workflow_phase"] = WorkflowPhase.SIGNAL_FUSION
            state["agent_states"]["portfolio_allocator"] = {"status": "running"}
            
            # Fuse all signals
            fused_signals = {}
            all_signals = []
            
            # Collect all signals
            for strategy, signals in state.get("raw_signals", {}).items():
                all_signals.extend(signals)
            
            # Group signals by symbol
            signals_by_symbol = {}
            for signal in all_signals:
                if signal.symbol not in signals_by_symbol:
                    signals_by_symbol[signal.symbol] = []
                signals_by_symbol[signal.symbol].append(signal)
            
            # Fuse signals for each symbol
            for symbol, signals in signals_by_symbol.items():
                try:
                    fused_signal_data = await self.agents["portfolio_allocator"].fuse_signals(
                        symbol, [s.dict() for s in signals], state["market_regime"]
                    )
                    
                    if fused_signal_data:
                        fused_signal = Signal(
                            symbol=symbol,
                            signal_type="fused",
                            value=fused_signal_data["signal_strength"],
                            confidence=fused_signal_data["confidence"],
                            top_3_reasons=fused_signal_data.get("top_3_reasons", []),
                            timestamp=datetime.now(),
                            model_version="1.0.0",
                            agent_name="portfolio_allocator"
                        )
                        fused_signals[symbol] = fused_signal
                        
                except Exception as e:
                    logger.error(f"Failed signal fusion for {symbol}: {e}")
                    continue
            
            state["fused_signals"] = fused_signals
            state["agent_states"]["portfolio_allocator"] = {"status": "completed", "signals_fused": len(fused_signals)}
            
            self._log_execution(state, "portfolio_allocator", f"Fused signals for {len(fused_signals)} symbols")
            
        except Exception as e:
            self._log_error(state, "portfolio_allocator", str(e))
            state["agent_states"]["portfolio_allocator"] = {"status": "error", "error": str(e)}
        
        return state
    
    async def _risk_management_node(self, state: TradingSystemState) -> TradingSystemState:
        """Risk management node"""
        try:
            state["workflow_phase"] = WorkflowPhase.RISK_ASSESSMENT
            state["agent_states"]["risk_manager"] = {"status": "running"}
            
            # Assess risk for all fused signals
            risk_assessment = await self.agents["risk_manager"].assess_portfolio_risk(
                state.get("fused_signals", {}),
                state.get("portfolio_state", {}),
                state.get("positions", {})
            )
            
            # Update risk metrics
            state["risk_metrics"] = RiskMetrics(
                portfolio_value=risk_assessment.get("portfolio_value", 100000.0),
                daily_pnl=risk_assessment.get("daily_pnl", 0.0),
                var_95=risk_assessment.get("var_95", 0.0),
                max_drawdown=risk_assessment.get("max_drawdown", 0.0),
                position_count=risk_assessment.get("position_count", 0),
                leverage=risk_assessment.get("leverage", 1.0),
                risk_score=risk_assessment.get("risk_score", 0.0),
                alerts=risk_assessment.get("alerts", [])
            )
            
            state["agent_states"]["risk_manager"] = {"status": "completed", "risk_score": risk_assessment.get("risk_score", 0.0)}
            
            self._log_execution(state, "risk_manager", f"Risk assessment completed, score: {risk_assessment.get('risk_score', 0.0)}")
            
        except Exception as e:
            self._log_error(state, "risk_manager", str(e))
            state["agent_states"]["risk_manager"] = {"status": "error", "error": str(e)}
        
        return state
    
    async def _execution_node(self, state: TradingSystemState) -> TradingSystemState:
        """Order execution node"""
        try:
            state["workflow_phase"] = WorkflowPhase.ORDER_EXECUTION
            state["agent_states"]["execution_engine"] = {"status": "running"}
            
            # Generate orders from fused signals
            orders = []
            for symbol, signal in state.get("fused_signals", {}).items():
                try:
                    if signal.confidence > 0.7:  # Only execute high-confidence signals
                        order_data = await self.agents["execution_engine"].create_order(
                            symbol, signal.dict()
                        )
                        
                        if order_data:
                            order = Order(
                                symbol=symbol,
                                side="BUY" if signal.value > 0 else "SELL",
                                quantity=abs(order_data.get("quantity", 100)),
                                order_type=order_data.get("order_type", "MARKET"),
                                price=order_data.get("price"),
                                strategy=signal.signal_type,
                                signal_id=f"{signal.agent_name}_{symbol}_{signal.timestamp.isoformat()}"
                            )
                            orders.append(order)
                            
                except Exception as e:
                    logger.error(f"Failed to create order for {symbol}: {e}")
                    continue
            
            state["pending_orders"] = orders
            state["agent_states"]["execution_engine"] = {"status": "completed", "orders_created": len(orders)}
            
            self._log_execution(state, "execution_engine", f"Created {len(orders)} orders for execution")
            
        except Exception as e:
            self._log_error(state, "execution_engine", str(e))
            state["agent_states"]["execution_engine"] = {"status": "error", "error": str(e)}
        
        return state
    
    async def _monitoring_node(self, state: TradingSystemState) -> TradingSystemState:
        """Monitoring and learning node"""
        try:
            state["workflow_phase"] = WorkflowPhase.MONITORING
            
            # Update performance metrics
            performance_metrics = {
                "total_signals_generated": sum(len(signals) for signals in state.get("raw_signals", {}).values()),
                "signals_fused": len(state.get("fused_signals", {})),
                "orders_created": len(state.get("pending_orders", [])),
                "risk_score": state.get("risk_metrics", {}).get("risk_score", 0.0),
                "workflow_completion_time": datetime.now().isoformat()
            }
            
            state["performance_metrics"] = performance_metrics
            
            self._log_execution(state, "monitor_and_learn", "Workflow cycle completed successfully")
            
        except Exception as e:
            self._log_error(state, "monitor_and_learn", str(e))
        
        return state
    
    # Conditional Routing Functions
    
    def _route_to_portfolio_or_continue(self, state: TradingSystemState) -> str:
        """Route to portfolio allocator or continue to next strategy"""
        # Check if all strategy agents have completed
        completed_agents = 0
        total_strategy_agents = 3  # momentum, mean_reversion, options
        
        for agent in ["momentum_agent", "mean_reversion_agent", "options_agent"]:
            if state.get("agent_states", {}).get(agent, {}).get("status") == "completed":
                completed_agents += 1
        
        if completed_agents >= total_strategy_agents:
            return "portfolio_allocator"
        else:
            return "continue"
    
    def _route_risk_or_execution(self, state: TradingSystemState) -> str:
        """Route to risk manager, execution, or halt based on signals"""
        fused_signals = state.get("fused_signals", {})
        
        if not fused_signals:
            return "halt"
        
        # Check if any high-confidence signals exist
        high_confidence_signals = [s for s in fused_signals.values() if s.confidence > 0.7]
        
        if high_confidence_signals:
            return "risk_manager"
        else:
            return "halt"
    
    def _route_after_risk_check(self, state: TradingSystemState) -> str:
        """Route after risk assessment"""
        risk_metrics = state.get("risk_metrics")
        
        if not risk_metrics:
            return "halt"
        
        # Check for emergency conditions
        if risk_metrics.risk_score > 0.9:
            return "emergency_stop"
        elif risk_metrics.risk_score > 0.7:
            return "halt"
        else:
            return "execution_engine"
    
    # Utility Functions
    
    def _detect_market_regime(self, market_data: Dict[str, MarketData]) -> MarketRegime:
        """Detect current market regime based on market data"""
        if not market_data:
            return MarketRegime.NORMAL
        
        # Simple regime detection based on volatility and price action
        volatilities = []
        for data in market_data.values():
            if data.high > 0 and data.low > 0:
                daily_range = (data.high - data.low) / data.close
                volatilities.append(daily_range)
        
        if volatilities:
            avg_volatility = sum(volatilities) / len(volatilities)
            
            if avg_volatility > 0.05:
                return MarketRegime.HIGH_VOLATILITY
            elif avg_volatility < 0.02:
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.NORMAL
        
        return MarketRegime.NORMAL
    
    def _log_execution(self, state: TradingSystemState, component: str, message: str):
        """Log execution event"""
        if "execution_log" not in state:
            state["execution_log"] = []
        
        state["execution_log"].append({
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "message": message,
            "workflow_phase": state.get("workflow_phase", "unknown")
        })
    
    def _log_error(self, state: TradingSystemState, component: str, error: str):
        """Log error event"""
        if "error_log" not in state:
            state["error_log"] = []
        
        state["error_log"].append({
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "error": error,
            "workflow_phase": state.get("workflow_phase", "unknown")
        })
    
    # Public API Methods
    
    async def run_workflow(self, initial_state: Optional[TradingSystemState] = None) -> TradingSystemState:
        """Run the complete workflow"""
        if not self.graph:
            raise RuntimeError("Workflow graph not initialized")
        
        # Initialize state if not provided
        if initial_state is None:
            initial_state = self._create_initial_state()
        
        try:
            # Run the workflow
            config = {"configurable": {"thread_id": "trading_session_" + str(int(datetime.now().timestamp()))}}
            
            result = await self.graph.ainvoke(initial_state, config=config)
            
            logger.info("Workflow execution completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def _create_initial_state(self) -> TradingSystemState:
        """Create initial state for workflow"""
        return TradingSystemState(
            market_data={},
            historical_data={},
            news_articles=[],
            sentiment_scores={},
            market_events=[],
            raw_signals={},
            fused_signals={},
            signal_conflicts=[],
            portfolio_state={},
            positions={},
            risk_metrics=RiskMetrics(
                portfolio_value=100000.0,
                daily_pnl=0.0,
                var_95=0.0,
                max_drawdown=0.0,
                position_count=0,
                leverage=1.0,
                risk_score=0.0
            ),
            risk_limits={"max_position_size": 10000, "max_daily_loss": 5000},
            pending_orders=[],
            executed_orders=[],
            execution_reports=[],
            market_regime=MarketRegime.NORMAL,
            workflow_phase=WorkflowPhase.DATA_INGESTION,
            system_alerts=[],
            performance_metrics={},
            symbols_universe=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
            active_strategies=["momentum", "mean_reversion", "options"],
            model_versions={"momentum": "1.0.0", "mean_reversion": "1.0.0", "options": "1.0.0"},
            agent_states={},
            execution_log=[],
            error_log=[]
        )
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "graph_compiled": self.graph is not None,
            "agents_initialized": len(self.agents),
            "monitoring_enabled": self.monitoring_enabled,
            "agent_list": list(self.agents.keys())
        }
    
    def enable_monitoring(self):
        """Enable workflow monitoring"""
        self.monitoring_enabled = True
        logger.info("Workflow monitoring enabled")
    
    def disable_monitoring(self):
        """Disable workflow monitoring"""
        self.monitoring_enabled = False
        logger.info("Workflow monitoring disabled")


class ConditionalRouter:
    """
    Advanced conditional routing system for market-based decisions.
    Routes workflow execution based on market conditions, agent performance,
    and system state.
    """
    
    def __init__(self):
        self.routing_rules = {}
        self.performance_history = {}
        self.market_condition_cache = {}
        
        # Setup default routing rules
        self._setup_default_routing_rules()
        
        logger.info("Conditional router initialized")
    
    def _setup_default_routing_rules(self):
        """Setup default routing rules based on market conditions"""
        self.routing_rules = {
            "high_volatility": {
                "preferred_strategies": ["momentum", "options"],
                "risk_multiplier": 0.5,
                "execution_priority": "fast"
            },
            "low_volatility": {
                "preferred_strategies": ["mean_reversion", "long_term"],
                "risk_multiplier": 1.2,
                "execution_priority": "optimal"
            },
            "trending": {
                "preferred_strategies": ["momentum"],
                "risk_multiplier": 1.0,
                "execution_priority": "fast"
            },
            "mean_reverting": {
                "preferred_strategies": ["mean_reversion", "pairs"],
                "risk_multiplier": 0.8,
                "execution_priority": "patient"
            },
            "news_driven": {
                "preferred_strategies": ["sentiment", "momentum"],
                "risk_multiplier": 0.6,
                "execution_priority": "immediate"
            },
            "crisis": {
                "preferred_strategies": ["risk_off"],
                "risk_multiplier": 0.2,
                "execution_priority": "emergency"
            }
        }
    
    def route_based_on_market_regime(self, state: TradingSystemState) -> Dict[str, Any]:
        """Route workflow based on detected market regime"""
        market_regime = state.get("market_regime", MarketRegime.NORMAL)
        
        # Get routing rules for current regime
        regime_rules = self.routing_rules.get(market_regime.lower(), {
            "preferred_strategies": ["momentum", "mean_reversion"],
            "risk_multiplier": 1.0,
            "execution_priority": "normal"
        })
        
        # Analyze current performance
        performance_scores = self._calculate_strategy_performance(state)
        
        # Combine regime preferences with performance
        routing_decision = {
            "market_regime": market_regime,
            "preferred_strategies": regime_rules["preferred_strategies"],
            "risk_adjustment": regime_rules["risk_multiplier"],
            "execution_priority": regime_rules["execution_priority"],
            "strategy_weights": self._calculate_strategy_weights(
                regime_rules["preferred_strategies"], 
                performance_scores
            ),
            "routing_confidence": self._calculate_routing_confidence(state)
        }
        
        return routing_decision
    
    def _calculate_strategy_performance(self, state: TradingSystemState) -> Dict[str, float]:
        """Calculate recent performance scores for each strategy"""
        performance_scores = {}
        
        # Get recent execution history
        execution_log = state.get("execution_log", [])
        recent_executions = execution_log[-50:]  # Last 50 executions
        
        # Calculate success rates by strategy
        strategy_stats = {}
        for execution in recent_executions:
            component = execution.get("component", "unknown")
            if "agent" in component:
                strategy = component.replace("_agent", "")
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {"success": 0, "total": 0}
                
                strategy_stats[strategy]["total"] += 1
                if "error" not in execution.get("message", "").lower():
                    strategy_stats[strategy]["success"] += 1
        
        # Convert to performance scores
        for strategy, stats in strategy_stats.items():
            if stats["total"] > 0:
                performance_scores[strategy] = stats["success"] / stats["total"]
            else:
                performance_scores[strategy] = 0.5  # Neutral score
        
        return performance_scores
    
    def _calculate_strategy_weights(self, preferred_strategies: List[str], performance_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate dynamic strategy weights"""
        weights = {}
        
        # Base weights for preferred strategies
        base_weight = 1.0 / len(preferred_strategies) if preferred_strategies else 0.0
        
        for strategy in preferred_strategies:
            performance = performance_scores.get(strategy, 0.5)
            # Adjust weight based on performance (0.5x to 1.5x multiplier)
            weight_multiplier = 0.5 + performance
            weights[strategy] = base_weight * weight_multiplier
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_routing_confidence(self, state: TradingSystemState) -> float:
        """Calculate confidence in routing decision"""
        # Factors affecting confidence:
        # 1. Data quality
        # 2. Market regime clarity
        # 3. Recent performance consistency
        
        data_quality = self._assess_data_quality(state)
        regime_clarity = self._assess_regime_clarity(state)
        performance_consistency = self._assess_performance_consistency(state)
        
        # Weighted average
        confidence = (data_quality * 0.4 + regime_clarity * 0.3 + performance_consistency * 0.3)
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _assess_data_quality(self, state: TradingSystemState) -> float:
        """Assess quality of available market data"""
        market_data = state.get("market_data", {})
        
        if not market_data:
            return 0.0
        
        # Check data completeness and freshness
        complete_data_count = 0
        total_symbols = len(market_data)
        
        for symbol, data in market_data.items():
            if all([data.open, data.high, data.low, data.close, data.volume]):
                complete_data_count += 1
        
        return complete_data_count / total_symbols if total_symbols > 0 else 0.0
    
    def _assess_regime_clarity(self, state: TradingSystemState) -> float:
        """Assess clarity of market regime detection"""
        market_regime = state.get("market_regime", MarketRegime.NORMAL)
        
        # Simple heuristic: some regimes are clearer than others
        clarity_scores = {
            MarketRegime.CRISIS: 0.9,
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.LOW_VOLATILITY: 0.8,
            MarketRegime.TRENDING: 0.7,
            MarketRegime.MEAN_REVERTING: 0.7,
            MarketRegime.NEWS_DRIVEN: 0.6,
            MarketRegime.NORMAL: 0.5
        }
        
        return clarity_scores.get(market_regime, 0.5)
    
    def _assess_performance_consistency(self, state: TradingSystemState) -> float:
        """Assess consistency of recent performance"""
        execution_log = state.get("execution_log", [])
        
        if len(execution_log) < 10:
            return 0.5  # Not enough data
        
        recent_executions = execution_log[-20:]
        success_count = sum(1 for ex in recent_executions if "error" not in ex.get("message", "").lower())
        
        return success_count / len(recent_executions)


class WorkflowOrchestrator:
    """
    High-level orchestrator that combines workflow, monitoring, and communication.
    Provides a unified interface for the complete trading system.
    """
    
    def __init__(self):
        self.workflow = LangGraphTradingWorkflow()
        self.message_bus = None
        self.coordinator = None
        self.monitor = None
        self.router = ConditionalRouter()
        
        self.running = False
        self.orchestration_task = None
        
        logger.info("Workflow orchestrator initialized")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize communication infrastructure
            from .communication_protocols import create_message_bus, create_agent_coordinator
            from .workflow_monitoring import create_workflow_monitor, MonitoringLevel
            
            self.message_bus = create_message_bus()
            self.coordinator = create_agent_coordinator(self.message_bus)
            self.monitor = create_workflow_monitor(MonitoringLevel.DETAILED)
            
            # Start services
            await self.message_bus.start()
            await self.monitor.start_monitoring()
            
            # Register agents with coordinator
            for agent_id, agent in self.workflow.agents.items():
                role = self._determine_agent_role(agent_id)
                capabilities = self._get_agent_capabilities(agent_id)
                self.coordinator.register_agent(agent_id, role, capabilities)
            
            logger.info("Workflow orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    def _determine_agent_role(self, agent_id: str) -> AgentRole:
        """Determine agent role based on agent ID"""
        role_mapping = {
            "market_data_ingestor": AgentRole.DATA_PROVIDER,
            "sentiment_agent": AgentRole.DATA_PROVIDER,
            "momentum_agent": AgentRole.SIGNAL_GENERATOR,
            "mean_reversion_agent": AgentRole.SIGNAL_GENERATOR,
            "options_agent": AgentRole.SIGNAL_GENERATOR,
            "portfolio_allocator": AgentRole.PORTFOLIO_MANAGER,
            "risk_manager": AgentRole.RISK_MANAGER,
            "execution_engine": AgentRole.EXECUTION_ENGINE
        }
        
        return role_mapping.get(agent_id, AgentRole.SIGNAL_GENERATOR)
    
    def _get_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Get agent capabilities"""
        capabilities = {
            "market_data_ingestor": {"data_sources": ["polygon", "alpaca"], "symbols": 1000},
            "sentiment_agent": {"news_sources": ["finbert", "gemini"], "languages": ["en"]},
            "momentum_agent": {"indicators": ["ema", "rsi", "macd"], "timeframes": ["1m", "5m", "1h"]},
            "mean_reversion_agent": {"indicators": ["bollinger", "zscore"], "pairs_trading": True},
            "options_agent": {"greeks": True, "iv_analysis": True, "strategies": ["straddle", "strangle"]},
            "portfolio_allocator": {"fusion_methods": ["weighted", "voting"], "explainability": True},
            "risk_manager": {"var_models": ["historical", "parametric"], "limits": True},
            "execution_engine": {"brokers": ["alpaca"], "order_types": ["market", "limit", "stop"]}
        }
        
        return capabilities.get(agent_id, {})
    
    async def start_orchestration(self):
        """Start the orchestration loop"""
        if self.running:
            logger.warning("Orchestration already running")
            return
        
        self.running = True
        self.orchestration_task = asyncio.create_task(self._orchestration_loop())
        logger.info("Workflow orchestration started")
    
    async def stop_orchestration(self):
        """Stop the orchestration"""
        self.running = False
        
        if self.orchestration_task:
            self.orchestration_task.cancel()
            try:
                await self.orchestration_task
            except asyncio.CancelledError:
                pass
        
        # Stop services
        if self.message_bus:
            await self.message_bus.stop()
        if self.monitor:
            await self.monitor.stop_monitoring()
        
        logger.info("Workflow orchestration stopped")
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.running:
            try:
                # Run workflow cycle
                with MonitoredExecution(self.monitor, "workflow_cycle"):
                    result = await self.workflow.run_workflow()
                
                # Analyze results and adjust routing
                routing_decision = self.router.route_based_on_market_regime(result)
                
                # Log cycle completion
                logger.info(f"Workflow cycle completed. Market regime: {routing_decision['market_regime']}")
                
                # Wait before next cycle (configurable)
                await asyncio.sleep(60)  # 1 minute between cycles
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def run_single_cycle(self) -> Dict[str, Any]:
        """Run a single workflow cycle"""
        try:
            with MonitoredExecution(self.monitor, "single_cycle"):
                result = await self.workflow.run_workflow()
            
            routing_decision = self.router.route_based_on_market_regime(result)
            
            return {
                "workflow_result": result,
                "routing_decision": routing_decision,
                "monitoring_data": self.monitor.get_monitoring_dashboard() if self.monitor else {},
                "communication_stats": self.message_bus.get_stats() if self.message_bus else {},
                "agent_status": self.coordinator.get_agent_status() if self.coordinator else {}
            }
            
        except Exception as e:
            logger.error(f"Single cycle execution failed: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "orchestrator": {
                "running": self.running,
                "initialized": all([self.message_bus, self.coordinator, self.monitor])
            },
            "workflow": self.workflow.get_workflow_status(),
            "monitoring": self.monitor.get_monitoring_dashboard() if self.monitor else {},
            "communication": self.message_bus.get_stats() if self.message_bus else {},
            "coordination": self.coordinator.get_agent_status() if self.coordinator else {}
        }


# Factory functions
def create_trading_workflow() -> LangGraphTradingWorkflow:
    """Create and return a new trading workflow instance"""
    return LangGraphTradingWorkflow()


def create_workflow_orchestrator() -> WorkflowOrchestrator:
    """Create and return a new workflow orchestrator instance"""
    return WorkflowOrchestrator()


# Main execution
if __name__ == "__main__":
    async def main():
        # Create and initialize orchestrator
        orchestrator = create_workflow_orchestrator()
        
        try:
            await orchestrator.initialize()
            
            # Run a single cycle for testing
            result = await orchestrator.run_single_cycle()
            
            print("Workflow execution completed successfully!")
            print(f"Market regime detected: {result['routing_decision']['market_regime']}")
            print(f"Preferred strategies: {result['routing_decision']['preferred_strategies']}")
            
            # Print system status
            status = orchestrator.get_system_status()
            print("\nSystem Status:")
            print(json.dumps(status, indent=2, default=str))
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
        finally:
            await orchestrator.stop_orchestration()
    
    asyncio.run(main())