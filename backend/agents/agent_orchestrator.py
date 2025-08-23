"""
Agent Orchestrator for Bloomberg Terminal
Coordinates and manages multiple specialized trading agents using Mixture of Agent Experts (MoAE).
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import pandas as pd
from collections import defaultdict

from agents.base_agent import BaseAgent, TradingSignal, SignalType, AgentStatus
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.sentiment_agent import SentimentAgent
from agents.arbitrage_agent import ArbitrageAgent
from agents.volatility_agent import VolatilityAgent
from agents.risk_manager_agent import RiskManagerAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Mixture of Agent Experts (MoAE) orchestrator that:
    - Manages multiple specialized trading agents
    - Aggregates and ranks signals from all agents
    - Dynamically weights agent contributions based on performance
    - Handles agent lifecycle and coordination
    - Provides unified signal generation interface
    - Manages agent conflicts and consensus
    - Tracks performance and adjusts weights
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        self.symbols = symbols
        
        default_config = {
            'agent_types': ['momentum', 'mean_reversion', 'sentiment', 'arbitrage', 'volatility', 'risk_manager'],
            'max_signals_per_symbol': 5,
            'signal_aggregation_method': 'weighted_average',  # weighted_average, majority_vote, best_agent
            'performance_lookback_days': 30,
            'min_confidence_threshold': 0.5,
            'consensus_threshold': 0.6,  # Minimum agreement for consensus signals
            'agent_weight_decay': 0.95,  # Daily decay for agent weights
            'initial_agent_weight': 1.0,
            'max_agent_weight': 2.0,
            'min_agent_weight': 0.1,
            'signal_timeout_minutes': 60,  # Signal expiration time
            'enable_agent_competition': True,
            'risk_manager_override': True,  # Risk manager can override other signals
            'performance_update_frequency': 3600,  # Update performance every hour
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_weights: Dict[str, float] = {}
        self.agent_performance: Dict[str, Dict] = {}
        self.signal_history: Dict[str, List[TradingSignal]] = {}
        self.consensus_signals: Dict[str, TradingSignal] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {
            'total_signals_generated': 0,
            'consensus_signals_count': 0,
            'agent_agreement_rate': 0.0,
            'avg_signal_confidence': 0.0,
            'last_performance_update': datetime.now()
        }
        
        self.is_initialized = False
        self.is_running = False
        
    async def initialize(self) -> None:
        """Initialize all agents and orchestrator components."""
        try:
            logger.info("Initializing Agent Orchestrator with MoAE architecture")
            
            # Initialize individual agents
            await self._initialize_agents()
            
            # Initialize performance tracking
            await self._initialize_performance_tracking()
            
            # Initialize signal history
            for symbol in self.symbols:
                self.signal_history[symbol] = []
            
            self.is_initialized = True
            logger.info(f"Agent Orchestrator initialized with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error initializing Agent Orchestrator: {e}")
            raise
    
    async def start(self) -> None:
        """Start the orchestrator and all managed agents."""
        if not self.is_initialized:
            await self.initialize()
        
        self.is_running = True
        logger.info("Agent Orchestrator started")
        
        # Start background tasks
        asyncio.create_task(self._performance_update_loop())
        asyncio.create_task(self._signal_cleanup_loop())
    
    async def stop(self) -> None:
        """Stop the orchestrator and cleanup all agents."""
        self.is_running = False
        
        # Cleanup all agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.cleanup()
                logger.info(f"Cleaned up agent: {agent_name}")
            except Exception as e:
                logger.error(f"Error cleaning up agent {agent_name}: {e}")
        
        logger.info("Agent Orchestrator stopped")
    
    async def generate_consensus_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate consensus signal from all agents for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Consensus trading signal or None
        """
        try:
            if not self.is_running:
                logger.warning("Orchestrator not running")
                return None
            
            # Generate signals from all agents
            agent_signals = await self._collect_agent_signals(symbol)
            
            if not agent_signals:
                return None
            
            # Check for risk manager override
            if self.config['risk_manager_override']:
                risk_signal = await self._check_risk_manager_override(agent_signals)
                if risk_signal:
                    return risk_signal
            
            # Generate consensus signal
            consensus_signal = await self._generate_consensus_signal(symbol, agent_signals)
            
            if consensus_signal:
                self.consensus_signals[symbol] = consensus_signal
                self.performance_metrics['consensus_signals_count'] += 1
                
                # Store in history
                self.signal_history[symbol].append(consensus_signal)
                if len(self.signal_history[symbol]) > 100:
                    self.signal_history[symbol] = self.signal_history[symbol][-100:]
            
            return consensus_signal
            
        except Exception as e:
            logger.error(f"Error generating consensus signal for {symbol}: {e}")
            return None
    
    async def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents."""
        summary = {
            'orchestrator_metrics': self.performance_metrics.copy(),
            'agent_weights': self.agent_weights.copy(),
            'agent_performance': {},
            'agent_status': {}
        }
        
        for agent_name, agent in self.agents.items():
            perf = self.agent_performance.get(agent_name, {})
            summary['agent_performance'][agent_name] = {
                'accuracy': perf.get('accuracy', 0.0),
                'total_signals': perf.get('total_signals', 0),
                'avg_confidence': perf.get('avg_confidence', 0.0),
                'last_signal_time': perf.get('last_signal_time', None)
            }
            
            summary['agent_status'][agent_name] = {
                'status': agent.status.value,
                'last_updated': agent.last_updated.isoformat() if agent.last_updated else None
            }
        
        return summary
    
    async def _initialize_agents(self) -> None:
        """Initialize all specialized agents."""
        try:
            agent_configs = {
                'momentum': {},
                'mean_reversion': {},
                'sentiment': {'enable_social_media': False, 'enable_options_sentiment': False},
                'arbitrage': {'max_pairs': 10},
                'volatility': {'volatility_lookback': 30},
                'risk_manager': {'max_portfolio_var': 0.05}
            }
            
            for agent_type in self.config['agent_types']:
                if agent_type == 'momentum':
                    agent = MomentumAgent(self.symbols, agent_configs.get('momentum', {}))
                elif agent_type == 'mean_reversion':
                    agent = MeanReversionAgent(self.symbols, agent_configs.get('mean_reversion', {}))
                elif agent_type == 'sentiment':
                    agent = SentimentAgent(self.symbols, agent_configs.get('sentiment', {}))
                elif agent_type == 'arbitrage':
                    agent = ArbitrageAgent(self.symbols, agent_configs.get('arbitrage', {}))
                elif agent_type == 'volatility':
                    agent = VolatilityAgent(self.symbols, agent_configs.get('volatility', {}))
                elif agent_type == 'risk_manager':
                    agent = RiskManagerAgent(self.symbols, agent_configs.get('risk_manager', {}))
                else:
                    logger.warning(f"Unknown agent type: {agent_type}")
                    continue
                
                # Initialize agent
                await agent.initialize()
                
                self.agents[agent_type] = agent
                self.agent_weights[agent_type] = self.config['initial_agent_weight']
                
                logger.info(f"Initialized {agent_type} agent")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    async def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking for all agents."""
        for agent_name in self.agents.keys():
            self.agent_performance[agent_name] = {
                'accuracy': 0.0,
                'total_signals': 0,
                'correct_predictions': 0,
                'avg_confidence': 0.0,
                'last_signal_time': None,
                'weight_history': [self.config['initial_agent_weight']],
                'performance_trend': 0.0
            }
    
    async def _collect_agent_signals(self, symbol: str) -> Dict[str, TradingSignal]:
        """Collect signals from all agents for a symbol."""
        agent_signals = {}
        
        # Collect signals concurrently
        tasks = []
        for agent_name, agent in self.agents.items():
            task = asyncio.create_task(self._get_agent_signal(agent_name, agent, symbol))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            agent_name = list(self.agents.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error getting signal from {agent_name}: {result}")
                continue
            
            if result is not None:
                agent_signals[agent_name] = result
                
                # Update performance tracking
                perf = self.agent_performance[agent_name]
                perf['total_signals'] += 1
                perf['last_signal_time'] = datetime.now()
                
                # Update average confidence
                current_avg = perf['avg_confidence']
                total_signals = perf['total_signals']
                perf['avg_confidence'] = ((current_avg * (total_signals - 1)) + result.confidence) / total_signals
        
        self.performance_metrics['total_signals_generated'] += len(agent_signals)
        
        return agent_signals
    
    async def _get_agent_signal(self, agent_name: str, agent: BaseAgent, symbol: str) -> Optional[TradingSignal]:
        """Get signal from a specific agent with error handling."""
        try:
            signal = await agent.generate_signal(symbol)
            if signal and signal.confidence >= self.config['min_confidence_threshold']:
                return signal
            return None
        except Exception as e:
            logger.error(f"Error getting signal from {agent_name} for {symbol}: {e}")
            return None
    
    async def _check_risk_manager_override(self, agent_signals: Dict[str, TradingSignal]) -> Optional[TradingSignal]:
        """Check if risk manager should override other signals."""
        if 'risk_manager' not in agent_signals:
            return None
        
        risk_signal = agent_signals['risk_manager']
        
        # Risk manager override conditions
        if (risk_signal.confidence > 0.8 and 
            risk_signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]):
            
            logger.info(f"Risk manager override: {risk_signal.signal_type.value} for {risk_signal.symbol}")
            return risk_signal
        
        return None
    
    async def _generate_consensus_signal(
        self, 
        symbol: str, 
        agent_signals: Dict[str, TradingSignal]
    ) -> Optional[TradingSignal]:
        """Generate consensus signal from multiple agent signals."""
        try:
            if len(agent_signals) < 2:
                # If only one signal, return it if confidence is high enough
                if len(agent_signals) == 1:
                    signal = list(agent_signals.values())[0]
                    if signal.confidence >= 0.7:
                        return signal
                return None
            
            method = self.config['signal_aggregation_method']
            
            if method == 'weighted_average':
                consensus = await self._weighted_average_consensus(symbol, agent_signals)
            elif method == 'majority_vote':
                consensus = await self._majority_vote_consensus(symbol, agent_signals)
            elif method == 'best_agent':
                consensus = await self._best_agent_consensus(symbol, agent_signals)
            else:
                logger.error(f"Unknown aggregation method: {method}")
                return None
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error generating consensus signal: {e}")
            return None
    
    async def _weighted_average_consensus(
        self, 
        symbol: str, 
        agent_signals: Dict[str, TradingSignal]
    ) -> Optional[TradingSignal]:
        """Generate consensus using weighted average of agent signals."""
        try:
            # Calculate weighted signal strength and direction
            total_weight = 0.0
            weighted_strength = 0.0
            weighted_confidence = 0.0
            signal_directions = defaultdict(float)
            
            for agent_name, signal in agent_signals.items():
                agent_weight = self.agent_weights.get(agent_name, 1.0)
                total_weight += agent_weight
                
                weighted_strength += signal.strength * agent_weight
                weighted_confidence += signal.confidence * agent_weight
                
                # Weight signal directions
                signal_directions[signal.signal_type] += agent_weight
            
            if total_weight == 0:
                return None
            
            # Normalize weights
            avg_strength = weighted_strength / total_weight
            avg_confidence = weighted_confidence / total_weight
            
            # Determine consensus signal type
            consensus_signal_type = max(signal_directions.items(), key=lambda x: x[1])[0]
            consensus_weight = signal_directions[consensus_signal_type] / total_weight
            
            # Check consensus threshold
            if consensus_weight < self.config['consensus_threshold']:
                return None
            
            # Calculate target price and stop loss (weighted average)
            total_target_weight = 0.0
            weighted_target_price = 0.0
            weighted_stop_loss = 0.0
            
            for agent_name, signal in agent_signals.items():
                if signal.signal_type == consensus_signal_type:
                    agent_weight = self.agent_weights.get(agent_name, 1.0)
                    total_target_weight += agent_weight
                    
                    if signal.target_price:
                        weighted_target_price += signal.target_price * agent_weight
                    if signal.stop_loss:
                        weighted_stop_loss += signal.stop_loss * agent_weight
            
            target_price = weighted_target_price / total_target_weight if total_target_weight > 0 else None
            stop_loss = weighted_stop_loss / total_target_weight if total_target_weight > 0 else None
            
            # Aggregate reasoning
            reasoning = {
                'consensus_method': 'weighted_average',
                'agent_contributions': {},
                'consensus_weight': consensus_weight,
                'contributing_agents': len(agent_signals)
            }
            
            for agent_name, signal in agent_signals.items():
                reasoning['agent_contributions'][agent_name] = {
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'strength': signal.strength,
                    'weight': self.agent_weights.get(agent_name, 1.0)
                }
            
            # Create consensus signal
            consensus_signal = TradingSignal(
                id=str(uuid.uuid4()),
                agent_name="AgentOrchestrator",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type=consensus_signal_type,
                confidence=min(avg_confidence * consensus_weight, 0.95),
                strength=avg_strength,
                reasoning=reasoning,
                features_used={},
                prediction_horizon=max([s.prediction_horizon for s in agent_signals.values()], default=60),
                target_price=target_price,
                stop_loss=stop_loss,
                risk_score=sum([s.risk_score for s in agent_signals.values()]) / len(agent_signals),
                expected_return=sum([s.expected_return for s in agent_signals.values()]) / len(agent_signals)
            )
            
            # Update agreement rate
            agreement_rate = consensus_weight
            current_avg = self.performance_metrics['agent_agreement_rate']
            total_consensus = self.performance_metrics['consensus_signals_count']
            self.performance_metrics['agent_agreement_rate'] = ((current_avg * total_consensus) + agreement_rate) / (total_consensus + 1)
            
            return consensus_signal
            
        except Exception as e:
            logger.error(f"Error in weighted average consensus: {e}")
            return None
    
    async def _majority_vote_consensus(
        self, 
        symbol: str, 
        agent_signals: Dict[str, TradingSignal]
    ) -> Optional[TradingSignal]:
        """Generate consensus using majority vote."""
        try:
            signal_votes = defaultdict(list)
            
            # Collect votes
            for agent_name, signal in agent_signals.items():
                signal_votes[signal.signal_type].append((agent_name, signal))
            
            # Find majority
            majority_signal_type = max(signal_votes.items(), key=lambda x: len(x[1]))[0]
            majority_signals = signal_votes[majority_signal_type]
            
            # Check if true majority
            if len(majority_signals) <= len(agent_signals) / 2:
                return None
            
            # Calculate average metrics from majority
            avg_confidence = np.mean([s[1].confidence for s in majority_signals])
            avg_strength = np.mean([s[1].strength for s in majority_signals])
            
            # Use signal with highest confidence as base
            best_signal = max(majority_signals, key=lambda x: x[1].confidence)[1]
            
            # Create consensus signal
            reasoning = {
                'consensus_method': 'majority_vote',
                'majority_count': len(majority_signals),
                'total_agents': len(agent_signals),
                'majority_agents': [s[0] for s in majority_signals]
            }
            
            consensus_signal = TradingSignal(
                id=str(uuid.uuid4()),
                agent_name="AgentOrchestrator",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type=majority_signal_type,
                confidence=avg_confidence,
                strength=avg_strength,
                reasoning=reasoning,
                features_used=best_signal.features_used,
                prediction_horizon=best_signal.prediction_horizon,
                target_price=best_signal.target_price,
                stop_loss=best_signal.stop_loss,
                risk_score=best_signal.risk_score,
                expected_return=best_signal.expected_return
            )
            
            return consensus_signal
            
        except Exception as e:
            logger.error(f"Error in majority vote consensus: {e}")
            return None
    
    async def _best_agent_consensus(
        self, 
        symbol: str, 
        agent_signals: Dict[str, TradingSignal]
    ) -> Optional[TradingSignal]:
        """Use signal from best performing agent."""
        try:
            # Find agent with highest weight (best performance)
            best_agent = max(self.agent_weights.items(), key=lambda x: x[1])[0]
            
            if best_agent in agent_signals:
                signal = agent_signals[best_agent]
                
                # Add orchestrator reasoning
                signal.reasoning = signal.reasoning or {}
                signal.reasoning['consensus_method'] = 'best_agent'
                signal.reasoning['selected_agent'] = best_agent
                signal.reasoning['agent_weight'] = self.agent_weights[best_agent]
                
                return signal
            
            # Fallback to highest confidence signal
            best_signal = max(agent_signals.values(), key=lambda s: s.confidence)
            return best_signal
            
        except Exception as e:
            logger.error(f"Error in best agent consensus: {e}")
            return None
    
    async def _performance_update_loop(self) -> None:
        """Background loop to update agent performance metrics."""
        while self.is_running:
            try:
                await self._update_agent_performance()
                await asyncio.sleep(self.config['performance_update_frequency'])
            except Exception as e:
                logger.error(f"Error in performance update loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _signal_cleanup_loop(self) -> None:
        """Background loop to cleanup expired signals."""
        while self.is_running:
            try:
                await self._cleanup_expired_signals()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Error in signal cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_agent_performance(self) -> None:
        """Update agent performance weights based on historical accuracy."""
        try:
            for agent_name in self.agents.keys():
                perf = self.agent_performance[agent_name]
                
                # Calculate accuracy (simplified - would use actual trade outcomes)
                total_signals = perf['total_signals']
                if total_signals > 10:  # Need minimum signals for meaningful performance
                    # Simulate accuracy based on confidence levels (placeholder)
                    avg_confidence = perf['avg_confidence']
                    simulated_accuracy = min(avg_confidence + np.random.normal(0, 0.1), 1.0)
                    perf['accuracy'] = max(simulated_accuracy, 0.0)
                    
                    # Update agent weight based on accuracy
                    current_weight = self.agent_weights[agent_name]
                    
                    if perf['accuracy'] > 0.6:  # Good performance
                        new_weight = min(current_weight * 1.05, self.config['max_agent_weight'])
                    elif perf['accuracy'] < 0.4:  # Poor performance
                        new_weight = max(current_weight * 0.95, self.config['min_agent_weight'])
                    else:
                        new_weight = current_weight * self.config['agent_weight_decay']  # Neutral decay
                    
                    self.agent_weights[agent_name] = new_weight
                    perf['weight_history'].append(new_weight)
                    
                    # Keep limited history
                    if len(perf['weight_history']) > 100:
                        perf['weight_history'] = perf['weight_history'][-100:]
                    
                    # Calculate performance trend
                    if len(perf['weight_history']) > 5:
                        recent_trend = np.mean(perf['weight_history'][-5:])
                        older_trend = np.mean(perf['weight_history'][-10:-5])
                        perf['performance_trend'] = (recent_trend - older_trend) / older_trend if older_trend > 0 else 0.0
            
            self.performance_metrics['last_performance_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating agent performance: {e}")
    
    async def _cleanup_expired_signals(self) -> None:
        """Remove expired signals from history."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=self.config['signal_timeout_minutes'])
            
            for symbol in self.signal_history:
                original_count = len(self.signal_history[symbol])
                self.signal_history[symbol] = [
                    signal for signal in self.signal_history[symbol]
                    if signal.timestamp > cutoff_time
                ]
                
                removed_count = original_count - len(self.signal_history[symbol])
                if removed_count > 0:
                    logger.debug(f"Cleaned up {removed_count} expired signals for {symbol}")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired signals: {e}")


# Convenience function for creating orchestrator
def create_agent_orchestrator(symbols: List[str], **kwargs) -> AgentOrchestrator:
    """Create an agent orchestrator with default configuration."""
    return AgentOrchestrator(symbols, kwargs)