"""
Signal Coordinator for Bloomberg Terminal
Coordinates trading signals between agents and the event system.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
import json

from agents.agent_orchestrator import AgentOrchestrator
from events.event_bus import EventBus, Event, EventType, get_event_bus
from agents.base_agent import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class SignalCoordinator:
    """
    Coordinates trading signals between agents and the broader system.
    
    Responsibilities:
    - Collect signals from agent orchestrator
    - Publish signals to event bus
    - Handle signal conflicts and prioritization
    - Coordinate with risk management
    - Manage signal lifecycle
    - Monitor signal performance
    """
    
    def __init__(self, orchestrator: AgentOrchestrator, config: Dict[str, Any] = None):
        self.orchestrator = orchestrator
        
        default_config = {
            'signal_generation_interval': 30,  # seconds
            'max_signals_per_interval': 20,
            'signal_priority_weights': {
                'STRONG_BUY': 10,
                'STRONG_SELL': 10,
                'BUY': 5,
                'SELL': 5,
                'HOLD': 1
            },
            'min_signal_confidence': 0.5,
            'risk_override_enabled': True,
            'signal_deduplication_window': 300,  # 5 minutes
            'performance_tracking_enabled': True
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.event_bus: EventBus = get_event_bus()
        
        # State tracking
        self.active_signals: Dict[str, List[TradingSignal]] = {}  # symbol -> signals
        self.signal_history: List[TradingSignal] = []
        self.performance_metrics: Dict[str, Any] = {
            'signals_generated': 0,
            'signals_published': 0,
            'signals_deduplicated': 0,
            'risk_overrides': 0,
            'last_update': datetime.now()
        }
        
        self.is_running = False
        
    async def initialize(self) -> None:
        """Initialize the signal coordinator."""
        try:
            logger.info("Initializing Signal Coordinator")
            
            # Initialize event bus if needed
            if not self.event_bus.is_running:
                await self.event_bus.start()
            
            # Subscribe to relevant events
            await self._setup_event_subscriptions()
            
            # Initialize active signals tracking
            for symbol in self.orchestrator.symbols:
                self.active_signals[symbol] = []
            
            logger.info("Signal Coordinator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Signal Coordinator: {e}")
            raise
    
    async def start(self) -> None:
        """Start the signal coordinator."""
        if not self.event_bus.is_running:
            await self.initialize()
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._signal_generation_loop())
        asyncio.create_task(self._signal_cleanup_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("Signal Coordinator started")
    
    async def stop(self) -> None:
        """Stop the signal coordinator."""
        self.is_running = False
        logger.info("Signal Coordinator stopped")
    
    async def generate_and_publish_signals(self) -> Dict[str, Any]:
        """
        Generate signals from orchestrator and publish to event bus.
        
        Returns:
            Summary of signal generation results
        """
        try:
            results = {
                'symbols_processed': 0,
                'signals_generated': 0,
                'signals_published': 0,
                'signals_deduplicated': 0,
                'errors': []
            }
            
            # Generate signals for all symbols
            for symbol in self.orchestrator.symbols:
                try:
                    # Generate consensus signal
                    signal = await self.orchestrator.generate_consensus_signal(symbol)
                    
                    if signal:
                        results['signals_generated'] += 1
                        
                        # Check if signal should be published
                        should_publish = await self._should_publish_signal(signal)
                        
                        if should_publish:
                            # Publish signal to event bus
                            success = await self._publish_signal(signal)
                            
                            if success:
                                results['signals_published'] += 1
                                
                                # Store in active signals
                                await self._store_active_signal(signal)
                            else:
                                results['errors'].append(f"Failed to publish signal for {symbol}")
                        else:
                            results['signals_deduplicated'] += 1
                
                except Exception as e:
                    error_msg = f"Error processing {symbol}: {e}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                
                results['symbols_processed'] += 1
            
            # Update performance metrics
            self.performance_metrics['signals_generated'] += results['signals_generated']
            self.performance_metrics['signals_published'] += results['signals_published']
            self.performance_metrics['signals_deduplicated'] += results['signals_deduplicated']
            self.performance_metrics['last_update'] = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in signal generation: {e}")
            return {'error': str(e)}
    
    async def handle_risk_override(self, symbol: str, risk_action: str) -> bool:
        """
        Handle risk management override signals.
        
        Args:
            symbol: Trading symbol
            risk_action: Risk action (REDUCE, STOP, EMERGENCY_CLOSE)
            
        Returns:
            Success status
        """
        try:
            # Create risk override event
            risk_data = {
                'symbol': symbol,
                'action': risk_action,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'SignalCoordinator'
            }
            
            # Publish risk alert
            success = await self.event_bus.publish_risk_alert(risk_data, 'SignalCoordinator')
            
            if success:
                self.performance_metrics['risk_overrides'] += 1
                
                # Cancel conflicting signals
                await self._cancel_conflicting_signals(symbol, risk_action)
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling risk override: {e}")
            return False
    
    async def get_active_signals_for_symbol(self, symbol: str) -> List[TradingSignal]:
        """Get currently active signals for a symbol."""
        return self.active_signals.get(symbol, [])
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get signal coordinator performance metrics."""
        return {
            **self.performance_metrics,
            'active_signals_count': sum(len(signals) for signals in self.active_signals.values()),
            'orchestrator_metrics': await self.orchestrator.get_agent_performance_summary()
        }
    
    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for coordination."""
        try:
            # Subscribe to risk alerts
            await self.event_bus.subscribe(
                [EventType.RISK_ALERT],
                self._handle_risk_alert_event
            )
            
            # Subscribe to order events for signal performance tracking
            await self.event_bus.subscribe(
                [EventType.ORDER_FILLED, EventType.ORDER_REJECTED],
                self._handle_order_event
            )
            
            logger.info("Event subscriptions setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup event subscriptions: {e}")
            raise
    
    async def _handle_risk_alert_event(self, event: Event) -> None:
        """Handle incoming risk alert events."""
        try:
            risk_data = event.data
            symbol = risk_data.get('symbol')
            action = risk_data.get('action')
            
            if symbol and action:
                logger.info(f"Processing risk alert for {symbol}: {action}")
                await self.handle_risk_override(symbol, action)
        
        except Exception as e:
            logger.error(f"Error handling risk alert event: {e}")
    
    async def _handle_order_event(self, event: Event) -> None:
        """Handle order events for signal performance tracking."""
        try:
            if not self.config['performance_tracking_enabled']:
                return
            
            order_data = event.data
            symbol = order_data.get('symbol')
            
            if symbol and symbol in self.active_signals:
                # Update signal performance based on order outcome
                # This is a simplified version - full implementation would track
                # signal performance over time
                logger.debug(f"Order event for {symbol}: {event.event_type.value}")
        
        except Exception as e:
            logger.error(f"Error handling order event: {e}")
    
    async def _should_publish_signal(self, signal: TradingSignal) -> bool:
        """
        Determine if a signal should be published based on various criteria.
        
        Args:
            signal: Trading signal to evaluate
            
        Returns:
            Whether signal should be published
        """
        try:
            # Check minimum confidence threshold
            if signal.confidence < self.config['min_signal_confidence']:
                return False
            
            # Check for recent duplicate signals
            if await self._is_duplicate_signal(signal):
                return False
            
            # Check risk override conditions
            if self.config['risk_override_enabled']:
                if await self._check_risk_restrictions(signal):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if signal should be published: {e}")
            return False
    
    async def _is_duplicate_signal(self, signal: TradingSignal) -> bool:
        """Check if signal is a duplicate of recent signals."""
        try:
            symbol = signal.symbol
            recent_signals = self.active_signals.get(symbol, [])
            
            # Check for signals within deduplication window
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                seconds=self.config['signal_deduplication_window']
            )
            
            for existing_signal in recent_signals:
                if (existing_signal.timestamp > cutoff_time and
                    existing_signal.signal_type == signal.signal_type and
                    abs(existing_signal.confidence - signal.confidence) < 0.1):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for duplicate signal: {e}")
            return False
    
    async def _check_risk_restrictions(self, signal: TradingSignal) -> bool:
        """Check if signal violates risk restrictions."""
        try:
            # This would integrate with the risk management system
            # For now, implement basic checks
            
            symbol = signal.symbol
            
            # Check if symbol is under risk restriction
            # (This would query the risk management system)
            
            # Placeholder implementation
            return False
            
        except Exception as e:
            logger.error(f"Error checking risk restrictions: {e}")
            return False
    
    async def _publish_signal(self, signal: TradingSignal) -> bool:
        """Publish trading signal to event bus."""
        try:
            # Convert signal to event data
            signal_data = {
                'signal_id': signal.id,
                'agent_name': signal.agent_name,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'strength': signal.strength,
                'target_price': signal.target_price,
                'stop_loss': signal.stop_loss,
                'prediction_horizon': signal.prediction_horizon,
                'risk_score': signal.risk_score,
                'expected_return': signal.expected_return,
                'reasoning': signal.reasoning,
                'timestamp': signal.timestamp.isoformat()
            }
            
            # Publish to event bus
            success = await self.event_bus.publish_trading_signal(signal_data, 'SignalCoordinator')
            
            if success:
                logger.info(f"Published signal: {signal.signal_type.value} for {signal.symbol} (conf: {signal.confidence:.2f})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error publishing signal: {e}")
            return False
    
    async def _store_active_signal(self, signal: TradingSignal) -> None:
        """Store signal in active signals tracking."""
        try:
            symbol = signal.symbol
            
            if symbol not in self.active_signals:
                self.active_signals[symbol] = []
            
            self.active_signals[symbol].append(signal)
            
            # Keep only recent signals
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            self.active_signals[symbol] = [
                s for s in self.active_signals[symbol]
                if s.timestamp > cutoff_time
            ]
            
            # Store in history
            self.signal_history.append(signal)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
        
        except Exception as e:
            logger.error(f"Error storing active signal: {e}")
    
    async def _cancel_conflicting_signals(self, symbol: str, risk_action: str) -> None:
        """Cancel signals that conflict with risk action."""
        try:
            if symbol not in self.active_signals:
                return
            
            # Determine conflicting signal types based on risk action
            conflicting_types = set()
            
            if risk_action in ['REDUCE', 'STOP']:
                conflicting_types.update([SignalType.BUY, SignalType.STRONG_BUY])
            elif risk_action == 'EMERGENCY_CLOSE':
                conflicting_types.update([SignalType.BUY, SignalType.STRONG_BUY, SignalType.HOLD])
            
            # Remove conflicting signals
            original_count = len(self.active_signals[symbol])
            self.active_signals[symbol] = [
                signal for signal in self.active_signals[symbol]
                if signal.signal_type not in conflicting_types
            ]
            
            removed_count = original_count - len(self.active_signals[symbol])
            if removed_count > 0:
                logger.info(f"Cancelled {removed_count} conflicting signals for {symbol}")
        
        except Exception as e:
            logger.error(f"Error cancelling conflicting signals: {e}")
    
    async def _signal_generation_loop(self) -> None:
        """Background loop for regular signal generation."""
        while self.is_running:
            try:
                results = await self.generate_and_publish_signals()
                
                if results.get('errors'):
                    logger.warning(f"Signal generation had {len(results['errors'])} errors")
                
                await asyncio.sleep(self.config['signal_generation_interval'])
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _signal_cleanup_loop(self) -> None:
        """Background loop for cleaning up expired signals."""
        while self.is_running:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                
                for symbol in self.active_signals:
                    original_count = len(self.active_signals[symbol])
                    self.active_signals[symbol] = [
                        signal for signal in self.active_signals[symbol]
                        if signal.timestamp > cutoff_time
                    ]
                    
                    removed_count = original_count - len(self.active_signals[symbol])
                    if removed_count > 0:
                        logger.debug(f"Cleaned up {removed_count} expired signals for {symbol}")
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in signal cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self) -> None:
        """Background loop for performance monitoring."""
        while self.is_running:
            try:
                # Store performance metrics in event bus
                metrics_data = await self.get_performance_metrics()
                
                # This could be enhanced to publish performance events
                logger.debug(f"Performance metrics: {metrics_data['signals_generated']} signals generated")
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)


# Convenience function
def create_signal_coordinator(orchestrator: AgentOrchestrator, **kwargs) -> SignalCoordinator:
    """Create a signal coordinator with default configuration."""
    return SignalCoordinator(orchestrator, kwargs)