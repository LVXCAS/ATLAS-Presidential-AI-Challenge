"""
Orchestration Service for Bloomberg Terminal
Main service that coordinates all components of the trading system.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import signal
import sys

from agents.agent_orchestrator import AgentOrchestrator
from orchestration.signal_coordinator import SignalCoordinator
from events.event_bus import initialize_event_bus, EventBus
from services.market_data_service import MarketDataService
from services.order_service import OrderService

logger = logging.getLogger(__name__)


class OrchestrationService:
    """
    Main orchestration service that coordinates all trading system components.
    
    Responsibilities:
    - Initialize and manage all service components
    - Coordinate between market data, agents, and order execution
    - Handle system lifecycle and graceful shutdown
    - Monitor system health and performance
    - Manage configuration and environment
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            # System configuration
            'environment': os.getenv('TRADING_ENV', 'development'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            
            # Trading symbols
            'symbols': [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
                'NVDA', 'META', 'NFLX', 'AMD', 'INTC',
                'SPY', 'QQQ', 'VTI', 'BTC-USD', 'ETH-USD'
            ],
            
            # Redis configuration
            'redis': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
                'db': int(os.getenv('REDIS_DB', 0)),
                'decode_responses': True,
                'socket_keepalive': True,
                'retry_on_timeout': True,
                'connection_pool_kwargs': {
                    'max_connections': 50,
                    'retry_on_timeout': True
                }
            },
            
            # Market data configuration
            'market_data': {
                'provider': 'alpaca',
                'api_key': os.getenv('ALPACA_API_KEY'),
                'api_secret': os.getenv('ALPACA_API_SECRET'),
                'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                'websocket_url': os.getenv('ALPACA_WS_URL', 'wss://paper-api.alpaca.markets/stream'),
                'enable_streaming': True,
                'cache_ttl': 60
            },
            
            # Agent orchestrator configuration
            'orchestrator': {
                'agent_types': ['momentum', 'mean_reversion', 'sentiment', 'arbitrage', 'volatility', 'risk_manager'],
                'signal_aggregation_method': 'weighted_average',
                'min_confidence_threshold': 0.5,
                'consensus_threshold': 0.6,
                'performance_update_frequency': 3600
            },
            
            # Signal coordinator configuration
            'signal_coordinator': {
                'signal_generation_interval': 30,
                'max_signals_per_interval': 20,
                'min_signal_confidence': 0.5,
                'risk_override_enabled': True
            },
            
            # Order service configuration
            'order_service': {
                'paper_trading': True,
                'max_position_size': 0.1,  # 10% of portfolio
                'risk_checks_enabled': True,
                'order_timeout': 300
            },
            
            # Health monitoring
            'health_check_interval': 30,
            'metrics_update_interval': 60,
            'performance_logging_enabled': True
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.symbols = self.config['symbols']
        
        # Core components
        self.event_bus: Optional[EventBus] = None
        self.market_data_service: Optional[MarketDataService] = None
        self.order_service: Optional[OrderService] = None
        self.agent_orchestrator: Optional[AgentOrchestrator] = None
        self.signal_coordinator: Optional[SignalCoordinator] = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        
        # Performance metrics
        self.system_metrics = {
            'uptime_seconds': 0,
            'total_signals_processed': 0,
            'total_orders_executed': 0,
            'system_errors': 0,
            'last_health_check': None
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    async def initialize(self) -> None:
        """Initialize all system components."""
        try:
            logger.info("Initializing Bloomberg Terminal Orchestration Service")
            
            # Setup logging
            self._setup_logging()
            
            # Initialize event bus
            logger.info("Initializing event bus...")
            self.event_bus = await initialize_event_bus(self.config['redis'])
            
            # Initialize market data service
            logger.info("Initializing market data service...")
            self.market_data_service = MarketDataService(self.config['market_data'])
            await self.market_data_service.initialize()
            
            # Initialize order service
            logger.info("Initializing order service...")
            self.order_service = OrderService(self.config['order_service'])
            await self.order_service.initialize()
            
            # Initialize agent orchestrator
            logger.info("Initializing agent orchestrator...")
            self.agent_orchestrator = AgentOrchestrator(self.symbols, self.config['orchestrator'])
            await self.agent_orchestrator.initialize()
            
            # Setup services interconnections
            await self._setup_service_connections()
            
            # Initialize signal coordinator
            logger.info("Initializing signal coordinator...")
            self.signal_coordinator = SignalCoordinator(
                self.agent_orchestrator, 
                self.config['signal_coordinator']
            )
            await self.signal_coordinator.initialize()
            
            self.is_initialized = True
            logger.info("Orchestration Service initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Orchestration Service: {e}")
            raise
    
    async def start(self) -> None:
        """Start all system components."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info("Starting Bloomberg Terminal Trading System")
            
            # Start core services
            await self.market_data_service.start()
            await self.order_service.start()
            await self.agent_orchestrator.start()
            await self.signal_coordinator.start()
            
            # Start system monitoring
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._metrics_update_loop())
            asyncio.create_task(self._performance_logging_loop())
            
            self.is_running = True
            self.startup_time = datetime.now(timezone.utc)
            
            logger.info("Bloomberg Terminal Trading System started successfully")
            logger.info(f"Trading {len(self.symbols)} symbols: {', '.join(self.symbols[:5])}...")
            logger.info(f"Environment: {self.config['environment']}")
            
        except Exception as e:
            logger.error(f"Failed to start Orchestration Service: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop all system components gracefully."""
        logger.info("Stopping Bloomberg Terminal Trading System...")
        
        self.is_running = False
        
        # Stop components in reverse order
        try:
            if self.signal_coordinator:
                await self.signal_coordinator.stop()
            
            if self.agent_orchestrator:
                await self.agent_orchestrator.stop()
            
            if self.order_service:
                await self.order_service.stop()
            
            if self.market_data_service:
                await self.market_data_service.stop()
            
            if self.event_bus:
                await self.event_bus.stop()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Bloomberg Terminal Trading System stopped")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'is_running': self.is_running,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'uptime_seconds': self.system_metrics['uptime_seconds'],
            'environment': self.config['environment'],
            'symbols_count': len(self.symbols),
            'system_metrics': self.system_metrics.copy()
        }
        
        # Get component statuses
        if self.is_running:
            try:
                if self.event_bus:
                    status['event_bus'] = await self.event_bus.get_metrics()
                
                if self.agent_orchestrator:
                    status['agents'] = await self.agent_orchestrator.get_agent_performance_summary()
                
                if self.signal_coordinator:
                    status['signals'] = await self.signal_coordinator.get_performance_metrics()
                
                if self.market_data_service:
                    status['market_data'] = {
                        'connected': True,  # Would check actual connection
                        'symbols_streaming': len(self.symbols)
                    }
                
                if self.order_service:
                    status['orders'] = {
                        'service_running': True,
                        'paper_trading': self.config['order_service']['paper_trading']
                    }
            
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                status['status_error'] = str(e)
        
        return status
    
    async def get_active_signals(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get active trading signals."""
        try:
            if not self.signal_coordinator:
                return {'error': 'Signal coordinator not initialized'}
            
            if symbol:
                signals = await self.signal_coordinator.get_active_signals_for_symbol(symbol)
                return {
                    'symbol': symbol,
                    'signals': [self._serialize_signal(s) for s in signals]
                }
            else:
                all_signals = {}
                for sym in self.symbols:
                    signals = await self.signal_coordinator.get_active_signals_for_symbol(sym)
                    if signals:
                        all_signals[sym] = [self._serialize_signal(s) for s in signals]
                
                return {'signals_by_symbol': all_signals}
        
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return {'error': str(e)}
    
    def _serialize_signal(self, signal) -> Dict[str, Any]:
        """Serialize a trading signal for API response."""
        return {
            'id': signal.id,
            'agent_name': signal.agent_name,
            'symbol': signal.symbol,
            'signal_type': signal.signal_type.value,
            'confidence': signal.confidence,
            'strength': signal.strength,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'timestamp': signal.timestamp.isoformat(),
            'prediction_horizon': signal.prediction_horizon,
            'risk_score': signal.risk_score,
            'expected_return': signal.expected_return
        }
    
    async def _setup_service_connections(self) -> None:
        """Setup connections between services."""
        try:
            # Connect market data service to agents
            for agent in self.agent_orchestrator.agents.values():
                agent.market_data_service = self.market_data_service
            
            # Setup event subscriptions for coordination
            await self._setup_event_subscriptions()
            
            logger.info("Service connections established")
            
        except Exception as e:
            logger.error(f"Failed to setup service connections: {e}")
            raise
    
    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for system coordination."""
        try:
            from events.event_bus import EventType
            
            # Subscribe to system events
            await self.event_bus.subscribe(
                [EventType.EMERGENCY_STOP],
                self._handle_emergency_stop
            )
            
            await self.event_bus.subscribe(
                [EventType.SYSTEM_HEALTH],
                self._handle_system_health_event
            )
            
        except Exception as e:
            logger.error(f"Failed to setup event subscriptions: {e}")
    
    async def _handle_emergency_stop(self, event) -> None:
        """Handle emergency stop events."""
        logger.critical(f"Emergency stop triggered: {event.data}")
        await self.stop()
    
    async def _handle_system_health_event(self, event) -> None:
        """Handle system health events."""
        health_data = event.data
        logger.debug(f"System health update: {health_data}")
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self.is_running:
            try:
                # Update uptime
                if self.startup_time:
                    self.system_metrics['uptime_seconds'] = (
                        datetime.now(timezone.utc) - self.startup_time
                    ).total_seconds()
                
                # Perform health checks
                await self._perform_health_checks()
                
                self.system_metrics['last_health_check'] = datetime.now()
                
                await asyncio.sleep(self.config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                self.system_metrics['system_errors'] += 1
                await asyncio.sleep(10)
    
    async def _perform_health_checks(self) -> None:
        """Perform system health checks."""
        try:
            # Check Redis connection
            if self.event_bus and self.event_bus.redis_client:
                await self.event_bus.redis_client.ping()
            
            # Check if all components are running
            if not all([
                self.market_data_service and self.market_data_service.is_running,
                self.agent_orchestrator and self.agent_orchestrator.is_running,
                self.signal_coordinator and self.signal_coordinator.is_running
            ]):
                logger.warning("Some components are not running")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.system_metrics['system_errors'] += 1
    
    async def _metrics_update_loop(self) -> None:
        """Background metrics update loop."""
        while self.is_running:
            try:
                # Collect metrics from all components
                await self._collect_system_metrics()
                
                await asyncio.sleep(self.config['metrics_update_interval'])
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self) -> None:
        """Collect metrics from all system components."""
        try:
            # Get signal coordinator metrics
            if self.signal_coordinator:
                signal_metrics = await self.signal_coordinator.get_performance_metrics()
                self.system_metrics['total_signals_processed'] = signal_metrics.get('signals_published', 0)
            
            # Additional metrics collection would go here
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _performance_logging_loop(self) -> None:
        """Background performance logging loop."""
        while self.is_running:
            try:
                if self.config['performance_logging_enabled']:
                    status = await self.get_system_status()
                    logger.info(
                        f"System Performance - Uptime: {status['uptime_seconds']:.0f}s, "
                        f"Signals: {status['system_metrics']['total_signals_processed']}, "
                        f"Errors: {status['system_metrics']['system_errors']}"
                    )
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance logging: {e}")
                await asyncio.sleep(60)
    
    def _setup_logging(self) -> None:
        """Setup system logging configuration."""
        log_level = getattr(logging, self.config['log_level'].upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Set specific loggers
        logging.getLogger('agents').setLevel(log_level)
        logging.getLogger('events').setLevel(log_level)
        logging.getLogger('orchestration').setLevel(log_level)
        logging.getLogger('services').setLevel(log_level)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


# Main function for running the service
async def main():
    """Main entry point for the orchestration service."""
    service = OrchestrationService()
    
    try:
        await service.start()
        
        # Keep running until shutdown
        while service.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())