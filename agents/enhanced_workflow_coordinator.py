"""
Enhanced Workflow Coordinator

This module integrates the enhanced communication system (Kafka + Redis)
with the existing LangGraph workflow, providing high-throughput coordination,
load balancing, and failover capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
import uuid

from .enhanced_communication import (
    KafkaMessageBus, RedisStateManager, LoadBalancer, FailoverManager,
    NegotiationProtocol, AgentInstance, LoadBalancingStrategy, FailoverStrategy,
    create_kafka_message_bus, create_redis_state_manager, create_load_balancer,
    create_failover_manager, create_negotiation_protocol
)
from .communication_protocols import (
    Message, MessageType, MessagePriority, AgentRole
)
from .langgraph_workflow import TradingSystemState, MarketRegime, WorkflowPhase

logger = logging.getLogger(__name__)


@dataclass
class CoordinationConfig:
    """Configuration for enhanced coordination system"""
    kafka_servers: List[str] = None
    redis_config: Dict[str, Any] = None
    use_redis_sentinel: bool = False
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED
    failover_strategy: FailoverStrategy = FailoverStrategy.CIRCUIT_BREAKER
    enable_negotiation: bool = True
    heartbeat_interval: int = 30  # seconds
    coordination_timeout: int = 60  # seconds
    
    def __post_init__(self):
        if self.kafka_servers is None:
            self.kafka_servers = ['localhost:9092']
        if self.redis_config is None:
            self.redis_config = {'host': 'localhost', 'port': 6379, 'db': 0}


class EnhancedWorkflowCoordinator:
    """
    Enhanced workflow coordinator that integrates Kafka, Redis, load balancing,
    and failover mechanisms with the LangGraph trading system.
    """
    
    def __init__(self, config: CoordinationConfig = None):
        self.config = config or CoordinationConfig()
        
        # Initialize communication components
        self.kafka_bus: Optional[KafkaMessageBus] = None
        self.redis_state: Optional[RedisStateManager] = None
        self.load_balancer: Optional[LoadBalancer] = None
        self.failover_manager: Optional[FailoverManager] = None
        self.negotiation_protocol: Optional[NegotiationProtocol] = None
        
        # Agent registry
        self.registered_agents: Dict[str, AgentInstance] = {}
        self.agent_capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Workflow state
        self.current_phase: WorkflowPhase = WorkflowPhase.DATA_INGESTION
        self.market_regime: MarketRegime = MarketRegime.NORMAL
        self.system_state: Dict[str, Any] = {}
        
        # Coordination sessions
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            "coordinations_initiated": 0,
            "coordinations_completed": 0,
            "coordinations_failed": 0,
            "agent_failures": 0,
            "failovers_triggered": 0,
            "negotiations_completed": 0
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info("Enhanced workflow coordinator initialized")
    
    async def start(self):
        """Start the enhanced coordination system"""
        try:
            # Initialize Kafka message bus
            self.kafka_bus = create_kafka_message_bus(
                bootstrap_servers=self.config.kafka_servers,
                config={
                    'producer': {'acks': 'all', 'retries': 3},
                    'consumer': {'group_id': 'trading_system_coordinator'}
                }
            )
            await self.kafka_bus.start()
            
            # Initialize Redis state manager
            self.redis_state = create_redis_state_manager(
                redis_config=self.config.redis_config,
                use_sentinel=self.config.use_redis_sentinel
            )
            
            # Initialize load balancer
            self.load_balancer = create_load_balancer(
                strategy=self.config.load_balancing_strategy
            )
            
            # Initialize failover manager
            self.failover_manager = create_failover_manager(
                strategy=self.config.failover_strategy
            )
            
            # Initialize negotiation protocol
            if self.config.enable_negotiation:
                self.negotiation_protocol = create_negotiation_protocol(self.redis_state)
                self._setup_negotiation_handlers()
            
            # Setup message handlers
            self._setup_message_handlers()
            
            # Start background tasks
            self.running = True
            self._start_background_tasks()
            
            logger.info("Enhanced coordination system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start enhanced coordination system: {e}")
            raise
    
    async def stop(self):
        """Stop the enhanced coordination system"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Stop Kafka bus
        if self.kafka_bus:
            await self.kafka_bus.stop()
        
        logger.info("Enhanced coordination system stopped")
    
    def _setup_message_handlers(self):
        """Setup Kafka message handlers"""
        if not self.kafka_bus:
            return
        
        # Subscribe to coordination topics
        self.kafka_bus.subscribe('agent_coordination', self._handle_coordination_message)
        self.kafka_bus.subscribe('agent_heartbeat', self._handle_heartbeat_message)
        self.kafka_bus.subscribe('signal_fusion', self._handle_signal_fusion_message)
        self.kafka_bus.subscribe('risk_alerts', self._handle_risk_alert_message)
        self.kafka_bus.subscribe('system_events', self._handle_system_event_message)
    
    def _setup_negotiation_handlers(self):
        """Setup negotiation protocol handlers"""
        if not self.negotiation_protocol:
            return
        
        self.negotiation_protocol.register_negotiation_handler(
            'signal_fusion', self._handle_signal_fusion_negotiation
        )
        self.negotiation_protocol.register_negotiation_handler(
            'resource_allocation', self._handle_resource_allocation_negotiation
        )
        self.negotiation_protocol.register_negotiation_handler(
            'risk_consensus', self._handle_risk_consensus_negotiation
        )
    
    def _start_background_tasks(self):
        """Start background coordination tasks"""
        self.background_tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._coordination_monitor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._failover_monitor())
        ]
    
    async def register_agent(self, 
                           agent_id: str, 
                           role: AgentRole, 
                           host: str = "localhost", 
                           port: int = 8000,
                           capabilities: Dict[str, Any] = None) -> bool:
        """Register an agent with the coordination system"""
        try:
            instance = AgentInstance(
                agent_id=agent_id,
                role=role,
                host=host,
                port=port,
                capabilities=capabilities or {}
            )
            
            # Register with load balancer
            self.load_balancer.register_agent_instance(instance)
            
            # Store in local registry
            self.registered_agents[agent_id] = instance
            self.agent_capabilities[agent_id] = capabilities or {}
            
            # Store in Redis for persistence
            await self.redis_state.set_agent_state(agent_id, {
                'role': role.value,
                'host': host,
                'port': port,
                'status': 'active',
                'registered_at': datetime.now().isoformat(),
                'capabilities': json.dumps(capabilities or {})
            })
            
            # Publish registration event
            await self.kafka_bus.publish('system_events', {
                'event_type': 'agent_registered',
                'agent_id': agent_id,
                'role': role.value,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Agent registered: {agent_id} ({role.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the coordination system"""
        try:
            if agent_id not in self.registered_agents:
                logger.warning(f"Agent not found for unregistration: {agent_id}")
                return False
            
            instance = self.registered_agents[agent_id]
            
            # Unregister from load balancer
            self.load_balancer.unregister_agent_instance(agent_id, instance.role)
            
            # Remove from local registry
            del self.registered_agents[agent_id]
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]
            
            # Update Redis state
            await self.redis_state.set_agent_state(agent_id, {
                'status': 'unregistered',
                'unregistered_at': datetime.now().isoformat()
            })
            
            # Publish unregistration event
            await self.kafka_bus.publish('system_events', {
                'event_type': 'agent_unregistered',
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Agent unregistered: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def coordinate_signal_fusion(self, 
                                     symbol: str, 
                                     signals: List[Dict[str, Any]], 
                                     timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Coordinate signal fusion across multiple agents"""
        coordination_id = str(uuid.uuid4())
        
        try:
            self.stats["coordinations_initiated"] += 1
            
            # Find relevant agents for signal fusion
            portfolio_agents = [
                agent_id for agent_id, instance in self.registered_agents.items()
                if instance.role == AgentRole.PORTFOLIO_MANAGER and instance.status == "active"
            ]
            
            if not portfolio_agents:
                logger.error("No active portfolio manager agents available")
                return None
            
            # Select agent using load balancer
            selected_instance = self.load_balancer.select_agent_instance(
                AgentRole.PORTFOLIO_MANAGER,
                {'symbol': symbol, 'signal_count': len(signals)}
            )
            
            if not selected_instance:
                logger.error("Failed to select portfolio manager agent")
                return None
            
            # Check failover status
            if not self.failover_manager.can_execute(selected_instance.agent_id):
                logger.warning(f"Agent {selected_instance.agent_id} is in circuit breaker state")
                return None
            
            # Create coordination session
            coordination_session = {
                'coordination_id': coordination_id,
                'type': 'signal_fusion',
                'symbol': symbol,
                'signals': signals,
                'selected_agent': selected_instance.agent_id,
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'timeout': timeout
            }
            
            self.active_coordinations[coordination_id] = coordination_session
            
            # Store in Redis
            await self.redis_state.set_shared_data(
                f"coordination:{coordination_id}",
                coordination_session,
                ttl=timeout + 60
            )
            
            # Use negotiation protocol if enabled
            if self.negotiation_protocol:
                negotiation_id = await self.negotiation_protocol.initiate_negotiation(
                    negotiation_type='signal_fusion',
                    initiator='coordinator',
                    participants=[selected_instance.agent_id],
                    proposal={
                        'symbol': symbol,
                        'signals': signals,
                        'coordination_id': coordination_id
                    },
                    timeout_seconds=timeout
                )
                
                coordination_session['negotiation_id'] = negotiation_id
            
            # Publish coordination request
            await self.kafka_bus.publish('signal_fusion', {
                'coordination_id': coordination_id,
                'symbol': symbol,
                'signals': signals,
                'target_agent': selected_instance.agent_id,
                'timeout': timeout,
                'timestamp': datetime.now().isoformat()
            })
            
            # Wait for result with timeout
            result = await self._wait_for_coordination_result(coordination_id, timeout)
            
            if result:
                self.failover_manager.record_success(selected_instance.agent_id)
                self.stats["coordinations_completed"] += 1
                logger.info(f"Signal fusion coordination completed: {coordination_id}")
            else:
                self.failover_manager.record_failure(selected_instance.agent_id)
                self.stats["coordinations_failed"] += 1
                logger.warning(f"Signal fusion coordination failed: {coordination_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in signal fusion coordination: {e}")
            self.stats["coordinations_failed"] += 1
            return None
        finally:
            # Clean up coordination session
            if coordination_id in self.active_coordinations:
                del self.active_coordinations[coordination_id]
    
    async def coordinate_risk_assessment(self, 
                                       portfolio_state: Dict[str, Any], 
                                       timeout: int = 15) -> Optional[Dict[str, Any]]:
        """Coordinate risk assessment across risk manager agents"""
        coordination_id = str(uuid.uuid4())
        
        try:
            self.stats["coordinations_initiated"] += 1
            
            # Find active risk manager agents
            risk_agents = [
                agent_id for agent_id, instance in self.registered_agents.items()
                if instance.role == AgentRole.RISK_MANAGER and instance.status == "active"
            ]
            
            if not risk_agents:
                logger.error("No active risk manager agents available")
                return None
            
            # Select agent using load balancer
            selected_instance = self.load_balancer.select_agent_instance(
                AgentRole.RISK_MANAGER,
                {'portfolio_size': len(portfolio_state.get('positions', []))}
            )
            
            if not selected_instance or not self.failover_manager.can_execute(selected_instance.agent_id):
                logger.error("No suitable risk manager agent available")
                return None
            
            # Publish risk assessment request
            await self.kafka_bus.publish('risk_alerts', {
                'coordination_id': coordination_id,
                'request_type': 'risk_assessment',
                'portfolio_state': portfolio_state,
                'target_agent': selected_instance.agent_id,
                'timeout': timeout,
                'timestamp': datetime.now().isoformat()
            })
            
            # Wait for result
            result = await self._wait_for_coordination_result(coordination_id, timeout)
            
            if result:
                self.failover_manager.record_success(selected_instance.agent_id)
                self.stats["coordinations_completed"] += 1
            else:
                self.failover_manager.record_failure(selected_instance.agent_id)
                self.stats["coordinations_failed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in risk assessment coordination: {e}")
            self.stats["coordinations_failed"] += 1
            return None
    
    async def _wait_for_coordination_result(self, coordination_id: str, timeout: int) -> Optional[Dict[str, Any]]:
        """Wait for coordination result with timeout"""
        end_time = datetime.now() + timedelta(seconds=timeout)
        
        while datetime.now() < end_time:
            # Check Redis for result
            result = await self.redis_state.get_shared_data(f"coordination_result:{coordination_id}")
            if result:
                return result
            
            # Brief sleep to avoid busy waiting
            await asyncio.sleep(0.1)
        
        logger.warning(f"Coordination {coordination_id} timed out")
        return None
    
    async def _handle_coordination_message(self, message: Dict[str, Any]):
        """Handle coordination messages from Kafka"""
        try:
            coordination_id = message.get('coordination_id')
            message_type = message.get('type', 'unknown')
            
            if message_type == 'result':
                # Store coordination result
                await self.redis_state.set_shared_data(
                    f"coordination_result:{coordination_id}",
                    message.get('result', {}),
                    ttl=300  # 5 minutes
                )
                
                logger.debug(f"Coordination result received: {coordination_id}")
            
        except Exception as e:
            logger.error(f"Error handling coordination message: {e}")
    
    async def _handle_heartbeat_message(self, message: Dict[str, Any]):
        """Handle agent heartbeat messages"""
        try:
            agent_id = message.get('agent_id')
            metrics = message.get('metrics', {})
            
            if agent_id in self.registered_agents:
                # Update load balancer metrics
                self.load_balancer.update_instance_metrics(agent_id, metrics)
                
                # Update agent state in Redis
                await self.redis_state.set_agent_state(agent_id, {
                    'last_heartbeat': datetime.now().isoformat(),
                    'metrics': json.dumps(metrics)
                })
                
                logger.debug(f"Heartbeat received from {agent_id}")
            
        except Exception as e:
            logger.error(f"Error handling heartbeat message: {e}")
    
    async def _handle_signal_fusion_message(self, message: Dict[str, Any]):
        """Handle signal fusion messages"""
        # This would be implemented based on specific signal fusion logic
        pass
    
    async def _handle_risk_alert_message(self, message: Dict[str, Any]):
        """Handle risk alert messages"""
        try:
            alert_type = message.get('alert_type')
            severity = message.get('severity', 'medium')
            
            if severity == 'critical':
                # Trigger emergency procedures
                await self._handle_critical_risk_alert(message)
            
            logger.info(f"Risk alert received: {alert_type} ({severity})")
            
        except Exception as e:
            logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_system_event_message(self, message: Dict[str, Any]):
        """Handle system event messages"""
        try:
            event_type = message.get('event_type')
            
            if event_type == 'agent_failure':
                agent_id = message.get('agent_id')
                if agent_id:
                    self.failover_manager.record_failure(agent_id)
                    self.stats["agent_failures"] += 1
            
            logger.debug(f"System event: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling system event: {e}")
    
    async def _handle_critical_risk_alert(self, alert: Dict[str, Any]):
        """Handle critical risk alerts with immediate action"""
        try:
            # Implement emergency risk procedures
            logger.critical(f"Critical risk alert: {alert}")
            
            # Could trigger:
            # - Emergency position closure
            # - Trading halt
            # - Notification to human operators
            
        except Exception as e:
            logger.error(f"Error handling critical risk alert: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and detect failures"""
        while self.running:
            try:
                current_time = datetime.now()
                heartbeat_timeout = timedelta(seconds=self.config.heartbeat_interval * 2)
                
                for agent_id, instance in self.registered_agents.items():
                    if current_time - instance.last_heartbeat > heartbeat_timeout:
                        logger.warning(f"Agent {agent_id} heartbeat timeout")
                        
                        # Mark as inactive
                        instance.status = "inactive"
                        
                        # Record failure
                        self.failover_manager.record_failure(agent_id)
                        
                        # Publish failure event
                        await self.kafka_bus.publish('system_events', {
                            'event_type': 'agent_failure',
                            'agent_id': agent_id,
                            'reason': 'heartbeat_timeout',
                            'timestamp': current_time.isoformat()
                        })
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)
    
    async def _coordination_monitor(self):
        """Monitor active coordinations and handle timeouts"""
        while self.running:
            try:
                current_time = datetime.now()
                
                expired_coordinations = []
                for coord_id, session in self.active_coordinations.items():
                    created_at = datetime.fromisoformat(session['created_at'])
                    timeout = session.get('timeout', 60)
                    
                    if current_time - created_at > timedelta(seconds=timeout):
                        expired_coordinations.append(coord_id)
                
                # Clean up expired coordinations
                for coord_id in expired_coordinations:
                    logger.warning(f"Coordination {coord_id} expired")
                    del self.active_coordinations[coord_id]
                    self.stats["coordinations_failed"] += 1
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in coordination monitor: {e}")
                await asyncio.sleep(5)
    
    async def _performance_monitor(self):
        """Monitor system performance and adjust load balancing"""
        while self.running:
            try:
                # Collect performance metrics
                kafka_stats = self.kafka_bus.get_stats() if self.kafka_bus else {}
                lb_stats = self.load_balancer.get_stats() if self.load_balancer else {}
                
                # Store metrics in Redis
                await self.redis_state.set_shared_data('system_performance', {
                    'timestamp': datetime.now().isoformat(),
                    'kafka_stats': kafka_stats,
                    'load_balancer_stats': lb_stats,
                    'coordination_stats': self.stats
                }, ttl=3600)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(30)
    
    async def _failover_monitor(self):
        """Monitor failover conditions and trigger failovers"""
        while self.running:
            try:
                # Check circuit breaker states
                for agent_id in self.registered_agents:
                    status = self.failover_manager.get_agent_status(agent_id)
                    
                    if status['circuit_breaker']['state'] == 'open':
                        logger.info(f"Agent {agent_id} in circuit breaker state")
                        self.stats["failovers_triggered"] += 1
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in failover monitor: {e}")
                await asyncio.sleep(15)
    
    # Negotiation handlers
    async def _handle_signal_fusion_negotiation(self, negotiation) -> Dict[str, Any]:
        """Handle signal fusion negotiation"""
        # Implement signal fusion consensus logic
        return {"status": "completed", "result": "consensus_reached"}
    
    async def _handle_resource_allocation_negotiation(self, negotiation) -> Dict[str, Any]:
        """Handle resource allocation negotiation"""
        # Implement resource allocation consensus logic
        return {"status": "completed", "result": "resources_allocated"}
    
    async def _handle_risk_consensus_negotiation(self, negotiation) -> Dict[str, Any]:
        """Handle risk consensus negotiation"""
        # Implement risk consensus logic
        return {"status": "completed", "result": "risk_consensus_reached"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "running": self.running,
            "registered_agents": len(self.registered_agents),
            "active_agents": len([a for a in self.registered_agents.values() if a.status == "active"]),
            "active_coordinations": len(self.active_coordinations),
            "current_phase": self.current_phase.value,
            "market_regime": self.market_regime.value,
            "statistics": self.stats,
            "kafka_stats": self.kafka_bus.get_stats() if self.kafka_bus else {},
            "load_balancer_stats": self.load_balancer.get_stats() if self.load_balancer else {}
        }


# Factory function
def create_enhanced_workflow_coordinator(config: CoordinationConfig = None) -> EnhancedWorkflowCoordinator:
    """Create enhanced workflow coordinator instance"""
    return EnhancedWorkflowCoordinator(config)


if __name__ == "__main__":
    # Example usage
    async def main():
        config = CoordinationConfig(
            kafka_servers=['localhost:9092'],
            redis_config={'host': 'localhost', 'port': 6379},
            load_balancing_strategy=LoadBalancingStrategy.PERFORMANCE_BASED
        )
        
        coordinator = create_enhanced_workflow_coordinator(config)
        
        try:
            await coordinator.start()
            
            # Register some test agents
            await coordinator.register_agent(
                "momentum_agent_1", 
                AgentRole.SIGNAL_GENERATOR,
                capabilities={"strategies": ["momentum", "fibonacci"]}
            )
            
            await coordinator.register_agent(
                "portfolio_manager_1",
                AgentRole.PORTFOLIO_MANAGER,
                capabilities={"fusion_methods": ["weighted_average", "consensus"]}
            )
            
            print("Enhanced workflow coordinator started")
            print("System status:", coordinator.get_system_status())
            
            # Keep running for demonstration
            await asyncio.sleep(5)
            
        finally:
            await coordinator.stop()
    
    asyncio.run(main())