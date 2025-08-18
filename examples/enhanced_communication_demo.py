"""
Enhanced Communication System Demo

This script demonstrates the enhanced agent coordination and message passing
system with Kafka, Redis, load balancing, and failover capabilities.
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced communication system
from agents.enhanced_communication import (
    KafkaMessageBus, RedisStateManager, LoadBalancer, FailoverManager,
    NegotiationProtocol, AgentInstance, LoadBalancingStrategy, FailoverStrategy,
    create_kafka_message_bus, create_redis_state_manager, create_load_balancer,
    create_failover_manager, create_negotiation_protocol
)
from agents.enhanced_workflow_coordinator import (
    EnhancedWorkflowCoordinator, CoordinationConfig
)
from agents.communication_protocols import AgentRole


class MockAgent:
    """Mock agent for demonstration purposes"""
    
    def __init__(self, agent_id: str, role: AgentRole, host: str = "localhost", port: int = 8000):
        self.agent_id = agent_id
        self.role = role
        self.host = host
        self.port = port
        self.status = "active"
        self.performance_score = 0.8
        self.connections = 0
        self.resource_usage = {"cpu_percent": 20, "memory_percent": 30}
        
        logger.info(f"Mock agent created: {agent_id} ({role.value})")
    
    async def process_signal_fusion(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock signal fusion processing"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Simple fusion logic for demo
        total_value = sum(signal.get('value', 0) for signal in signals)
        avg_confidence = sum(signal.get('confidence', 0) for signal in signals) / len(signals)
        
        result = {
            'fused_signal': {
                'value': total_value / len(signals),
                'confidence': avg_confidence,
                'method': 'simple_average',
                'contributing_signals': len(signals)
            },
            'processing_time': 0.1,
            'agent_id': self.agent_id
        }
        
        logger.info(f"Signal fusion completed by {self.agent_id}: {result['fused_signal']}")
        return result
    
    async def assess_risk(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Mock risk assessment"""
        await asyncio.sleep(0.05)  # Simulate processing time
        
        positions = portfolio_state.get('positions', [])
        total_value = portfolio_state.get('total_value', 0)
        
        # Simple risk calculation for demo
        risk_score = min(len(positions) * 0.1, 1.0)  # More positions = more risk
        
        result = {
            'risk_assessment': {
                'risk_score': risk_score,
                'risk_level': 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high',
                'recommendations': ['diversify'] if len(positions) < 5 else ['monitor_closely'],
                'var_estimate': total_value * risk_score * 0.05
            },
            'processing_time': 0.05,
            'agent_id': self.agent_id
        }
        
        logger.info(f"Risk assessment completed by {self.agent_id}: {result['risk_assessment']}")
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics for load balancing"""
        return {
            'load': min(self.connections * 0.1, 1.0),
            'connections': self.connections,
            'performance_score': self.performance_score,
            'resource_usage': self.resource_usage
        }


async def demo_basic_components():
    """Demonstrate basic communication components"""
    logger.info("=== Demo: Basic Communication Components ===")
    
    try:
        # Demo Redis State Manager (fallback to mock if Redis not available)
        logger.info("Testing Redis State Manager...")
        try:
            redis_manager = create_redis_state_manager()
            
            # Test basic operations
            await redis_manager.set_agent_state('demo_agent', {
                'status': 'active',
                'last_seen': datetime.now().isoformat()
            })
            
            state = await redis_manager.get_agent_state('demo_agent')
            logger.info(f"Retrieved agent state: {state}")
            
            # Test shared data
            await redis_manager.set_shared_data('demo_key', {'test': 'data'})
            data = await redis_manager.get_shared_data('demo_key')
            logger.info(f"Retrieved shared data: {data}")
            
            # Test distributed lock
            lock_acquired = await redis_manager.acquire_lock('demo_lock', timeout=5)
            logger.info(f"Lock acquired: {lock_acquired}")
            
            if lock_acquired:
                await redis_manager.release_lock('demo_lock')
                logger.info("Lock released")
            
            # Test counter
            count = await redis_manager.increment_counter('demo_counter', 5)
            logger.info(f"Counter value: {count}")
            
        except Exception as e:
            logger.warning(f"Redis not available, skipping Redis demo: {e}")
        
        # Demo Load Balancer
        logger.info("Testing Load Balancer...")
        load_balancer = create_load_balancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        # Create sample agents
        agents = [
            AgentInstance("agent_1", AgentRole.SIGNAL_GENERATOR, "host1", 8001, connections=3, performance_score=0.8),
            AgentInstance("agent_2", AgentRole.SIGNAL_GENERATOR, "host2", 8002, connections=5, performance_score=0.9),
            AgentInstance("agent_3", AgentRole.SIGNAL_GENERATOR, "host3", 8003, connections=2, performance_score=0.7)
        ]
        
        for agent in agents:
            load_balancer.register_agent_instance(agent)
        
        # Test round-robin selection
        logger.info("Round-robin selections:")
        for i in range(6):
            selected = load_balancer.select_agent_instance(AgentRole.SIGNAL_GENERATOR)
            logger.info(f"  Selection {i+1}: {selected.agent_id}")
        
        # Test different strategies
        strategies = [
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.PERFORMANCE_BASED
        ]
        
        for strategy in strategies:
            lb = create_load_balancer(strategy)
            for agent in agents:
                lb.register_agent_instance(agent)
            
            selected = lb.select_agent_instance(AgentRole.SIGNAL_GENERATOR)
            logger.info(f"{strategy.value} selection: {selected.agent_id}")
        
        # Demo Failover Manager
        logger.info("Testing Failover Manager...")
        failover_manager = create_failover_manager()
        
        agent_id = "test_agent"
        
        # Test normal operation
        logger.info(f"Can execute (initial): {failover_manager.can_execute(agent_id)}")
        
        # Record some failures
        for i in range(3):
            failover_manager.record_failure(agent_id)
            logger.info(f"After {i+1} failures - can execute: {failover_manager.can_execute(agent_id)}")
        
        # Record more failures to trigger circuit breaker
        for i in range(3):
            failover_manager.record_failure(agent_id)
        
        logger.info(f"After 6 failures - can execute: {failover_manager.can_execute(agent_id)}")
        
        # Record success to reset
        failover_manager.record_success(agent_id)
        logger.info(f"After success - can execute: {failover_manager.can_execute(agent_id)}")
        
        status = failover_manager.get_agent_status(agent_id)
        logger.info(f"Agent status: {status}")
        
    except Exception as e:
        logger.error(f"Error in basic components demo: {e}")


async def demo_kafka_message_bus():
    """Demonstrate Kafka message bus (if available)"""
    logger.info("=== Demo: Kafka Message Bus ===")
    
    try:
        # Try to create Kafka message bus
        kafka_bus = create_kafka_message_bus(['localhost:9092'])
        
        await kafka_bus.start()
        logger.info("Kafka message bus started")
        
        # Test message publishing
        test_messages = [
            {'type': 'market_data', 'symbol': 'AAPL', 'price': 150.0},
            {'type': 'signal', 'symbol': 'GOOGL', 'value': 0.8},
            {'type': 'risk_alert', 'level': 'medium', 'message': 'Portfolio exposure high'}
        ]
        
        for i, message in enumerate(test_messages):
            await kafka_bus.publish(f'test_topic_{i}', message, f'key_{i}')
            logger.info(f"Published message {i+1}: {message}")
        
        # Show statistics
        stats = kafka_bus.get_stats()
        logger.info(f"Kafka stats: {stats}")
        
        await kafka_bus.stop()
        logger.info("Kafka message bus stopped")
        
    except Exception as e:
        logger.warning(f"Kafka not available or error occurred: {e}")


async def demo_negotiation_protocol():
    """Demonstrate negotiation protocol"""
    logger.info("=== Demo: Negotiation Protocol ===")
    
    try:
        # Create mock state manager for negotiation
        class MockStateManager:
            def __init__(self):
                self.data = {}
            
            async def set_shared_data(self, key, data, ttl=None):
                self.data[key] = data
            
            async def get_shared_data(self, key):
                return self.data.get(key)
        
        mock_state = MockStateManager()
        negotiation_protocol = create_negotiation_protocol(mock_state)
        
        # Register a simple negotiation handler
        async def signal_fusion_handler(negotiation):
            responses = negotiation.responses
            logger.info(f"Processing negotiation with {len(responses)} responses")
            
            # Simple consensus logic
            accept_count = sum(1 for r in responses.values() if r.get('accept', False))
            total_confidence = sum(r.get('confidence', 0) for r in responses.values())
            
            return {
                'consensus': accept_count > len(responses) / 2,
                'average_confidence': total_confidence / len(responses) if responses else 0,
                'participant_count': len(responses)
            }
        
        negotiation_protocol.register_negotiation_handler('signal_fusion', signal_fusion_handler)
        
        # Initiate negotiation
        request_id = await negotiation_protocol.initiate_negotiation(
            negotiation_type='signal_fusion',
            initiator='coordinator',
            participants=['agent_1', 'agent_2', 'agent_3'],
            proposal={'symbol': 'AAPL', 'signals': [{'value': 0.8}, {'value': 0.6}]},
            timeout_seconds=30
        )
        
        logger.info(f"Negotiation initiated: {request_id}")
        
        # Simulate responses from participants
        responses = [
            ('agent_1', {'accept': True, 'confidence': 0.9}),
            ('agent_2', {'accept': True, 'confidence': 0.7}),
            ('agent_3', {'accept': False, 'confidence': 0.4})
        ]
        
        for agent_id, response in responses:
            success = await negotiation_protocol.respond_to_negotiation(request_id, agent_id, response)
            logger.info(f"Response from {agent_id}: {response} (success: {success})")
        
        # Wait a moment for processing
        await asyncio.sleep(0.1)
        
        # Get result
        result = await negotiation_protocol.get_negotiation_result(request_id)
        logger.info(f"Negotiation result: {result}")
        
    except Exception as e:
        logger.error(f"Error in negotiation protocol demo: {e}")


async def demo_enhanced_workflow_coordinator():
    """Demonstrate enhanced workflow coordinator"""
    logger.info("=== Demo: Enhanced Workflow Coordinator ===")
    
    try:
        # Create configuration (using fallback settings for demo)
        config = CoordinationConfig(
            kafka_servers=['localhost:9092'],
            redis_config={'host': 'localhost', 'port': 6379},
            load_balancing_strategy=LoadBalancingStrategy.PERFORMANCE_BASED,
            failover_strategy=FailoverStrategy.CIRCUIT_BREAKER
        )
        
        coordinator = EnhancedWorkflowCoordinator(config)
        
        # Create mock agents
        mock_agents = [
            MockAgent('momentum_agent', AgentRole.SIGNAL_GENERATOR),
            MockAgent('mean_reversion_agent', AgentRole.SIGNAL_GENERATOR),
            MockAgent('portfolio_manager', AgentRole.PORTFOLIO_MANAGER),
            MockAgent('risk_manager', AgentRole.RISK_MANAGER)
        ]
        
        try:
            await coordinator.start()
            logger.info("Enhanced workflow coordinator started")
            
            # Register agents
            for agent in mock_agents:
                success = await coordinator.register_agent(
                    agent.agent_id,
                    agent.role,
                    agent.host,
                    agent.port,
                    capabilities={'mock': True}
                )
                logger.info(f"Agent registration {agent.agent_id}: {'success' if success else 'failed'}")
            
            # Show system status
            status = coordinator.get_system_status()
            logger.info(f"System status: {json.dumps(status, indent=2, default=str)}")
            
            # Simulate signal fusion coordination
            logger.info("Testing signal fusion coordination...")
            signals = [
                {'symbol': 'AAPL', 'value': 0.8, 'confidence': 0.9, 'agent': 'momentum_agent'},
                {'symbol': 'AAPL', 'value': 0.6, 'confidence': 0.7, 'agent': 'mean_reversion_agent'}
            ]
            
            # This will likely timeout in demo environment, but shows the coordination attempt
            result = await coordinator.coordinate_signal_fusion('AAPL', signals, timeout=2)
            
            if result:
                logger.info(f"Signal fusion result: {result}")
            else:
                logger.info("Signal fusion timed out (expected in demo environment)")
            
            # Simulate risk assessment coordination
            logger.info("Testing risk assessment coordination...")
            portfolio_state = {
                'positions': [
                    {'symbol': 'AAPL', 'quantity': 100, 'value': 15000},
                    {'symbol': 'GOOGL', 'quantity': 50, 'value': 12500}
                ],
                'total_value': 27500
            }
            
            risk_result = await coordinator.coordinate_risk_assessment(portfolio_state, timeout=2)
            
            if risk_result:
                logger.info(f"Risk assessment result: {risk_result}")
            else:
                logger.info("Risk assessment timed out (expected in demo environment)")
            
            # Show final statistics
            final_status = coordinator.get_system_status()
            logger.info(f"Final statistics: {final_status['statistics']}")
            
        finally:
            await coordinator.stop()
            logger.info("Enhanced workflow coordinator stopped")
            
    except Exception as e:
        logger.error(f"Error in enhanced workflow coordinator demo: {e}")


async def demo_load_balancing_scenarios():
    """Demonstrate various load balancing scenarios"""
    logger.info("=== Demo: Load Balancing Scenarios ===")
    
    try:
        # Create agents with different characteristics
        agents = [
            AgentInstance("fast_agent", AgentRole.SIGNAL_GENERATOR, "host1", 8001, 
                         connections=2, performance_score=0.95, 
                         resource_usage={"cpu_percent": 15, "memory_percent": 20}),
            AgentInstance("busy_agent", AgentRole.SIGNAL_GENERATOR, "host2", 8002, 
                         connections=10, performance_score=0.75, 
                         resource_usage={"cpu_percent": 80, "memory_percent": 70}),
            AgentInstance("balanced_agent", AgentRole.SIGNAL_GENERATOR, "host3", 8003, 
                         connections=5, performance_score=0.85, 
                         resource_usage={"cpu_percent": 40, "memory_percent": 45})
        ]
        
        strategies = [
            LoadBalancingStrategy.ROUND_ROBIN,
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.PERFORMANCE_BASED,
            LoadBalancingStrategy.RESOURCE_BASED
        ]
        
        for strategy in strategies:
            logger.info(f"\nTesting {strategy.value} strategy:")
            
            lb = create_load_balancer(strategy)
            for agent in agents:
                lb.register_agent_instance(agent)
            
            # Make multiple selections
            selections = []
            for i in range(10):
                selected = lb.select_agent_instance(AgentRole.SIGNAL_GENERATOR)
                selections.append(selected.agent_id)
                
                # Simulate connection increase
                selected.connections += 1
                lb.update_instance_metrics(selected.agent_id, selected.get_metrics())
            
            # Count selections per agent
            selection_counts = {}
            for agent_id in selections:
                selection_counts[agent_id] = selection_counts.get(agent_id, 0) + 1
            
            logger.info(f"  Selection distribution: {selection_counts}")
            
            # Show final agent states
            for agent in agents:
                logger.info(f"  {agent.agent_id}: connections={agent.connections}, "
                          f"performance={agent.performance_score}, "
                          f"cpu={agent.resource_usage['cpu_percent']}%")
                # Reset connections for next test
                agent.connections = agents[0].connections if agent == agents[0] else \
                                  agents[1].connections if agent == agents[1] else \
                                  agents[2].connections
        
    except Exception as e:
        logger.error(f"Error in load balancing scenarios demo: {e}")


async def demo_failover_scenarios():
    """Demonstrate failover scenarios"""
    logger.info("=== Demo: Failover Scenarios ===")
    
    try:
        failover_manager = create_failover_manager(FailoverStrategy.CIRCUIT_BREAKER)
        
        agents = ["reliable_agent", "unreliable_agent", "recovering_agent"]
        
        # Simulate different failure patterns
        scenarios = [
            ("reliable_agent", [True] * 10),  # All successes
            ("unreliable_agent", [False] * 8 + [True] * 2),  # Mostly failures
            ("recovering_agent", [False] * 5 + [True] * 5)  # Failures then recovery
        ]
        
        for agent_id, operations in scenarios:
            logger.info(f"\nTesting {agent_id}:")
            
            for i, success in enumerate(operations):
                can_execute_before = failover_manager.can_execute(agent_id)
                
                if success:
                    failover_manager.record_success(agent_id)
                else:
                    failover_manager.record_failure(agent_id)
                
                can_execute_after = failover_manager.can_execute(agent_id)
                status = failover_manager.get_agent_status(agent_id)
                
                logger.info(f"  Op {i+1}: {'SUCCESS' if success else 'FAILURE'} - "
                          f"Can execute: {can_execute_before} -> {can_execute_after} - "
                          f"Failures: {status['failure_count']} - "
                          f"Circuit: {status['circuit_breaker']['state']}")
        
        # Demonstrate circuit breaker recovery
        logger.info("\nTesting circuit breaker recovery:")
        
        # Force circuit breaker open
        for _ in range(5):
            failover_manager.record_failure("recovery_test_agent")
        
        logger.info(f"Circuit breaker opened: {not failover_manager.can_execute('recovery_test_agent')}")
        
        # Simulate timeout by modifying the opened time
        if "recovery_test_agent" in failover_manager.circuit_breakers:
            failover_manager.circuit_breakers["recovery_test_agent"]["opened_at"] = \
                datetime.now() - timedelta(seconds=70)
        
        # Should transition to half-open
        can_execute = failover_manager.can_execute("recovery_test_agent")
        logger.info(f"After timeout - can execute (half-open): {can_execute}")
        
        if can_execute:
            cb_state = failover_manager.circuit_breakers["recovery_test_agent"]["state"]
            logger.info(f"Circuit breaker state: {cb_state}")
        
    except Exception as e:
        logger.error(f"Error in failover scenarios demo: {e}")


async def main():
    """Run all demonstrations"""
    logger.info("Starting Enhanced Communication System Demo")
    logger.info("=" * 60)
    
    demos = [
        demo_basic_components,
        demo_kafka_message_bus,
        demo_negotiation_protocol,
        demo_load_balancing_scenarios,
        demo_failover_scenarios,
        demo_enhanced_workflow_coordinator
    ]
    
    for demo in demos:
        try:
            await demo()
            logger.info("")  # Add spacing between demos
        except Exception as e:
            logger.error(f"Demo failed: {demo.__name__} - {e}")
            logger.info("")
    
    logger.info("Enhanced Communication System Demo completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())