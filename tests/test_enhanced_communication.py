"""
Tests for Enhanced Communication System

This module contains comprehensive tests for the enhanced agent coordination
and message passing system including Kafka, Redis, load balancing, and failover.
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import the modules to test
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


class TestKafkaMessageBus:
    """Test Kafka message bus functionality"""
    
    @pytest.fixture
    def mock_kafka_producer(self):
        """Mock Kafka producer"""
        with patch('agents.enhanced_communication.KafkaProducer') as mock:
            producer_instance = Mock()
            mock.return_value = producer_instance
            
            # Mock send method
            future_mock = Mock()
            future_mock.get.return_value = Mock(partition=0, offset=123)
            producer_instance.send.return_value = future_mock
            
            yield producer_instance
    
    @pytest.fixture
    def mock_kafka_consumer(self):
        """Mock Kafka consumer"""
        with patch('agents.enhanced_communication.KafkaConsumer') as mock:
            consumer_instance = Mock()
            mock.return_value = consumer_instance
            
            # Mock poll method
            consumer_instance.poll.return_value = {}
            
            yield consumer_instance
    
    @pytest.mark.asyncio
    async def test_kafka_message_bus_initialization(self, mock_kafka_producer):
        """Test Kafka message bus initialization"""
        with patch('agents.enhanced_communication.KAFKA_AVAILABLE', True):
            bus = KafkaMessageBus(['localhost:9092'])
            
            assert bus.bootstrap_servers == ['localhost:9092']
            assert bus.running is False
            assert bus.stats['messages_sent'] == 0
    
    @pytest.mark.asyncio
    async def test_kafka_message_bus_publish(self, mock_kafka_producer):
        """Test message publishing"""
        with patch('agents.enhanced_communication.KAFKA_AVAILABLE', True):
            bus = KafkaMessageBus(['localhost:9092'])
            
            await bus.start()
            
            test_message = {'type': 'test', 'data': 'test_data'}
            await bus.publish('test_topic', test_message, 'test_key')
            
            # Verify producer.send was called
            bus.producer.send.assert_called_once_with(
                'test_topic', 
                value=test_message, 
                key='test_key'
            )
            
            assert bus.stats['messages_sent'] == 1
            
            await bus.stop()
    
    @pytest.mark.asyncio
    async def test_kafka_message_bus_subscribe(self, mock_kafka_consumer):
        """Test message subscription"""
        with patch('agents.enhanced_communication.KAFKA_AVAILABLE', True):
            bus = KafkaMessageBus(['localhost:9092'])
            
            handler_called = False
            
            def test_handler(message):
                nonlocal handler_called
                handler_called = True
            
            bus.subscribe('test_topic', test_handler)
            
            assert 'test_topic' in bus.message_handlers
            assert test_handler in bus.message_handlers['test_topic']
            assert bus.stats['topics_subscribed'] == 1
    
    def test_kafka_not_available_fallback(self):
        """Test fallback when Kafka is not available"""
        with patch('agents.enhanced_communication.KAFKA_AVAILABLE', False):
            with pytest.raises(ImportError):
                KafkaMessageBus(['localhost:9092'])


class TestRedisStateManager:
    """Test Redis state manager functionality"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        with patch('agents.enhanced_communication.redis.Redis') as mock:
            redis_instance = Mock()
            mock.return_value = redis_instance
            
            # Mock basic operations
            redis_instance.ping.return_value = True
            redis_instance.hset.return_value = True
            redis_instance.hgetall.return_value = {'key': 'value'}
            redis_instance.set.return_value = True
            redis_instance.get.return_value = '{"test": "data"}'
            redis_instance.incrby.return_value = 1
            redis_instance.lpush.return_value = 1
            redis_instance.rpop.return_value = '{"test": "item"}'
            
            yield redis_instance
    
    @pytest.mark.asyncio
    async def test_redis_state_manager_initialization(self, mock_redis):
        """Test Redis state manager initialization"""
        with patch('agents.enhanced_communication.REDIS_AVAILABLE', True):
            manager = RedisStateManager({'host': 'localhost', 'port': 6379})
            
            assert manager.redis_client is not None
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_get_agent_state(self, mock_redis):
        """Test setting and getting agent state"""
        with patch('agents.enhanced_communication.REDIS_AVAILABLE', True):
            manager = RedisStateManager()
            
            test_state = {'status': 'active', 'load': 0.5}
            await manager.set_agent_state('test_agent', test_state)
            
            # Verify hset was called
            manager.redis_client.hset.assert_called()
            
            # Test getting state
            state = await manager.get_agent_state('test_agent')
            assert isinstance(state, dict)
    
    @pytest.mark.asyncio
    async def test_shared_data_operations(self, mock_redis):
        """Test shared data operations"""
        with patch('agents.enhanced_communication.REDIS_AVAILABLE', True):
            manager = RedisStateManager()
            
            test_data = {'key': 'value', 'number': 123}
            await manager.set_shared_data('test_key', test_data)
            
            # Verify set was called
            manager.redis_client.set.assert_called()
            
            # Test getting data
            data = await manager.get_shared_data('test_key')
            assert data == {'test': 'data'}  # From mock return value
    
    @pytest.mark.asyncio
    async def test_distributed_lock(self, mock_redis):
        """Test distributed lock operations"""
        with patch('agents.enhanced_communication.REDIS_AVAILABLE', True):
            manager = RedisStateManager()
            
            # Mock successful lock acquisition
            manager.redis_client.set.return_value = True
            
            success = await manager.acquire_lock('test_lock', timeout=10)
            assert success is True
            
            # Test lock release
            manager.redis_client.eval.return_value = 1
            released = await manager.release_lock('test_lock', 'identifier')
            assert released is True
    
    @pytest.mark.asyncio
    async def test_counter_operations(self, mock_redis):
        """Test atomic counter operations"""
        with patch('agents.enhanced_communication.REDIS_AVAILABLE', True):
            manager = RedisStateManager()
            
            result = await manager.increment_counter('test_counter', 5)
            assert result == 1  # From mock return value
            
            manager.redis_client.incrby.assert_called_with('counter:test_counter', 5)
    
    @pytest.mark.asyncio
    async def test_queue_operations(self, mock_redis):
        """Test Redis queue operations"""
        with patch('agents.enhanced_communication.REDIS_AVAILABLE', True):
            manager = RedisStateManager()
            
            test_item = {'type': 'test', 'data': 'queue_item'}
            result = await manager.push_to_queue('test_queue', test_item)
            assert result == 1
            
            item = await manager.pop_from_queue('test_queue')
            assert item == {'test': 'item'}  # From mock return value
    
    def test_redis_not_available_fallback(self):
        """Test fallback when Redis is not available"""
        with patch('agents.enhanced_communication.REDIS_AVAILABLE', False):
            with pytest.raises(ImportError):
                RedisStateManager()


class TestLoadBalancer:
    """Test load balancer functionality"""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer instance"""
        return LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agent instances"""
        return [
            AgentInstance(
                agent_id="agent_1",
                role=AgentRole.SIGNAL_GENERATOR,
                host="host1",
                port=8001,
                connections=5,
                performance_score=0.8
            ),
            AgentInstance(
                agent_id="agent_2",
                role=AgentRole.SIGNAL_GENERATOR,
                host="host2",
                port=8002,
                connections=3,
                performance_score=0.9
            ),
            AgentInstance(
                agent_id="agent_3",
                role=AgentRole.SIGNAL_GENERATOR,
                host="host3",
                port=8003,
                connections=7,
                performance_score=0.7
            )
        ]
    
    def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initialization"""
        assert load_balancer.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert len(load_balancer.agent_instances) == 0
        assert load_balancer.stats['requests_routed'] == 0
    
    def test_register_agent_instances(self, load_balancer, sample_agents):
        """Test agent instance registration"""
        for agent in sample_agents:
            load_balancer.register_agent_instance(agent)
        
        role = AgentRole.SIGNAL_GENERATOR.value
        assert len(load_balancer.agent_instances[role]) == 3
        assert load_balancer.round_robin_counters[role] == 0
    
    def test_round_robin_selection(self, load_balancer, sample_agents):
        """Test round-robin load balancing"""
        for agent in sample_agents:
            load_balancer.register_agent_instance(agent)
        
        # Test multiple selections
        selected_agents = []
        for _ in range(6):  # Two full rounds
            selected = load_balancer.select_agent_instance(AgentRole.SIGNAL_GENERATOR)
            selected_agents.append(selected.agent_id)
        
        # Should cycle through agents
        expected = ["agent_1", "agent_2", "agent_3", "agent_1", "agent_2", "agent_3"]
        assert selected_agents == expected
    
    def test_least_connections_selection(self, sample_agents):
        """Test least connections load balancing"""
        lb = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        for agent in sample_agents:
            lb.register_agent_instance(agent)
        
        selected = lb.select_agent_instance(AgentRole.SIGNAL_GENERATOR)
        assert selected.agent_id == "agent_2"  # Has least connections (3)
    
    def test_performance_based_selection(self, sample_agents):
        """Test performance-based load balancing"""
        lb = LoadBalancer(LoadBalancingStrategy.PERFORMANCE_BASED)
        
        for agent in sample_agents:
            lb.register_agent_instance(agent)
        
        selected = lb.select_agent_instance(AgentRole.SIGNAL_GENERATOR)
        assert selected.agent_id == "agent_2"  # Has highest performance score (0.9)
    
    def test_no_agents_available(self, load_balancer):
        """Test selection when no agents are available"""
        selected = load_balancer.select_agent_instance(AgentRole.SIGNAL_GENERATOR)
        assert selected is None
    
    def test_update_instance_metrics(self, load_balancer, sample_agents):
        """Test updating instance metrics"""
        agent = sample_agents[0]
        load_balancer.register_agent_instance(agent)
        
        new_metrics = {
            'load': 0.6,
            'connections': 10,
            'performance_score': 0.95
        }
        
        load_balancer.update_instance_metrics(agent.agent_id, new_metrics)
        
        # Verify metrics were updated
        role = agent.role.value
        updated_agent = load_balancer.agent_instances[role][0]
        assert updated_agent.load == 0.6
        assert updated_agent.connections == 10
        assert updated_agent.performance_score == 0.95


class TestFailoverManager:
    """Test failover manager functionality"""
    
    @pytest.fixture
    def failover_manager(self):
        """Create failover manager instance"""
        return FailoverManager(FailoverStrategy.CIRCUIT_BREAKER)
    
    def test_failover_manager_initialization(self, failover_manager):
        """Test failover manager initialization"""
        assert failover_manager.strategy == FailoverStrategy.CIRCUIT_BREAKER
        assert len(failover_manager.circuit_breakers) == 0
        assert len(failover_manager.failure_counts) == 0
    
    def test_record_success(self, failover_manager):
        """Test recording successful operations"""
        agent_id = "test_agent"
        
        # First record some failures
        for _ in range(3):
            failover_manager.record_failure(agent_id)
        
        assert failover_manager.failure_counts[agent_id] == 3
        
        # Record success - should reset failure count
        failover_manager.record_success(agent_id)
        assert failover_manager.failure_counts[agent_id] == 0
    
    def test_record_failure_and_circuit_breaker(self, failover_manager):
        """Test recording failures and circuit breaker activation"""
        agent_id = "test_agent"
        
        # Record failures up to threshold
        for i in range(5):
            failover_manager.record_failure(agent_id)
            
            if i < 4:
                assert failover_manager.can_execute(agent_id) is True
            else:
                # Circuit breaker should open after 5th failure
                assert agent_id in failover_manager.circuit_breakers
                assert failover_manager.circuit_breakers[agent_id]['state'] == 'open'
                assert failover_manager.can_execute(agent_id) is False
    
    def test_circuit_breaker_recovery(self, failover_manager):
        """Test circuit breaker recovery after timeout"""
        agent_id = "test_agent"
        
        # Trigger circuit breaker
        for _ in range(5):
            failover_manager.record_failure(agent_id)
        
        assert failover_manager.can_execute(agent_id) is False
        
        # Simulate timeout by modifying opened_at time
        failover_manager.circuit_breakers[agent_id]['opened_at'] = datetime.now() - timedelta(seconds=70)
        
        # Should transition to half-open
        assert failover_manager.can_execute(agent_id) is True
        assert failover_manager.circuit_breakers[agent_id]['state'] == 'half_open'
    
    def test_half_open_state_limits(self, failover_manager):
        """Test half-open state call limits"""
        agent_id = "test_agent"
        
        # Set up half-open state
        failover_manager.circuit_breakers[agent_id] = {
            'state': 'half_open',
            'opened_at': datetime.now(),
            'half_open_calls': 0
        }
        
        # Should allow limited calls
        for i in range(3):
            assert failover_manager.can_execute(agent_id) is True
        
        # Should reject further calls
        assert failover_manager.can_execute(agent_id) is False
    
    def test_get_agent_status(self, failover_manager):
        """Test getting agent failover status"""
        agent_id = "test_agent"
        
        failover_manager.record_failure(agent_id)
        status = failover_manager.get_agent_status(agent_id)
        
        assert status['failure_count'] == 1
        assert 'last_failure' in status
        assert 'circuit_breaker' in status


class TestNegotiationProtocol:
    """Test negotiation protocol functionality"""
    
    @pytest.fixture
    async def mock_state_manager(self):
        """Mock Redis state manager"""
        manager = Mock(spec=RedisStateManager)
        manager.set_shared_data = AsyncMock()
        manager.get_shared_data = AsyncMock(return_value=None)
        return manager
    
    @pytest.fixture
    def negotiation_protocol(self, mock_state_manager):
        """Create negotiation protocol instance"""
        return NegotiationProtocol(mock_state_manager)
    
    @pytest.mark.asyncio
    async def test_initiate_negotiation(self, negotiation_protocol):
        """Test negotiation initiation"""
        request_id = await negotiation_protocol.initiate_negotiation(
            negotiation_type='signal_fusion',
            initiator='coordinator',
            participants=['agent_1', 'agent_2'],
            proposal={'symbol': 'AAPL', 'signals': []},
            timeout_seconds=30
        )
        
        assert request_id in negotiation_protocol.active_negotiations
        negotiation = negotiation_protocol.active_negotiations[request_id]
        assert negotiation.negotiation_type == 'signal_fusion'
        assert negotiation.initiator == 'coordinator'
        assert len(negotiation.participants) == 2
    
    @pytest.mark.asyncio
    async def test_respond_to_negotiation(self, negotiation_protocol):
        """Test responding to negotiation"""
        # First initiate a negotiation
        request_id = await negotiation_protocol.initiate_negotiation(
            negotiation_type='signal_fusion',
            initiator='coordinator',
            participants=['agent_1', 'agent_2'],
            proposal={'symbol': 'AAPL'},
            timeout_seconds=30
        )
        
        # Respond as agent_1
        response1 = {'accept': True, 'confidence': 0.8}
        success = await negotiation_protocol.respond_to_negotiation(
            request_id, 'agent_1', response1
        )
        
        assert success is True
        negotiation = negotiation_protocol.active_negotiations[request_id]
        assert 'agent_1' in negotiation.responses
        assert negotiation.responses['agent_1'] == response1
    
    @pytest.mark.asyncio
    async def test_invalid_respondent(self, negotiation_protocol):
        """Test response from invalid participant"""
        request_id = await negotiation_protocol.initiate_negotiation(
            negotiation_type='signal_fusion',
            initiator='coordinator',
            participants=['agent_1', 'agent_2'],
            proposal={'symbol': 'AAPL'},
            timeout_seconds=30
        )
        
        # Try to respond as non-participant
        success = await negotiation_protocol.respond_to_negotiation(
            request_id, 'agent_3', {'accept': True}
        )
        
        assert success is False
    
    def test_register_negotiation_handler(self, negotiation_protocol):
        """Test registering negotiation handlers"""
        def test_handler(negotiation):
            return {'result': 'test'}
        
        negotiation_protocol.register_negotiation_handler('test_type', test_handler)
        
        assert 'test_type' in negotiation_protocol.negotiation_handlers
        assert negotiation_protocol.negotiation_handlers['test_type'] == test_handler


class TestEnhancedWorkflowCoordinator:
    """Test enhanced workflow coordinator"""
    
    @pytest.fixture
    def coordination_config(self):
        """Create coordination configuration"""
        return CoordinationConfig(
            kafka_servers=['localhost:9092'],
            redis_config={'host': 'localhost', 'port': 6379},
            load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
            failover_strategy=FailoverStrategy.CIRCUIT_BREAKER
        )
    
    @pytest.fixture
    def mock_components(self):
        """Mock all communication components"""
        with patch.multiple(
            'agents.enhanced_workflow_coordinator',
            create_kafka_message_bus=Mock(return_value=Mock()),
            create_redis_state_manager=Mock(return_value=Mock()),
            create_load_balancer=Mock(return_value=Mock()),
            create_failover_manager=Mock(return_value=Mock()),
            create_negotiation_protocol=Mock(return_value=Mock())
        ) as mocks:
            # Setup async methods
            kafka_mock = mocks['create_kafka_message_bus'].return_value
            kafka_mock.start = AsyncMock()
            kafka_mock.stop = AsyncMock()
            kafka_mock.publish = AsyncMock()
            kafka_mock.get_stats.return_value = {}
            
            redis_mock = mocks['create_redis_state_manager'].return_value
            redis_mock.set_agent_state = AsyncMock()
            redis_mock.set_shared_data = AsyncMock()
            redis_mock.get_shared_data = AsyncMock(return_value=None)
            
            lb_mock = mocks['create_load_balancer'].return_value
            lb_mock.register_agent_instance = Mock()
            lb_mock.select_agent_instance = Mock(return_value=Mock(agent_id='test_agent'))
            lb_mock.get_stats.return_value = {}
            
            fm_mock = mocks['create_failover_manager'].return_value
            fm_mock.can_execute.return_value = True
            fm_mock.record_success = Mock()
            fm_mock.record_failure = Mock()
            
            yield mocks
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordination_config):
        """Test coordinator initialization"""
        coordinator = EnhancedWorkflowCoordinator(coordination_config)
        
        assert coordinator.config == coordination_config
        assert coordinator.running is False
        assert len(coordinator.registered_agents) == 0
    
    @pytest.mark.asyncio
    async def test_coordinator_start_stop(self, coordination_config, mock_components):
        """Test coordinator start and stop"""
        coordinator = EnhancedWorkflowCoordinator(coordination_config)
        
        await coordinator.start()
        assert coordinator.running is True
        assert coordinator.kafka_bus is not None
        assert coordinator.redis_state is not None
        
        await coordinator.stop()
        assert coordinator.running is False
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, coordination_config, mock_components):
        """Test agent registration"""
        coordinator = EnhancedWorkflowCoordinator(coordination_config)
        await coordinator.start()
        
        success = await coordinator.register_agent(
            agent_id='test_agent',
            role=AgentRole.SIGNAL_GENERATOR,
            host='localhost',
            port=8000,
            capabilities={'strategies': ['momentum']}
        )
        
        assert success is True
        assert 'test_agent' in coordinator.registered_agents
        assert coordinator.registered_agents['test_agent'].role == AgentRole.SIGNAL_GENERATOR
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_agent_unregistration(self, coordination_config, mock_components):
        """Test agent unregistration"""
        coordinator = EnhancedWorkflowCoordinator(coordination_config)
        await coordinator.start()
        
        # First register an agent
        await coordinator.register_agent(
            'test_agent', AgentRole.SIGNAL_GENERATOR, 'localhost', 8000
        )
        
        # Then unregister
        success = await coordinator.unregister_agent('test_agent')
        
        assert success is True
        assert 'test_agent' not in coordinator.registered_agents
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_signal_fusion_coordination(self, coordination_config, mock_components):
        """Test signal fusion coordination"""
        coordinator = EnhancedWorkflowCoordinator(coordination_config)
        await coordinator.start()
        
        # Register a portfolio manager agent
        await coordinator.register_agent(
            'portfolio_agent', AgentRole.PORTFOLIO_MANAGER, 'localhost', 8000
        )
        
        # Mock coordination result
        coordinator.redis_state.get_shared_data = AsyncMock(
            return_value={'fused_signal': {'value': 0.8, 'confidence': 0.9}}
        )
        
        signals = [
            {'symbol': 'AAPL', 'value': 0.7, 'confidence': 0.8},
            {'symbol': 'AAPL', 'value': 0.9, 'confidence': 0.9}
        ]
        
        result = await coordinator.coordinate_signal_fusion('AAPL', signals, timeout=5)
        
        # Should return the mocked result
        assert result is not None
        assert 'fused_signal' in result
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_risk_assessment_coordination(self, coordination_config, mock_components):
        """Test risk assessment coordination"""
        coordinator = EnhancedWorkflowCoordinator(coordination_config)
        await coordinator.start()
        
        # Register a risk manager agent
        await coordinator.register_agent(
            'risk_agent', AgentRole.RISK_MANAGER, 'localhost', 8000
        )
        
        # Mock risk assessment result
        coordinator.redis_state.get_shared_data = AsyncMock(
            return_value={'risk_score': 0.3, 'recommendations': ['reduce_exposure']}
        )
        
        portfolio_state = {
            'positions': [{'symbol': 'AAPL', 'quantity': 100}],
            'total_value': 50000
        }
        
        result = await coordinator.coordinate_risk_assessment(portfolio_state, timeout=5)
        
        assert result is not None
        assert 'risk_score' in result
        
        await coordinator.stop()
    
    def test_get_system_status(self, coordination_config, mock_components):
        """Test getting system status"""
        coordinator = EnhancedWorkflowCoordinator(coordination_config)
        
        status = coordinator.get_system_status()
        
        assert 'running' in status
        assert 'registered_agents' in status
        assert 'statistics' in status
        assert status['running'] is False
        assert status['registered_agents'] == 0


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_coordination_flow(self):
        """Test complete coordination flow with mocked components"""
        # This would test the entire flow from agent registration
        # through coordination to result delivery
        
        # Mock all external dependencies
        with patch.multiple(
            'agents.enhanced_communication',
            KAFKA_AVAILABLE=True,
            REDIS_AVAILABLE=True
        ):
            with patch('agents.enhanced_communication.KafkaProducer'), \
                 patch('agents.enhanced_communication.KafkaConsumer'), \
                 patch('agents.enhanced_communication.redis.Redis') as mock_redis:
                
                # Setup Redis mock
                redis_instance = Mock()
                mock_redis.return_value = redis_instance
                redis_instance.ping.return_value = True
                redis_instance.hset.return_value = True
                redis_instance.set.return_value = True
                redis_instance.get.return_value = '{"result": "success"}'
                
                # Create coordinator
                config = CoordinationConfig()
                coordinator = EnhancedWorkflowCoordinator(config)
                
                try:
                    await coordinator.start()
                    
                    # Register agents
                    await coordinator.register_agent(
                        'signal_agent', AgentRole.SIGNAL_GENERATOR, 'localhost', 8001
                    )
                    await coordinator.register_agent(
                        'portfolio_agent', AgentRole.PORTFOLIO_MANAGER, 'localhost', 8002
                    )
                    
                    # Verify agents are registered
                    assert len(coordinator.registered_agents) == 2
                    
                    # Test coordination (would timeout in real scenario, but tests the setup)
                    signals = [{'symbol': 'AAPL', 'value': 0.8}]
                    
                    # This will timeout, but we're testing the setup
                    result = await coordinator.coordinate_signal_fusion('AAPL', signals, timeout=1)
                    
                    # In a real scenario with proper message handling, this would succeed
                    # For now, we just verify the coordination was attempted
                    assert coordinator.stats['coordinations_initiated'] > 0
                    
                finally:
                    await coordinator.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])