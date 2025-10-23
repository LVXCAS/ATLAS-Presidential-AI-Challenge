"""
Enhanced Agent Coordination and Message Passing System

This module implements high-throughput agent communication using Kafka,
Redis for shared state management, agent negotiation protocols,
and load balancing with failover mechanisms.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Kafka and Redis imports
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("Kafka not available, using fallback message bus")

try:
    import redis
    from redis.sentinel import Sentinel
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory state management")

from pydantic import BaseModel, Field

from .communication_protocols import (
    Message, MessageType, MessagePriority, AgentRole, MessageHandler
)

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    PERFORMANCE_BASED = "performance_based"


class FailoverStrategy(str, Enum):
    """Failover strategies"""
    IMMEDIATE = "immediate"
    GRACEFUL = "graceful"
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_WITH_BACKOFF = "retry_with_backoff"


@dataclass
class AgentInstance:
    """Agent instance information"""
    agent_id: str
    role: AgentRole
    host: str
    port: int
    status: str = "active"
    load: float = 0.0
    connections: int = 0
    last_heartbeat: datetime = None
    performance_score: float = 1.0
    capabilities: Dict[str, Any] = None
    resource_usage: Dict[str, float] = None
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()
        if self.capabilities is None:
            self.capabilities = {}
        if self.resource_usage is None:
            self.resource_usage = {}


@dataclass
class NegotiationRequest:
    """Agent negotiation request"""
    request_id: str
    initiator: str
    participants: List[str]
    negotiation_type: str
    proposal: Dict[str, Any]
    deadline: datetime
    status: str = "pending"
    responses: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.responses is None:
            self.responses = {}


class KafkaMessageBus:
    """
    High-throughput message bus using Apache Kafka
    """
    
    def __init__(self, bootstrap_servers: List[str] = None, config: Dict[str, Any] = None):
        if not KAFKA_AVAILABLE:
            raise ImportError("Kafka is not available. Install kafka-python package.")
        
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        self.config = config or {}
        
        # Kafka producer configuration
        producer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v, default=str).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'batch_size': 16384,
            'linger_ms': 10,
            'buffer_memory': 33554432,
            **self.config.get('producer', {})
        }
        
        # Kafka consumer configuration
        consumer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda k: k.decode('utf-8') if k else None,
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True,
            'group_id': 'trading_system',
            **self.config.get('consumer', {})
        }
        
        self.producer = KafkaProducer(**producer_config)
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.running = False
        self.consumer_threads: Dict[str, threading.Thread] = {}
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "topics_subscribed": 0,
            "active_consumers": 0
        }
        
        logger.info(f"Kafka message bus initialized with servers: {self.bootstrap_servers}")
    
    async def start(self):
        """Start the Kafka message bus"""
        self.running = True
        logger.info("Kafka message bus started")
    
    async def stop(self):
        """Stop the Kafka message bus"""
        self.running = False
        
        # Stop all consumers
        for topic, consumer in self.consumers.items():
            consumer.close()
        
        # Stop all consumer threads
        for thread in self.consumer_threads.values():
            thread.join(timeout=5.0)
        
        # Close producer
        self.producer.close()
        
        logger.info("Kafka message bus stopped")
    
    def subscribe(self, topic: str, handler: Callable, consumer_group: str = None):
        """Subscribe to a Kafka topic"""
        if topic not in self.consumers:
            consumer_config = {
                'bootstrap_servers': self.bootstrap_servers,
                'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
                'key_deserializer': lambda k: k.decode('utf-8') if k else None,
                'auto_offset_reset': 'latest',
                'enable_auto_commit': True,
                'group_id': consumer_group or f'trading_system_{topic}',
            }
            
            consumer = KafkaConsumer(topic, **consumer_config)
            self.consumers[topic] = consumer
            
            # Start consumer thread
            thread = threading.Thread(
                target=self._consume_messages,
                args=(topic, consumer),
                daemon=True
            )
            thread.start()
            self.consumer_threads[topic] = thread
            
            self.stats["topics_subscribed"] += 1
            self.stats["active_consumers"] += 1
        
        # Add handler
        if topic not in self.message_handlers:
            self.message_handlers[topic] = []
        self.message_handlers[topic].append(handler)
        
        logger.debug(f"Subscribed to topic: {topic}")
    
    def _consume_messages(self, topic: str, consumer):
        """Consume messages from Kafka topic"""
        while self.running:
            try:
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Process message
                            handlers = self.message_handlers.get(topic, [])
                            for handler in handlers:
                                if asyncio.iscoroutinefunction(handler):
                                    # Run async handler in event loop
                                    asyncio.create_task(handler(message.value))
                                else:
                                    handler(message.value)
                            
                            self.stats["messages_received"] += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing message from {topic}: {e}")
                            self.stats["messages_failed"] += 1
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error consuming from {topic}: {e}")
                    time.sleep(1)  # Brief pause before retry
    
    async def publish(self, topic: str, message: Dict[str, Any], key: str = None):
        """Publish message to Kafka topic"""
        try:
            future = self.producer.send(topic, value=message, key=key)
            record_metadata = future.get(timeout=10)
            
            self.stats["messages_sent"] += 1
            
            logger.debug(f"Message sent to {topic}: partition {record_metadata.partition}, offset {record_metadata.offset}")
            
        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            self.stats["messages_failed"] += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Kafka message bus statistics"""
        return self.stats.copy()


class RedisStateManager:
    """
    Shared state management using Redis
    """
    
    def __init__(self, redis_config: Dict[str, Any] = None, use_sentinel: bool = False):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install redis package.")
        
        self.config = redis_config or {}
        self.use_sentinel = use_sentinel
        
        if use_sentinel:
            # Redis Sentinel for high availability
            sentinels = self.config.get('sentinels', [('localhost', 26379)])
            service_name = self.config.get('service_name', 'mymaster')
            
            sentinel = Sentinel(sentinels)
            self.redis_client = sentinel.master_for(service_name, socket_timeout=0.1)
            self.redis_slave = sentinel.slave_for(service_name, socket_timeout=0.1)
        else:
            # Single Redis instance
            redis_config = {
                'host': self.config.get('host', 'localhost'),
                'port': self.config.get('port', 6379),
                'db': self.config.get('db', 0),
                'decode_responses': True,
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'retry_on_timeout': True,
                **self.config
            }
            
            self.redis_client = redis.Redis(**redis_config)
            self.redis_slave = self.redis_client  # Same instance for reads
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Redis state manager initialized successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # State prefixes
        self.prefixes = {
            'agent_state': 'agent:state:',
            'shared_data': 'shared:data:',
            'coordination': 'coord:',
            'locks': 'lock:',
            'counters': 'counter:',
            'queues': 'queue:'
        }
    
    async def set_agent_state(self, agent_id: str, state: Dict[str, Any], ttl: int = None):
        """Set agent state in Redis"""
        key = f"{self.prefixes['agent_state']}{agent_id}"
        
        try:
            pipeline = self.redis_client.pipeline()
            pipeline.hset(key, mapping=state)
            if ttl:
                pipeline.expire(key, ttl)
            pipeline.execute()
            
            logger.debug(f"Agent state set for {agent_id}")
            
        except redis.RedisError as e:
            logger.error(f"Failed to set agent state for {agent_id}: {e}")
            raise
    
    async def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get agent state from Redis"""
        key = f"{self.prefixes['agent_state']}{agent_id}"
        
        try:
            state = self.redis_slave.hgetall(key)
            return dict(state) if state else {}
            
        except redis.RedisError as e:
            logger.error(f"Failed to get agent state for {agent_id}: {e}")
            return {}
    
    async def set_shared_data(self, data_key: str, data: Any, ttl: int = None):
        """Set shared data in Redis"""
        key = f"{self.prefixes['shared_data']}{data_key}"
        
        try:
            serialized_data = json.dumps(data, default=str)
            
            if ttl:
                self.redis_client.setex(key, ttl, serialized_data)
            else:
                self.redis_client.set(key, serialized_data)
            
            logger.debug(f"Shared data set: {data_key}")
            
        except (redis.RedisError, json.JSONEncodeError) as e:
            logger.error(f"Failed to set shared data {data_key}: {e}")
            raise
    
    async def get_shared_data(self, data_key: str) -> Any:
        """Get shared data from Redis"""
        key = f"{self.prefixes['shared_data']}{data_key}"
        
        try:
            data = self.redis_slave.get(key)
            if data:
                return json.loads(data)
            return None
            
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get shared data {data_key}: {e}")
            return None
    
    async def acquire_lock(self, lock_name: str, timeout: int = 10, blocking_timeout: int = 5) -> bool:
        """Acquire distributed lock"""
        key = f"{self.prefixes['locks']}{lock_name}"
        identifier = str(uuid.uuid4())
        
        try:
            # Try to acquire lock with timeout
            end_time = time.time() + blocking_timeout
            
            while time.time() < end_time:
                if self.redis_client.set(key, identifier, nx=True, ex=timeout):
                    logger.debug(f"Lock acquired: {lock_name}")
                    return True
                
                time.sleep(0.001)  # 1ms sleep
            
            logger.debug(f"Failed to acquire lock: {lock_name}")
            return False
            
        except redis.RedisError as e:
            logger.error(f"Error acquiring lock {lock_name}: {e}")
            return False
    
    async def release_lock(self, lock_name: str, identifier: str = None) -> bool:
        """Release distributed lock"""
        key = f"{self.prefixes['locks']}{lock_name}"
        
        try:
            # Lua script for atomic lock release
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            
            result = self.redis_client.eval(lua_script, 1, key, identifier or "")
            
            if result:
                logger.debug(f"Lock released: {lock_name}")
                return True
            else:
                logger.warning(f"Failed to release lock (not owner): {lock_name}")
                return False
                
        except redis.RedisError as e:
            logger.error(f"Error releasing lock {lock_name}: {e}")
            return False
    
    async def increment_counter(self, counter_name: str, amount: int = 1) -> int:
        """Increment atomic counter"""
        key = f"{self.prefixes['counters']}{counter_name}"
        
        try:
            result = self.redis_client.incrby(key, amount)
            return result
            
        except redis.RedisError as e:
            logger.error(f"Error incrementing counter {counter_name}: {e}")
            return 0
    
    async def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        key = f"{self.prefixes['counters']}{counter_name}"
        
        try:
            result = self.redis_slave.get(key)
            return int(result) if result else 0
            
        except (redis.RedisError, ValueError) as e:
            logger.error(f"Error getting counter {counter_name}: {e}")
            return 0
    
    async def push_to_queue(self, queue_name: str, item: Any) -> int:
        """Push item to Redis queue"""
        key = f"{self.prefixes['queues']}{queue_name}"
        
        try:
            serialized_item = json.dumps(item, default=str)
            result = self.redis_client.lpush(key, serialized_item)
            return result
            
        except (redis.RedisError, json.JSONEncodeError) as e:
            logger.error(f"Error pushing to queue {queue_name}: {e}")
            return 0
    
    async def pop_from_queue(self, queue_name: str, timeout: int = 0) -> Any:
        """Pop item from Redis queue"""
        key = f"{self.prefixes['queues']}{queue_name}"
        
        try:
            if timeout > 0:
                result = self.redis_client.brpop(key, timeout=timeout)
                if result:
                    return json.loads(result[1])
            else:
                result = self.redis_client.rpop(key)
                if result:
                    return json.loads(result)
            
            return None
            
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error popping from queue {queue_name}: {e}")
            return None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get Redis connection information"""
        try:
            info = self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                "redis_version": info.get("redis_version", "unknown")
            }
        except redis.RedisError as e:
            logger.error(f"Error getting Redis info: {e}")
            return {}


class LoadBalancer:
    """
    Load balancer for distributing requests across agent instances
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.agent_instances: Dict[str, List[AgentInstance]] = {}
        self.round_robin_counters: Dict[str, int] = {}
        self.stats = {
            "requests_routed": 0,
            "failed_routes": 0,
            "agent_failures": 0
        }
        
        logger.info(f"Load balancer initialized with strategy: {strategy}")
    
    def register_agent_instance(self, instance: AgentInstance):
        """Register an agent instance"""
        role = instance.role.value
        
        if role not in self.agent_instances:
            self.agent_instances[role] = []
            self.round_robin_counters[role] = 0
        
        self.agent_instances[role].append(instance)
        logger.info(f"Agent instance registered: {instance.agent_id} ({role})")
    
    def unregister_agent_instance(self, agent_id: str, role: AgentRole):
        """Unregister an agent instance"""
        role_str = role.value
        
        if role_str in self.agent_instances:
            self.agent_instances[role_str] = [
                instance for instance in self.agent_instances[role_str]
                if instance.agent_id != agent_id
            ]
            logger.info(f"Agent instance unregistered: {agent_id} ({role_str})")
    
    def select_agent_instance(self, role: AgentRole, request_context: Dict[str, Any] = None) -> Optional[AgentInstance]:
        """Select an agent instance based on load balancing strategy"""
        role_str = role.value
        
        if role_str not in self.agent_instances or not self.agent_instances[role_str]:
            logger.warning(f"No instances available for role: {role_str}")
            return None
        
        instances = [inst for inst in self.agent_instances[role_str] if inst.status == "active"]
        
        if not instances:
            logger.warning(f"No active instances available for role: {role_str}")
            return None
        
        try:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                selected = self._round_robin_selection(role_str, instances)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                selected = self._least_connections_selection(instances)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                selected = self._weighted_round_robin_selection(instances)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                selected = self._resource_based_selection(instances)
            elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
                selected = self._performance_based_selection(instances)
            else:
                selected = instances[0]  # Fallback
            
            self.stats["requests_routed"] += 1
            return selected
            
        except Exception as e:
            logger.error(f"Error selecting agent instance: {e}")
            self.stats["failed_routes"] += 1
            return None
    
    def _round_robin_selection(self, role: str, instances: List[AgentInstance]) -> AgentInstance:
        """Round-robin selection"""
        counter = self.round_robin_counters[role]
        selected = instances[counter % len(instances)]
        self.round_robin_counters[role] = (counter + 1) % len(instances)
        return selected
    
    def _least_connections_selection(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select instance with least connections"""
        return min(instances, key=lambda x: x.connections)
    
    def _weighted_round_robin_selection(self, instances: List[AgentInstance]) -> AgentInstance:
        """Weighted round-robin based on performance score"""
        # Simple implementation - could be enhanced with proper weighted selection
        weights = [inst.performance_score for inst in instances]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return instances[0]
        
        # Normalize weights and select
        normalized_weights = [w / total_weight for w in weights]
        
        import random
        r = random.random()
        cumulative = 0
        
        for i, weight in enumerate(normalized_weights):
            cumulative += weight
            if r <= cumulative:
                return instances[i]
        
        return instances[-1]  # Fallback
    
    def _resource_based_selection(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select instance with lowest resource usage"""
        def resource_score(instance):
            usage = instance.resource_usage
            cpu = usage.get('cpu_percent', 0)
            memory = usage.get('memory_percent', 0)
            return cpu + memory  # Simple scoring
        
        return min(instances, key=resource_score)
    
    def _performance_based_selection(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select instance with highest performance score"""
        return max(instances, key=lambda x: x.performance_score)
    
    def update_instance_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Update instance metrics for load balancing decisions"""
        for role_instances in self.agent_instances.values():
            for instance in role_instances:
                if instance.agent_id == agent_id:
                    instance.load = metrics.get('load', instance.load)
                    instance.connections = metrics.get('connections', instance.connections)
                    instance.performance_score = metrics.get('performance_score', instance.performance_score)
                    instance.resource_usage = metrics.get('resource_usage', instance.resource_usage)
                    instance.last_heartbeat = datetime.now()
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        total_instances = sum(len(instances) for instances in self.agent_instances.values())
        active_instances = sum(
            len([inst for inst in instances if inst.status == "active"])
            for instances in self.agent_instances.values()
        )
        
        return {
            **self.stats,
            "total_instances": total_instances,
            "active_instances": active_instances,
            "roles": list(self.agent_instances.keys())
        }


class FailoverManager:
    """
    Manages failover scenarios and circuit breaker patterns
    """
    
    def __init__(self, strategy: FailoverStrategy = FailoverStrategy.CIRCUIT_BREAKER):
        self.strategy = strategy
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, datetime] = {}
        
        # Circuit breaker configuration
        self.circuit_config = {
            'failure_threshold': 5,
            'recovery_timeout': 60,  # seconds
            'half_open_max_calls': 3
        }
        
        logger.info(f"Failover manager initialized with strategy: {strategy}")
    
    def record_success(self, agent_id: str):
        """Record successful operation"""
        if agent_id in self.failure_counts:
            self.failure_counts[agent_id] = 0
        
        # Reset circuit breaker if in half-open state
        if agent_id in self.circuit_breakers:
            cb = self.circuit_breakers[agent_id]
            if cb['state'] == 'half_open':
                cb['state'] = 'closed'
                cb['half_open_calls'] = 0
                logger.info(f"Circuit breaker closed for {agent_id}")
    
    def record_failure(self, agent_id: str, error: Exception = None):
        """Record failed operation"""
        self.failure_counts[agent_id] = self.failure_counts.get(agent_id, 0) + 1
        self.last_failure_time[agent_id] = datetime.now()
        
        # Check if circuit breaker should open
        if self.failure_counts[agent_id] >= self.circuit_config['failure_threshold']:
            self._open_circuit_breaker(agent_id)
        
        logger.warning(f"Failure recorded for {agent_id}: {error}")
    
    def _open_circuit_breaker(self, agent_id: str):
        """Open circuit breaker for agent"""
        self.circuit_breakers[agent_id] = {
            'state': 'open',
            'opened_at': datetime.now(),
            'half_open_calls': 0
        }
        
        logger.warning(f"Circuit breaker opened for {agent_id}")
    
    def can_execute(self, agent_id: str) -> bool:
        """Check if operation can be executed on agent"""
        if agent_id not in self.circuit_breakers:
            return True
        
        cb = self.circuit_breakers[agent_id]
        
        if cb['state'] == 'closed':
            return True
        elif cb['state'] == 'open':
            # Check if recovery timeout has passed
            if datetime.now() - cb['opened_at'] > timedelta(seconds=self.circuit_config['recovery_timeout']):
                cb['state'] = 'half_open'
                cb['half_open_calls'] = 0
                logger.info(f"Circuit breaker half-opened for {agent_id}")
                return True
            return False
        elif cb['state'] == 'half_open':
            # Allow limited calls in half-open state
            if cb['half_open_calls'] < self.circuit_config['half_open_max_calls']:
                cb['half_open_calls'] += 1
                return True
            return False
        
        return False
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent failover status"""
        return {
            'failure_count': self.failure_counts.get(agent_id, 0),
            'last_failure': self.last_failure_time.get(agent_id),
            'circuit_breaker': self.circuit_breakers.get(agent_id, {'state': 'closed'})
        }


class NegotiationProtocol:
    """
    Implements agent negotiation and consensus protocols
    """
    
    def __init__(self, state_manager: RedisStateManager):
        self.state_manager = state_manager
        self.active_negotiations: Dict[str, NegotiationRequest] = {}
        self.negotiation_handlers: Dict[str, Callable] = {}
        
        logger.info("Negotiation protocol initialized")
    
    def register_negotiation_handler(self, negotiation_type: str, handler: Callable):
        """Register handler for specific negotiation type"""
        self.negotiation_handlers[negotiation_type] = handler
        logger.debug(f"Negotiation handler registered for: {negotiation_type}")
    
    async def initiate_negotiation(self, 
                                 negotiation_type: str,
                                 initiator: str,
                                 participants: List[str],
                                 proposal: Dict[str, Any],
                                 timeout_seconds: int = 30) -> str:
        """Initiate negotiation between agents"""
        request_id = str(uuid.uuid4())
        deadline = datetime.now() + timedelta(seconds=timeout_seconds)
        
        negotiation = NegotiationRequest(
            request_id=request_id,
            initiator=initiator,
            participants=participants,
            negotiation_type=negotiation_type,
            proposal=proposal,
            deadline=deadline
        )
        
        self.active_negotiations[request_id] = negotiation
        
        # Store in Redis for persistence
        await self.state_manager.set_shared_data(
            f"negotiation:{request_id}",
            asdict(negotiation),
            ttl=timeout_seconds + 60
        )
        
        logger.info(f"Negotiation initiated: {request_id} ({negotiation_type})")
        return request_id
    
    async def respond_to_negotiation(self, 
                                   request_id: str,
                                   respondent: str,
                                   response: Dict[str, Any]) -> bool:
        """Respond to negotiation request"""
        # Get negotiation from Redis if not in memory
        if request_id not in self.active_negotiations:
            negotiation_data = await self.state_manager.get_shared_data(f"negotiation:{request_id}")
            if not negotiation_data:
                logger.warning(f"Negotiation not found: {request_id}")
                return False
            
            self.active_negotiations[request_id] = NegotiationRequest(**negotiation_data)
        
        negotiation = self.active_negotiations[request_id]
        
        # Check if respondent is valid participant
        if respondent not in negotiation.participants:
            logger.warning(f"Invalid respondent {respondent} for negotiation {request_id}")
            return False
        
        # Check if negotiation is still active
        if datetime.now() > negotiation.deadline:
            logger.warning(f"Negotiation {request_id} has expired")
            return False
        
        # Record response
        negotiation.responses[respondent] = response
        
        # Update in Redis
        await self.state_manager.set_shared_data(
            f"negotiation:{request_id}",
            asdict(negotiation)
        )
        
        # Check if all participants have responded
        if len(negotiation.responses) == len(negotiation.participants):
            await self._finalize_negotiation(request_id)
        
        logger.debug(f"Response recorded for negotiation {request_id} from {respondent}")
        return True
    
    async def _finalize_negotiation(self, request_id: str):
        """Finalize negotiation when all responses received"""
        negotiation = self.active_negotiations[request_id]
        
        # Get handler for negotiation type
        handler = self.negotiation_handlers.get(negotiation.negotiation_type)
        
        if handler:
            try:
                result = await handler(negotiation)
                negotiation.status = "completed"
                
                # Store final result
                await self.state_manager.set_shared_data(
                    f"negotiation_result:{request_id}",
                    result,
                    ttl=3600  # Keep result for 1 hour
                )
                
                logger.info(f"Negotiation completed: {request_id}")
                
            except Exception as e:
                logger.error(f"Error finalizing negotiation {request_id}: {e}")
                negotiation.status = "failed"
        else:
            logger.warning(f"No handler for negotiation type: {negotiation.negotiation_type}")
            negotiation.status = "failed"
        
        # Clean up
        if request_id in self.active_negotiations:
            del self.active_negotiations[request_id]
    
    async def get_negotiation_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get negotiation result"""
        return await self.state_manager.get_shared_data(f"negotiation_result:{request_id}")


# Factory functions for creating components
def create_kafka_message_bus(bootstrap_servers: List[str] = None, config: Dict[str, Any] = None) -> KafkaMessageBus:
    """Create Kafka message bus instance"""
    return KafkaMessageBus(bootstrap_servers, config)


def create_redis_state_manager(redis_config: Dict[str, Any] = None, use_sentinel: bool = False) -> RedisStateManager:
    """Create Redis state manager instance"""
    return RedisStateManager(redis_config, use_sentinel)


def create_load_balancer(strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> LoadBalancer:
    """Create load balancer instance"""
    return LoadBalancer(strategy)


def create_failover_manager(strategy: FailoverStrategy = FailoverStrategy.CIRCUIT_BREAKER) -> FailoverManager:
    """Create failover manager instance"""
    return FailoverManager(strategy)


def create_negotiation_protocol(state_manager: RedisStateManager) -> NegotiationProtocol:
    """Create negotiation protocol instance"""
    return NegotiationProtocol(state_manager)


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        try:
            # Create components
            kafka_bus = create_kafka_message_bus()
            redis_state = create_redis_state_manager()
            load_balancer = create_load_balancer()
            failover_manager = create_failover_manager()
            negotiation_protocol = create_negotiation_protocol(redis_state)
            
            print("Enhanced communication system initialized successfully")
            print("Kafka stats:", kafka_bus.get_stats())
            print("Redis info:", redis_state.get_connection_info())
            print("Load balancer stats:", load_balancer.get_stats())
            
        except Exception as e:
            print(f"Error initializing system: {e}")
    
    asyncio.run(main())