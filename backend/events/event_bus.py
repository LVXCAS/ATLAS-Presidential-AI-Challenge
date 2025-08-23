"""
Redis-based Event Bus for Bloomberg Terminal
High-performance pub/sub event system for sub-50ms latency.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the trading system."""
    # Market data events
    MARKET_DATA_UPDATE = "market_data_update"
    PRICE_TICK = "price_tick"
    VOLUME_UPDATE = "volume_update"
    
    # Trading signal events
    TRADING_SIGNAL = "trading_signal"
    SIGNAL_CONSENSUS = "signal_consensus"
    SIGNAL_CONFLICT = "signal_conflict"
    
    # Order events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Risk events
    RISK_ALERT = "risk_alert"
    POSITION_UPDATE = "position_update"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    
    # Agent events
    AGENT_SIGNAL = "agent_signal"
    AGENT_STATUS_CHANGE = "agent_status_change"
    AGENT_PERFORMANCE_UPDATE = "agent_performance_update"
    
    # System events
    SYSTEM_HEALTH = "system_health"
    EMERGENCY_STOP = "emergency_stop"
    MARKET_HOURS_CHANGE = "market_hours_change"


@dataclass
class Event:
    """Base event class for all system events."""
    id: str
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=critical
    ttl_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'id': self.id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'correlation_id': self.correlation_id,
            'priority': self.priority,
            'ttl_seconds': self.ttl_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            id=data['id'],
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            data=data['data'],
            correlation_id=data.get('correlation_id'),
            priority=data.get('priority', 0),
            ttl_seconds=data.get('ttl_seconds')
        )


class EventBus:
    """
    High-performance Redis-based event bus with sub-50ms latency.
    
    Features:
    - Asynchronous pub/sub with Redis
    - Event filtering and routing
    - Priority queuing
    - Dead letter queues
    - Event replay capabilities
    - Circuit breaker for resilience
    - Performance monitoring
    """
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        default_redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'decode_responses': True,
            'socket_keepalive': True,
            'socket_keepalive_options': {},
            'retry_on_timeout': True,
            'connection_pool_kwargs': {
                'max_connections': 50,
                'retry_on_timeout': True
            }
        }
        
        if redis_config:
            default_redis_config.update(redis_config)
        
        self.redis_config = default_redis_config
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Event handling
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.channel_subscribers: Dict[str, Set[str]] = {}
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            'events_published': 0,
            'events_consumed': 0,
            'avg_latency_ms': 0.0,
            'failed_publishes': 0,
            'circuit_breaker_trips': 0,
            'last_health_check': None
        }
        
        # Circuit breaker
        self.circuit_breaker = {
            'failures': 0,
            'failure_threshold': 5,
            'recovery_timeout': 30,
            'last_failure': None,
            'state': 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        }
        
    async def initialize(self) -> None:
        """Initialize Redis connection and event bus."""
        try:
            logger.info("Initializing Redis Event Bus")
            
            # Create Redis connection
            self.redis_client = redis.Redis(**self.redis_config)
            
            # Test connection
            await self.redis_client.ping()
            
            # Create pub/sub client
            self.pubsub = self.redis_client.pubsub()
            
            logger.info("Redis Event Bus initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis Event Bus: {e}")
            raise
    
    async def start(self) -> None:
        """Start the event bus."""
        if not self.redis_client:
            await self.initialize()
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._metrics_update_loop())
        
        logger.info("Event Bus started")
    
    async def stop(self) -> None:
        """Stop the event bus and cleanup connections."""
        self.is_running = False
        
        if self.pubsub:
            await self.pubsub.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Event Bus stopped")
    
    async def publish(self, event: Event, channel: Optional[str] = None) -> bool:
        """
        Publish event to Redis with high performance.
        
        Args:
            event: Event to publish
            channel: Optional specific channel (defaults to event type)
            
        Returns:
            Success status
        """
        try:
            # Check circuit breaker
            if not self._circuit_breaker_allow():
                logger.warning("Circuit breaker OPEN - rejecting publish")
                return False
            
            start_time = asyncio.get_event_loop().time()
            
            # Determine channel
            if not channel:
                channel = f"events:{event.event_type.value}"
            
            # Serialize event
            event_data = json.dumps(event.to_dict())
            
            # Publish with priority handling
            if event.priority > 0:
                # High priority events go to priority channel
                priority_channel = f"{channel}:priority"
                result = await self.redis_client.publish(priority_channel, event_data)
            else:
                result = await self.redis_client.publish(channel, event_data)
            
            # Store in event log for replay capability
            await self._store_event_for_replay(event, channel)
            
            # Update metrics
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            await self._update_latency_metrics(latency_ms)
            
            self.metrics['events_published'] += 1
            
            # Reset circuit breaker on success
            await self._circuit_breaker_success()
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.id}: {e}")
            self.metrics['failed_publishes'] += 1
            await self._circuit_breaker_failure()
            return False
    
    async def subscribe(
        self, 
        event_types: List[EventType], 
        handler: Callable[[Event], None],
        channel_pattern: Optional[str] = None
    ) -> str:
        """
        Subscribe to specific event types with a handler.
        
        Args:
            event_types: List of event types to subscribe to
            handler: Async function to handle events
            channel_pattern: Optional channel pattern for filtering
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        
        try:
            # Register handlers
            for event_type in event_types:
                if event_type not in self.event_handlers:
                    self.event_handlers[event_type] = []
                self.event_handlers[event_type].append(handler)
                
                # Subscribe to Redis channels
                channel = channel_pattern or f"events:{event_type.value}"
                await self.pubsub.subscribe(channel)
                
                # Track subscription
                if channel not in self.channel_subscribers:
                    self.channel_subscribers[channel] = set()
                self.channel_subscribers[channel].add(subscription_id)
            
            # Start message consumption if not already running
            if not hasattr(self, '_consumer_task'):
                self._consumer_task = asyncio.create_task(self._consume_messages())
            
            logger.info(f"Subscribed to {len(event_types)} event types with ID {subscription_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events."""
        try:
            # Remove from channel subscribers
            channels_to_remove = []
            for channel, subscribers in self.channel_subscribers.items():
                if subscription_id in subscribers:
                    subscribers.remove(subscription_id)
                    if not subscribers:
                        channels_to_remove.append(channel)
            
            # Unsubscribe from empty channels
            for channel in channels_to_remove:
                await self.pubsub.unsubscribe(channel)
                del self.channel_subscribers[channel]
            
            logger.info(f"Unsubscribed {subscription_id}")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe {subscription_id}: {e}")
    
    async def publish_trading_signal(self, signal_data: Dict[str, Any], source: str) -> bool:
        """Convenience method to publish trading signals."""
        event = Event(
            id=str(uuid.uuid4()),
            event_type=EventType.TRADING_SIGNAL,
            timestamp=datetime.now(timezone.utc),
            source=source,
            data=signal_data,
            priority=1  # High priority for trading signals
        )
        return await self.publish(event)
    
    async def publish_market_data(self, market_data: Dict[str, Any], source: str) -> bool:
        """Convenience method to publish market data updates."""
        event = Event(
            id=str(uuid.uuid4()),
            event_type=EventType.MARKET_DATA_UPDATE,
            timestamp=datetime.now(timezone.utc),
            source=source,
            data=market_data
        )
        return await self.publish(event)
    
    async def publish_risk_alert(self, alert_data: Dict[str, Any], source: str) -> bool:
        """Convenience method to publish risk alerts."""
        event = Event(
            id=str(uuid.uuid4()),
            event_type=EventType.RISK_ALERT,
            timestamp=datetime.now(timezone.utc),
            source=source,
            data=alert_data,
            priority=2  # Critical priority for risk alerts
        )
        return await self.publish(event)
    
    async def get_event_history(
        self, 
        event_type: EventType, 
        start_time: datetime, 
        end_time: datetime,
        limit: int = 1000
    ) -> List[Event]:
        """Get event history for replay/analysis."""
        try:
            key = f"event_log:{event_type.value}"
            
            # Query Redis sorted set by timestamp
            start_score = start_time.timestamp()
            end_score = end_time.timestamp()
            
            event_data = await self.redis_client.zrangebyscore(
                key, start_score, end_score, start=0, num=limit
            )
            
            events = []
            for data in event_data:
                try:
                    event_dict = json.loads(data)
                    events.append(Event.from_dict(event_dict))
                except Exception as e:
                    logger.error(f"Failed to deserialize event: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get event history: {e}")
            return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get event bus performance metrics."""
        return {
            **self.metrics,
            'circuit_breaker': self.circuit_breaker,
            'active_subscribers': sum(len(subs) for subs in self.channel_subscribers.values()),
            'subscribed_channels': len(self.channel_subscribers)
        }
    
    async def _consume_messages(self) -> None:
        """Background task to consume and dispatch messages."""
        try:
            async for message in self.pubsub.listen():
                if not self.is_running:
                    break
                
                if message['type'] != 'message':
                    continue
                
                try:
                    # Deserialize event
                    event_dict = json.loads(message['data'])
                    event = Event.from_dict(event_dict)
                    
                    # Dispatch to handlers
                    await self._dispatch_event(event)
                    
                    self.metrics['events_consumed'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
        
        except Exception as e:
            logger.error(f"Message consumption error: {e}")
            if self.is_running:
                # Restart consumption after a delay
                await asyncio.sleep(5)
                asyncio.create_task(self._consume_messages())
    
    async def _dispatch_event(self, event: Event) -> None:
        """Dispatch event to registered handlers."""
        handlers = self.event_handlers.get(event.event_type, [])
        
        if not handlers:
            return
        
        # Execute handlers concurrently
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._safe_handler_call(handler, event))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_handler_call(self, handler: Callable, event: Event) -> None:
        """Safely call event handler with error handling."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Handler error for event {event.id}: {e}")
    
    async def _store_event_for_replay(self, event: Event, channel: str) -> None:
        """Store event in Redis for replay capability."""
        try:
            key = f"event_log:{event.event_type.value}"
            score = event.timestamp.timestamp()
            value = json.dumps(event.to_dict())
            
            # Store in sorted set with timestamp as score
            await self.redis_client.zadd(key, {value: score})
            
            # Set expiration for event logs (7 days)
            await self.redis_client.expire(key, 7 * 24 * 3600)
            
        except Exception as e:
            logger.error(f"Failed to store event for replay: {e}")
    
    async def _update_latency_metrics(self, latency_ms: float) -> None:
        """Update latency metrics with exponential moving average."""
        current_avg = self.metrics['avg_latency_ms']
        alpha = 0.1  # Smoothing factor
        self.metrics['avg_latency_ms'] = alpha * latency_ms + (1 - alpha) * current_avg
    
    def _circuit_breaker_allow(self) -> bool:
        """Check if circuit breaker allows operation."""
        now = datetime.now().timestamp()
        
        if self.circuit_breaker['state'] == 'OPEN':
            # Check if recovery timeout has passed
            if (now - self.circuit_breaker['last_failure']) > self.circuit_breaker['recovery_timeout']:
                self.circuit_breaker['state'] = 'HALF_OPEN'
                return True
            return False
        
        return True
    
    async def _circuit_breaker_success(self) -> None:
        """Record successful operation for circuit breaker."""
        if self.circuit_breaker['state'] == 'HALF_OPEN':
            self.circuit_breaker['state'] = 'CLOSED'
            self.circuit_breaker['failures'] = 0
    
    async def _circuit_breaker_failure(self) -> None:
        """Record failure for circuit breaker."""
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = datetime.now().timestamp()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['failure_threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self.metrics['circuit_breaker_trips'] += 1
            logger.warning("Circuit breaker OPEN due to failures")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self.is_running:
            try:
                # Ping Redis
                await self.redis_client.ping()
                self.metrics['last_health_check'] = datetime.now()
                
                # Reset circuit breaker if healthy
                if self.circuit_breaker['state'] == 'OPEN':
                    self.circuit_breaker['failures'] = 0
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await self._circuit_breaker_failure()
            
            await asyncio.sleep(30)  # Health check every 30 seconds
    
    async def _metrics_update_loop(self) -> None:
        """Background metrics update loop."""
        while self.is_running:
            try:
                # Store metrics in Redis for monitoring
                metrics_key = "event_bus:metrics"
                metrics_data = json.dumps({
                    **self.metrics,
                    'timestamp': datetime.now().isoformat()
                })
                
                await self.redis_client.set(metrics_key, metrics_data, ex=3600)  # 1 hour expiry
                
            except Exception as e:
                logger.error(f"Failed to update metrics: {e}")
            
            await asyncio.sleep(60)  # Update metrics every minute


# Global event bus instance
event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global event_bus
    if event_bus is None:
        event_bus = EventBus()
    return event_bus


async def initialize_event_bus(redis_config: Dict[str, Any] = None) -> EventBus:
    """Initialize and start the global event bus."""
    global event_bus
    event_bus = EventBus(redis_config)
    await event_bus.initialize()
    await event_bus.start()
    return event_bus