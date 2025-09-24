import asyncio
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict
from datetime import datetime
import heapq
import threading
from contextlib import asynccontextmanager


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Event:
    name: str
    data: Any = None
    priority: Priority = Priority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return (self.priority.value, self.timestamp) < (other.priority.value, other.timestamp)


@dataclass
class EventSubscription:
    callback: Union[Callable, Coroutine]
    event_filter: Optional[Callable[[Event], bool]] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    dead_letter: bool = True


class EventBus:
    def __init__(self, max_queue_size: int = 10000, worker_count: int = 4):
        self.logger = logging.getLogger(__name__)
        self._max_queue_size = max_queue_size
        self._worker_count = worker_count
        
        self._event_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._subscribers: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._wildcard_subscribers: List[EventSubscription] = []
        self._dead_letter_queue: List[Event] = []
        
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'events_retried': 0
        }
        self._lock = threading.RLock()
        
        self._setup_logging()
    
    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def start(self):
        if self._running:
            self.logger.warning("EventBus is already running")
            return
        
        self._running = True
        self.logger.info(f"Starting EventBus with {self._worker_count} workers")
        
        for i in range(self._worker_count):
            worker = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self._workers.append(worker)
        
        self.logger.info("EventBus started successfully")
    
    async def stop(self, timeout: float = 5.0):
        if not self._running:
            return
        
        self.logger.info("Stopping EventBus...")
        self._running = False
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Workers did not stop gracefully, cancelling...")
            for worker in self._workers:
                worker.cancel()
        
        self._workers.clear()
        self.logger.info("EventBus stopped")
    
    async def publish(self, event_name: str, data: Any = None, 
                     priority: Priority = Priority.NORMAL,
                     correlation_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self._running:
            self.logger.error("Cannot publish event: EventBus is not running")
            return False
        
        event = Event(
            name=event_name,
            data=data,
            priority=priority,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        try:
            self._event_queue.put_nowait(event)
            with self._lock:
                self._stats['events_published'] += 1
            
            self.logger.debug(f"Published event: {event_name} (priority: {priority.name})")
            return True
            
        except asyncio.QueueFull:
            self.logger.error(f"Event queue is full. Dropping event: {event_name}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to publish event {event_name}: {e}")
            return False
    
    def subscribe(self, event_name: str, callback: Union[Callable, Coroutine],
                 event_filter: Optional[Callable[[Event], bool]] = None,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 dead_letter: bool = True):
        subscription = EventSubscription(
            callback=callback,
            event_filter=event_filter,
            max_retries=max_retries,
            retry_delay=retry_delay,
            dead_letter=dead_letter
        )
        
        with self._lock:
            if event_name == "*":
                self._wildcard_subscribers.append(subscription)
            else:
                self._subscribers[event_name].append(subscription)
        
        self.logger.debug(f"Subscribed to event: {event_name}")
    
    def unsubscribe(self, event_name: str, callback: Union[Callable, Coroutine]):
        with self._lock:
            if event_name == "*":
                self._wildcard_subscribers = [
                    sub for sub in self._wildcard_subscribers 
                    if sub.callback != callback
                ]
            else:
                self._subscribers[event_name] = [
                    sub for sub in self._subscribers[event_name] 
                    if sub.callback != callback
                ]
        
        self.logger.debug(f"Unsubscribed from event: {event_name}")
    
    async def _event_worker(self, worker_name: str):
        self.logger.debug(f"Event worker {worker_name} started")
        
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event, worker_name)
                self._event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.debug(f"Event worker {worker_name} stopped")
    
    async def _process_event(self, event: Event, worker_name: str):
        self.logger.debug(f"Processing event: {event.name} (worker: {worker_name})")
        
        try:
            subscribers = self._get_subscribers(event.name)
            
            if not subscribers:
                self.logger.debug(f"No subscribers for event: {event.name}")
                return
            
            tasks = []
            for subscription in subscribers:
                if subscription.event_filter and not subscription.event_filter(event):
                    continue
                
                task = asyncio.create_task(
                    self._execute_callback(event, subscription, worker_name)
                )
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            with self._lock:
                self._stats['events_processed'] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to process event {event.name}: {e}")
            with self._lock:
                self._stats['events_failed'] += 1
    
    def _get_subscribers(self, event_name: str) -> List[EventSubscription]:
        with self._lock:
            subscribers = list(self._subscribers.get(event_name, []))
            subscribers.extend(self._wildcard_subscribers)
            return subscribers
    
    async def _execute_callback(self, event: Event, subscription: EventSubscription, worker_name: str):
        retry_count = 0
        last_exception = None
        
        while retry_count <= subscription.max_retries:
            try:
                if asyncio.iscoroutinefunction(subscription.callback):
                    await subscription.callback(event)
                else:
                    subscription.callback(event)
                
                if retry_count > 0:
                    self.logger.info(f"Event {event.name} processed successfully after {retry_count} retries")
                
                return
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                if retry_count <= subscription.max_retries:
                    self.logger.warning(
                        f"Event {event.name} callback failed (attempt {retry_count}): {e}. Retrying..."
                    )
                    with self._lock:
                        self._stats['events_retried'] += 1
                    
                    await asyncio.sleep(subscription.retry_delay)
                else:
                    break
        
        self.logger.error(
            f"Event {event.name} callback failed after {subscription.max_retries} retries: {last_exception}"
        )
        
        if subscription.dead_letter:
            self._add_to_dead_letter_queue(event, last_exception)
        
        with self._lock:
            self._stats['events_failed'] += 1
    
    def _add_to_dead_letter_queue(self, event: Event, exception: Exception):
        with self._lock:
            dead_letter_event = Event(
                name=f"dead_letter.{event.name}",
                data={
                    'original_event': event,
                    'error': str(exception),
                    'traceback': traceback.format_exc(),
                    'failed_at': datetime.now()
                },
                priority=Priority.LOW,
                correlation_id=event.correlation_id
            )
            
            self._dead_letter_queue.append(dead_letter_event)
            
            if len(self._dead_letter_queue) > 1000:
                self._dead_letter_queue = self._dead_letter_queue[-1000:]
        
        self.logger.warning(f"Added event {event.name} to dead letter queue")
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats.copy(),
                'queue_size': self._event_queue.qsize(),
                'dead_letter_count': len(self._dead_letter_queue),
                'subscribers_count': sum(len(subs) for subs in self._subscribers.values()),
                'wildcard_subscribers': len(self._wildcard_subscribers),
                'running': self._running
            }
    
    def get_dead_letter_events(self, limit: int = 100) -> List[Event]:
        with self._lock:
            return list(self._dead_letter_queue[-limit:])
    
    def clear_dead_letter_queue(self):
        with self._lock:
            self._dead_letter_queue.clear()
        self.logger.info("Dead letter queue cleared")
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


class TradingEventBus(EventBus):
    MARKET_DATA = "market_data"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_ALERT = "risk_alert"
    SYSTEM_ERROR = "system_error"
    HEARTBEAT = "heartbeat"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_trading_logging()
    
    def _setup_trading_logging(self):
        trading_handler = logging.FileHandler('trading_events.log')
        trading_formatter = logging.Formatter(
            '%(asctime)s - TRADING - %(levelname)s - %(message)s'
        )
        trading_handler.setFormatter(trading_formatter)
        
        trading_logger = logging.getLogger('trading_events')
        trading_logger.addHandler(trading_handler)
        trading_logger.setLevel(logging.INFO)
        
        self.trading_logger = trading_logger
    
    async def publish_market_data(self, symbol: str, price: float, volume: int):
        return await self.publish(
            self.MARKET_DATA,
            {'symbol': symbol, 'price': price, 'volume': volume},
            priority=Priority.HIGH
        )
    
    async def publish_order_event(self, event_type: str, order_data: Dict[str, Any]):
        self.trading_logger.info(f"Order event: {event_type} - {order_data}")
        return await self.publish(event_type, order_data, priority=Priority.CRITICAL)
    
    async def publish_risk_alert(self, alert_data: Dict[str, Any]):
        self.trading_logger.warning(f"Risk alert: {alert_data}")
        return await self.publish(self.RISK_ALERT, alert_data, priority=Priority.CRITICAL)