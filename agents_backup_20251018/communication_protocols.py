"""
Agent Communication Protocols for LangGraph Trading System

This module implements the communication protocols, message bus, and coordination
mechanisms for inter-agent communication in the trading system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages in the system"""
    MARKET_DATA_UPDATE = "market_data_update"
    SIGNAL_GENERATED = "signal_generated"
    RISK_ALERT = "risk_alert"
    ORDER_REQUEST = "order_request"
    ORDER_FILLED = "order_filled"
    SYSTEM_ALERT = "system_alert"
    AGENT_STATUS = "agent_status"
    COORDINATION_REQUEST = "coordination_request"
    RESOURCE_REQUEST = "resource_request"
    LEARNING_UPDATE = "learning_update"


class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AgentRole(str, Enum):
    """Agent roles in the system"""
    DATA_PROVIDER = "data_provider"
    SIGNAL_GENERATOR = "signal_generator"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"
    EXECUTION_ENGINE = "execution_engine"
    COORDINATOR = "coordinator"
    MONITOR = "monitor"


@dataclass
class Message:
    """Base message structure for agent communication"""
    id: str
    type: MessageType
    sender: str
    recipient: Optional[str] = None  # None for broadcast
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = None
    data: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.data is None:
            self.data = {}
        if self.id is None:
            self.id = str(uuid.uuid4())


class MessageHandler(ABC):
    """Abstract base class for message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message and optionally return a response"""
        pass


class MessageBus:
    """
    Central message bus for agent communication.
    Implements publish-subscribe pattern with routing and filtering.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_handlers: Dict[str, MessageHandler] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_dropped": 0,
            "active_subscribers": 0
        }
        
        logger.info("Message bus initialized")
    
    async def start(self):
        """Start the message bus processing loop"""
        self.running = True
        asyncio.create_task(self._process_messages())
        logger.info("Message bus started")
    
    async def stop(self):
        """Stop the message bus"""
        self.running = False
        logger.info("Message bus stopped")
    
    def subscribe(self, message_type: MessageType, handler: Callable, agent_id: str = None):
        """Subscribe to messages of a specific type"""
        key = f"{message_type}:{agent_id}" if agent_id else str(message_type)
        
        if key not in self.subscribers:
            self.subscribers[key] = []
        
        self.subscribers[key].append(handler)
        self.stats["active_subscribers"] = len(self.subscribers)
        
        logger.debug(f"Agent subscribed to {message_type} messages")
    
    def unsubscribe(self, message_type: MessageType, handler: Callable, agent_id: str = None):
        """Unsubscribe from messages"""
        key = f"{message_type}:{agent_id}" if agent_id else str(message_type)
        
        if key in self.subscribers and handler in self.subscribers[key]:
            self.subscribers[key].remove(handler)
            if not self.subscribers[key]:
                del self.subscribers[key]
            
            self.stats["active_subscribers"] = len(self.subscribers)
            logger.debug(f"Agent unsubscribed from {message_type} messages")
    
    async def publish(self, message: Message):
        """Publish a message to the bus"""
        try:
            await self.message_queue.put(message)
            self.stats["messages_sent"] += 1
            
            logger.debug(f"Message published: {message.type} from {message.sender}")
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            self.stats["messages_dropped"] += 1
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                await self._route_message(message)
                self.stats["messages_received"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.stats["messages_dropped"] += 1
    
    async def _route_message(self, message: Message):
        """Route message to appropriate subscribers"""
        # Check TTL
        if message.ttl and (datetime.now() - message.timestamp).seconds > message.ttl:
            logger.warning(f"Message {message.id} expired, dropping")
            self.stats["messages_dropped"] += 1
            return
        
        # Find subscribers
        subscribers = []
        
        # Direct recipient
        if message.recipient:
            key = f"{message.type}:{message.recipient}"
            subscribers.extend(self.subscribers.get(key, []))
        
        # Broadcast subscribers
        subscribers.extend(self.subscribers.get(str(message.type), []))
        
        # Deliver to all subscribers
        for handler in subscribers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return self.stats.copy()


class AgentCoordinator:
    """
    Coordinates agent interactions and manages resource allocation.
    Implements negotiation protocols and conflict resolution.
    """
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.resource_pool = {
            "cpu_cores": 8,
            "memory_gb": 16,
            "api_calls_per_minute": 1000,
            "data_bandwidth_mbps": 100
        }
        self.resource_allocations: Dict[str, Dict[str, float]] = {}
        self.coordination_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Subscribe to coordination messages
        self.message_bus.subscribe(MessageType.COORDINATION_REQUEST, self._handle_coordination_request)
        self.message_bus.subscribe(MessageType.RESOURCE_REQUEST, self._handle_resource_request)
        self.message_bus.subscribe(MessageType.AGENT_STATUS, self._handle_agent_status)
        
        logger.info("Agent coordinator initialized")
    
    def register_agent(self, agent_id: str, role: AgentRole, capabilities: Dict[str, Any]):
        """Register an agent with the coordinator"""
        self.agents[agent_id] = {
            "role": role,
            "capabilities": capabilities,
            "status": "active",
            "last_seen": datetime.now(),
            "performance_metrics": {},
            "resource_usage": {}
        }
        
        # Initialize resource allocation
        self.resource_allocations[agent_id] = {
            "cpu_cores": 1.0,
            "memory_gb": 2.0,
            "api_calls_per_minute": 100,
            "data_bandwidth_mbps": 10
        }
        
        logger.info(f"Agent {agent_id} registered with role {role}")
    
    async def _handle_coordination_request(self, message: Message):
        """Handle coordination requests between agents"""
        try:
            request_type = message.data.get("request_type")
            requesting_agent = message.sender
            
            if request_type == "signal_fusion":
                await self._coordinate_signal_fusion(message)
            elif request_type == "risk_assessment":
                await self._coordinate_risk_assessment(message)
            elif request_type == "order_execution":
                await self._coordinate_order_execution(message)
            else:
                logger.warning(f"Unknown coordination request type: {request_type}")
                
        except Exception as e:
            logger.error(f"Error handling coordination request: {e}")
    
    async def _coordinate_signal_fusion(self, message: Message):
        """Coordinate signal fusion between multiple agents"""
        session_id = str(uuid.uuid4())
        signals = message.data.get("signals", [])
        symbol = message.data.get("symbol")
        
        # Create coordination session
        self.coordination_sessions[session_id] = {
            "type": "signal_fusion",
            "symbol": symbol,
            "participants": [],
            "signals": signals,
            "status": "active",
            "created_at": datetime.now()
        }
        
        # Find relevant agents for signal fusion
        relevant_agents = []
        for agent_id, agent_info in self.agents.items():
            if agent_info["role"] in [AgentRole.SIGNAL_GENERATOR, AgentRole.PORTFOLIO_MANAGER]:
                relevant_agents.append(agent_id)
        
        # Send coordination messages to relevant agents
        for agent_id in relevant_agents:
            coord_message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.COORDINATION_REQUEST,
                sender="coordinator",
                recipient=agent_id,
                priority=MessagePriority.HIGH,
                data={
                    "session_id": session_id,
                    "coordination_type": "signal_fusion",
                    "symbol": symbol,
                    "signals": signals,
                    "deadline": (datetime.now().timestamp() + 30)  # 30 second deadline
                },
                correlation_id=message.id
            )
            
            await self.message_bus.publish(coord_message)
        
        logger.info(f"Signal fusion coordination initiated for {symbol}")
    
    async def _handle_resource_request(self, message: Message):
        """Handle resource allocation requests"""
        try:
            requesting_agent = message.sender
            requested_resources = message.data.get("resources", {})
            priority = message.data.get("priority", "normal")
            duration = message.data.get("duration", 60)  # seconds
            
            # Check if resources are available
            available = self._check_resource_availability(requested_resources)
            
            if available:
                # Allocate resources
                self._allocate_resources(requesting_agent, requested_resources, duration)
                
                response = Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.SYSTEM_ALERT,
                    sender="coordinator",
                    recipient=requesting_agent,
                    data={
                        "status": "approved",
                        "resources": requested_resources,
                        "duration": duration
                    },
                    correlation_id=message.id
                )
            else:
                # Negotiate alternative allocation
                alternative = self._negotiate_resources(requesting_agent, requested_resources)
                
                response = Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.SYSTEM_ALERT,
                    sender="coordinator",
                    recipient=requesting_agent,
                    data={
                        "status": "negotiated",
                        "alternative_resources": alternative,
                        "reason": "Insufficient resources available"
                    },
                    correlation_id=message.id
                )
            
            await self.message_bus.publish(response)
            
        except Exception as e:
            logger.error(f"Error handling resource request: {e}")
    
    def _check_resource_availability(self, requested: Dict[str, float]) -> bool:
        """Check if requested resources are available"""
        for resource, amount in requested.items():
            if resource in self.resource_pool:
                allocated = sum(
                    allocation.get(resource, 0) 
                    for allocation in self.resource_allocations.values()
                )
                available = self.resource_pool[resource] - allocated
                
                if amount > available:
                    return False
        
        return True
    
    def _allocate_resources(self, agent_id: str, resources: Dict[str, float], duration: int):
        """Allocate resources to an agent"""
        if agent_id not in self.resource_allocations:
            self.resource_allocations[agent_id] = {}
        
        for resource, amount in resources.items():
            self.resource_allocations[agent_id][resource] = amount
        
        # Schedule resource deallocation
        asyncio.create_task(self._deallocate_resources_after_delay(agent_id, resources, duration))
        
        logger.info(f"Resources allocated to {agent_id}: {resources}")
    
    async def _deallocate_resources_after_delay(self, agent_id: str, resources: Dict[str, float], delay: int):
        """Deallocate resources after specified delay"""
        await asyncio.sleep(delay)
        
        for resource in resources:
            if agent_id in self.resource_allocations and resource in self.resource_allocations[agent_id]:
                del self.resource_allocations[agent_id][resource]
        
        logger.info(f"Resources deallocated from {agent_id}: {resources}")
    
    def _negotiate_resources(self, agent_id: str, requested: Dict[str, float]) -> Dict[str, float]:
        """Negotiate alternative resource allocation"""
        alternative = {}
        
        for resource, amount in requested.items():
            if resource in self.resource_pool:
                allocated = sum(
                    allocation.get(resource, 0) 
                    for allocation in self.resource_allocations.values()
                )
                available = self.resource_pool[resource] - allocated
                
                # Offer 80% of available resources
                alternative[resource] = min(amount, available * 0.8)
        
        return alternative
    
    async def _handle_agent_status(self, message: Message):
        """Handle agent status updates"""
        agent_id = message.sender
        status_data = message.data
        
        if agent_id in self.agents:
            self.agents[agent_id]["status"] = status_data.get("status", "unknown")
            self.agents[agent_id]["last_seen"] = datetime.now()
            self.agents[agent_id]["performance_metrics"] = status_data.get("performance_metrics", {})
            
            logger.debug(f"Agent {agent_id} status updated: {status_data.get('status')}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a["status"] == "active"]),
            "resource_utilization": self._calculate_resource_utilization(),
            "coordination_sessions": len(self.coordination_sessions),
            "agents": self.agents
        }
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization"""
        utilization = {}
        
        for resource, total in self.resource_pool.items():
            allocated = sum(
                allocation.get(resource, 0) 
                for allocation in self.resource_allocations.values()
            )
            utilization[resource] = (allocated / total) * 100 if total > 0 else 0
        
        return utilization


class ConflictResolver:
    """
    Resolves conflicts between agents using various strategies.
    """
    
    def __init__(self):
        self.resolution_strategies = {
            "weighted_average": self._weighted_average_resolution,
            "highest_confidence": self._highest_confidence_resolution,
            "expert_override": self._expert_override_resolution,
            "voting": self._voting_resolution,
            "performance_based": self._performance_based_resolution
        }
        
        logger.info("Conflict resolver initialized")
    
    def detect_conflicts(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts between signals"""
        conflicts = []
        
        # Group signals by symbol
        signals_by_symbol = {}
        for signal_id, signal in signals.items():
            symbol = signal.get("symbol")
            if symbol not in signals_by_symbol:
                signals_by_symbol[symbol] = []
            signals_by_symbol[symbol].append((signal_id, signal))
        
        # Check for conflicts within each symbol
        for symbol, symbol_signals in signals_by_symbol.items():
            if len(symbol_signals) > 1:
                # Check for opposing signals
                buy_signals = [s for s in symbol_signals if s[1].get("value", 0) > 0]
                sell_signals = [s for s in symbol_signals if s[1].get("value", 0) < 0]
                
                if buy_signals and sell_signals:
                    conflicts.append({
                        "symbol": symbol,
                        "type": "opposing_signals",
                        "buy_signals": buy_signals,
                        "sell_signals": sell_signals,
                        "severity": "high"
                    })
                
                # Check for confidence conflicts
                confidences = [s[1].get("confidence", 0) for s in symbol_signals]
                if max(confidences) - min(confidences) > 0.5:
                    conflicts.append({
                        "symbol": symbol,
                        "type": "confidence_conflict",
                        "signals": symbol_signals,
                        "severity": "medium"
                    })
        
        return conflicts
    
    def resolve_conflict(self, conflict: Dict[str, Any], strategy: str = "weighted_average") -> Dict[str, Any]:
        """Resolve a conflict using specified strategy"""
        if strategy not in self.resolution_strategies:
            logger.warning(f"Unknown resolution strategy: {strategy}, using weighted_average")
            strategy = "weighted_average"
        
        resolver = self.resolution_strategies[strategy]
        return resolver(conflict)
    
    def _weighted_average_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using weighted average based on confidence"""
        signals = conflict.get("signals", [])
        if not signals:
            return {"resolved_signal": None, "method": "weighted_average"}
        
        total_weight = 0
        weighted_value = 0
        
        for signal_id, signal in signals:
            confidence = signal.get("confidence", 0)
            value = signal.get("value", 0)
            
            weighted_value += value * confidence
            total_weight += confidence
        
        if total_weight > 0:
            resolved_value = weighted_value / total_weight
        else:
            resolved_value = 0
        
        return {
            "resolved_signal": {
                "value": resolved_value,
                "confidence": total_weight / len(signals),
                "method": "weighted_average",
                "contributing_signals": len(signals)
            },
            "method": "weighted_average"
        }
    
    def _highest_confidence_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by selecting highest confidence signal"""
        signals = conflict.get("signals", [])
        if not signals:
            return {"resolved_signal": None, "method": "highest_confidence"}
        
        best_signal = max(signals, key=lambda x: x[1].get("confidence", 0))
        
        return {
            "resolved_signal": best_signal[1],
            "method": "highest_confidence"
        }
    
    def _expert_override_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using expert system rules"""
        # Implement expert system logic here
        # For now, use weighted average as fallback
        return self._weighted_average_resolution(conflict)
    
    def _voting_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using majority voting"""
        signals = conflict.get("signals", [])
        if not signals:
            return {"resolved_signal": None, "method": "voting"}
        
        # Count votes for buy/sell
        buy_votes = sum(1 for _, signal in signals if signal.get("value", 0) > 0)
        sell_votes = sum(1 for _, signal in signals if signal.get("value", 0) < 0)
        
        if buy_votes > sell_votes:
            # Find average of buy signals
            buy_signals = [signal for _, signal in signals if signal.get("value", 0) > 0]
            avg_value = sum(s.get("value", 0) for s in buy_signals) / len(buy_signals)
            avg_confidence = sum(s.get("confidence", 0) for s in buy_signals) / len(buy_signals)
        elif sell_votes > buy_votes:
            # Find average of sell signals
            sell_signals = [signal for _, signal in signals if signal.get("value", 0) < 0]
            avg_value = sum(s.get("value", 0) for s in sell_signals) / len(sell_signals)
            avg_confidence = sum(s.get("confidence", 0) for s in sell_signals) / len(sell_signals)
        else:
            # Tie - no action
            avg_value = 0
            avg_confidence = 0.5
        
        return {
            "resolved_signal": {
                "value": avg_value,
                "confidence": avg_confidence,
                "method": "voting",
                "buy_votes": buy_votes,
                "sell_votes": sell_votes
            },
            "method": "voting"
        }
    
    def _performance_based_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict based on historical agent performance"""
        # This would require historical performance data
        # For now, use weighted average as fallback
        return self._weighted_average_resolution(conflict)


# Factory functions
def create_message_bus() -> MessageBus:
    """Create and return a new message bus instance"""
    return MessageBus()


def create_agent_coordinator(message_bus: MessageBus) -> AgentCoordinator:
    """Create and return a new agent coordinator instance"""
    return AgentCoordinator(message_bus)


def create_conflict_resolver() -> ConflictResolver:
    """Create and return a new conflict resolver instance"""
    return ConflictResolver()


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create communication infrastructure
        message_bus = create_message_bus()
        coordinator = create_agent_coordinator(message_bus)
        resolver = create_conflict_resolver()
        
        await message_bus.start()
        
        # Register some test agents
        coordinator.register_agent("momentum_agent", AgentRole.SIGNAL_GENERATOR, {"strategies": ["momentum"]})
        coordinator.register_agent("risk_manager", AgentRole.RISK_MANAGER, {"risk_models": ["var", "drawdown"]})
        
        print("Communication system initialized")
        print("Agent Status:", coordinator.get_agent_status())
        print("Message Bus Stats:", message_bus.get_stats())
        
        await message_bus.stop()
    
    asyncio.run(main())