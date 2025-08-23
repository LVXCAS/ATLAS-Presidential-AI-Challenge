"""
WebSocket Manager for Bloomberg Terminal
Real-time data streaming to frontend clients with high performance.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from core.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """WebSocket client connection information."""
    client_id: str
    websocket: WebSocket
    connected_at: datetime
    subscribed_symbols: Set[str]
    last_ping: datetime
    message_count: int = 0


class WebSocketManager:
    """High-performance WebSocket connection manager for real-time data streaming."""
    
    def __init__(self):
        self.clients: Dict[str, ClientConnection] = {}
        self.symbol_subscribers: Dict[str, Set[str]] = {}  # symbol -> set of client_ids
        self.redis_manager = get_redis_manager()
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self.total_messages_sent = 0
        self.total_clients_connected = 0
        self.start_time = time.time()
    
    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        """Accept and manage new WebSocket connection."""
        try:
            await websocket.accept()
            
            connection = ClientConnection(
                client_id=client_id,
                websocket=websocket,
                connected_at=datetime.now(),
                subscribed_symbols=set(),
                last_ping=datetime.now()
            )
            
            self.clients[client_id] = connection
            self.total_clients_connected += 1
            
            logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
            
            # Send welcome message
            await self._send_to_client(client_id, {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "server_time": time.time(),
                "message": "Welcome to Bloomberg Terminal WebSocket"
            })
            
            # Start monitoring tasks for this client if not already running
            if not self.is_running:
                await self._start_background_tasks()
            
        except Exception as e:
            logger.error(f"Error connecting client {client_id}: {e}")
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def disconnect(self, client_id: str) -> None:
        """Handle client disconnection."""
        if client_id in self.clients:
            connection = self.clients[client_id]
            
            # Remove from all symbol subscriptions
            for symbol in connection.subscribed_symbols:
                if symbol in self.symbol_subscribers:
                    self.symbol_subscribers[symbol].discard(client_id)
                    if not self.symbol_subscribers[symbol]:
                        del self.symbol_subscribers[symbol]
            
            # Close WebSocket connection
            try:
                await connection.websocket.close()
            except:
                pass  # Connection might already be closed
            
            del self.clients[client_id]
            logger.info(f"Client {client_id} disconnected. Total clients: {len(self.clients)}")
            
            # Stop background tasks if no clients
            if not self.clients and self.is_running:
                await self._stop_background_tasks()
    
    async def subscribe_to_symbols(self, client_id: str, symbols: List[str]) -> None:
        """Subscribe client to market data for specific symbols."""
        if client_id not in self.clients:
            logger.warning(f"Cannot subscribe unknown client {client_id}")
            return
        
        connection = self.clients[client_id]
        
        for symbol in symbols:
            symbol = symbol.upper().strip()
            
            # Add to client's subscriptions
            connection.subscribed_symbols.add(symbol)
            
            # Add to symbol subscribers
            if symbol not in self.symbol_subscribers:
                self.symbol_subscribers[symbol] = set()
            self.symbol_subscribers[symbol].add(client_id)
        
        logger.info(f"Client {client_id} subscribed to {len(symbols)} symbols")
        
        # Send confirmation
        await self._send_to_client(client_id, {
            "type": "subscription",
            "action": "subscribed",
            "symbols": symbols,
            "total_subscriptions": len(connection.subscribed_symbols)
        })
        
        # Send latest data for newly subscribed symbols
        await self._send_initial_data(client_id, symbols)
    
    async def unsubscribe_from_symbols(self, client_id: str, symbols: List[str]) -> None:
        """Unsubscribe client from market data for specific symbols."""
        if client_id not in self.clients:
            return
        
        connection = self.clients[client_id]
        
        for symbol in symbols:
            symbol = symbol.upper().strip()
            
            # Remove from client's subscriptions
            connection.subscribed_symbols.discard(symbol)
            
            # Remove from symbol subscribers
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(client_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
        
        logger.info(f"Client {client_id} unsubscribed from {len(symbols)} symbols")
        
        # Send confirmation
        await self._send_to_client(client_id, {
            "type": "subscription",
            "action": "unsubscribed",
            "symbols": symbols,
            "total_subscriptions": len(connection.subscribed_symbols)
        })
    
    async def broadcast_market_data(self, symbol: str, data: Any) -> None:
        """Broadcast market data update to all subscribed clients."""
        if symbol not in self.symbol_subscribers:
            return
        
        message = {
            "type": "market_data",
            "symbol": symbol,
            "data": data if isinstance(data, dict) else asdict(data) if hasattr(data, '__dict__') else str(data),
            "timestamp": time.time()
        }
        
        # Send to all subscribers of this symbol
        for client_id in self.symbol_subscribers[symbol].copy():  # Copy to avoid modification during iteration
            await self._send_to_client(client_id, message)
    
    async def broadcast_portfolio_update(self, portfolio_data: Dict[str, Any]) -> None:
        """Broadcast portfolio update to all connected clients."""
        message = {
            "type": "portfolio_update",
            "data": portfolio_data,
            "timestamp": time.time()
        }
        
        await self._broadcast_to_all_clients(message)
    
    async def broadcast_risk_alert(self, risk_data: Dict[str, Any]) -> None:
        """Broadcast risk alert to all connected clients."""
        message = {
            "type": "risk_alert",
            "data": risk_data,
            "timestamp": time.time(),
            "severity": risk_data.get("severity", "MEDIUM")
        }
        
        await self._broadcast_to_all_clients(message)
    
    async def broadcast_agent_signal(self, agent_name: str, symbol: str, signal_data: Dict[str, Any]) -> None:
        """Broadcast agent trading signal to subscribed clients."""
        message = {
            "type": "agent_signal",
            "agent": agent_name,
            "symbol": symbol,
            "data": signal_data,
            "timestamp": time.time()
        }
        
        # Send to clients subscribed to this symbol
        if symbol in self.symbol_subscribers:
            for client_id in self.symbol_subscribers[symbol]:
                await self._send_to_client(client_id, message)
    
    async def send_system_status(self, status_data: Dict[str, Any]) -> None:
        """Send system status update to all clients."""
        message = {
            "type": "system_status",
            "data": status_data,
            "timestamp": time.time()
        }
        
        await self._broadcast_to_all_clients(message)
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket manager and disconnect all clients."""
        logger.info("Shutting down WebSocket manager...")
        
        # Disconnect all clients
        for client_id in list(self.clients.keys()):
            await self.disconnect(client_id)
        
        # Stop background tasks
        await self._stop_background_tasks()
        
        logger.info("WebSocket manager shutdown complete")
    
    # Private methods
    
    async def _send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific client with error handling."""
        if client_id not in self.clients:
            return False
        
        connection = self.clients[client_id]
        
        try:
            await connection.websocket.send_text(json.dumps(message))
            connection.message_count += 1
            self.total_messages_sent += 1
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")
            # Disconnect the client if send fails
            await self.disconnect(client_id)
            return False
    
    async def _broadcast_to_all_clients(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected clients."""
        sent_count = 0
        
        for client_id in list(self.clients.keys()):  # Copy to avoid modification during iteration
            if await self._send_to_client(client_id, message):
                sent_count += 1
        
        return sent_count
    
    async def _send_initial_data(self, client_id: str, symbols: List[str]) -> None:
        """Send initial data to client for newly subscribed symbols."""
        try:
            redis_client = await self.redis_manager.get_client()
            
            for symbol in symbols:
                # Get latest price
                price_data = await redis_client.hgetall(f"price:{symbol}")
                if price_data:
                    await self._send_to_client(client_id, {
                        "type": "market_data",
                        "symbol": symbol,
                        "data": {k: float(v) if k != 'timestamp' else v for k, v in price_data.items()},
                        "timestamp": time.time(),
                        "initial": True
                    })
                
                # Get latest quote
                quote_data = await redis_client.hgetall(f"quote:{symbol}")
                if quote_data:
                    await self._send_to_client(client_id, {
                        "type": "quote_data",
                        "symbol": symbol,
                        "data": {k: float(v) if k != 'timestamp' else v for k, v in quote_data.items()},
                        "timestamp": time.time(),
                        "initial": True
                    })
                
                # Get technical indicators
                indicators_data = await redis_client.hgetall(f"indicators:{symbol}")
                if indicators_data:
                    await self._send_to_client(client_id, {
                        "type": "indicators",
                        "symbol": symbol,
                        "data": {k: float(v) if k != 'timestamp' else v for k, v in indicators_data.items()},
                        "timestamp": time.time(),
                        "initial": True
                    })
        
        except Exception as e:
            logger.error(f"Error sending initial data to client {client_id}: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._ping_clients()),
            asyncio.create_task(self._monitor_performance()),
            asyncio.create_task(self._stream_redis_data())
        ]
        
        logger.info("WebSocket background tasks started")
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        self.is_running = False
        
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        logger.info("WebSocket background tasks stopped")
    
    async def _ping_clients(self) -> None:
        """Periodically ping clients to maintain connection health."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for client_id in list(self.clients.keys()):
                    connection = self.clients[client_id]
                    
                    # Send ping every 30 seconds
                    if (current_time - connection.last_ping).total_seconds() > 30:
                        ping_sent = await self._send_to_client(client_id, {
                            "type": "ping",
                            "timestamp": time.time()
                        })
                        
                        if ping_sent:
                            connection.last_ping = current_time
                
                await asyncio.sleep(30)  # Ping every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_performance(self) -> None:
        """Monitor WebSocket performance and log metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                uptime = time.time() - self.start_time
                avg_messages_per_second = self.total_messages_sent / uptime if uptime > 0 else 0
                
                logger.info(
                    f"WebSocket performance: {len(self.clients)} clients, "
                    f"{self.total_messages_sent} messages sent, "
                    f"{avg_messages_per_second:.1f} msg/sec avg, "
                    f"{len(self.symbol_subscribers)} symbols tracked"
                )
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    async def _stream_redis_data(self) -> None:
        """Stream real-time data from Redis to WebSocket clients."""
        try:
            redis_client = await self.redis_manager.get_pubsub_client()
            pubsub = redis_client.pubsub()
            
            # Subscribe to market data channels
            await pubsub.subscribe(
                "market_data:*",
                "portfolio_updates",
                "risk_alerts",
                "agent_signals:*"
            )
            
            while self.is_running:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    
                    if message and message['type'] == 'message':
                        channel = message['channel']
                        data = json.loads(message['data'])
                        
                        # Route message based on channel
                        if channel.startswith('market_data:'):
                            symbol = channel.split(':')[1]
                            await self.broadcast_market_data(symbol, data)
                        
                        elif channel == 'portfolio_updates':
                            await self.broadcast_portfolio_update(data)
                        
                        elif channel == 'risk_alerts':
                            await self.broadcast_risk_alert(data)
                        
                        elif channel.startswith('agent_signals:'):
                            parts = channel.split(':')
                            agent_name = parts[1]
                            symbol = parts[2] if len(parts) > 2 else 'UNKNOWN'
                            await self.broadcast_agent_signal(agent_name, symbol, data)
                
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue listening
                except Exception as e:
                    logger.error(f"Error in Redis stream: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Error setting up Redis stream: {e}")
        finally:
            try:
                await pubsub.unsubscribe()
                await pubsub.close()
            except:
                pass
    
    # Public status methods
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_clients": len(self.clients),
            "total_messages_sent": self.total_messages_sent,
            "total_clients_connected": self.total_clients_connected,
            "symbols_tracked": len(self.symbol_subscribers),
            "uptime_seconds": time.time() - self.start_time,
            "avg_messages_per_second": self.total_messages_sent / (time.time() - self.start_time) if time.time() > self.start_time else 0
        }
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific client."""
        if client_id not in self.clients:
            return None
        
        connection = self.clients[client_id]
        return {
            "client_id": client_id,
            "connected_at": connection.connected_at.isoformat(),
            "subscribed_symbols": list(connection.subscribed_symbols),
            "message_count": connection.message_count,
            "last_ping": connection.last_ping.isoformat()
        }