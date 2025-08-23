"""
Redis Manager for Bloomberg Terminal API
High-performance Redis connection management with connection pooling.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisManager:
    """High-performance Redis connection manager."""
    
    def __init__(self):
        self.settings = settings
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.pubsub_client: Optional[redis.Redis] = None
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize Redis connections with connection pooling."""
        if self.is_initialized:
            return
        
        try:
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                self.settings.redis.url,
                encoding=self.settings.redis.encoding,
                decode_responses=self.settings.redis.decode_responses,
                max_connections=self.settings.redis.max_connections,
                socket_keepalive=self.settings.redis.socket_keepalive,
                socket_keepalive_options=self.settings.redis.socket_keepalive_options,
                health_check_interval=30
            )
            
            # Create main Redis client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Create separate client for pub/sub
            self.pubsub_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            logger.info("Redis connection initialized successfully")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
    
    async def close(self) -> None:
        """Close all Redis connections."""
        try:
            if self.client:
                await self.client.close()
            if self.pubsub_client:
                await self.pubsub_client.close()
            if self.pool:
                await self.pool.disconnect()
            
            logger.info("Redis connections closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")
        finally:
            self.is_initialized = False
    
    async def get_client(self) -> redis.Redis:
        """Get main Redis client."""
        if not self.is_initialized:
            await self.initialize()
        return self.client
    
    async def get_pubsub_client(self) -> redis.Redis:
        """Get pub/sub Redis client."""
        if not self.is_initialized:
            await self.initialize()
        return self.pubsub_client
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            if not self.is_initialized:
                return False
            
            await self.client.ping()
            return True
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    # High-level operations
    
    async def set_with_expiry(self, key: str, value: Any, expiry: int = 300) -> bool:
        """Set value with expiry time."""
        try:
            client = await self.get_client()
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await client.setex(key, expiry, value)
            return True
            
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get and deserialize JSON value."""
        try:
            client = await self.get_client()
            value = await client.get(key)
            
            if value is None:
                return None
            
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value  # Return as string if not JSON
                
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    async def push_to_stream(self, stream: str, data: Dict[str, Any], max_len: int = 10000) -> str:
        """Push data to Redis stream with size limit."""
        try:
            client = await self.get_client()
            
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = str(int(asyncio.get_event_loop().time() * 1000))
            
            # Add to stream
            message_id = await client.xadd(stream, data, maxlen=max_len, approximate=True)
            
            return message_id
            
        except Exception as e:
            logger.error(f"Error pushing to stream {stream}: {e}")
            return None
    
    async def read_from_stream(self, stream: str, last_id: str = "0", count: int = 100) -> List[Dict]:
        """Read messages from Redis stream."""
        try:
            client = await self.get_client()
            
            # Read from stream
            messages = await client.xread({stream: last_id}, count=count, block=1000)
            
            result = []
            for stream_name, stream_messages in messages:
                for message_id, fields in stream_messages:
                    result.append({
                        'id': message_id,
                        'data': fields
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading from stream {stream}: {e}")
            return []
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to Redis channel."""
        try:
            client = await self.get_pubsub_client()
            
            if isinstance(message, (dict, list)):
                message = json.dumps(message)
            
            return await client.publish(channel, message)
            
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            return 0
    
    async def subscribe(self, *channels) -> redis.client.PubSub:
        """Subscribe to Redis channels."""
        try:
            client = await self.get_pubsub_client()
            pubsub = client.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub
            
        except Exception as e:
            logger.error(f"Error subscribing to channels {channels}: {e}")
            return None
    
    async def cache_market_data(self, symbol: str, data: Dict[str, Any], ttl: int = 300) -> None:
        """Cache market data with optimized structure."""
        try:
            client = await self.get_client()
            
            # Cache latest price
            price_key = f"price:{symbol}"
            await client.hset(price_key, mapping={
                k: str(v) for k, v in data.items()
            })
            await client.expire(price_key, ttl)
            
            # Add to time series for charting
            ts_key = f"timeseries:{symbol}"
            timestamp = data.get('timestamp', int(asyncio.get_event_loop().time() * 1000))
            price = data.get('price') or data.get('close')
            
            if price:
                await client.zadd(ts_key, {f"{price}:{timestamp}": timestamp})
                # Keep only last 1000 points
                await client.zremrangebyrank(ts_key, 0, -1001)
                await client.expire(ts_key, 3600)  # 1 hour
            
        except Exception as e:
            logger.error(f"Error caching market data for {symbol}: {e}")
    
    async def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get price history from cache."""
        try:
            client = await self.get_client()
            ts_key = f"timeseries:{symbol}"
            
            # Get latest points
            data = await client.zrevrange(ts_key, 0, limit-1, withscores=True)
            
            result = []
            for entry, timestamp in data:
                price_str, entry_timestamp = entry.split(':')
                result.append({
                    'price': float(price_str),
                    'timestamp': int(timestamp)
                })
            
            return result[::-1]  # Return in chronological order
            
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []
    
    async def cache_agent_signal(self, agent_name: str, symbol: str, signal: Dict[str, Any]) -> None:
        """Cache agent trading signal."""
        try:
            client = await self.get_client()
            
            signal_key = f"signal:{agent_name}:{symbol}"
            
            # Add timestamp if not present
            if 'timestamp' not in signal:
                signal['timestamp'] = int(asyncio.get_event_loop().time() * 1000)
            
            await client.hset(signal_key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in signal.items()
            })
            await client.expire(signal_key, 3600)  # 1 hour
            
            # Add to agent's signal history
            history_key = f"signals:{agent_name}"
            await client.lpush(history_key, json.dumps({
                'symbol': symbol,
                'signal': signal
            }))
            await client.ltrim(history_key, 0, 99)  # Keep last 100 signals
            await client.expire(history_key, 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Error caching agent signal: {e}")
    
    async def get_latest_signals(self, agent_name: str = None, symbol: str = None) -> List[Dict]:
        """Get latest agent signals."""
        try:
            client = await self.get_client()
            
            if agent_name and symbol:
                # Get specific signal
                signal_key = f"signal:{agent_name}:{symbol}"
                signal_data = await client.hgetall(signal_key)
                if signal_data:
                    return [signal_data]
                return []
            
            elif agent_name:
                # Get all signals for agent
                history_key = f"signals:{agent_name}"
                signals = await client.lrange(history_key, 0, 9)  # Last 10 signals
                return [json.loads(s) for s in signals]
            
            else:
                # Get all latest signals
                pattern = "signal:*"
                keys = await client.keys(pattern)
                
                signals = []
                for key in keys[:50]:  # Limit to 50 latest
                    signal_data = await client.hgetall(key)
                    if signal_data:
                        signals.append(signal_data)
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting latest signals: {e}")
            return []
    
    async def cache_portfolio_metrics(self, metrics: Dict[str, Any]) -> None:
        """Cache portfolio performance metrics."""
        try:
            client = await self.get_client()
            
            metrics_key = "portfolio:metrics"
            
            # Add timestamp if not present
            if 'timestamp' not in metrics:
                metrics['timestamp'] = int(asyncio.get_event_loop().time() * 1000)
            
            await client.hset(metrics_key, mapping={
                k: str(v) for k, v in metrics.items()
            })
            await client.expire(metrics_key, 300)  # 5 minutes
            
            # Add to history for charting
            history_key = "portfolio:history"
            await client.lpush(history_key, json.dumps(metrics))
            await client.ltrim(history_key, 0, 999)  # Keep last 1000 points
            await client.expire(history_key, 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Error caching portfolio metrics: {e}")
    
    async def get_portfolio_metrics(self) -> Optional[Dict]:
        """Get latest portfolio metrics."""
        try:
            client = await self.get_client()
            metrics_key = "portfolio:metrics"
            
            metrics_data = await client.hgetall(metrics_key)
            if metrics_data:
                # Convert string values back to appropriate types
                result = {}
                for k, v in metrics_data.items():
                    try:
                        # Try to convert to float for numeric values
                        result[k] = float(v)
                    except (ValueError, TypeError):
                        result[k] = v
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
            return None


@lru_cache()
def get_redis_manager() -> RedisManager:
    """Get cached Redis manager instance."""
    return RedisManager()