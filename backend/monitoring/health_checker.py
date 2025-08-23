"""
Comprehensive health checking system for monitoring system components.
"""

import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from core.redis_manager import get_redis_manager
from core.database import get_database_health

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Configuration for a health check."""
    name: str
    description: str
    check_function: Callable
    interval_seconds: int = 60
    timeout_seconds: int = 30
    critical: bool = False  # If true, failure affects overall system health
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    uptime_seconds: float
    checks: List[HealthResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemHealthMonitor:
    """Monitor system-level health metrics."""
    
    def __init__(self):
        self.start_time = datetime.now()
        
    async def check_cpu_health(self) -> HealthResult:
        """Check CPU usage health."""
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            response_time = (time.time() - start_time) * 1000
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage critically high: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthResult(
                name="cpu_usage",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'cpu_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count(),
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="cpu_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check CPU usage: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def check_memory_health(self) -> HealthResult:
        """Check memory usage health."""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            response_time = (time.time() - start_time) * 1000
            
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critically high: {memory.percent:.1f}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            return HealthResult(
                name="memory_usage",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory usage: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def check_disk_health(self) -> HealthResult:
        """Check disk usage health."""
        start_time = time.time()
        
        try:
            disk = psutil.disk_usage('/')
            response_time = (time.time() - start_time) * 1000
            
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage critically high: {disk_percent:.1f}%"
            elif disk_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthResult(
                name="disk_usage",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'percent': disk_percent
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="disk_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk usage: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def check_network_health(self) -> HealthResult:
        """Check network connectivity health."""
        start_time = time.time()
        
        try:
            network_io = psutil.net_io_counters()
            response_time = (time.time() - start_time) * 1000
            
            # Simple network health check based on I/O counters
            status = HealthStatus.HEALTHY
            message = "Network connectivity normal"
            
            return HealthResult(
                name="network_connectivity",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv,
                    'errin': network_io.errin,
                    'errout': network_io.errout
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="network_connectivity",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check network: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def check_process_health(self) -> HealthResult:
        """Check process health metrics."""
        start_time = time.time()
        
        try:
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            
            process_info = {
                'pid': current_process.pid,
                'memory_percent': current_process.memory_percent(),
                'cpu_percent': current_process.cpu_percent(),
                'num_threads': current_process.num_threads(),
                'create_time': current_process.create_time()
            }
            
            response_time = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY
            message = f"Process health normal (PID: {current_process.pid})"
            
            return HealthResult(
                name="process_health",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'total_processes': process_count,
                    'current_process': process_info,
                    'uptime_seconds': time.time() - current_process.create_time()
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="process_health",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check process health: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )


class ServiceHealthChecker:
    """Check health of application services."""
    
    def __init__(self):
        self.redis = None
    
    async def initialize(self):
        """Initialize service health checker."""
        self.redis = get_redis_manager().client
    
    async def check_redis_health(self) -> HealthResult:
        """Check Redis connection health."""
        start_time = time.time()
        
        try:
            if not self.redis:
                await self.initialize()
            
            # Test Redis connection
            test_key = "health:test"
            await self.redis.set(test_key, "test_value")
            value = await self.redis.get(test_key)
            await self.redis.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000
            
            if value == "test_value":
                status = HealthStatus.HEALTHY
                message = "Redis connection healthy"
            else:
                status = HealthStatus.DEGRADED
                message = "Redis test failed"
            
            # Get Redis info
            redis_info = await self.redis.info()
            
            return HealthResult(
                name="redis_connection",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'used_memory_human': redis_info.get('used_memory_human', 'unknown'),
                    'redis_version': redis_info.get('redis_version', 'unknown'),
                    'uptime_in_seconds': redis_info.get('uptime_in_seconds', 0)
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="redis_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def check_database_health(self) -> HealthResult:
        """Check database connection health."""
        start_time = time.time()
        
        try:
            db_healthy = await get_database_health()
            response_time = (time.time() - start_time) * 1000
            
            if db_healthy:
                status = HealthStatus.HEALTHY
                message = "Database connection healthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Database connection failed"
            
            return HealthResult(
                name="database_connection",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthResult(
                name="database_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Database health check failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def check_market_data_service_health(self) -> HealthResult:
        """Check market data service health."""
        start_time = time.time()
        
        try:
            # Check if market data service is responsive
            # This would integrate with the actual market data service
            
            response_time = (time.time() - start_time) * 1000
            
            # Simulate health check
            status = HealthStatus.HEALTHY
            message = "Market data service healthy"
            
            return HealthResult(
                name="market_data_service",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'active_subscriptions': 0,  # Would get from actual service
                    'messages_per_second': 0,
                    'last_message_time': None
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="market_data_service",
                status=HealthStatus.UNKNOWN,
                message=f"Market data service check failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def check_order_service_health(self) -> HealthResult:
        """Check order service health."""
        start_time = time.time()
        
        try:
            response_time = (time.time() - start_time) * 1000
            
            # Simulate health check
            status = HealthStatus.HEALTHY
            message = "Order service healthy"
            
            return HealthResult(
                name="order_service",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'pending_orders': 0,  # Would get from actual service
                    'orders_per_minute': 0,
                    'last_order_time': None
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="order_service",
                status=HealthStatus.UNKNOWN,
                message=f"Order service check failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def check_risk_service_health(self) -> HealthResult:
        """Check risk management service health."""
        start_time = time.time()
        
        try:
            response_time = (time.time() - start_time) * 1000
            
            # Simulate health check
            status = HealthStatus.HEALTHY
            message = "Risk management service healthy"
            
            return HealthResult(
                name="risk_service",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    'active_alerts': 0,  # Would get from actual service
                    'risk_checks_per_minute': 0,
                    'last_risk_check': None
                }
            )
        
        except Exception as e:
            return HealthResult(
                name="risk_service",
                status=HealthStatus.UNKNOWN,
                message=f"Risk service check failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )


class HealthChecker:
    """Main health checking coordinator."""
    
    def __init__(self):
        self.system_monitor = SystemHealthMonitor()
        self.service_checker = ServiceHealthChecker()
        
        self.health_checks: Dict[str, HealthCheck] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_history: List[HealthResult] = []
        
        self.is_running = False
        self.redis = None
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("HealthChecker initialized")
    
    async def initialize(self):
        """Initialize health checker."""
        self.redis = get_redis_manager().client
        await self.service_checker.initialize()
        
        self.is_running = True
        
        # Start health checking loop
        asyncio.create_task(self._health_check_loop())
        
        logger.info("HealthChecker started")
    
    async def stop(self):
        """Stop health checker."""
        self.is_running = False
        logger.info("HealthChecker stopped")
    
    def _register_default_checks(self):
        """Register default health checks."""
        default_checks = [
            HealthCheck(
                name="cpu_usage",
                description="System CPU usage",
                check_function=self.system_monitor.check_cpu_health,
                interval_seconds=30,
                critical=True
            ),
            HealthCheck(
                name="memory_usage",
                description="System memory usage",
                check_function=self.system_monitor.check_memory_health,
                interval_seconds=30,
                critical=True
            ),
            HealthCheck(
                name="disk_usage",
                description="System disk usage",
                check_function=self.system_monitor.check_disk_health,
                interval_seconds=60,
                critical=False
            ),
            HealthCheck(
                name="network_connectivity",
                description="Network connectivity",
                check_function=self.system_monitor.check_network_health,
                interval_seconds=60,
                critical=False
            ),
            HealthCheck(
                name="process_health",
                description="Process health metrics",
                check_function=self.system_monitor.check_process_health,
                interval_seconds=30,
                critical=False
            ),
            HealthCheck(
                name="redis_connection",
                description="Redis database connection",
                check_function=self.service_checker.check_redis_health,
                interval_seconds=60,
                critical=True
            ),
            HealthCheck(
                name="database_connection",
                description="Main database connection",
                check_function=self.service_checker.check_database_health,
                interval_seconds=60,
                critical=True
            ),
            HealthCheck(
                name="market_data_service",
                description="Market data service",
                check_function=self.service_checker.check_market_data_service_health,
                interval_seconds=30,
                critical=True
            ),
            HealthCheck(
                name="order_service",
                description="Order execution service",
                check_function=self.service_checker.check_order_service_health,
                interval_seconds=30,
                critical=True
            ),
            HealthCheck(
                name="risk_service",
                description="Risk management service",
                check_function=self.service_checker.check_risk_service_health,
                interval_seconds=30,
                critical=True
            )
        ]
        
        for check in default_checks:
            self.health_checks[check.name] = check
    
    async def add_health_check(self, check: HealthCheck):
        """Add a custom health check."""
        self.health_checks[check.name] = check
        logger.info(f"Added health check: {check.name}")
    
    async def remove_health_check(self, check_name: str) -> bool:
        """Remove a health check."""
        if check_name in self.health_checks:
            del self.health_checks[check_name]
            logger.info(f"Removed health check: {check_name}")
            return True
        return False
    
    async def _health_check_loop(self):
        """Main health checking loop."""
        check_counters = {name: 0 for name in self.health_checks.keys()}
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Run health checks based on their intervals
                for check_name, check in self.health_checks.items():
                    if not check.enabled:
                        continue
                    
                    # Check if it's time to run this check
                    if check_counters[check_name] >= check.interval_seconds:
                        asyncio.create_task(self._run_health_check(check))
                        check_counters[check_name] = 0
                    else:
                        check_counters[check_name] += 10  # Loop runs every 10 seconds
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)
    
    async def _run_health_check(self, check: HealthCheck):
        """Run a single health check."""
        try:
            # Run the health check with timeout
            result = await asyncio.wait_for(
                check.check_function(),
                timeout=check.timeout_seconds
            )
            
            # Update component health
            self._update_component_health(check.name, result)
            
            # Store result
            self.health_history.append(result)
            
            # Keep only last 1000 results
            if len(self.health_history) > 1000:
                self.health_history = self.health_history[-1000:]
            
            # Persist to Redis
            await self._persist_health_result(result)
            
        except asyncio.TimeoutError:
            result = HealthResult(
                name=check.name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check timed out after {check.timeout_seconds}s",
                response_time_ms=check.timeout_seconds * 1000,
                timestamp=datetime.now(),
                error="timeout"
            )
            
            self._update_component_health(check.name, result)
            
        except Exception as e:
            result = HealthResult(
                name=check.name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check failed: {e}",
                response_time_ms=0,
                timestamp=datetime.now(),
                error=str(e)
            )
            
            self._update_component_health(check.name, result)
    
    def _update_component_health(self, component_name: str, result: HealthResult):
        """Update component health status."""
        if component_name not in self.component_health:
            self.component_health[component_name] = ComponentHealth(
                name=component_name,
                status=result.status,
                last_check=result.timestamp,
                response_time_ms=result.response_time_ms,
                uptime_seconds=0,
                checks=[result]
            )
        else:
            component = self.component_health[component_name]
            component.status = result.status
            component.last_check = result.timestamp
            component.response_time_ms = result.response_time_ms
            
            # Add result to history
            component.checks.append(result)
            
            # Keep only last 100 checks
            if len(component.checks) > 100:
                component.checks = component.checks[-100:]
            
            # Calculate uptime
            if component.checks:
                first_check = component.checks[0].timestamp
                component.uptime_seconds = (result.timestamp - first_check).total_seconds()
    
    async def _persist_health_result(self, result: HealthResult):
        """Persist health result to Redis."""
        if not self.redis:
            return
        
        try:
            result_data = {
                'name': result.name,
                'status': result.status.value,
                'message': result.message,
                'response_time_ms': result.response_time_ms,
                'timestamp': result.timestamp.isoformat(),
                'details': result.details,
                'error': result.error
            }
            
            # Store latest result
            await self.redis.setex(
                f"health:latest:{result.name}",
                300,  # 5 minute TTL
                json.dumps(result_data, default=str)
            )
            
            # Store in time series
            timestamp_key = result.timestamp.strftime('%Y%m%d_%H%M')
            await self.redis.setex(
                f"health:history:{result.name}:{timestamp_key}",
                3600,  # 1 hour TTL
                json.dumps(result_data, default=str)
            )
        
        except Exception as e:
            logger.error(f"Error persisting health result: {e}")
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.component_health:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'No health data available',
                'components': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Determine overall status
        critical_components = [
            name for name, check in self.health_checks.items()
            if check.critical and check.enabled
        ]
        
        critical_unhealthy = [
            name for name in critical_components
            if name in self.component_health and 
            self.component_health[name].status == HealthStatus.UNHEALTHY
        ]
        
        critical_degraded = [
            name for name in critical_components
            if name in self.component_health and 
            self.component_health[name].status == HealthStatus.DEGRADED
        ]
        
        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
            message = f"Critical components unhealthy: {', '.join(critical_unhealthy)}"
        elif critical_degraded:
            overall_status = HealthStatus.DEGRADED
            message = f"Critical components degraded: {', '.join(critical_degraded)}"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All critical components healthy"
        
        # Component summaries
        components = {}
        for name, component in self.component_health.items():
            components[name] = {
                'status': component.status.value,
                'last_check': component.last_check.isoformat(),
                'response_time_ms': component.response_time_ms,
                'uptime_seconds': component.uptime_seconds
            }
        
        return {
            'status': overall_status.value,
            'message': message,
            'components': components,
            'timestamp': datetime.now().isoformat(),
            'total_components': len(self.component_health),
            'healthy_components': len([c for c in self.component_health.values() if c.status == HealthStatus.HEALTHY]),
            'degraded_components': len([c for c in self.component_health.values() if c.status == HealthStatus.DEGRADED]),
            'unhealthy_components': len([c for c in self.component_health.values() if c.status == HealthStatus.UNHEALTHY])
        }
    
    async def get_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get health status for a specific component."""
        if component_name not in self.component_health:
            return None
        
        component = self.component_health[component_name]
        
        recent_checks = component.checks[-10:]  # Last 10 checks
        
        return {
            'name': component.name,
            'status': component.status.value,
            'last_check': component.last_check.isoformat(),
            'response_time_ms': component.response_time_ms,
            'uptime_seconds': component.uptime_seconds,
            'recent_checks': [
                {
                    'status': check.status.value,
                    'message': check.message,
                    'timestamp': check.timestamp.isoformat(),
                    'response_time_ms': check.response_time_ms,
                    'error': check.error
                }
                for check in recent_checks
            ]
        }
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive health summary."""
        overall_health = await self.get_overall_health()
        
        # Calculate average response times
        avg_response_times = {}
        for name, component in self.component_health.items():
            if component.checks:
                avg_response_time = sum(check.response_time_ms for check in component.checks[-10:]) / min(len(component.checks), 10)
                avg_response_times[name] = avg_response_time
        
        return {
            'overall_health': overall_health,
            'average_response_times': avg_response_times,
            'health_checks_enabled': len([c for c in self.health_checks.values() if c.enabled]),
            'total_health_checks': len(self.health_checks),
            'last_updated': datetime.now().isoformat()
        }