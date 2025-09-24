"""
Comprehensive Logging and Monitoring System

Enterprise-grade logging, metrics collection, alerting, and observability
for institutional trading systems with real-time monitoring capabilities.
"""

import asyncio
import logging
import json
import time
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import socket
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import queue
import warnings
warnings.filterwarnings('ignore')

# Metrics and monitoring
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import redis
import sqlite3

# Structured logging
import structlog
from pythonjsonlogger import jsonlogger

# Alerting
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import slack_sdk

# Performance monitoring
import cProfile
import pstats
from functools import wraps
import traceback

class LogLevel(Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LogEvent:
    """Structured log event"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class MetricEvent:
    """System metric event"""
    timestamp: datetime
    metric_name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass
class AlertEvent:
    """Alert event"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    source: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    acknowledged: bool = False

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    open_file_descriptors: int
    thread_count: int
    process_count: int
    load_average: List[float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class StructuredLogger:
    """Enhanced structured logging with correlation tracking"""

    def __init__(self, name: str, log_level: LogLevel = LogLevel.INFO):
        self.name = name
        self.log_level = log_level
        self.correlation_id = None
        self.user_id = None
        self.session_id = None

        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger(name)

    def set_context(self, correlation_id: str = None, user_id: str = None, session_id: str = None):
        """Set logging context"""
        self.correlation_id = correlation_id
        self.user_id = user_id
        self.session_id = session_id

    def _create_log_event(self, level: LogLevel, message: str, **kwargs) -> LogEvent:
        """Create structured log event"""
        frame = sys._getframe(2)

        return LogEvent(
            timestamp=datetime.now(timezone.utc),
            level=level,
            logger_name=self.name,
            message=message,
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            correlation_id=self.correlation_id,
            user_id=self.user_id,
            session_id=self.session_id,
            extra_data=kwargs,
            exception_info=kwargs.get('exc_info'),
            stack_trace=traceback.format_stack() if kwargs.get('include_stack') else None
        )

    def trace(self, message: str, **kwargs):
        """Log trace level message"""
        if self.log_level.value <= LogLevel.TRACE.value:
            event = self._create_log_event(LogLevel.TRACE, message, **kwargs)
            self.logger.debug(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level message"""
        if self.log_level.value <= LogLevel.DEBUG.value:
            event = self._create_log_event(LogLevel.DEBUG, message, **kwargs)
            self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info level message"""
        event = self._create_log_event(LogLevel.INFO, message, **kwargs)
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning level message"""
        event = self._create_log_event(LogLevel.WARNING, message, **kwargs)
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error level message"""
        event = self._create_log_event(LogLevel.ERROR, message, **kwargs)
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical level message"""
        event = self._create_log_event(LogLevel.CRITICAL, message, **kwargs)
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with stack trace"""
        kwargs['exc_info'] = True
        self.error(message, **kwargs)

class MetricsCollector:
    """Prometheus-compatible metrics collection"""

    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        self.metrics: Dict[str, Any] = {}

        # Default system metrics
        self._setup_system_metrics()

    def _setup_system_metrics(self):
        """Setup default system metrics"""

        # Trading system metrics
        self.trading_orders_total = Counter(
            'trading_orders_total',
            'Total number of trading orders',
            ['strategy', 'symbol', 'side', 'status'],
            registry=self.registry
        )

        self.trading_pnl_total = Gauge(
            'trading_pnl_total',
            'Total P&L across all strategies',
            ['strategy'],
            registry=self.registry
        )

        self.trading_positions_current = Gauge(
            'trading_positions_current',
            'Current number of open positions',
            ['strategy', 'symbol'],
            registry=self.registry
        )

        self.market_data_latency = Histogram(
            'market_data_latency_seconds',
            'Market data latency in seconds',
            ['provider', 'symbol'],
            registry=self.registry
        )

        self.order_execution_time = Histogram(
            'order_execution_time_seconds',
            'Order execution time in seconds',
            ['broker', 'order_type'],
            registry=self.registry
        )

        # System performance metrics
        self.cpu_usage_percent = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage_percent = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )

        self.disk_usage_percent = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            ['device'],
            registry=self.registry
        )

        # Application metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )

    def counter(self, name: str, description: str, labels: List[str] = None) -> Counter:
        """Create or get counter metric"""
        if name not in self.metrics:
            self.metrics[name] = Counter(name, description, labels or [], registry=self.registry)
        return self.metrics[name]

    def gauge(self, name: str, description: str, labels: List[str] = None) -> Gauge:
        """Create or get gauge metric"""
        if name not in self.metrics:
            self.metrics[name] = Gauge(name, description, labels or [], registry=self.registry)
        return self.metrics[name]

    def histogram(self, name: str, description: str, labels: List[str] = None, buckets: List[float] = None) -> Histogram:
        """Create or get histogram metric"""
        if name not in self.metrics:
            kwargs = {'name': name, 'documentation': description, 'labelnames': labels or [], 'registry': self.registry}
            if buckets:
                kwargs['buckets'] = buckets
            self.metrics[name] = Histogram(**kwargs)
        return self.metrics[name]

    def summary(self, name: str, description: str, labels: List[str] = None) -> Summary:
        """Create or get summary metric"""
        if name not in self.metrics:
            self.metrics[name] = Summary(name, description, labels or [], registry=self.registry)
        return self.metrics[name]

    def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_usage_mb = memory.used / (1024 * 1024)

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # Network I/O
            network_io = psutil.net_io_counters()._asdict()

            # Process metrics
            process = psutil.Process()
            open_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            thread_count = process.num_threads()

            # System load
            load_average = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]

            # Update Prometheus metrics
            self.cpu_usage_percent.set(cpu_percent)
            self.memory_usage_percent.set(memory_percent)
            self.disk_usage_percent.labels(device='root').set(disk_usage_percent)

            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_usage_mb=memory_usage_mb,
                disk_usage_percent=disk_usage_percent,
                network_io=network_io,
                open_file_descriptors=open_fds,
                thread_count=thread_count,
                process_count=len(psutil.pids()),
                load_average=load_average
            )

        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, {}, 0, 0, 0, [0, 0, 0])

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

class AlertManager:
    """Alert management and notification system"""

    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: Dict[str, Callable] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=10000)

        # Default alert rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default alerting rules"""

        # System resource alerts
        self.add_rule(
            "high_cpu_usage",
            metric_name="system_cpu_usage_percent",
            threshold=80.0,
            operator=">=",
            severity=AlertSeverity.HIGH,
            description="CPU usage is above 80%"
        )

        self.add_rule(
            "high_memory_usage",
            metric_name="system_memory_usage_percent",
            threshold=85.0,
            operator=">=",
            severity=AlertSeverity.HIGH,
            description="Memory usage is above 85%"
        )

        # Trading system alerts
        self.add_rule(
            "high_order_rejection_rate",
            metric_name="trading_order_rejection_rate",
            threshold=0.1,
            operator=">=",
            severity=AlertSeverity.CRITICAL,
            description="Order rejection rate is above 10%"
        )

        self.add_rule(
            "high_market_data_latency",
            metric_name="market_data_latency_seconds",
            threshold=1.0,
            operator=">=",
            severity=AlertSeverity.MEDIUM,
            description="Market data latency is above 1 second"
        )

    def add_rule(self, rule_id: str, metric_name: str, threshold: float,
                operator: str, severity: AlertSeverity, description: str = "",
                duration: int = 60, labels: Dict[str, str] = None):
        """Add alerting rule"""

        self.alert_rules[rule_id] = {
            'metric_name': metric_name,
            'threshold': threshold,
            'operator': operator,
            'severity': severity,
            'description': description,
            'duration': duration,
            'labels': labels or {},
            'enabled': True
        }

    def add_notification_channel(self, channel_name: str, handler: Callable):
        """Add notification channel"""
        self.notification_channels[channel_name] = handler

    def evaluate_rules(self, metrics: Dict[str, float]):
        """Evaluate alerting rules against current metrics"""

        for rule_id, rule in self.alert_rules.items():
            if not rule['enabled']:
                continue

            metric_name = rule['metric_name']
            if metric_name not in metrics:
                continue

            current_value = metrics[metric_name]
            threshold = rule['threshold']
            operator = rule['operator']

            # Evaluate condition
            triggered = False
            if operator == ">=":
                triggered = current_value >= threshold
            elif operator == "<=":
                triggered = current_value <= threshold
            elif operator == ">":
                triggered = current_value > threshold
            elif operator == "<":
                triggered = current_value < threshold
            elif operator == "==":
                triggered = current_value == threshold
            elif operator == "!=":
                triggered = current_value != threshold

            if triggered:
                self._trigger_alert(rule_id, rule, current_value)
            else:
                self._resolve_alert(rule_id)

    def _trigger_alert(self, rule_id: str, rule: Dict[str, Any], current_value: float):
        """Trigger alert"""

        # Check if alert is already active
        if rule_id in self.active_alerts:
            return

        alert = AlertEvent(
            alert_id=f"{rule_id}_{int(time.time())}",
            timestamp=datetime.now(timezone.utc),
            severity=rule['severity'],
            title=f"Alert: {rule_id}",
            message=rule['description'],
            source="alert_manager",
            metric_name=rule['metric_name'],
            current_value=current_value,
            threshold=rule['threshold'],
            labels=rule['labels']
        )

        self.active_alerts[rule_id] = alert
        self.alert_history.append(alert)

        # Send notifications
        self._send_notifications(alert)

        # Store in Redis if available
        if self.redis_client:
            self.redis_client.setex(
                f"alert:{rule_id}",
                3600,  # 1 hour expiry
                json.dumps(asdict(alert), default=str)
            )

    def _resolve_alert(self, rule_id: str):
        """Resolve alert"""

        if rule_id in self.active_alerts:
            alert = self.active_alerts[rule_id]
            alert.resolved = True
            alert.timestamp = datetime.now(timezone.utc)

            # Send resolution notification
            self._send_notifications(alert, resolved=True)

            # Remove from active alerts
            del self.active_alerts[rule_id]

            # Remove from Redis
            if self.redis_client:
                self.redis_client.delete(f"alert:{rule_id}")

    def _send_notifications(self, alert: AlertEvent, resolved: bool = False):
        """Send alert notifications"""

        for channel_name, handler in self.notification_channels.items():
            try:
                handler(alert, resolved)
            except Exception as e:
                logging.error(f"Error sending notification via {channel_name}: {e}")

    def get_active_alerts(self) -> List[AlertEvent]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())

    def acknowledge_alert(self, rule_id: str, acknowledged_by: str):
        """Acknowledge alert"""
        if rule_id in self.active_alerts:
            self.active_alerts[rule_id].acknowledged = True
            logging.info(f"Alert {rule_id} acknowledged by {acknowledged_by}")

class PerformanceProfiler:
    """Application performance profiling"""

    def __init__(self):
        self.profiles: Dict[str, cProfile.Profile] = {}
        self.execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def profile_function(self, func_name: str = None):
        """Decorator for profiling function execution"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                profile_name = func_name or f"{func.__module__}.{func.__name__}"

                start_time = time.time()

                # Start profiling
                profiler = cProfile.Profile()
                profiler.enable()

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profiler.disable()

                    # Record execution time
                    execution_time = time.time() - start_time
                    self.execution_times[profile_name].append(execution_time)

                    # Store profile
                    self.profiles[profile_name] = profiler

            return wrapper
        return decorator

    def get_profile_stats(self, func_name: str) -> Optional[str]:
        """Get profile statistics for function"""

        if func_name not in self.profiles:
            return None

        profiler = self.profiles[func_name]
        stats = pstats.Stats(profiler)

        # Format stats as string
        import io
        output = io.StringIO()
        stats.print_stats(output)
        return output.getvalue()

    def get_execution_stats(self, func_name: str) -> Dict[str, float]:
        """Get execution time statistics"""

        if func_name not in self.execution_times:
            return {}

        times = list(self.execution_times[func_name])
        if not times:
            return {}

        return {
            'count': len(times),
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times),
            'total': sum(times)
        }

class MonitoringSystem:
    """Central monitoring system coordinating all components"""

    def __init__(self, redis_client: redis.Redis = None, db_path: str = "monitoring.db"):
        self.redis_client = redis_client
        self.db_path = db_path

        # Initialize components
        self.logger = StructuredLogger("monitoring_system")
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(redis_client)
        self.performance_profiler = PerformanceProfiler()

        # Setup database
        self._setup_database()

        # Setup notification channels
        self._setup_notifications()

        # Start background monitoring
        self.monitoring_thread = None
        self.is_monitoring = False

    def _setup_database(self):
        """Setup monitoring database"""

        with sqlite3.connect(self.db_path) as conn:
            # System metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp TEXT,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_usage_mb REAL,
                    disk_usage_percent REAL,
                    network_io TEXT,
                    open_file_descriptors INTEGER,
                    thread_count INTEGER,
                    process_count INTEGER,
                    load_average TEXT
                )
            """)

            # Application metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS application_metrics (
                    timestamp TEXT,
                    metric_name TEXT,
                    metric_type TEXT,
                    value REAL,
                    labels TEXT
                )
            """)

            # Performance profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_profiles (
                    timestamp TEXT,
                    function_name TEXT,
                    execution_time REAL,
                    profile_data TEXT
                )
            """)

    def _setup_notifications(self):
        """Setup notification channels"""

        # Email notifications
        def email_handler(alert: AlertEvent, resolved: bool = False):
            # Implementation would send email
            self.logger.info(f"Email notification: {alert.title}",
                           alert_id=alert.alert_id, resolved=resolved)

        # Slack notifications
        def slack_handler(alert: AlertEvent, resolved: bool = False):
            # Implementation would send Slack message
            self.logger.info(f"Slack notification: {alert.title}",
                           alert_id=alert.alert_id, resolved=resolved)

        # Console notifications for development
        def console_handler(alert: AlertEvent, resolved: bool = False):
            status = "RESOLVED" if resolved else "TRIGGERED"
            self.logger.warning(f"ALERT {status}: {alert.title} - {alert.message}",
                              alert_id=alert.alert_id,
                              severity=alert.severity.value,
                              metric_value=alert.current_value)

        self.alert_manager.add_notification_channel("email", email_handler)
        self.alert_manager.add_notification_channel("slack", slack_handler)
        self.alert_manager.add_notification_channel("console", console_handler)

    def start_monitoring(self, interval: int = 60):
        """Start background monitoring"""

        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Monitoring system started", interval=interval)

    def stop_monitoring(self):
        """Stop background monitoring"""

        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

        self.logger.info("Monitoring system stopped")

    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""

        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()

                # Store metrics in database
                self._store_system_metrics(system_metrics)

                # Store metrics in Redis for real-time access
                if self.redis_client:
                    self.redis_client.setex(
                        "system_metrics:current",
                        300,  # 5 minutes
                        json.dumps(asdict(system_metrics), default=str)
                    )

                # Evaluate alert rules
                metrics_dict = {
                    'system_cpu_usage_percent': system_metrics.cpu_percent,
                    'system_memory_usage_percent': system_metrics.memory_percent,
                    'system_disk_usage_percent': system_metrics.disk_usage_percent
                }

                self.alert_manager.evaluate_rules(metrics_dict)

                # Log monitoring cycle
                self.logger.debug("Monitoring cycle completed",
                                cpu_percent=system_metrics.cpu_percent,
                                memory_percent=system_metrics.memory_percent)

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(interval)

    def _store_system_metrics(self, metrics: PerformanceMetrics):
        """Store system metrics in database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_metrics
                    (timestamp, cpu_percent, memory_percent, memory_usage_mb,
                     disk_usage_percent, network_io, open_file_descriptors,
                     thread_count, process_count, load_average)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.isoformat(),
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.memory_usage_mb,
                    metrics.disk_usage_percent,
                    json.dumps(metrics.network_io),
                    metrics.open_file_descriptors,
                    metrics.thread_count,
                    metrics.process_count,
                    json.dumps(metrics.load_average)
                ))

        except Exception as e:
            self.logger.error(f"Error storing system metrics: {e}")

    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None):
        """Record custom application metric"""

        labels = labels or {}

        # Update Prometheus metrics
        if metric_type == MetricType.COUNTER:
            metric = self.metrics_collector.counter(name, f"Custom counter: {name}")
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

        elif metric_type == MetricType.GAUGE:
            metric = self.metrics_collector.gauge(name, f"Custom gauge: {name}")
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO application_metrics
                    (timestamp, metric_name, metric_type, value, labels)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    name,
                    metric_type.value,
                    value,
                    json.dumps(labels)
                ))
        except Exception as e:
            self.logger.error(f"Error storing application metric: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""

        # Get current metrics
        current_metrics = self.metrics_collector.collect_system_metrics()

        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()

        # Determine health status
        health_status = "healthy"
        if any(alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] for alert in active_alerts):
            health_status = "critical"
        elif any(alert.severity == AlertSeverity.MEDIUM for alert in active_alerts):
            health_status = "warning"
        elif active_alerts:
            health_status = "degraded"

        return {
            'status': health_status,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': asdict(current_metrics),
            'active_alerts': len(active_alerts),
            'alert_details': [
                {
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'metric': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold
                }
                for alert in active_alerts
            ]
        }

    def get_metrics_export(self) -> str:
        """Get metrics in Prometheus format for scraping"""
        return self.metrics_collector.get_prometheus_metrics()

# Performance monitoring decorators
def monitor_execution_time(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor function execution time"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time

                # Record metric (assuming global monitoring system)
                if hasattr(wrapper, '_monitoring_system'):
                    wrapper._monitoring_system.record_metric(
                        f"{name}_execution_time",
                        execution_time,
                        MetricType.HISTOGRAM,
                        labels
                    )

        return wrapper
    return decorator

def monitor_function_calls(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor function call counts"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}_calls"

            # Record metric (assuming global monitoring system)
            if hasattr(wrapper, '_monitoring_system'):
                wrapper._monitoring_system.record_metric(
                    name,
                    1,
                    MetricType.COUNTER,
                    labels
                )

            return func(*args, **kwargs)

        return wrapper
    return decorator

# Example usage and setup
def setup_monitoring_system():
    """Setup and configure monitoring system"""

    # Initialize Redis connection
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # Create monitoring system
    monitoring = MonitoringSystem(redis_client)

    # Start monitoring
    monitoring.start_monitoring(interval=30)  # Monitor every 30 seconds

    return monitoring

if __name__ == "__main__":
    # Example usage
    monitoring = setup_monitoring_system()

    # Example of recording custom metrics
    monitoring.record_metric("trading_orders_executed", 150, MetricType.COUNTER, {"strategy": "momentum"})
    monitoring.record_metric("portfolio_value", 1050000.0, MetricType.GAUGE)

    # Example performance monitoring
    @monitor_execution_time("example_function")
    @monitor_function_calls("example_function")
    def example_function():
        time.sleep(0.1)  # Simulate work
        return "completed"

    # Set monitoring system for decorators
    example_function._monitoring_system = monitoring

    # Test function
    result = example_function()

    # Get system health
    health = monitoring.get_system_health()
    print(f"System health: {health['status']}")

    # Get metrics export
    metrics_export = monitoring.get_metrics_export()
    print("Prometheus metrics available for scraping")

    # Keep running for a while to see monitoring in action
    import signal
    import sys

    def signal_handler(sig, frame):
        monitoring.stop_monitoring()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("Monitoring system running... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitoring.stop_monitoring()