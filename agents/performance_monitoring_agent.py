#!/usr/bin/env python3
"""
Performance Monitoring Agent - Task 8.1 Implementation

This agent implements comprehensive performance monitoring including:
- Basic performance dashboards with real-time metrics
- Real-time P&L tracking and portfolio monitoring
- Latency monitoring (p50/p95/p99) for all system components
- Basic alerting for system failures and performance degradation
- Integration with existing monitoring infrastructure

Requirements: Requirement 9 (Monitoring and Observability)
Task: 8.1 Performance Monitoring
"""

import sys
import os
from pathlib import Path

# Add project root to Python path to ensure local config is imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
import psutil
import threading
from collections import defaultdict, deque

# LangGraph imports
from langgraph.graph import StateGraph, END

# Database imports
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Configuration
from config.settings import settings
from config.secure_config import get_api_keys

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    PNL = "pnl"
    POSITION = "position"
    RISK = "risk"
    SYSTEM = "system"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of system alerts"""
    LATENCY_THRESHOLD = "latency_threshold"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_SPACE = "disk_space"
    PNL_DRAWDOWN = "pnl_drawdown"
    POSITION_LIMIT = "position_limit"
    SYSTEM_FAILURE = "system_failure"
    DATA_FEED_ISSUE = "data_feed_issue"
    BROKER_CONNECTION = "broker_connection"


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    metric_type: MetricType
    component: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyMetric:
    """Latency-specific metric"""
    component: str
    p50: float
    p95: float
    p99: float
    min_latency: float
    max_latency: float
    avg_latency: float
    sample_count: int
    timestamp: datetime


@dataclass
class PnLMetric:
    """P&L tracking metric"""
    symbol: str
    position_size: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    entry_price: float
    current_price: float
    timestamp: datetime


@dataclass
class SystemAlert:
    """System alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    component: str
    message: str
    value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Dashboard data structure"""
    timestamp: datetime
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    pnl_summary: Dict[str, Any]
    alerts: List[SystemAlert]
    latency_summary: Dict[str, LatencyMetric]
    resource_usage: Dict[str, float]


class LatencyMonitor:
    """Monitors latency metrics for system components"""
    
    def __init__(self, retention_samples: int = 1000):
        self.retention_samples = retention_samples
        self.latency_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=retention_samples)
        )
        self.lock = threading.Lock()
    
    def record_latency(self, component: str, latency: float):
        """Record latency measurement for a component"""
        with self.lock:
            self.latency_data[component].append(latency)
    
    def get_latency_metrics(self, component: str) -> Optional[LatencyMetric]:
        """Get latency metrics for a component"""
        with self.lock:
            if component not in self.latency_data or not self.latency_data[component]:
                return None
            
            latencies = list(self.latency_data[component])
            if not latencies:
                return None
            
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            return LatencyMetric(
                component=component,
                p50=sorted_latencies[int(n * 0.5)] if n > 0 else 0,
                p95=sorted_latencies[int(n * 0.95)] if n > 0 else 0,
                p99=sorted_latencies[int(n * 0.99)] if n > 0 else 0,
                min_latency=min(latencies),
                max_latency=max(latencies),
                avg_latency=sum(latencies) / len(latencies),
                sample_count=len(latencies),
                timestamp=datetime.now()
            )
    
    def get_all_latency_metrics(self) -> Dict[str, LatencyMetric]:
        """Get latency metrics for all components"""
        metrics = {}
        for component in self.latency_data.keys():
            metric = self.get_latency_metrics(component)
            if metric:
                metrics[component] = metric
        return metrics


class PnLMonitor:
    """Monitors P&L and portfolio metrics"""
    
    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        self.pnl_history: deque = deque(maxlen=10000)
        self.daily_pnl: Dict[str, float] = defaultdict(float)
        self.lock = threading.Lock()
    
    def update_position(self, symbol: str, position_data: Dict):
        """Update position information"""
        with self.lock:
            self.positions[symbol] = position_data.copy()
            self.positions[symbol]['timestamp'] = datetime.now()
    
    def record_pnl(self, symbol: str, pnl: float, pnl_type: str = "total"):
        """Record P&L change"""
        with self.lock:
            timestamp = datetime.now()
            date_key = timestamp.strftime('%Y-%m-%d')
            
            # Record in history
            self.pnl_history.append({
                'symbol': symbol,
                'pnl': pnl,
                'type': pnl_type,
                'timestamp': timestamp
            })
            
            # Update daily P&L
            self.daily_pnl[date_key] += pnl
    
    def get_pnl_summary(self) -> Dict[str, Any]:
        """Get comprehensive P&L summary"""
        with self.lock:
            if not self.pnl_history:
                return {
                    'total_pnl': 0.0,
                    'daily_pnl': {},
                    'position_count': 0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0
                }
            
            # Calculate total P&L
            total_pnl = sum(record['pnl'] for record in self.pnl_history)
            
            # Calculate position metrics
            unrealized_pnl = sum(
                pos.get('unrealized_pnl', 0) for pos in self.positions.values()
            )
            realized_pnl = sum(
                pos.get('realized_pnl', 0) for pos in self.positions.values()
            )
            
            return {
                'total_pnl': total_pnl,
                'daily_pnl': dict(self.daily_pnl),
                'position_count': len(self.positions),
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'positions': self.positions.copy()
            }
    
    def get_drawdown_metrics(self) -> Dict[str, float]:
        """Calculate drawdown metrics"""
        with self.lock:
            if not self.pnl_history:
                return {'max_drawdown': 0.0, 'current_drawdown': 0.0}
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            running_total = 0
            for record in self.pnl_history:
                running_total += record['pnl']
                cumulative_pnl.append(running_total)
            
            if not cumulative_pnl:
                return {'max_drawdown': 0.0, 'current_drawdown': 0.0}
            
            # Calculate drawdown
            peak = cumulative_pnl[0]
            max_drawdown = 0.0
            current_drawdown = 0.0
            
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                drawdown = (peak - pnl) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                if pnl == cumulative_pnl[-1]:  # Current value
                    current_drawdown = drawdown
            
            return {
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown
            }


class ResourceMonitor:
    """Monitors system resource usage"""
    
    def __init__(self):
        self.last_check = datetime.now()
        self.resource_history: deque = deque(maxlen=1000)
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            
            # Network I/O
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process count
            process_count = len(psutil.pids())
            
            resources = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free_gb,
                'network_bytes_sent': network_bytes_sent,
                'network_bytes_recv': network_bytes_recv,
                'process_count': process_count,
                'timestamp': datetime.now()
            }
            
            # Store in history
            self.resource_history.append(resources)
            
            return resources
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_used_gb': 0.0,
                'disk_percent': 0.0,
                'disk_free_gb': 0.0,
                'network_bytes_sent': 0,
                'network_bytes_recv': 0,
                'process_count': 0,
                'timestamp': datetime.now()
            }
    
    def get_resource_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get resource usage trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        trends = defaultdict(list)
        timestamps = []
        
        for resource_data in self.resource_history:
            if resource_data['timestamp'] >= cutoff_time:
                for key, value in resource_data.items():
                    if key != 'timestamp':
                        trends[key].append(value)
                timestamps.append(resource_data['timestamp'])
        
        return dict(trends)


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[SystemAlert] = []
        self.alert_rules: Dict[str, Dict] = {}
        self.alert_handlers: Dict[AlertType, callable] = {}
        self.lock = threading.Lock()
        
        # Initialize default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        self.alert_rules = {
            'latency_p99': {
                'threshold': 1000.0,  # 1 second
                'severity': AlertSeverity.WARNING,
                'component': 'all'
            },
            'memory_usage': {
                'threshold': 90.0,  # 90%
                'severity': AlertSeverity.WARNING,
                'component': 'system'
            },
            'cpu_usage': {
                'threshold': 95.0,  # 95%
                'severity': AlertSeverity.WARNING,
                'component': 'system'
            },
            'disk_usage': {
                'threshold': 85.0,  # 85%
                'severity': AlertSeverity.WARNING,
                'component': 'system'
            },
            'pnl_drawdown': {
                'threshold': 0.10,  # 10%
                'severity': AlertSeverity.ERROR,
                'component': 'trading'
            }
        }
    
    def add_alert_rule(self, rule_name: str, rule_config: Dict):
        """Add a new alert rule"""
        with self.lock:
            self.alert_rules[rule_name] = rule_config
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[SystemAlert]:
        """Check metrics against alert rules and generate alerts"""
        new_alerts = []
        
        with self.lock:
            # Check latency alerts
            if 'latency_p99' in metrics:
                for component, latency_metric in metrics['latency_p99'].items():
                    if latency_metric.p99 > self.alert_rules['latency_p99']['threshold']:
                        alert = SystemAlert(
                            alert_id=f"latency_{component}_{int(time.time())}",
                            alert_type=AlertType.LATENCY_THRESHOLD,
                            severity=self.alert_rules['latency_p99']['severity'],
                            component=component,
                            message=f"P99 latency {latency_metric.p99:.2f}ms exceeds threshold {self.alert_rules['latency_p99']['threshold']}ms",
                            value=latency_metric.p99,
                            threshold=self.alert_rules['latency_p99']['threshold'],
                            timestamp=datetime.now()
                        )
                        new_alerts.append(alert)
            
            # Check resource usage alerts
            if 'resources' in metrics:
                resources = metrics['resources']
                
                # Memory usage
                if resources.get('memory_percent', 0) > self.alert_rules['memory_usage']['threshold']:
                    alert = SystemAlert(
                        alert_id=f"memory_{int(time.time())}",
                        alert_type=AlertType.MEMORY_USAGE,
                        severity=self.alert_rules['memory_usage']['severity'],
                        component='system',
                        message=f"Memory usage {resources['memory_percent']:.1f}% exceeds threshold {self.alert_rules['memory_usage']['threshold']}%",
                        value=resources['memory_percent'],
                        threshold=self.alert_rules['memory_usage']['threshold'],
                        timestamp=datetime.now()
                    )
                    new_alerts.append(alert)
                
                # CPU usage
                if resources.get('cpu_percent', 0) > self.alert_rules['cpu_usage']['threshold']:
                    alert = SystemAlert(
                        alert_id=f"cpu_{int(time.time())}",
                        alert_type=AlertType.CPU_USAGE,
                        severity=self.alert_rules['cpu_usage']['severity'],
                        component='system',
                        message=f"CPU usage {resources['cpu_percent']:.1f}% exceeds threshold {self.alert_rules['cpu_usage']['threshold']}%",
                        value=resources['cpu_percent'],
                        threshold=self.alert_rules['cpu_usage']['threshold'],
                        timestamp=datetime.now()
                    )
                    new_alerts.append(alert)
                
                # Disk usage
                if resources.get('disk_percent', 0) > self.alert_rules['disk_usage']['threshold']:
                    alert = SystemAlert(
                        alert_id=f"disk_{int(time.time())}",
                        alert_type=AlertType.DISK_SPACE,
                        severity=self.alert_rules['disk_usage']['severity'],
                        component='system',
                        message=f"Disk usage {resources['disk_percent']:.1f}% exceeds threshold {self.alert_rules['disk_usage']['threshold']}%",
                        value=resources['disk_percent'],
                        threshold=self.alert_rules['disk_usage']['threshold'],
                        timestamp=datetime.now()
                    )
                    new_alerts.append(alert)
            
            # Check P&L alerts
            if 'pnl' in metrics:
                pnl_data = metrics['pnl']
                if 'drawdown' in pnl_data:
                    drawdown = pnl_data['drawdown'].get('current_drawdown', 0)
                    if drawdown > self.alert_rules['pnl_drawdown']['threshold']:
                        alert = SystemAlert(
                            alert_id=f"pnl_drawdown_{int(time.time())}",
                            alert_type=AlertType.PNL_DRAWDOWN,
                            severity=self.alert_rules['pnl_drawdown']['severity'],
                            component='trading',
                            message=f"Current drawdown {drawdown:.2%} exceeds threshold {self.alert_rules['pnl_drawdown']['threshold']:.2%}",
                            value=drawdown,
                            threshold=self.alert_rules['pnl_drawdown']['threshold'],
                            timestamp=datetime.now()
                        )
                        new_alerts.append(alert)
        
        # Add new alerts to the list
        with self.lock:
            self.alerts.extend(new_alerts)
        
        return new_alerts
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    break
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    break
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts"""
        with self.lock:
            total_alerts = len(self.alerts)
            active_alerts = len([a for a in self.alerts if not a.resolved])
            acknowledged_alerts = len([a for a in self.alerts if a.acknowledged and not a.resolved])
            resolved_alerts = len([a for a in self.alerts if a.resolved])
            
            # Count by severity
            severity_counts = defaultdict(int)
            for alert in self.alerts:
                severity_counts[alert.severity.value] += 1
            
            return {
                'total_alerts': total_alerts,
                'active_alerts': active_alerts,
                'acknowledged_alerts': acknowledged_alerts,
                'resolved_alerts': resolved_alerts,
                'severity_breakdown': dict(severity_counts)
            }


class PerformanceMonitoringAgent:
    """
    Main performance monitoring agent
    
    This agent provides comprehensive performance monitoring including:
    - Real-time performance dashboards
    - P&L tracking and portfolio monitoring
    - Latency monitoring (p50/p95/p99)
    - System resource monitoring
    - Alert management and notification
    """
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.running = False
        
        # Initialize monitoring components
        self.latency_monitor = LatencyMonitor()
        self.pnl_monitor = PnLMonitor()
        self.resource_monitor = ResourceMonitor()
        self.alert_manager = AlertManager()
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=10000)
        self.start_time = datetime.now()
        
        # Monitoring thread
        self.monitoring_thread = None
        
        logger.info("Performance Monitoring Agent initialized")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect all metrics
                metrics = self._collect_metrics()
                
                # Check for alerts
                new_alerts = self.alert_manager.check_alerts(metrics)
                
                # Store performance data
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics,
                    'alerts': new_alerts
                })
                
                # Log critical alerts
                for alert in new_alerts:
                    if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                        logger.error(f"CRITICAL ALERT: {alert.message}")
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect all performance metrics"""
        try:
            # Collect latency metrics
            latency_metrics = self.latency_monitor.get_all_latency_metrics()
            
            # Collect P&L metrics
            pnl_summary = self.pnl_monitor.get_pnl_summary()
            drawdown_metrics = self.pnl_monitor.get_drawdown_metrics()
            
            # Collect system resources
            resources = self.resource_monitor.get_system_resources()
            
            # Collect alerts
            active_alerts = self.alert_manager.get_active_alerts()
            alert_summary = self.alert_manager.get_alert_summary()
            
            return {
                'latency_p99': latency_metrics,
                'pnl': {
                    'summary': pnl_summary,
                    'drawdown': drawdown_metrics
                },
                'resources': resources,
                'alerts': {
                    'active': active_alerts,
                    'summary': alert_summary
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def record_latency(self, component: str, latency: float):
        """Record latency measurement for a component"""
        self.latency_monitor.record_latency(component, latency)
    
    def update_position(self, symbol: str, position_data: Dict):
        """Update position information"""
        self.pnl_monitor.update_position(symbol, position_data)
    
    def record_pnl(self, symbol: str, pnl: float, pnl_type: str = "total"):
        """Record P&L change"""
        self.pnl_monitor.record_pnl(symbol, pnl, pnl_type)
    
    def get_dashboard_data(self) -> DashboardData:
        """Get comprehensive dashboard data"""
        try:
            # Collect current metrics
            metrics = self._collect_metrics()
            
            # Get system health status
            system_health = self._calculate_system_health(metrics)
            
            # Get performance summary
            performance_summary = self._calculate_performance_summary(metrics)
            
            # Get P&L summary
            pnl_summary = metrics.get('pnl', {}).get('summary', {})
            
            # Get active alerts
            active_alerts = metrics.get('alerts', {}).get('active', [])
            
            # Get latency summary
            latency_summary = metrics.get('latency_p99', {})
            
            # Get resource usage
            resources = metrics.get('resources', {})
            resource_usage = {
                'cpu_percent': resources.get('cpu_percent', 0),
                'memory_percent': resources.get('memory_percent', 0),
                'disk_percent': resources.get('disk_percent', 0),
                'process_count': resources.get('process_count', 0)
            }
            
            return DashboardData(
                timestamp=datetime.now(),
                system_health=system_health,
                performance_metrics=performance_summary,
                pnl_summary=pnl_summary,
                alerts=active_alerts,
                latency_summary=latency_summary,
                resource_usage=resource_usage
            )
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return DashboardData(
                timestamp=datetime.now(),
                system_health={'status': 'error', 'message': str(e)},
                performance_metrics={},
                pnl_summary={},
                alerts=[],
                latency_summary={},
                resource_usage={}
            )
    
    def _calculate_system_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health status"""
        try:
            health_score = 100
            issues = []
            
            # Check latency
            latency_metrics = metrics.get('latency_p99', {})
            for component, latency in latency_metrics.items():
                if latency.p99 > 1000:  # 1 second threshold
                    health_score -= 20
                    issues.append(f"High latency in {component}: {latency.p99:.2f}ms")
            
            # Check resource usage
            resources = metrics.get('resources', {})
            if resources.get('memory_percent', 0) > 90:
                health_score -= 15
                issues.append(f"High memory usage: {resources['memory_percent']:.1f}%")
            
            if resources.get('cpu_percent', 0) > 95:
                health_score -= 15
                issues.append(f"High CPU usage: {resources['cpu_percent']:.1f}%")
            
            if resources.get('disk_percent', 0) > 85:
                health_score -= 10
                issues.append(f"High disk usage: {resources['disk_percent']:.1f}%")
            
            # Check alerts
            alerts = metrics.get('alerts', {}).get('active', [])
            critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
            error_alerts = [a for a in alerts if a.severity == AlertSeverity.ERROR]
            
            health_score -= len(critical_alerts) * 25
            health_score -= len(error_alerts) * 15
            
            # Determine status
            if health_score >= 80:
                status = 'healthy'
            elif health_score >= 60:
                status = 'warning'
            elif health_score >= 40:
                status = 'critical'
            else:
                status = 'down'
            
            return {
                'status': status,
                'health_score': max(0, health_score),
                'issues': issues,
                'critical_alerts': len(critical_alerts),
                'error_alerts': len(error_alerts),
                'total_alerts': len(alerts)
            }
            
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return {
                'status': 'error',
                'health_score': 0,
                'issues': [f"Error calculating health: {str(e)}"],
                'critical_alerts': 0,
                'error_alerts': 0,
                'total_alerts': 0
            }
    
    def _calculate_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance summary metrics"""
        try:
            # Latency summary
            latency_metrics = metrics.get('latency_p99', {})
            if latency_metrics:
                all_p99s = [lat.p99 for lat in latency_metrics.values()]
                avg_p99 = sum(all_p99s) / len(all_p99s)
                max_p99 = max(all_p99s)
            else:
                avg_p99 = 0
                max_p99 = 0
            
            # Throughput summary (mock for now)
            throughput_summary = {
                'orders_per_second': 0,  # Would come from execution engine
                'signals_per_second': 0,  # Would come from signal generation
                'data_points_per_second': 0  # Would come from data ingestion
            }
            
            return {
                'latency': {
                    'avg_p99': avg_p99,
                    'max_p99': max_p99,
                    'components_monitored': len(latency_metrics)
                },
                'throughput': throughput_summary,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'monitoring_active': self.running
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {
                'error': str(e),
                'monitoring_active': self.running
            }
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance report for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter performance history
            relevant_data = [
                data for data in self.performance_history
                if data['timestamp'] >= cutoff_time
            ]
            
            if not relevant_data:
                return {
                    'period_hours': hours,
                    'data_points': 0,
                    'message': 'No data available for specified period'
                }
            
            # Calculate statistics
            latency_data = []
            pnl_data = []
            alert_counts = []
            
            for data in relevant_data:
                metrics = data.get('metrics', {})
                
                # Collect latency data
                latency_metrics = metrics.get('latency_p99', {})
                for component, latency in latency_metrics.items():
                    latency_data.append({
                        'component': component,
                        'p99': latency.p99,
                        'timestamp': data['timestamp']
                    })
                
                # Collect P&L data
                pnl_summary = metrics.get('pnl', {}).get('summary', {})
                if 'total_pnl' in pnl_summary:
                    pnl_data.append({
                        'pnl': pnl_summary['total_pnl'],
                        'timestamp': data['timestamp']
                    })
                
                # Collect alert counts
                alerts = metrics.get('alerts', {}).get('active', [])
                alert_counts.append(len(alerts))
            
            # Calculate statistics
            report = {
                'period_hours': hours,
                'data_points': len(relevant_data),
                'latency_stats': {},
                'pnl_stats': {},
                'alert_stats': {}
            }
            
            # Latency statistics
            if latency_data:
                components = set(item['component'] for item in latency_data)
                for component in components:
                    component_data = [item for item in latency_data if item['component'] == component]
                    p99_values = [item['p99'] for item in component_data]
                    
                    report['latency_stats'][component] = {
                        'avg_p99': sum(p99_values) / len(p99_values),
                        'max_p99': max(p99_values),
                        'min_p99': min(p99_values),
                        'samples': len(p99_values)
                    }
            
            # P&L statistics
            if pnl_data:
                pnl_values = [item['pnl'] for item in pnl_data]
                report['pnl_stats'] = {
                    'total_pnl': sum(pnl_values),
                    'avg_pnl': sum(pnl_values) / len(pnl_values),
                    'max_pnl': max(pnl_values),
                    'min_pnl': min(pnl_values),
                    'samples': len(pnl_values)
                }
            
            # Alert statistics
            if alert_counts:
                report['alert_stats'] = {
                    'total_alerts': sum(alert_counts),
                    'avg_alerts_per_check': sum(alert_counts) / len(alert_counts),
                    'max_alerts': max(alert_counts),
                    'min_alerts': min(alert_counts)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {
                'error': str(e),
                'period_hours': hours
            }
    
    def export_dashboard_data(self, output_path: str = None) -> str:
        """Export dashboard data to JSON file"""
        try:
            if output_path is None:
                output_path = f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            dashboard_data = self.get_dashboard_data()
            
            # Convert to serializable format
            export_data = {
                'timestamp': dashboard_data.timestamp.isoformat(),
                'system_health': dashboard_data.system_health,
                'performance_metrics': dashboard_data.performance_metrics,
                'pnl_summary': dashboard_data.pnl_summary,
                'alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'alert_type': alert.alert_type.value,
                        'severity': alert.severity.value,
                        'component': alert.component,
                        'message': alert.message,
                        'value': alert.value,
                        'threshold': alert.threshold,
                        'timestamp': alert.timestamp.isoformat(),
                        'acknowledged': alert.acknowledged,
                        'resolved': alert.resolved
                    } for alert in dashboard_data.alerts
                ],
                'latency_summary': {
                    component: {
                        'p50': metric.p50,
                        'p95': metric.p95,
                        'p99': metric.p99,
                        'avg_latency': metric.avg_latency,
                        'sample_count': metric.sample_count,
                        'timestamp': metric.timestamp.isoformat()
                    } for component, metric in dashboard_data.latency_summary.items()
                },
                'resource_usage': dashboard_data.resource_usage
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize monitoring agent
        monitor = PerformanceMonitoringAgent(update_interval=5)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate some activity
        print("Simulating monitoring activity...")
        
        # Record some latency measurements
        monitor.record_latency("order_execution", 150.0)
        monitor.record_latency("signal_generation", 45.0)
        monitor.record_latency("data_ingestion", 25.0)
        
        # Update some positions
        monitor.update_position("AAPL", {
            'quantity': 100,
            'entry_price': 150.0,
            'current_price': 155.0,
            'unrealized_pnl': 500.0,
            'realized_pnl': 0.0
        })
        
        monitor.update_position("TSLA", {
            'quantity': 50,
            'entry_price': 200.0,
            'current_price': 195.0,
            'unrealized_pnl': -250.0,
            'realized_pnl': 100.0
        })
        
        # Record some P&L
        monitor.record_pnl("AAPL", 500.0, "unrealized")
        monitor.record_pnl("TSLA", -250.0, "unrealized")
        monitor.record_pnl("TSLA", 100.0, "realized")
        
        # Wait for monitoring to collect data
        await asyncio.sleep(10)
        
        # Get dashboard data
        dashboard = monitor.get_dashboard_data()
        print(f"\nDashboard Status: {dashboard.system_health['status']}")
        print(f"Health Score: {dashboard.system_health['health_score']}")
        print(f"Active Alerts: {len(dashboard.alerts)}")
        
        # Get performance report
        report = monitor.get_performance_report(hours=1)
        print(f"\nPerformance Report: {report['data_points']} data points")
        
        # Export dashboard data
        export_path = monitor.export_dashboard_data()
        print(f"\nDashboard exported to: {export_path}")
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        print("\nPerformance monitoring demo completed!")
    
    # Run the demo
    asyncio.run(main()) 