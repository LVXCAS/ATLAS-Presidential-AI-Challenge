"""
Workflow Monitoring and Debugging Tools for LangGraph Trading System

This module provides comprehensive monitoring, debugging, and observability
tools for the LangGraph workflow execution.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time
import psutil
import threading
from collections import defaultdict, deque

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MonitoringLevel(str, Enum):
    """Monitoring detail levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"
    TRACE = "trace"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Performance metrics for workflow components"""
    component_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    error_count: int
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    timestamp: datetime


@dataclass
class WorkflowEvent:
    """Workflow execution event"""
    event_id: str
    event_type: str
    component: str
    timestamp: datetime
    duration: Optional[float] = None
    status: str = "success"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None


@dataclass
class SystemAlert:
    """System alert for monitoring"""
    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.retention_hours = retention_hours
        self.start_time = datetime.now()
        
        # Performance tracking
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("Metrics collector initialized")
    
    def record_execution_time(self, component: str, execution_time: float):
        """Record execution time for a component"""
        self.execution_times[component].append(execution_time)
        
        # Keep only recent measurements (last 100)
        if len(self.execution_times[component]) > 100:
            self.execution_times[component] = self.execution_times[component][-100:]
    
    def record_success(self, component: str):
        """Record successful execution"""
        self.success_counts[component] += 1
    
    def record_error(self, component: str):
        """Record error"""
        self.error_counts[component] += 1
    
    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """Get metrics for a specific component"""
        execution_times = self.execution_times.get(component, [])
        total_executions = self.success_counts[component] + self.error_counts[component]
        
        if not execution_times:
            return {
                "component": component,
                "total_executions": total_executions,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "error_count": self.error_counts[component]
            }
        
        # Calculate percentiles
        sorted_times = sorted(execution_times)
        n = len(sorted_times)
        
        p50 = sorted_times[int(n * 0.5)] if n > 0 else 0
        p95 = sorted_times[int(n * 0.95)] if n > 0 else 0
        p99 = sorted_times[int(n * 0.99)] if n > 0 else 0
        
        return {
            "component": component,
            "total_executions": total_executions,
            "success_rate": self.success_counts[component] / total_executions if total_executions > 0 else 0,
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            "error_count": self.error_counts[component],
            "success_count": self.success_counts[component]
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics"""
        # System resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network stats (if available)
        try:
            network = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        except:
            network_stats = {}
        
        return {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "network_stats": network_stats,
            "total_components": len(self.execution_times),
            "total_executions": sum(self.success_counts.values()) + sum(self.error_counts.values()),
            "total_errors": sum(self.error_counts.values())
        }
    
    def cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        for component_metrics in self.metrics.values():
            while component_metrics and component_metrics[0].timestamp < cutoff_time:
                component_metrics.popleft()


class EventLogger:
    """Logs and tracks workflow events"""
    
    def __init__(self, max_events: int = 10000):
        self.events: deque = deque(maxlen=max_events)
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        
        logger.info("Event logger initialized")
    
    def log_event(self, event: WorkflowEvent):
        """Log a workflow event"""
        with self.lock:
            self.events.append(event)
            self.event_counts[event.event_type] += 1
            
            # Log to standard logger based on status
            if event.status == "error":
                logger.error(f"[{event.component}] {event.event_type}: {event.error}")
            elif event.status == "warning":
                logger.warning(f"[{event.component}] {event.event_type}")
            else:
                logger.debug(f"[{event.component}] {event.event_type}")
    
    def get_recent_events(self, count: int = 100, event_type: str = None) -> List[WorkflowEvent]:
        """Get recent events, optionally filtered by type"""
        with self.lock:
            events = list(self.events)
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return events[-count:]
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of logged events"""
        with self.lock:
            total_events = len(self.events)
            error_events = len([e for e in self.events if e.status == "error"])
            warning_events = len([e for e in self.events if e.status == "warning"])
            
            return {
                "total_events": total_events,
                "error_events": error_events,
                "warning_events": warning_events,
                "success_rate": (total_events - error_events) / total_events if total_events > 0 else 0,
                "event_types": dict(self.event_counts),
                "recent_errors": [
                    {"component": e.component, "error": e.error, "timestamp": e.timestamp.isoformat()}
                    for e in list(self.events)[-10:] if e.status == "error"
                ]
            }


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, SystemAlert] = {}
        self.alert_handlers: List[Callable] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Default alert rules
        self._setup_default_rules()
        
        logger.info("Alert manager initialized")
    
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        self.alert_rules = {
            "high_error_rate": {
                "condition": lambda metrics: metrics.get("error_rate", 0) > 0.1,
                "severity": AlertSeverity.WARNING,
                "message": "High error rate detected: {error_rate:.2%}"
            },
            "high_latency": {
                "condition": lambda metrics: metrics.get("p95_latency", 0) > 5.0,
                "severity": AlertSeverity.WARNING,
                "message": "High latency detected: {p95_latency:.2f}s"
            },
            "system_overload": {
                "condition": lambda metrics: metrics.get("cpu_usage_percent", 0) > 90,
                "severity": AlertSeverity.CRITICAL,
                "message": "System overload: CPU usage {cpu_usage_percent:.1f}%"
            },
            "memory_pressure": {
                "condition": lambda metrics: metrics.get("memory_usage_percent", 0) > 85,
                "severity": AlertSeverity.WARNING,
                "message": "Memory pressure: {memory_usage_percent:.1f}% used"
            }
        }
    
    def add_alert_handler(self, handler: Callable[[SystemAlert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def create_alert(self, component: str, severity: AlertSeverity, message: str, data: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        alert_id = f"{component}_{severity}_{int(time.time())}"
        
        alert = SystemAlert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now(),
            data=data or {}
        )
        
        self.alerts[alert_id] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.info(f"Alert created: [{severity}] {component}: {message}")
        return alert_id
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")
    
    def check_alert_rules(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule["condition"](metrics):
                    # Check if similar alert already exists
                    existing_alerts = [
                        a for a in self.alerts.values()
                        if not a.resolved and rule_name in a.alert_id
                    ]
                    
                    if not existing_alerts:
                        message = rule["message"].format(**metrics)
                        self.create_alert("system", rule["severity"], message, metrics)
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_alerts = self.get_active_alerts()
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity] += 1
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "severity_breakdown": dict(severity_counts),
            "recent_alerts": [
                {
                    "severity": a.severity,
                    "component": a.component,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }


class WorkflowDebugger:
    """Debugging tools for workflow execution"""
    
    def __init__(self):
        self.breakpoints: Dict[str, Dict[str, Any]] = {}
        self.step_mode = False
        self.trace_enabled = False
        self.execution_stack: List[Dict[str, Any]] = []
        
        logger.info("Workflow debugger initialized")
    
    def set_breakpoint(self, component: str, condition: Callable = None):
        """Set a breakpoint for a component"""
        self.breakpoints[component] = {
            "condition": condition,
            "hit_count": 0,
            "enabled": True
        }
        logger.info(f"Breakpoint set for component: {component}")
    
    def remove_breakpoint(self, component: str):
        """Remove a breakpoint"""
        if component in self.breakpoints:
            del self.breakpoints[component]
            logger.info(f"Breakpoint removed for component: {component}")
    
    def enable_step_mode(self):
        """Enable step-by-step execution"""
        self.step_mode = True
        logger.info("Step mode enabled")
    
    def disable_step_mode(self):
        """Disable step-by-step execution"""
        self.step_mode = False
        logger.info("Step mode disabled")
    
    def enable_trace(self):
        """Enable execution tracing"""
        self.trace_enabled = True
        logger.info("Execution tracing enabled")
    
    def disable_trace(self):
        """Disable execution tracing"""
        self.trace_enabled = False
        logger.info("Execution tracing disabled")
    
    def check_breakpoint(self, component: str, state: Dict[str, Any]) -> bool:
        """Check if execution should pause at breakpoint"""
        if component not in self.breakpoints:
            return False
        
        breakpoint = self.breakpoints[component]
        
        if not breakpoint["enabled"]:
            return False
        
        # Check condition if specified
        if breakpoint["condition"]:
            try:
                if not breakpoint["condition"](state):
                    return False
            except Exception as e:
                logger.error(f"Error evaluating breakpoint condition: {e}")
                return False
        
        breakpoint["hit_count"] += 1
        logger.info(f"Breakpoint hit for component: {component} (count: {breakpoint['hit_count']})")
        return True
    
    def trace_execution(self, component: str, operation: str, data: Dict[str, Any]):
        """Trace execution step"""
        if not self.trace_enabled:
            return
        
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "operation": operation,
            "data": data
        }
        
        self.execution_stack.append(trace_entry)
        
        # Keep only recent traces
        if len(self.execution_stack) > 1000:
            self.execution_stack = self.execution_stack[-1000:]
        
        logger.debug(f"TRACE: [{component}] {operation}")
    
    def get_execution_trace(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution trace"""
        return self.execution_stack[-count:]
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information"""
        return {
            "step_mode": self.step_mode,
            "trace_enabled": self.trace_enabled,
            "breakpoints": {
                name: {
                    "hit_count": bp["hit_count"],
                    "enabled": bp["enabled"]
                }
                for name, bp in self.breakpoints.items()
            },
            "execution_stack_size": len(self.execution_stack)
        }


class WorkflowMonitor:
    """Main workflow monitoring system"""
    
    def __init__(self, monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED):
        self.monitoring_level = monitoring_level
        self.metrics_collector = MetricsCollector()
        self.event_logger = EventLogger()
        self.alert_manager = AlertManager()
        self.debugger = WorkflowDebugger()
        
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Setup default alert handler
        self.alert_manager.add_alert_handler(self._default_alert_handler)
        
        logger.info(f"Workflow monitor initialized with level: {monitoring_level}")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Workflow monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Workflow monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.get_system_metrics()
                
                # Check alert rules
                self.alert_manager.check_alert_rules(system_metrics)
                
                # Cleanup old data
                self.metrics_collector.cleanup_old_metrics()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    def _default_alert_handler(self, alert: SystemAlert):
        """Default alert handler - logs alerts"""
        level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        
        logger.log(
            level_map.get(alert.severity, logging.INFO),
            f"ALERT [{alert.severity}] {alert.component}: {alert.message}"
        )
    
    def record_component_execution(self, component: str, execution_time: float, success: bool = True, error: str = None):
        """Record component execution metrics"""
        self.metrics_collector.record_execution_time(component, execution_time)
        
        if success:
            self.metrics_collector.record_success(component)
        else:
            self.metrics_collector.record_error(component)
        
        # Log event
        event = WorkflowEvent(
            event_id=f"{component}_{int(time.time())}",
            event_type="component_execution",
            component=component,
            timestamp=datetime.now(),
            duration=execution_time,
            status="success" if success else "error",
            error=error
        )
        
        self.event_logger.log_event(event)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        return {
            "system_metrics": self.metrics_collector.get_system_metrics(),
            "event_summary": self.event_logger.get_event_summary(),
            "alert_summary": self.alert_manager.get_alert_summary(),
            "debug_info": self.debugger.get_debug_info(),
            "monitoring_status": {
                "active": self.monitoring_active,
                "level": self.monitoring_level,
                "uptime": (datetime.now() - self.metrics_collector.start_time).total_seconds()
            }
        }
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component"""
        metrics = self.metrics_collector.get_component_metrics(component)
        recent_events = self.event_logger.get_recent_events(count=10, event_type="component_execution")
        component_events = [e for e in recent_events if e.component == component]
        
        # Calculate health score
        success_rate = metrics.get("success_rate", 0)
        avg_latency = metrics.get("avg_execution_time", 0)
        error_count = metrics.get("error_count", 0)
        
        health_score = success_rate * 0.5 + (1 - min(avg_latency / 10, 1)) * 0.3 + (1 - min(error_count / 10, 1)) * 0.2
        
        return {
            "component": component,
            "health_score": health_score,
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "unhealthy",
            "metrics": metrics,
            "recent_events": len(component_events),
            "last_execution": component_events[0].timestamp.isoformat() if component_events else None
        }


# Context manager for monitoring component execution
class MonitoredExecution:
    """Context manager for monitoring component execution"""
    
    def __init__(self, monitor: WorkflowMonitor, component: str):
        self.monitor = monitor
        self.component = component
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        
        if self.monitor.debugger.trace_enabled:
            self.monitor.debugger.trace_execution(self.component, "start", {})
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        
        self.monitor.record_component_execution(self.component, execution_time, success, error)
        
        if self.monitor.debugger.trace_enabled:
            self.monitor.debugger.trace_execution(
                self.component, 
                "complete", 
                {"execution_time": execution_time, "success": success}
            )


# Factory function
def create_workflow_monitor(level: MonitoringLevel = MonitoringLevel.DETAILED) -> WorkflowMonitor:
    """Create and return a new workflow monitor instance"""
    return WorkflowMonitor(level)


if __name__ == "__main__":
    # Example usage
    async def main():
        monitor = create_workflow_monitor(MonitoringLevel.DEBUG)
        
        await monitor.start_monitoring()
        
        # Simulate some component executions
        with MonitoredExecution(monitor, "test_component"):
            await asyncio.sleep(0.1)  # Simulate work
        
        # Get dashboard data
        dashboard = monitor.get_monitoring_dashboard()
        print("Monitoring Dashboard:")
        print(json.dumps(dashboard, indent=2, default=str))
        
        await monitor.stop_monitoring()
    
    asyncio.run(main())