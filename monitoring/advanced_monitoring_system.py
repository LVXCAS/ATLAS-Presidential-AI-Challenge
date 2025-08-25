#!/usr/bin/env python3
"""
HIVE TRADE - Advanced Monitoring and Alerting System
Comprehensive monitoring with real-time alerts, anomaly detection, and performance tracking
"""

import asyncio
import logging
import json
import time
import smtplib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import statistics
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class MonitorType(Enum):
    """Types of monitoring checks"""
    SYSTEM_HEALTH = "SYSTEM_HEALTH"
    TRADING_PERFORMANCE = "TRADING_PERFORMANCE"
    RISK_METRICS = "RISK_METRICS"
    CONNECTIVITY = "CONNECTIVITY"
    DATA_QUALITY = "DATA_QUALITY"
    EXECUTION_LATENCY = "EXECUTION_LATENCY"
    POSITION_MONITORING = "POSITION_MONITORING"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    monitor_type: MonitorType
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    source: str
    
    # Alert metadata
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    escalated: bool = False
    escalation_count: int = 0
    
    # Context data
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'monitor_type': self.monitor_type.value,
            'title': self.title,
            'description': self.description,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'source': self.source,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None,
            'escalated': self.escalated,
            'escalation_count': self.escalation_count,
            'context': self.context
        }

@dataclass 
class MonitoringRule:
    """Monitoring rule configuration"""
    name: str
    monitor_type: MonitorType
    severity: AlertSeverity
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq', 'ne', 'between'
    enabled: bool = True
    
    # Advanced rule settings
    window_size: int = 5  # Number of data points to consider
    trigger_count: int = 1  # Number of threshold breaches to trigger alert
    cooldown_minutes: int = 15  # Minimum time between alerts
    
    # Notification settings
    email_enabled: bool = True
    slack_enabled: bool = False
    sms_enabled: bool = False
    
    # Auto-resolution
    auto_resolve: bool = True
    resolve_threshold: Optional[float] = None
    
    def evaluate(self, value: float, history: List[float]) -> bool:
        """Evaluate if rule should trigger alert"""
        if not self.enabled:
            return False
        
        # Check basic threshold
        triggered = False
        
        if self.comparison == 'gt':
            triggered = value > self.threshold
        elif self.comparison == 'lt':
            triggered = value < self.threshold
        elif self.comparison == 'eq':
            triggered = abs(value - self.threshold) < 1e-6
        elif self.comparison == 'ne':
            triggered = abs(value - self.threshold) >= 1e-6
        elif self.comparison == 'between':
            # Threshold should be a tuple (min, max)
            if isinstance(self.threshold, (list, tuple)) and len(self.threshold) == 2:
                triggered = not (self.threshold[0] <= value <= self.threshold[1])
        
        if not triggered:
            return False
        
        # Check trigger count requirement
        if len(history) < self.trigger_count:
            return False
        
        # Count recent threshold breaches
        recent_breaches = 0
        for hist_value in history[-self.trigger_count:]:
            if self.comparison == 'gt' and hist_value > self.threshold:
                recent_breaches += 1
            elif self.comparison == 'lt' and hist_value < self.threshold:
                recent_breaches += 1
            elif self.comparison == 'eq' and abs(hist_value - self.threshold) < 1e-6:
                recent_breaches += 1
            elif self.comparison == 'ne' and abs(hist_value - self.threshold) >= 1e-6:
                recent_breaches += 1
        
        return recent_breaches >= self.trigger_count

class AnomalyDetector:
    """Statistical anomaly detection for trading metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_data_point(self, metric_name: str, value: float):
        """Add data point for anomaly detection"""
        self.data_history[metric_name].append(value)
    
    def detect_anomaly(self, metric_name: str, value: float, 
                      method: str = 'zscore', threshold: float = 3.0) -> bool:
        """Detect if value is anomalous"""
        history = list(self.data_history[metric_name])
        
        if len(history) < 10:  # Need minimum data
            return False
        
        if method == 'zscore':
            mean = statistics.mean(history)
            stdev = statistics.stdev(history)
            
            if stdev == 0:
                return False
            
            zscore = abs((value - mean) / stdev)
            return zscore > threshold
        
        elif method == 'iqr':
            sorted_data = sorted(history)
            q1 = sorted_data[len(sorted_data) // 4]
            q3 = sorted_data[3 * len(sorted_data) // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            return value < lower_bound or value > upper_bound
        
        elif method == 'moving_average':
            if len(history) < 20:
                return False
            
            recent_avg = statistics.mean(history[-10:])
            long_avg = statistics.mean(history[-20:])
            
            deviation = abs(recent_avg - long_avg) / long_avg if long_avg != 0 else 0
            return deviation > threshold
        
        return False

class NotificationManager:
    """Manage alert notifications across multiple channels"""
    
    def __init__(self):
        self.email_config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'from_email': 'alerts@hivetrade.com',
            'to_emails': []
        }
        
        self.slack_config = {
            'webhook_url': '',
            'channel': '#trading-alerts'
        }
        
        self.notification_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.worker_thread.start()
    
    def _notification_worker(self):
        """Background worker for sending notifications"""
        while True:
            try:
                alert, channels = self.notification_queue.get(timeout=1)
                self._send_notifications(alert, channels)
                self.notification_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Notification worker error: {e}")
    
    def send_alert(self, alert: Alert, channels: List[str]):
        """Queue alert for notification"""
        self.notification_queue.put((alert, channels))
    
    def _send_notifications(self, alert: Alert, channels: List[str]):
        """Send notifications to specified channels"""
        for channel in channels:
            try:
                if channel == 'email':
                    self._send_email(alert)
                elif channel == 'slack':
                    self._send_slack(alert)
                elif channel == 'log':
                    self._log_alert(alert)
                elif channel == 'console':
                    self._console_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send {channel} notification for alert {alert.id}: {e}")
    
    def _send_email(self, alert: Alert):
        """Send email notification"""
        if not self.email_config['to_emails']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"HIVE TRADE ALERT: {alert.severity.value} - {alert.title}"
            
            body = f"""
HIVE TRADE Alert Notification

Alert ID: {alert.id}
Timestamp: {alert.timestamp}
Severity: {alert.severity.value}
Monitor Type: {alert.monitor_type.value}
Source: {alert.source}

Title: {alert.title}
Description: {alert.description}

Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold}

Context: {json.dumps(alert.context, indent=2)}

This is an automated message from the HIVE TRADE monitoring system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: Email sending would require proper SMTP configuration
            logger.info(f"Email notification prepared for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email for alert {alert.id}: {e}")
    
    def _send_slack(self, alert: Alert):
        """Send Slack notification"""
        # Slack integration would go here
        logger.info(f"Slack notification prepared for alert {alert.id}")
    
    def _log_alert(self, alert: Alert):
        """Log alert notification"""
        log_level = getattr(logging, alert.severity.value, logging.INFO)
        logger.log(log_level, f"ALERT: {alert.title} - {alert.description}")
    
    def _console_alert(self, alert: Alert):
        """Console alert notification"""
        severity_colors = {
            AlertSeverity.INFO: '\033[94m',      # Blue
            AlertSeverity.WARNING: '\033[93m',   # Yellow
            AlertSeverity.ERROR: '\033[91m',     # Red
            AlertSeverity.CRITICAL: '\033[95m',  # Magenta
            AlertSeverity.EMERGENCY: '\033[97m\033[41m'  # White on Red
        }
        
        color = severity_colors.get(alert.severity, '\033[0m')
        reset = '\033[0m'
        
        print(f"\n{color}[{alert.severity.value}] HIVE TRADE ALERT{reset}")
        print(f"{color}Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}{reset}")
        print(f"{color}Title: {alert.title}{reset}")
        print(f"{color}Description: {alert.description}{reset}")
        print(f"{color}Source: {alert.source}{reset}")
        print(f"{color}Metric: {alert.metric_name} = {alert.current_value} (threshold: {alert.threshold}){reset}\n")

class AdvancedMonitoringSystem:
    """Comprehensive monitoring and alerting system"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Monitoring data
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Components
        self.anomaly_detector = AnomalyDetector()
        self.notification_manager = NotificationManager()
        
        # Monitoring state
        self.is_running = False
        self.monitoring_threads = []
        
        # Performance tracking
        self.system_metrics = {
            'alerts_generated': 0,
            'alerts_resolved': 0,
            'monitoring_cycles': 0,
            'avg_processing_time': 0.0,
            'last_update': datetime.now()
        }
        
        self.init_database()
        self.setup_default_rules()
        
        logger.info("Advanced Monitoring System initialized")

    def init_database(self):
        """Initialize monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    severity TEXT,
                    monitor_type TEXT,
                    title TEXT,
                    description TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    threshold REAL,
                    source TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0,
                    resolution_time DATETIME,
                    context TEXT
                )
            ''')
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    metric_name TEXT,
                    value REAL,
                    source TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def setup_default_rules(self):
        """Setup default monitoring rules"""
        default_rules = [
            # System Health Rules
            MonitoringRule(
                name="high_cpu_usage",
                monitor_type=MonitorType.SYSTEM_HEALTH,
                severity=AlertSeverity.WARNING,
                threshold=80.0,
                comparison='gt',
                window_size=3,
                trigger_count=2,
                cooldown_minutes=10
            ),
            
            MonitoringRule(
                name="high_memory_usage",
                monitor_type=MonitorType.SYSTEM_HEALTH,
                severity=AlertSeverity.WARNING,
                threshold=85.0,
                comparison='gt',
                window_size=3,
                trigger_count=2,
                cooldown_minutes=10
            ),
            
            # Trading Performance Rules
            MonitoringRule(
                name="high_daily_loss",
                monitor_type=MonitorType.TRADING_PERFORMANCE,
                severity=AlertSeverity.ERROR,
                threshold=-5000.0,
                comparison='lt',
                window_size=1,
                trigger_count=1,
                cooldown_minutes=5
            ),
            
            MonitoringRule(
                name="execution_latency_high",
                monitor_type=MonitorType.EXECUTION_LATENCY,
                severity=AlertSeverity.WARNING,
                threshold=1000.0,  # milliseconds
                comparison='gt',
                window_size=5,
                trigger_count=3,
                cooldown_minutes=15
            ),
            
            # Risk Metrics Rules
            MonitoringRule(
                name="var_exceeded",
                monitor_type=MonitorType.RISK_METRICS,
                severity=AlertSeverity.CRITICAL,
                threshold=10000.0,
                comparison='gt',
                window_size=1,
                trigger_count=1,
                cooldown_minutes=1
            ),
            
            MonitoringRule(
                name="max_drawdown_exceeded",
                monitor_type=MonitorType.RISK_METRICS,
                severity=AlertSeverity.ERROR,
                threshold=-15.0,  # percentage
                comparison='lt',
                window_size=1,
                trigger_count=1,
                cooldown_minutes=5
            ),
            
            # Connectivity Rules
            MonitoringRule(
                name="api_connection_lost",
                monitor_type=MonitorType.CONNECTIVITY,
                severity=AlertSeverity.CRITICAL,
                threshold=0.0,  # 0 = disconnected, 1 = connected
                comparison='eq',
                window_size=1,
                trigger_count=1,
                cooldown_minutes=1
            ),
            
            # Data Quality Rules
            MonitoringRule(
                name="stale_market_data",
                monitor_type=MonitorType.DATA_QUALITY,
                severity=AlertSeverity.WARNING,
                threshold=300.0,  # seconds
                comparison='gt',
                window_size=1,
                trigger_count=1,
                cooldown_minutes=5
            )
        ]
        
        for rule in default_rules:
            self.add_monitoring_rule(rule)

    def add_monitoring_rule(self, rule: MonitoringRule):
        """Add or update monitoring rule"""
        self.monitoring_rules[rule.name] = rule
        logger.info(f"Added monitoring rule: {rule.name}")

    def remove_monitoring_rule(self, rule_name: str):
        """Remove monitoring rule"""
        if rule_name in self.monitoring_rules:
            del self.monitoring_rules[rule_name]
            logger.info(f"Removed monitoring rule: {rule_name}")

    def record_metric(self, metric_name: str, value: float, source: str = "unknown"):
        """Record a metric value"""
        timestamp = datetime.now()
        
        # Store in memory
        self.metric_history[metric_name].append(value)
        
        # Store in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO metrics (timestamp, metric_name, value, source)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, metric_name, value, source))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store metric {metric_name}: {e}")
        
        # Add to anomaly detector
        self.anomaly_detector.add_data_point(metric_name, value)
        
        # Check monitoring rules
        self._check_rules(metric_name, value, source)

    def _check_rules(self, metric_name: str, value: float, source: str):
        """Check all monitoring rules for a metric"""
        current_time = datetime.now()
        
        for rule_name, rule in self.monitoring_rules.items():
            try:
                # Skip if rule doesn't apply to this metric
                if rule_name not in metric_name and metric_name not in rule_name:
                    continue
                
                # Check cooldown
                last_alert_key = f"{rule_name}_{metric_name}"
                if last_alert_key in self.last_alert_time:
                    time_since_last = current_time - self.last_alert_time[last_alert_key]
                    if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                        continue
                
                # Get metric history
                history = list(self.metric_history[metric_name])
                
                # Evaluate rule
                if rule.evaluate(value, history):
                    self._generate_alert(rule, metric_name, value, source, current_time)
                    self.last_alert_time[last_alert_key] = current_time
                
                # Check for auto-resolution
                elif rule.auto_resolve and rule.resolve_threshold is not None:
                    self._check_auto_resolution(rule, metric_name, value)
                    
            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {e}")

    def _generate_alert(self, rule: MonitoringRule, metric_name: str, 
                       value: float, source: str, timestamp: datetime):
        """Generate new alert"""
        alert_id = f"{rule.name}_{metric_name}_{int(timestamp.timestamp())}"
        
        # Check for anomaly
        is_anomaly = self.anomaly_detector.detect_anomaly(metric_name, value)
        
        alert = Alert(
            id=alert_id,
            timestamp=timestamp,
            severity=rule.severity,
            monitor_type=rule.monitor_type,
            title=f"{rule.name.replace('_', ' ').title()}: {metric_name}",
            description=f"Metric {metric_name} value {value} exceeded threshold {rule.threshold}",
            metric_name=metric_name,
            current_value=value,
            threshold=rule.threshold,
            source=source,
            context={
                'rule_name': rule.name,
                'comparison': rule.comparison,
                'anomaly_detected': is_anomaly,
                'metric_history': list(self.metric_history[metric_name])[-10:]  # Last 10 values
            }
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Save to database
        self._save_alert_to_db(alert)
        
        # Send notifications
        notification_channels = []
        if rule.email_enabled:
            notification_channels.append('email')
        if rule.slack_enabled:
            notification_channels.append('slack')
        
        notification_channels.extend(['log', 'console'])  # Always log and console
        
        self.notification_manager.send_alert(alert, notification_channels)
        
        # Update metrics
        self.system_metrics['alerts_generated'] += 1
        self.system_metrics['last_update'] = timestamp
        
        logger.warning(f"Generated alert: {alert.id} - {alert.title}")

    def _check_auto_resolution(self, rule: MonitoringRule, metric_name: str, value: float):
        """Check if any active alerts should be auto-resolved"""
        for alert_id, alert in list(self.active_alerts.items()):
            if (alert.metric_name == metric_name and not alert.resolved and
                rule.resolve_threshold is not None):
                
                should_resolve = False
                
                if rule.comparison == 'gt' and value <= rule.resolve_threshold:
                    should_resolve = True
                elif rule.comparison == 'lt' and value >= rule.resolve_threshold:
                    should_resolve = True
                
                if should_resolve:
                    self.resolve_alert(alert_id, "Auto-resolved: metric returned to normal")

    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            # Update in database
            self._update_alert_in_db(alert)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Update metrics
            self.system_metrics['alerts_resolved'] += 1
            
            logger.info(f"Resolved alert: {alert_id} - {resolution_note}")

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            
            # Update in database
            self._update_alert_in_db(alert)
            
            logger.info(f"Acknowledged alert: {alert_id}")

    def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (id, timestamp, severity, monitor_type, title, description,
                                  metric_name, current_value, threshold, source, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.timestamp, alert.severity.value, alert.monitor_type.value,
                alert.title, alert.description, alert.metric_name, alert.current_value,
                alert.threshold, alert.source, json.dumps(alert.context)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save alert to database: {e}")

    def _update_alert_in_db(self, alert: Alert):
        """Update alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE alerts SET acknowledged = ?, resolved = ?, resolution_time = ?
                WHERE id = ?
            ''', (alert.acknowledged, alert.resolved, alert.resolution_time, alert.id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update alert in database: {e}")

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        current_time = datetime.now()
        
        # Alert statistics
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        acknowledged_alerts = len([a for a in self.active_alerts.values() if a.acknowledged])
        
        # Severity breakdown
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        # Recent metrics
        recent_metrics = {}
        for metric_name, history in self.metric_history.items():
            if history:
                recent_metrics[metric_name] = {
                    'current_value': history[-1],
                    'avg_value': statistics.mean(history),
                    'min_value': min(history),
                    'max_value': max(history),
                    'data_points': len(history)
                }
        
        # System performance
        uptime_seconds = (current_time - self.system_metrics['last_update']).total_seconds()
        
        return {
            'timestamp': current_time.isoformat(),
            'monitoring_system': 'Advanced Monitoring System v1.0',
            'system_status': 'OPERATIONAL' if self.is_running else 'STOPPED',
            'uptime_seconds': uptime_seconds,
            
            'alert_statistics': {
                'total_alerts_generated': self.system_metrics['alerts_generated'],
                'total_alerts_resolved': self.system_metrics['alerts_resolved'],
                'active_alerts': active_alerts,
                'acknowledged_alerts': acknowledged_alerts,
                'unacknowledged_alerts': active_alerts - acknowledged_alerts,
                'severity_breakdown': dict(severity_counts)
            },
            
            'active_alerts': [alert.to_dict() for alert in self.active_alerts.values()],
            
            'monitoring_rules': {
                'total_rules': len(self.monitoring_rules),
                'enabled_rules': len([r for r in self.monitoring_rules.values() if r.enabled]),
                'rules': [
                    {
                        'name': rule.name,
                        'monitor_type': rule.monitor_type.value,
                        'severity': rule.severity.value,
                        'threshold': rule.threshold,
                        'enabled': rule.enabled,
                        'cooldown_minutes': rule.cooldown_minutes
                    }
                    for rule in self.monitoring_rules.values()
                ]
            },
            
            'recent_metrics': recent_metrics,
            
            'system_metrics': {
                'monitoring_cycles': self.system_metrics['monitoring_cycles'],
                'avg_processing_time_ms': self.system_metrics['avg_processing_time'] * 1000,
                'alerts_generated': self.system_metrics['alerts_generated'],
                'alerts_resolved': self.system_metrics['alerts_resolved'],
                'last_update': self.system_metrics['last_update'].isoformat()
            }
        }

    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_running:
            logger.warning("Monitoring system already running")
            return
        
        self.is_running = True
        
        # Start background monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        self.monitoring_threads.append(monitor_thread)
        
        logger.info("Advanced Monitoring System started")

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        logger.info("Advanced Monitoring System stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Simulate collecting system metrics
                self._collect_system_metrics()
                
                # Update processing time
                processing_time = time.time() - start_time
                self.system_metrics['avg_processing_time'] = (
                    (self.system_metrics['avg_processing_time'] * self.system_metrics['monitoring_cycles'] + 
                     processing_time) / (self.system_metrics['monitoring_cycles'] + 1)
                )
                self.system_metrics['monitoring_cycles'] += 1
                
                # Sleep before next cycle
                time.sleep(10)  # Run every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error

    def _collect_system_metrics(self):
        """Collect system metrics for monitoring"""
        import psutil
        
        try:
            # CPU Usage
            cpu_usage = psutil.cpu_percent(interval=1)
            self.record_metric('cpu_usage_percent', cpu_usage, 'system')
            
            # Memory Usage
            memory = psutil.virtual_memory()
            self.record_metric('memory_usage_percent', memory.percent, 'system')
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            self.record_metric('disk_usage_percent', disk_usage_percent, 'system')
            
        except ImportError:
            # psutil not available, use mock data
            import random
            self.record_metric('cpu_usage_percent', random.uniform(10, 90), 'system')
            self.record_metric('memory_usage_percent', random.uniform(20, 80), 'system')
            self.record_metric('disk_usage_percent', random.uniform(30, 70), 'system')

def main():
    """Demonstrate advanced monitoring system"""
    print("HIVE TRADE - Advanced Monitoring System Demo")
    print("=" * 60)
    
    # Initialize monitoring system
    monitor = AdvancedMonitoringSystem()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some metrics that will trigger alerts
    print("\nSimulating system metrics...")
    
    # Normal metrics
    for i in range(5):
        monitor.record_metric('cpu_usage_percent', 45.0 + i, 'demo')
        monitor.record_metric('memory_usage_percent', 60.0 + i * 2, 'demo')
        time.sleep(0.1)
    
    # High CPU usage (should trigger alert)
    print("\nSimulating high CPU usage...")
    for i in range(3):
        monitor.record_metric('cpu_usage_percent', 85.0 + i, 'demo')
        time.sleep(0.1)
    
    # High daily loss (should trigger alert)
    print("\nSimulating high daily loss...")
    monitor.record_metric('high_daily_loss', -6000.0, 'trading_system')
    
    # VaR exceeded (should trigger critical alert)
    print("\nSimulating VaR exceeded...")
    monitor.record_metric('var_exceeded', 12000.0, 'risk_system')
    
    # Wait for alerts to be processed
    time.sleep(2)
    
    # Get dashboard
    print("\nGenerating monitoring dashboard...")
    dashboard = monitor.get_monitoring_dashboard()
    
    # Display results
    print(f"\nMONITORING DASHBOARD:")
    print(f"  System Status: {dashboard['system_status']}")
    print(f"  Active Alerts: {dashboard['alert_statistics']['active_alerts']}")
    print(f"  Total Alerts Generated: {dashboard['alert_statistics']['total_alerts_generated']}")
    print(f"  Monitoring Rules: {dashboard['monitoring_rules']['enabled_rules']}/{dashboard['monitoring_rules']['total_rules']} enabled")
    
    if dashboard['active_alerts']:
        print(f"\nACTIVE ALERTS:")
        for alert in dashboard['active_alerts']:
            print(f"  {alert['severity']} - {alert['title']}")
            print(f"    {alert['description']}")
            print(f"    Current: {alert['current_value']}, Threshold: {alert['threshold']}")
    
    print(f"\nSEVERITY BREAKDOWN:")
    for severity, count in dashboard['alert_statistics']['severity_breakdown'].items():
        print(f"  {severity}: {count}")
    
    print(f"\nSYSTEM METRICS:")
    sys_metrics = dashboard['system_metrics']
    print(f"  Monitoring Cycles: {sys_metrics['monitoring_cycles']}")
    print(f"  Avg Processing Time: {sys_metrics['avg_processing_time_ms']:.2f}ms")
    print(f"  Alerts Generated: {sys_metrics['alerts_generated']}")
    
    # Save dashboard
    results_file = 'monitoring_dashboard.json'
    with open(results_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"\nMonitoring dashboard saved to {results_file}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("\nAdvanced Monitoring System demonstration completed!")

if __name__ == "__main__":
    main()