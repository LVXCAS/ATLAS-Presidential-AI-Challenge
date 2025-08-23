"""
Comprehensive alerting system for monitoring critical system events and metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from core.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Expression to evaluate
    threshold_value: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    evaluation_window: int  # seconds
    alert_channels: List[AlertChannel]
    cooldown_period: int = 300  # seconds
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """An active alert instance."""
    id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    current_value: float
    threshold_value: float
    first_triggered: datetime
    last_updated: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    email: Dict[str, str] = field(default_factory=dict)
    slack: Dict[str, str] = field(default_factory=dict)
    webhook: Dict[str, str] = field(default_factory=dict)
    sms: Dict[str, str] = field(default_factory=dict)


class MetricEvaluator:
    """Evaluates metrics against alert rules."""
    
    def __init__(self):
        self.metric_cache: Dict[str, List[Tuple[datetime, float]]] = {}
        
    def add_metric_value(self, metric_name: str, value: float, timestamp: datetime = None):
        """Add a metric value for evaluation."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if metric_name not in self.metric_cache:
            self.metric_cache[metric_name] = []
        
        self.metric_cache[metric_name].append((timestamp, value))
        
        # Keep only last 1 hour of data
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.metric_cache[metric_name] = [
            (ts, val) for ts, val in self.metric_cache[metric_name]
            if ts >= cutoff_time
        ]
    
    def evaluate_rule(self, rule: AlertRule) -> Optional[float]:
        """Evaluate an alert rule against cached metrics."""
        # Extract metric name from condition (simplified)
        # In production, this would use a proper expression parser
        metric_name = self._extract_metric_name(rule.condition)
        
        if metric_name not in self.metric_cache:
            return None
        
        # Get values within evaluation window
        window_start = datetime.now() - timedelta(seconds=rule.evaluation_window)
        window_values = [
            value for timestamp, value in self.metric_cache[metric_name]
            if timestamp >= window_start
        ]
        
        if not window_values:
            return None
        
        # Calculate aggregated value based on condition
        if "avg(" in rule.condition:
            return sum(window_values) / len(window_values)
        elif "max(" in rule.condition:
            return max(window_values)
        elif "min(" in rule.condition:
            return min(window_values)
        elif "count(" in rule.condition:
            return len(window_values)
        else:
            # Default to latest value
            return window_values[-1] if window_values else None
    
    def _extract_metric_name(self, condition: str) -> str:
        """Extract metric name from condition string."""
        # Simplified extraction - would need proper parsing in production
        for term in condition.split():
            if "(" in term and ")" in term:
                start = term.find("(") + 1
                end = term.find(")")
                return term[start:end]
        
        return condition.strip()
    
    def check_threshold(self, current_value: float, threshold: float, comparison: str) -> bool:
        """Check if current value meets threshold condition."""
        if comparison == "gt":
            return current_value > threshold
        elif comparison == "lt":
            return current_value < threshold
        elif comparison == "gte":
            return current_value >= threshold
        elif comparison == "lte":
            return current_value <= threshold
        elif comparison == "eq":
            return abs(current_value - threshold) < 0.001  # Float comparison
        else:
            return False


class NotificationManager:
    """Manages sending notifications through various channels."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
        logger.info("NotificationManager initialized")
    
    async def send_alert(self, alert: Alert, channels: List[AlertChannel]):
        """Send alert through specified channels."""
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert)
                elif channel == AlertChannel.SMS:
                    await self._send_sms_alert(alert)
                elif channel == AlertChannel.DASHBOARD:
                    await self._send_dashboard_alert(alert)
                
                logger.info(f"Alert {alert.id} sent via {channel.value}")
                
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        if not self.config.email:
            logger.warning("Email configuration not available")
            return
        
        smtp_server = self.config.email.get('smtp_server', 'localhost')
        smtp_port = int(self.config.email.get('smtp_port', 587))
        username = self.config.email.get('username')
        password = self.config.email.get('password')
        from_email = self.config.email.get('from_email')
        to_emails = self.config.email.get('to_emails', '').split(',')
        
        if not all([from_email, to_emails, username, password]):
            logger.warning("Incomplete email configuration")
            return
        
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
        message["From"] = from_email
        message["To"] = ", ".join(to_emails)
        
        # Create email body
        html_body = self._create_email_html(alert)
        text_body = self._create_email_text(alert)
        
        message.attach(MIMEText(text_body, "plain"))
        message.attach(MIMEText(html_body, "html"))
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(username, password)
            server.send_message(message, from_email, to_emails)
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert."""
        if not self.config.slack:
            logger.warning("Slack configuration not available")
            return
        
        webhook_url = self.config.slack.get('webhook_url')
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        # Create Slack message
        color = self._get_slack_color(alert.severity)
        
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": str(alert.current_value),
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": str(alert.threshold_value),
                            "short": True
                        },
                        {
                            "title": "Triggered",
                            "value": alert.first_triggered.strftime("%Y-%m-%d %H:%M:%S"),
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        # Send to Slack (would use aiohttp in production)
        import json
        import urllib.request
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    logger.error(f"Slack webhook returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert."""
        if not self.config.webhook:
            logger.warning("Webhook configuration not available")
            return
        
        webhook_url = self.config.webhook.get('url')
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return
        
        # Create webhook payload
        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "first_triggered": alert.first_triggered.isoformat(),
            "labels": alert.labels,
            "annotations": alert.annotations
        }
        
        # Send webhook (simplified - would use aiohttp in production)
        logger.info(f"Would send webhook to {webhook_url}: {payload}")
    
    async def _send_sms_alert(self, alert: Alert):
        """Send SMS alert."""
        if not self.config.sms:
            logger.warning("SMS configuration not available")
            return
        
        # SMS implementation would integrate with services like Twilio
        logger.info(f"Would send SMS alert: {alert.message}")
    
    async def _send_dashboard_alert(self, alert: Alert):
        """Send alert to dashboard via Redis."""
        redis = get_redis_manager().client
        
        alert_data = {
            'id': alert.id,
            'rule_name': alert.rule_name,
            'severity': alert.severity.value,
            'status': alert.status.value,
            'message': alert.message,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'first_triggered': alert.first_triggered.isoformat(),
            'labels': alert.labels
        }
        
        # Publish to dashboard channel
        await redis.publish("alerts:dashboard", json.dumps(alert_data))
        
        # Store in active alerts set
        await redis.setex(
            f"alert:active:{alert.id}",
            3600,  # 1 hour TTL
            json.dumps(alert_data)
        )
    
    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack color for severity level."""
        colors = {
            AlertSeverity.INFO: "#36a64f",      # Green
            AlertSeverity.WARNING: "#ff9900",   # Orange
            AlertSeverity.CRITICAL: "#ff0000",  # Red
            AlertSeverity.EMERGENCY: "#8b0000"  # Dark Red
        }
        return colors.get(severity, "#cccccc")
    
    def _create_email_html(self, alert: Alert) -> str:
        """Create HTML email body."""
        return f"""
        <html>
        <body>
            <h2 style="color: {'red' if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else 'orange'};">
                Alert: {alert.rule_name}
            </h2>
            
            <table>
                <tr><td><b>Severity:</b></td><td>{alert.severity.value.upper()}</td></tr>
                <tr><td><b>Status:</b></td><td>{alert.status.value.upper()}</td></tr>
                <tr><td><b>Message:</b></td><td>{alert.message}</td></tr>
                <tr><td><b>Current Value:</b></td><td>{alert.current_value}</td></tr>
                <tr><td><b>Threshold:</b></td><td>{alert.threshold_value}</td></tr>
                <tr><td><b>First Triggered:</b></td><td>{alert.first_triggered.strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
            </table>
            
            <p>This alert was generated by the Bloomberg Terminal monitoring system.</p>
        </body>
        </html>
        """
    
    def _create_email_text(self, alert: Alert) -> str:
        """Create plain text email body."""
        return f"""
ALERT: {alert.rule_name}

Severity: {alert.severity.value.upper()}
Status: {alert.status.value.upper()}
Message: {alert.message}
Current Value: {alert.current_value}
Threshold: {alert.threshold_value}
First Triggered: {alert.first_triggered.strftime("%Y-%m-%d %H:%M:%S")}

This alert was generated by the Bloomberg Terminal monitoring system.
        """


class AlertManager:
    """Main alerting system coordinator."""
    
    def __init__(self, notification_config: NotificationConfig = None):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        self.evaluator = MetricEvaluator()
        self.notification_manager = NotificationManager(notification_config or NotificationConfig())
        
        self.redis = None
        self.is_running = False
        
        # Default alert rules
        self._create_default_rules()
        
        logger.info("AlertManager initialized")
    
    async def initialize(self):
        """Initialize alert manager."""
        self.redis = get_redis_manager().client
        
        # Load alert rules from Redis
        await self._load_alert_rules()
        
        # Load active alerts from Redis
        await self._load_active_alerts()
        
        self.is_running = True
        
        # Start evaluation loop
        asyncio.create_task(self._evaluation_loop())
        
        logger.info("AlertManager started")
    
    async def stop(self):
        """Stop alert manager."""
        self.is_running = False
        
        # Save state to Redis
        await self._save_alert_rules()
        await self._save_active_alerts()
        
        logger.info("AlertManager stopped")
    
    def _create_default_rules(self):
        """Create default alert rules."""
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage above 80%",
                severity=AlertSeverity.WARNING,
                condition="avg(system_cpu_usage_percent)",
                threshold_value=80.0,
                comparison="gt",
                evaluation_window=300,  # 5 minutes
                alert_channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL]
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage", 
                description="Memory usage above 85%",
                severity=AlertSeverity.WARNING,
                condition="avg(system_memory_usage_percent)",
                threshold_value=85.0,
                comparison="gt",
                evaluation_window=300,
                alert_channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL]
            ),
            AlertRule(
                id="trading_loss_threshold",
                name="Trading Loss Threshold",
                description="Total P&L below -$10,000",
                severity=AlertSeverity.CRITICAL,
                condition="latest(trading_total_pnl)",
                threshold_value=-10000.0,
                comparison="lt",
                evaluation_window=60,
                alert_channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL, AlertChannel.SLACK]
            ),
            AlertRule(
                id="high_risk_var",
                name="High Value at Risk",
                description="VaR exceeds $50,000",
                severity=AlertSeverity.CRITICAL,
                condition="latest(risk_current_var)",
                threshold_value=50000.0,
                comparison="gt",
                evaluation_window=60,
                alert_channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL, AlertChannel.SLACK]
            ),
            AlertRule(
                id="max_drawdown_exceeded",
                name="Maximum Drawdown Exceeded",
                description="Drawdown exceeds 15%",
                severity=AlertSeverity.EMERGENCY,
                condition="max(risk_max_drawdown)",
                threshold_value=0.15,
                comparison="gt",
                evaluation_window=300,
                alert_channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS]
            ),
            AlertRule(
                id="low_win_rate",
                name="Low Trading Win Rate",
                description="Win rate below 40%",
                severity=AlertSeverity.WARNING,
                condition="avg(trading_win_rate)",
                threshold_value=0.40,
                comparison="lt",
                evaluation_window=1800,  # 30 minutes
                alert_channels=[AlertChannel.DASHBOARD]
            ),
            AlertRule(
                id="api_error_rate_high",
                name="High API Error Rate",
                description="API error rate above 5%",
                severity=AlertSeverity.WARNING,
                condition="avg(system_error_rate)",
                threshold_value=0.05,
                comparison="gt",
                evaluation_window=300,
                alert_channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL]
            ),
            AlertRule(
                id="websocket_disconnections",
                name="WebSocket Disconnections",
                description="WebSocket connections dropping",
                severity=AlertSeverity.WARNING,
                condition="min(system_active_connections)",
                threshold_value=1.0,
                comparison="lt",
                evaluation_window=60,
                alert_channels=[AlertChannel.DASHBOARD]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    async def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules[rule.id] = rule
        await self._save_alert_rules()
        
        logger.info(f"Added alert rule: {rule.name}")
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            await self._save_alert_rules()
            
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        
        return False
    
    async def update_alert_rule(self, rule: AlertRule):
        """Update an existing alert rule."""
        self.alert_rules[rule.id] = rule
        await self._save_alert_rules()
        
        logger.info(f"Updated alert rule: {rule.name}")
    
    async def record_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Record a metric value for alert evaluation."""
        self.evaluator.add_metric_value(metric_name, value, timestamp)
    
    async def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self.is_running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(30)
    
    async def _evaluate_all_rules(self):
        """Evaluate all alert rules."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        current_value = self.evaluator.evaluate_rule(rule)
        
        if current_value is None:
            return  # No data available
        
        # Check if threshold is breached
        threshold_breached = self.evaluator.check_threshold(
            current_value, rule.threshold_value, rule.comparison
        )
        
        existing_alert = self._get_active_alert_for_rule(rule.id)
        
        if threshold_breached:
            if not existing_alert:
                # Create new alert
                await self._create_alert(rule, current_value)
            else:
                # Update existing alert
                existing_alert.current_value = current_value
                existing_alert.last_updated = datetime.now()
        else:
            if existing_alert and existing_alert.status == AlertStatus.ACTIVE:
                # Resolve alert
                await self._resolve_alert(existing_alert)
    
    async def _create_alert(self, rule: AlertRule, current_value: float):
        """Create a new alert."""
        alert_id = f"{rule.id}_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"{rule.description}. Current value: {current_value}, Threshold: {rule.threshold_value}",
            current_value=current_value,
            threshold_value=rule.threshold_value,
            first_triggered=datetime.now(),
            last_updated=datetime.now(),
            labels={"rule_id": rule.id},
            annotations={"description": rule.description}
        )
        
        self.active_alerts[alert_id] = alert
        
        # Send notifications
        await self.notification_manager.send_alert(alert, rule.alert_channels)
        
        logger.info(f"Created alert: {alert.rule_name} [{alert.id}]")
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an active alert."""
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.last_updated = datetime.now()
        
        # Move to history
        self.alert_history.append(alert)
        
        if alert.id in self.active_alerts:
            del self.active_alerts[alert.id]
        
        # Send resolution notification
        alert.message = f"RESOLVED: {alert.message}"
        await self.notification_manager.send_alert(alert, [AlertChannel.DASHBOARD])
        
        logger.info(f"Resolved alert: {alert.rule_name} [{alert.id}]")
    
    def _get_active_alert_for_rule(self, rule_id: str) -> Optional[Alert]:
        """Get active alert for a rule."""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                return alert
        return None
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            alert.last_updated = datetime.now()
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        
        return False
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    async def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]
    
    async def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return list(self.alert_rules.values())
    
    async def _load_alert_rules(self):
        """Load alert rules from Redis."""
        if not self.redis:
            return
        
        try:
            rules_data = await self.redis.get("alerts:rules")
            if rules_data:
                rules_dict = json.loads(rules_data)
                for rule_data in rules_dict:
                    rule = AlertRule(**rule_data)
                    self.alert_rules[rule.id] = rule
                
                logger.info(f"Loaded {len(rules_dict)} alert rules from Redis")
        except Exception as e:
            logger.error(f"Error loading alert rules: {e}")
    
    async def _save_alert_rules(self):
        """Save alert rules to Redis."""
        if not self.redis:
            return
        
        try:
            rules_data = [
                {
                    'id': rule.id,
                    'name': rule.name,
                    'description': rule.description,
                    'severity': rule.severity.value,
                    'condition': rule.condition,
                    'threshold_value': rule.threshold_value,
                    'comparison': rule.comparison,
                    'evaluation_window': rule.evaluation_window,
                    'alert_channels': [ch.value for ch in rule.alert_channels],
                    'cooldown_period': rule.cooldown_period,
                    'enabled': rule.enabled,
                    'metadata': rule.metadata
                }
                for rule in self.alert_rules.values()
            ]
            
            await self.redis.setex(
                "alerts:rules",
                86400,  # 24 hours TTL
                json.dumps(rules_data)
            )
        except Exception as e:
            logger.error(f"Error saving alert rules: {e}")
    
    async def _load_active_alerts(self):
        """Load active alerts from Redis."""
        if not self.redis:
            return
        
        try:
            alert_keys = await self.redis.keys("alert:active:*")
            for key in alert_keys:
                alert_data = await self.redis.get(key)
                if alert_data:
                    alert_dict = json.loads(alert_data)
                    # Convert back to Alert object (simplified)
                    alert_id = alert_dict['id']
                    self.active_alerts[alert_id] = alert_dict  # Store as dict for now
            
            logger.info(f"Loaded {len(self.active_alerts)} active alerts from Redis")
        except Exception as e:
            logger.error(f"Error loading active alerts: {e}")
    
    async def _save_active_alerts(self):
        """Save active alerts to Redis."""
        if not self.redis:
            return
        
        try:
            for alert_id, alert in self.active_alerts.items():
                alert_data = {
                    'id': alert.id,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity.value,
                    'status': alert.status.value,
                    'message': alert.message,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'first_triggered': alert.first_triggered.isoformat(),
                    'last_updated': alert.last_updated.isoformat()
                }
                
                await self.redis.setex(
                    f"alert:active:{alert_id}",
                    3600,  # 1 hour TTL
                    json.dumps(alert_data)
                )
        except Exception as e:
            logger.error(f"Error saving active alerts: {e}")