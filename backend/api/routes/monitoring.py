"""
Monitoring API routes for observability and system health.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import uuid

from monitoring import (
    MetricsAggregator, DistributedTracing, AlertManager, HealthChecker,
    AlertRule, AlertSeverity, AlertChannel, NotificationConfig,
    HealthCheck, HealthStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global monitoring system components
metrics_aggregator: Optional[MetricsAggregator] = None
distributed_tracing: Optional[DistributedTracing] = None
alert_manager: Optional[AlertManager] = None
health_checker: Optional[HealthChecker] = None


class AlertRuleRequest(BaseModel):
    """Alert rule creation/update request."""
    name: str
    description: str
    severity: str  # info, warning, critical, emergency
    condition: str
    threshold_value: float
    comparison: str  # gt, lt, eq, gte, lte
    evaluation_window: int = Field(default=300, ge=60, le=3600)
    alert_channels: List[str] = Field(default=["dashboard"])
    cooldown_period: int = Field(default=300, ge=60, le=3600)
    enabled: bool = True


class NotificationConfigRequest(BaseModel):
    """Notification configuration request."""
    email: Dict[str, str] = Field(default_factory=dict)
    slack: Dict[str, str] = Field(default_factory=dict)
    webhook: Dict[str, str] = Field(default_factory=dict)
    sms: Dict[str, str] = Field(default_factory=dict)


async def get_metrics_aggregator() -> MetricsAggregator:
    """Get or create metrics aggregator instance."""
    global metrics_aggregator
    if metrics_aggregator is None:
        metrics_aggregator = MetricsAggregator()
        await metrics_aggregator.initialize()
    return metrics_aggregator


async def get_distributed_tracing() -> DistributedTracing:
    """Get or create distributed tracing instance."""
    global distributed_tracing
    if distributed_tracing is None:
        distributed_tracing = DistributedTracing()
        await distributed_tracing.initialize()
    return distributed_tracing


async def get_alert_manager() -> AlertManager:
    """Get or create alert manager instance."""
    global alert_manager
    if alert_manager is None:
        alert_manager = AlertManager()
        await alert_manager.initialize()
    return alert_manager


async def get_health_checker() -> HealthChecker:
    """Get or create health checker instance."""
    global health_checker
    if health_checker is None:
        health_checker = HealthChecker()
        await health_checker.initialize()
    return health_checker


# Metrics endpoints

@router.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics in text format."""
    try:
        aggregator = await get_metrics_aggregator()
        
        # Get Prometheus metrics if available
        prometheus_metrics = aggregator.prometheus_collector.get_metrics()
        
        return {
            'prometheus_metrics': prometheus_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/business")
async def get_business_metrics() -> Dict[str, Any]:
    """Get comprehensive business metrics."""
    try:
        aggregator = await get_metrics_aggregator()
        metrics = await aggregator.get_comprehensive_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting business metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/timeseries/{metric_name}")
async def get_time_series_data(
    metric_name: str,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """Get time series data for a specific metric."""
    try:
        aggregator = await get_metrics_aggregator()
        
        # Parse time parameters
        start_dt = None
        end_dt = None
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        time_series_data = await aggregator.get_time_series_data(
            metric_name, start_dt, end_dt
        )
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error getting time series data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/record/trade")
async def record_trade_metric(
    symbol: str,
    pnl: float,
    volume: float,
    commission: float,
    agent_name: str = "unknown"
) -> Dict[str, Any]:
    """Record a trade execution metric."""
    try:
        aggregator = await get_metrics_aggregator()
        
        await aggregator.record_trade_execution(
            symbol, pnl, volume, commission, agent_name
        )
        
        return {
            'status': 'recorded',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording trade metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/record/agent")
async def record_agent_metric(
    agent_name: str,
    signal_type: str,
    confidence: float,
    executed: bool = False
) -> Dict[str, Any]:
    """Record an agent activity metric."""
    try:
        aggregator = await get_metrics_aggregator()
        
        await aggregator.record_agent_activity(
            agent_name, signal_type, confidence, executed
        )
        
        return {
            'status': 'recorded',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording agent metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Tracing endpoints

@router.get("/tracing/traces")
async def get_recent_traces(limit: int = Query(default=100, le=1000)) -> List[Dict[str, Any]]:
    """Get recent distributed traces."""
    try:
        tracing = await get_distributed_tracing()
        traces = await tracing.get_recent_traces(limit)
        
        return traces
        
    except Exception as e:
        logger.error(f"Error getting traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tracing/traces/{trace_id}")
async def get_trace_details(trace_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific trace."""
    try:
        tracing = await get_distributed_tracing()
        trace_data = await tracing.get_trace(trace_id)
        
        if not trace_data:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        
        return trace_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Alerting endpoints

@router.get("/alerts")
async def get_active_alerts() -> List[Dict[str, Any]]:
    """Get all active alerts."""
    try:
        alert_mgr = await get_alert_manager()
        alerts = await alert_mgr.get_active_alerts()
        
        return [
            {
                'id': alert.id,
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'first_triggered': alert.first_triggered.isoformat(),
                'last_updated': alert.last_updated.isoformat(),
                'acknowledged_by': alert.acknowledged_by,
                'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
            }
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/history")
async def get_alert_history(limit: int = Query(default=100, le=1000)) -> List[Dict[str, Any]]:
    """Get alert history."""
    try:
        alert_mgr = await get_alert_manager()
        history = await alert_mgr.get_alert_history(limit)
        
        return [
            {
                'id': alert.id,
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'first_triggered': alert.first_triggered.isoformat(),
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in history
        ]
        
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str) -> Dict[str, Any]:
    """Acknowledge an active alert."""
    try:
        alert_mgr = await get_alert_manager()
        
        success = await alert_mgr.acknowledge_alert(alert_id, acknowledged_by)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        return {
            'alert_id': alert_id,
            'status': 'acknowledged',
            'acknowledged_by': acknowledged_by,
            'acknowledged_at': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/rules")
async def get_alert_rules() -> List[Dict[str, Any]]:
    """Get all alert rules."""
    try:
        alert_mgr = await get_alert_manager()
        rules = await alert_mgr.get_alert_rules()
        
        return [
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
                'enabled': rule.enabled
            }
            for rule in rules
        ]
        
    except Exception as e:
        logger.error(f"Error getting alert rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/rules")
async def create_alert_rule(request: AlertRuleRequest) -> Dict[str, Any]:
    """Create a new alert rule."""
    try:
        from monitoring.alerting_system import AlertRule, AlertSeverity, AlertChannel
        
        alert_mgr = await get_alert_manager()
        
        # Convert request to AlertRule
        rule = AlertRule(
            id=str(uuid.uuid4()),
            name=request.name,
            description=request.description,
            severity=AlertSeverity(request.severity),
            condition=request.condition,
            threshold_value=request.threshold_value,
            comparison=request.comparison,
            evaluation_window=request.evaluation_window,
            alert_channels=[AlertChannel(ch) for ch in request.alert_channels],
            cooldown_period=request.cooldown_period,
            enabled=request.enabled
        )
        
        await alert_mgr.add_alert_rule(rule)
        
        return {
            'rule_id': rule.id,
            'status': 'created',
            'created_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str) -> Dict[str, Any]:
    """Delete an alert rule."""
    try:
        alert_mgr = await get_alert_manager()
        
        success = await alert_mgr.remove_alert_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert rule {rule_id} not found")
        
        return {
            'rule_id': rule_id,
            'status': 'deleted',
            'deleted_at': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health checking endpoints

@router.get("/health")
async def get_overall_health() -> Dict[str, Any]:
    """Get overall system health status."""
    try:
        health_chk = await get_health_checker()
        health_status = await health_chk.get_overall_health()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting overall health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/summary")
async def get_health_summary() -> Dict[str, Any]:
    """Get comprehensive health summary."""
    try:
        health_chk = await get_health_checker()
        summary = await health_chk.get_health_summary()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/components/{component_name}")
async def get_component_health(component_name: str) -> Dict[str, Any]:
    """Get health status for a specific component."""
    try:
        health_chk = await get_health_checker()
        component_health = await health_chk.get_component_health(component_name)
        
        if not component_health:
            raise HTTPException(status_code=404, detail=f"Component {component_name} not found")
        
        return component_health
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting component health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Dashboard endpoints

@router.get("/dashboard/overview")
async def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard data."""
    try:
        # Get data from all monitoring components
        aggregator = await get_metrics_aggregator()
        alert_mgr = await get_alert_manager()
        health_chk = await get_health_checker()
        
        # Get metrics
        business_metrics = await aggregator.get_comprehensive_metrics()
        
        # Get alerts
        active_alerts = await alert_mgr.get_active_alerts()
        
        # Get health
        health_status = await health_chk.get_overall_health()
        
        dashboard_data = {
            'system_overview': {
                'status': health_status['status'],
                'total_components': health_status['total_components'],
                'healthy_components': health_status['healthy_components'],
                'active_alerts': len(active_alerts),
                'critical_alerts': len([a for a in active_alerts if a.severity.value == 'critical'])
            },
            'business_metrics': business_metrics['business_metrics'],
            'health_status': health_status,
            'active_alerts': [
                {
                    'id': alert.id,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'first_triggered': alert.first_triggered.isoformat()
                }
                for alert in active_alerts[:10]  # Top 10 alerts
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/charts/trading")
async def get_trading_charts_data() -> Dict[str, Any]:
    """Get trading metrics for dashboard charts."""
    try:
        aggregator = await get_metrics_aggregator()
        
        # Get time series data for trading metrics
        pnl_data = await aggregator.get_time_series_data('pnl')
        trades_data = await aggregator.get_time_series_data('trades')
        volume_data = await aggregator.get_time_series_data('volume')
        
        return {
            'pnl_chart': pnl_data,
            'trades_chart': trades_data,
            'volume_chart': volume_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting trading charts data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/charts/system")
async def get_system_charts_data() -> Dict[str, Any]:
    """Get system metrics for dashboard charts."""
    try:
        aggregator = await get_metrics_aggregator()
        
        # Get time series data for system metrics
        cpu_data = await aggregator.get_time_series_data('cpu_usage')
        memory_data = await aggregator.get_time_series_data('memory_usage')
        connections_data = await aggregator.get_time_series_data('active_connections')
        
        return {
            'cpu_chart': cpu_data,
            'memory_chart': memory_data,
            'connections_chart': connections_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system charts data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notifications/config")
async def update_notification_config(config: NotificationConfigRequest) -> Dict[str, Any]:
    """Update notification configuration."""
    try:
        # This would update the notification configuration
        # For now, return success status
        return {
            'status': 'updated',
            'updated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating notification config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_monitoring_system_status() -> Dict[str, Any]:
    """Get monitoring system component status."""
    try:
        status = {
            'metrics_aggregator': metrics_aggregator is not None,
            'distributed_tracing': distributed_tracing is not None,
            'alert_manager': alert_manager is not None,
            'health_checker': health_checker is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting monitoring system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))