"""
Comprehensive metrics collection system for the Bloomberg Terminal trading platform.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from collections import defaultdict, deque

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, push_to_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available")

from core.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"  
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric to collect."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[Dict[float, float]] = None  # For summaries


@dataclass
class MetricValue:
    """A metric value with metadata."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    unit: Optional[str] = None


class BusinessMetricsCollector:
    """Collects business-specific metrics for trading operations."""
    
    def __init__(self):
        self.redis = None
        self.metrics_buffer: List[MetricValue] = []
        self.collection_interval = 30  # seconds
        self.is_running = False
        
        # Business metric accumulators
        self.trade_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_volume': 0.0,
            'total_commission': 0.0
        }
        
        self.agent_metrics = defaultdict(lambda: {
            'signals_generated': 0,
            'signals_executed': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0
        })
        
        self.risk_metrics = {
            'current_var': 0.0,
            'max_drawdown': 0.0,
            'leverage_ratio': 0.0,
            'risk_alerts': 0,
            'positions_count': 0
        }
        
        self.system_health = {
            'uptime_seconds': 0,
            'memory_usage_bytes': 0,
            'cpu_usage_percent': 0.0,
            'active_connections': 0,
            'error_rate': 0.0
        }
        
        logger.info("BusinessMetricsCollector initialized")
    
    async def initialize(self):
        """Initialize metrics collector."""
        self.redis = get_redis_manager().client
        self.is_running = True
        
        # Start collection loop
        asyncio.create_task(self._collection_loop())
        
        logger.info("BusinessMetricsCollector started")
    
    async def stop(self):
        """Stop metrics collection."""
        self.is_running = False
        logger.info("BusinessMetricsCollector stopped")
    
    async def record_trade_metric(
        self, 
        symbol: str, 
        pnl: float, 
        volume: float, 
        commission: float,
        agent_name: str = "unknown"
    ):
        """Record a trade execution metric."""
        self.trade_metrics['total_trades'] += 1
        self.trade_metrics['total_pnl'] += pnl
        self.trade_metrics['total_volume'] += volume
        self.trade_metrics['total_commission'] += commission
        
        if pnl > 0:
            self.trade_metrics['winning_trades'] += 1
        elif pnl < 0:
            self.trade_metrics['losing_trades'] += 1
        
        # Record individual trade metric
        metric = MetricValue(
            name="trade_executed",
            value=1,
            labels={
                'symbol': symbol,
                'agent': agent_name,
                'outcome': 'win' if pnl > 0 else 'loss' if pnl < 0 else 'neutral'
            }
        )
        
        self.metrics_buffer.append(metric)
        
        # Record PnL metric
        pnl_metric = MetricValue(
            name="trade_pnl",
            value=pnl,
            labels={'symbol': symbol, 'agent': agent_name},
            unit="USD"
        )
        
        self.metrics_buffer.append(pnl_metric)
    
    async def record_agent_signal(
        self,
        agent_name: str,
        signal_type: str,
        confidence: float,
        executed: bool = False
    ):
        """Record an agent signal generation."""
        self.agent_metrics[agent_name]['signals_generated'] += 1
        
        if executed:
            self.agent_metrics[agent_name]['signals_executed'] += 1
        
        # Update average confidence
        current_avg = self.agent_metrics[agent_name]['avg_confidence']
        total_signals = self.agent_metrics[agent_name]['signals_generated']
        
        new_avg = ((current_avg * (total_signals - 1)) + confidence) / total_signals
        self.agent_metrics[agent_name]['avg_confidence'] = new_avg
        
        # Record signal metric
        metric = MetricValue(
            name="agent_signal_generated",
            value=1,
            labels={
                'agent': agent_name,
                'signal_type': signal_type,
                'executed': str(executed).lower()
            }
        )
        
        self.metrics_buffer.append(metric)
        
        # Record confidence metric
        confidence_metric = MetricValue(
            name="agent_signal_confidence",
            value=confidence,
            labels={'agent': agent_name, 'signal_type': signal_type}
        )
        
        self.metrics_buffer.append(confidence_metric)
    
    async def record_risk_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a risk management metric."""
        if metric_name in self.risk_metrics:
            self.risk_metrics[metric_name] = value
        
        metric = MetricValue(
            name=f"risk_{metric_name}",
            value=value,
            labels=labels or {}
        )
        
        self.metrics_buffer.append(metric)
    
    async def record_system_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a system health metric."""
        if metric_name in self.system_health:
            self.system_health[metric_name] = value
        
        metric = MetricValue(
            name=f"system_{metric_name}",
            value=value,
            labels=labels or {}
        )
        
        self.metrics_buffer.append(metric)
    
    async def _collection_loop(self):
        """Main metrics collection and persistence loop."""
        while self.is_running:
            try:
                await self._persist_metrics()
                await self._calculate_derived_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _persist_metrics(self):
        """Persist collected metrics to Redis."""
        if not self.metrics_buffer:
            return
        
        # Prepare metrics for storage
        metrics_data = []
        
        for metric in self.metrics_buffer:
            metric_data = {
                'name': metric.name,
                'value': metric.value,
                'labels': metric.labels,
                'timestamp': metric.timestamp.isoformat(),
                'unit': metric.unit
            }
            metrics_data.append(metric_data)
        
        # Store in Redis with timestamp key
        timestamp_key = datetime.now().strftime('%Y%m%d_%H%M%S')
        redis_key = f"metrics:business:{timestamp_key}"
        
        await self.redis.setex(
            redis_key,
            86400,  # 24 hour TTL
            json.dumps(metrics_data)
        )
        
        # Store latest metrics snapshot
        snapshot = {
            'trade_metrics': self.trade_metrics,
            'risk_metrics': self.risk_metrics,
            'system_health': self.system_health,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.redis.setex(
            "metrics:business:latest",
            3600,  # 1 hour TTL
            json.dumps(snapshot)
        )
        
        logger.debug(f"Persisted {len(self.metrics_buffer)} metrics to Redis")
        
        # Clear buffer
        self.metrics_buffer.clear()
    
    async def _calculate_derived_metrics(self):
        """Calculate derived business metrics."""
        # Calculate win rate
        total_trades = self.trade_metrics['total_trades']
        if total_trades > 0:
            win_rate = self.trade_metrics['winning_trades'] / total_trades
            
            win_rate_metric = MetricValue(
                name="trading_win_rate",
                value=win_rate,
                unit="percentage"
            )
            self.metrics_buffer.append(win_rate_metric)
        
        # Calculate average PnL per trade
        if total_trades > 0:
            avg_pnl = self.trade_metrics['total_pnl'] / total_trades
            
            avg_pnl_metric = MetricValue(
                name="trading_avg_pnl",
                value=avg_pnl,
                unit="USD"
            )
            self.metrics_buffer.append(avg_pnl_metric)
        
        # Calculate agent execution rates
        for agent_name, metrics in self.agent_metrics.items():
            if metrics['signals_generated'] > 0:
                execution_rate = metrics['signals_executed'] / metrics['signals_generated']
                
                execution_metric = MetricValue(
                    name="agent_execution_rate",
                    value=execution_rate,
                    labels={'agent': agent_name},
                    unit="percentage"
                )
                self.metrics_buffer.append(execution_metric)
    
    async def get_business_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current business metrics."""
        # Calculate derived metrics
        total_trades = self.trade_metrics['total_trades']
        win_rate = (self.trade_metrics['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = self.trade_metrics['total_pnl'] / total_trades if total_trades > 0 else 0
        
        return {
            'trading': {
                'total_trades': total_trades,
                'win_rate_percent': round(win_rate, 2),
                'total_pnl': round(self.trade_metrics['total_pnl'], 2),
                'average_pnl': round(avg_pnl, 2),
                'total_volume': round(self.trade_metrics['total_volume'], 2),
                'total_commission': round(self.trade_metrics['total_commission'], 2)
            },
            'risk': self.risk_metrics,
            'system': self.system_health,
            'agents': dict(self.agent_metrics),
            'timestamp': datetime.now().isoformat()
        }


class PrometheusMetricsCollector:
    """Prometheus metrics collection and exposure."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.registry = CollectorRegistry()
        self.metrics = {}
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - metrics will not be exported")
            return
        
        # Define core system metrics
        self._define_core_metrics()
        
        logger.info(f"PrometheusMetricsCollector initialized on port {port}")
    
    def _define_core_metrics(self):
        """Define core Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Trading metrics
        self.metrics['trades_total'] = Counter(
            'bloomberg_trades_total',
            'Total number of trades executed',
            ['symbol', 'agent', 'outcome'],
            registry=self.registry
        )
        
        self.metrics['trade_pnl'] = Histogram(
            'bloomberg_trade_pnl_usd',
            'P&L per trade in USD',
            ['symbol', 'agent'],
            buckets=[-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000, 5000],
            registry=self.registry
        )
        
        self.metrics['portfolio_value'] = Gauge(
            'bloomberg_portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        # Agent metrics
        self.metrics['agent_signals'] = Counter(
            'bloomberg_agent_signals_total',
            'Total signals generated by agents',
            ['agent', 'signal_type', 'executed'],
            registry=self.registry
        )
        
        self.metrics['agent_confidence'] = Histogram(
            'bloomberg_agent_confidence',
            'Agent signal confidence distribution',
            ['agent', 'signal_type'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Risk metrics
        self.metrics['risk_var'] = Gauge(
            'bloomberg_risk_var_usd',
            'Current Value at Risk in USD',
            registry=self.registry
        )
        
        self.metrics['risk_drawdown'] = Gauge(
            'bloomberg_risk_max_drawdown_percent',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        
        self.metrics['risk_leverage'] = Gauge(
            'bloomberg_risk_leverage_ratio',
            'Current leverage ratio',
            registry=self.registry
        )
        
        # System metrics
        self.metrics['http_requests'] = Counter(
            'bloomberg_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['http_duration'] = Histogram(
            'bloomberg_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.metrics['websocket_connections'] = Gauge(
            'bloomberg_websocket_connections_active',
            'Active WebSocket connections',
            registry=self.registry
        )
        
        self.metrics['market_data_latency'] = Histogram(
            'bloomberg_market_data_latency_seconds',
            'Market data processing latency',
            ['symbol', 'data_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        # ML Training metrics
        self.metrics['training_jobs'] = Counter(
            'bloomberg_training_jobs_total',
            'Total training jobs',
            ['model_type', 'status'],
            registry=self.registry
        )
        
        self.metrics['training_duration'] = Histogram(
            'bloomberg_training_duration_seconds',
            'Training job duration',
            ['model_type'],
            registry=self.registry
        )
        
        self.metrics['model_accuracy'] = Gauge(
            'bloomberg_model_accuracy',
            'Model accuracy score',
            ['model_name', 'model_type'],
            registry=self.registry
        )
    
    def start_server(self):
        """Start Prometheus metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Cannot start Prometheus server - client not available")
            return
        
        try:
            start_http_server(self.port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def record_trade(self, symbol: str, agent: str, outcome: str, pnl: float):
        """Record a trade metric."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['trades_total'].labels(
            symbol=symbol, 
            agent=agent, 
            outcome=outcome
        ).inc()
        
        self.metrics['trade_pnl'].labels(
            symbol=symbol, 
            agent=agent
        ).observe(pnl)
    
    def record_agent_signal(self, agent: str, signal_type: str, confidence: float, executed: bool):
        """Record an agent signal metric."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['agent_signals'].labels(
            agent=agent,
            signal_type=signal_type,
            executed=str(executed).lower()
        ).inc()
        
        self.metrics['agent_confidence'].labels(
            agent=agent,
            signal_type=signal_type
        ).observe(confidence)
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value gauge."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['portfolio_value'].set(value)
    
    def update_risk_metrics(self, var: float, drawdown: float, leverage: float):
        """Update risk metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['risk_var'].set(var)
        self.metrics['risk_drawdown'].set(drawdown)
        self.metrics['risk_leverage'].set(leverage)
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['http_requests'].labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.metrics['http_duration'].labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def update_websocket_connections(self, count: int):
        """Update WebSocket connections count."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['websocket_connections'].set(count)
    
    def record_market_data_latency(self, symbol: str, data_type: str, latency: float):
        """Record market data processing latency."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['market_data_latency'].labels(
            symbol=symbol,
            data_type=data_type
        ).observe(latency)
    
    def record_training_job(self, model_type: str, status: str, duration: Optional[float] = None):
        """Record training job metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['training_jobs'].labels(
            model_type=model_type,
            status=status
        ).inc()
        
        if duration is not None:
            self.metrics['training_duration'].labels(
                model_type=model_type
            ).observe(duration)
    
    def update_model_accuracy(self, model_name: str, model_type: str, accuracy: float):
        """Update model accuracy metric."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['model_accuracy'].labels(
            model_name=model_name,
            model_type=model_type
        ).set(accuracy)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        return generate_latest(self.registry).decode('utf-8')


class MetricsAggregator:
    """Aggregates metrics from multiple sources and provides unified access."""
    
    def __init__(self):
        self.business_collector = BusinessMetricsCollector()
        self.prometheus_collector = PrometheusMetricsCollector()
        
        # Time-series storage for trends
        self.time_series = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 points
        
        self.is_running = False
        
        logger.info("MetricsAggregator initialized")
    
    async def initialize(self):
        """Initialize metrics aggregator."""
        await self.business_collector.initialize()
        self.prometheus_collector.start_server()
        
        self.is_running = True
        
        # Start aggregation loop
        asyncio.create_task(self._aggregation_loop())
        
        logger.info("MetricsAggregator started")
    
    async def stop(self):
        """Stop metrics aggregation."""
        self.is_running = False
        await self.business_collector.stop()
        
        logger.info("MetricsAggregator stopped")
    
    async def record_trade_execution(
        self,
        symbol: str,
        pnl: float,
        volume: float,
        commission: float,
        agent_name: str = "unknown"
    ):
        """Record a trade execution across all collectors."""
        # Record in business metrics
        await self.business_collector.record_trade_metric(
            symbol, pnl, volume, commission, agent_name
        )
        
        # Record in Prometheus
        outcome = 'win' if pnl > 0 else 'loss' if pnl < 0 else 'neutral'
        self.prometheus_collector.record_trade(symbol, agent_name, outcome, pnl)
        
        # Store for time series
        timestamp = datetime.now()
        self.time_series['trades'].append((timestamp, 1))
        self.time_series['pnl'].append((timestamp, pnl))
        self.time_series['volume'].append((timestamp, volume))
    
    async def record_agent_activity(
        self,
        agent_name: str,
        signal_type: str,
        confidence: float,
        executed: bool = False
    ):
        """Record agent activity across all collectors."""
        # Record in business metrics
        await self.business_collector.record_agent_signal(
            agent_name, signal_type, confidence, executed
        )
        
        # Record in Prometheus
        self.prometheus_collector.record_agent_signal(
            agent_name, signal_type, confidence, executed
        )
        
        # Store for time series
        timestamp = datetime.now()
        self.time_series[f'agent_{agent_name}_signals'].append((timestamp, 1))
        self.time_series[f'agent_{agent_name}_confidence'].append((timestamp, confidence))
    
    async def update_portfolio_metrics(self, portfolio_value: float, positions_count: int):
        """Update portfolio-level metrics."""
        # Update Prometheus
        self.prometheus_collector.update_portfolio_value(portfolio_value)
        
        # Record in business metrics
        await self.business_collector.record_system_metric(
            'portfolio_value', portfolio_value
        )
        
        await self.business_collector.record_risk_metric(
            'positions_count', positions_count
        )
        
        # Store for time series
        timestamp = datetime.now()
        self.time_series['portfolio_value'].append((timestamp, portfolio_value))
        self.time_series['positions_count'].append((timestamp, positions_count))
    
    async def update_risk_metrics(
        self,
        var: float,
        max_drawdown: float,
        leverage: float,
        risk_alerts: int = 0
    ):
        """Update risk management metrics."""
        # Update Prometheus
        self.prometheus_collector.update_risk_metrics(var, max_drawdown, leverage)
        
        # Record in business metrics
        await self.business_collector.record_risk_metric('current_var', var)
        await self.business_collector.record_risk_metric('max_drawdown', max_drawdown)
        await self.business_collector.record_risk_metric('leverage_ratio', leverage)
        await self.business_collector.record_risk_metric('risk_alerts', risk_alerts)
        
        # Store for time series
        timestamp = datetime.now()
        self.time_series['var'].append((timestamp, var))
        self.time_series['drawdown'].append((timestamp, max_drawdown))
        self.time_series['leverage'].append((timestamp, leverage))
    
    async def record_system_performance(
        self,
        memory_usage: float,
        cpu_usage: float,
        active_connections: int,
        error_rate: float = 0.0
    ):
        """Record system performance metrics."""
        # Record in business metrics
        await self.business_collector.record_system_metric('memory_usage_bytes', memory_usage)
        await self.business_collector.record_system_metric('cpu_usage_percent', cpu_usage)
        await self.business_collector.record_system_metric('active_connections', active_connections)
        await self.business_collector.record_system_metric('error_rate', error_rate)
        
        # Store for time series
        timestamp = datetime.now()
        self.time_series['memory_usage'].append((timestamp, memory_usage))
        self.time_series['cpu_usage'].append((timestamp, cpu_usage))
        self.time_series['active_connections'].append((timestamp, active_connections))
    
    async def record_training_metrics(
        self,
        model_type: str,
        model_name: str,
        status: str,
        accuracy: Optional[float] = None,
        duration: Optional[float] = None
    ):
        """Record ML training metrics."""
        # Record in Prometheus
        self.prometheus_collector.record_training_job(model_type, status, duration)
        
        if accuracy is not None:
            self.prometheus_collector.update_model_accuracy(model_name, model_type, accuracy)
            
            # Store for time series
            timestamp = datetime.now()
            self.time_series[f'model_{model_name}_accuracy'].append((timestamp, accuracy))
    
    async def _aggregation_loop(self):
        """Main aggregation loop for calculating rolling metrics."""
        while self.is_running:
            try:
                await self._calculate_rolling_metrics()
                await asyncio.sleep(60)  # Calculate every minute
                
            except Exception as e:
                logger.error(f"Error in metrics aggregation loop: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_rolling_metrics(self):
        """Calculate rolling averages and trends."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # Calculate rolling metrics for the last hour
        for metric_name, data_points in self.time_series.items():
            if not data_points:
                continue
            
            # Filter last hour data
            recent_points = [(ts, val) for ts, val in data_points if ts >= one_hour_ago]
            
            if len(recent_points) > 1:
                values = [val for _, val in recent_points]
                
                # Calculate statistics
                avg_value = sum(values) / len(values)
                min_value = min(values)
                max_value = max(values)
                
                # Store rolling metrics
                rolling_key = f"{metric_name}_rolling_1h"
                self.time_series[f"{rolling_key}_avg"].append((now, avg_value))
                self.time_series[f"{rolling_key}_min"].append((now, min_value))
                self.time_series[f"{rolling_key}_max"].append((now, max_value))
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all sources."""
        # Get business metrics
        business_metrics = await self.business_collector.get_business_metrics_summary()
        
        # Get time series summaries
        time_series_summary = {}
        for metric_name, data_points in self.time_series.items():
            if data_points:
                latest_timestamp, latest_value = data_points[-1]
                time_series_summary[metric_name] = {
                    'latest_value': latest_value,
                    'latest_timestamp': latest_timestamp.isoformat(),
                    'data_points_count': len(data_points)
                }
        
        return {
            'business_metrics': business_metrics,
            'time_series_summary': time_series_summary,
            'collection_timestamp': datetime.now().isoformat(),
            'prometheus_available': PROMETHEUS_AVAILABLE
        }
    
    async def get_time_series_data(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get time series data for a specific metric."""
        if metric_name not in self.time_series:
            return []
        
        data_points = self.time_series[metric_name]
        
        # Filter by time range if provided
        if start_time or end_time:
            filtered_points = []
            for timestamp, value in data_points:
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_points.append((timestamp, value))
            data_points = filtered_points
        
        return [
            {
                'timestamp': timestamp.isoformat(),
                'value': value
            }
            for timestamp, value in data_points
        ]