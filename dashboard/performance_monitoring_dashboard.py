"""
Performance Monitoring and Visualization Dashboard

Real-time dashboard for monitoring trading system performance, risk metrics,
and system health with comprehensive visualizations and alerts.
"""

import asyncio
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
import json
import logging
from dataclasses import dataclass, field
import redis
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

# Financial libraries
import yfinance as yf
import talib
from scipy import stats
import quantstats as qs

# ML libraries
import torch
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Real-time data
import websocket
import requests

@dataclass
class AlertConfig:
    """Configuration for system alerts"""
    max_drawdown_threshold: float = 0.05
    sharpe_ratio_threshold: float = 1.0
    daily_loss_threshold: float = 0.02
    position_concentration_threshold: float = 0.25
    system_latency_threshold: float = 100  # milliseconds
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    alert_frequency: int = 300  # seconds

@dataclass
class DashboardMetrics:
    """Real-time metrics for dashboard display"""
    current_portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_positions: int = 0
    active_strategies: int = 0
    system_health: str = "Healthy"
    last_updated: datetime = field(default_factory=datetime.now)

class DataStreamManager:
    """Manages real-time data streams for the dashboard"""

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.data_queue = Queue()
        self.is_streaming = False
        self.stream_thread = None

    def start_streaming(self):
        """Start real-time data streaming"""
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def stop_streaming(self):
        """Stop data streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()

    def _stream_worker(self):
        """Worker thread for data streaming"""
        while self.is_streaming:
            try:
                # Get latest data from Redis
                portfolio_data = self.redis_client.get('portfolio:current')
                if portfolio_data:
                    data = json.loads(portfolio_data)
                    self.data_queue.put(('portfolio', data))

                # Get strategy performance
                strategy_data = self.redis_client.get('strategies:performance')
                if strategy_data:
                    data = json.loads(strategy_data)
                    self.data_queue.put(('strategies', data))

                # Get system metrics
                system_data = self.redis_client.get('system:metrics')
                if system_data:
                    data = json.loads(system_data)
                    self.data_queue.put(('system', data))

                time.sleep(1)  # Update every second

            except Exception as e:
                logging.error(f"Error in data streaming: {e}")
                time.sleep(5)

    def get_latest_data(self) -> Dict[str, Any]:
        """Get latest data from queue"""
        latest_data = {}
        while not self.data_queue.empty():
            data_type, data = self.data_queue.get()
            latest_data[data_type] = data
        return latest_data

class PerformanceCalculator:
    """Calculates real-time performance metrics"""

    def __init__(self, db_path: str = "trading_performance.db"):
        self.db_path = db_path
        self._setup_database()

    def _setup_database(self):
        """Setup performance tracking database"""
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                portfolio_value REAL,
                daily_return REAL,
                daily_pnl REAL,
                positions_count INTEGER,
                active_strategies INTEGER,
                trades_count INTEGER,
                drawdown REAL,
                volatility REAL,
                sharpe_ratio REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                date TEXT,
                trades_count INTEGER,
                win_rate REAL,
                total_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                avg_holding_period REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)

        conn.close()

    def update_daily_performance(self, portfolio_data: Dict[str, Any]):
        """Update daily performance metrics"""
        conn = sqlite3.connect(self.db_path)

        today = datetime.now().strftime('%Y-%m-%d')

        conn.execute("""
            INSERT OR REPLACE INTO daily_performance
            (date, portfolio_value, daily_return, daily_pnl, positions_count,
             active_strategies, trades_count, drawdown, volatility, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            today,
            portfolio_data.get('portfolio_value', 0),
            portfolio_data.get('daily_return', 0),
            portfolio_data.get('daily_pnl', 0),
            portfolio_data.get('positions_count', 0),
            portfolio_data.get('active_strategies', 0),
            portfolio_data.get('trades_count', 0),
            portfolio_data.get('drawdown', 0),
            portfolio_data.get('volatility', 0),
            portfolio_data.get('sharpe_ratio', 0)
        ))

        conn.commit()
        conn.close()

    def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Get performance history for specified days"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT * FROM daily_performance
            ORDER BY date DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(days,))
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

        return df

    def calculate_metrics(self, portfolio_data: Dict[str, Any]) -> DashboardMetrics:
        """Calculate comprehensive dashboard metrics"""

        # Get historical data for calculations
        hist_data = self.get_performance_history(252)  # 1 year

        current_value = portfolio_data.get('portfolio_value', 0)
        daily_pnl = portfolio_data.get('daily_pnl', 0)

        if len(hist_data) > 1:
            initial_value = hist_data['portfolio_value'].iloc[0]
            total_return = (current_value - initial_value) / initial_value if initial_value > 0 else 0

            # Calculate rolling metrics
            returns = hist_data['daily_return'].dropna()
            if len(returns) > 10:
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                running_max = hist_data['portfolio_value'].expanding().max()
                drawdown = (hist_data['portfolio_value'] - running_max) / running_max
                max_dd = drawdown.min()
            else:
                sharpe = 0
                max_dd = 0
        else:
            total_return = 0
            sharpe = 0
            max_dd = 0

        # System health assessment
        health_status = self._assess_system_health(portfolio_data, sharpe, max_dd)

        return DashboardMetrics(
            current_portfolio_value=current_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl / current_value if current_value > 0 else 0,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            current_positions=portfolio_data.get('positions_count', 0),
            active_strategies=portfolio_data.get('active_strategies', 0),
            system_health=health_status,
            last_updated=datetime.now()
        )

    def _assess_system_health(self, portfolio_data: Dict[str, Any],
                            sharpe: float, max_dd: float) -> str:
        """Assess overall system health"""
        issues = []

        # Check performance metrics
        if sharpe < 0.5:
            issues.append("Low Sharpe ratio")
        if max_dd < -0.1:
            issues.append("High drawdown")
        if portfolio_data.get('daily_pnl', 0) < -portfolio_data.get('portfolio_value', 1) * 0.05:
            issues.append("Large daily loss")

        # Check system metrics
        if portfolio_data.get('system_latency', 0) > 500:
            issues.append("High latency")
        if portfolio_data.get('error_rate', 0) > 0.01:
            issues.append("High error rate")

        if not issues:
            return "Healthy"
        elif len(issues) <= 2:
            return "Warning"
        else:
            return "Critical"

class AlertManager:
    """Manages system alerts and notifications"""

    def __init__(self, config: AlertConfig, performance_calc: PerformanceCalculator):
        self.config = config
        self.performance_calc = performance_calc
        self.last_alert_time = {}

    def check_alerts(self, metrics: DashboardMetrics, portfolio_data: Dict[str, Any]):
        """Check for alert conditions"""
        current_time = datetime.now()
        alerts = []

        # Drawdown alert
        if metrics.max_drawdown < -self.config.max_drawdown_threshold:
            if self._should_send_alert('drawdown', current_time):
                alerts.append({
                    'type': 'drawdown',
                    'severity': 'high',
                    'message': f"Max drawdown exceeded threshold: {metrics.max_drawdown:.2%}"
                })

        # Sharpe ratio alert
        if metrics.sharpe_ratio < self.config.sharpe_ratio_threshold:
            if self._should_send_alert('sharpe', current_time):
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Sharpe ratio below threshold: {metrics.sharpe_ratio:.2f}"
                })

        # Daily loss alert
        if metrics.daily_pnl_pct < -self.config.daily_loss_threshold:
            if self._should_send_alert('daily_loss', current_time):
                alerts.append({
                    'type': 'loss',
                    'severity': 'high',
                    'message': f"Daily loss exceeded threshold: {metrics.daily_pnl_pct:.2%}"
                })

        # System health alert
        if metrics.system_health in ['Warning', 'Critical']:
            if self._should_send_alert('health', current_time):
                alerts.append({
                    'type': 'system',
                    'severity': 'high' if metrics.system_health == 'Critical' else 'medium',
                    'message': f"System health: {metrics.system_health}"
                })

        # Store and send alerts
        for alert in alerts:
            self._store_alert(alert)
            self._send_notification(alert)

        return alerts

    def _should_send_alert(self, alert_type: str, current_time: datetime) -> bool:
        """Check if enough time has passed since last alert"""
        if alert_type not in self.last_alert_time:
            self.last_alert_time[alert_type] = current_time
            return True

        time_diff = (current_time - self.last_alert_time[alert_type]).total_seconds()
        if time_diff >= self.config.alert_frequency:
            self.last_alert_time[alert_type] = current_time
            return True

        return False

    def _store_alert(self, alert: Dict[str, Any]):
        """Store alert in database"""
        conn = sqlite3.connect(self.performance_calc.db_path)

        conn.execute("""
            INSERT INTO system_alerts (timestamp, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            alert['type'],
            alert['severity'],
            alert['message']
        ))

        conn.commit()
        conn.close()

    def _send_notification(self, alert: Dict[str, Any]):
        """Send notification via configured channels"""
        # Email notification
        if self.config.enable_email_alerts:
            self._send_email_alert(alert)

        # Slack notification
        if self.config.enable_slack_alerts:
            self._send_slack_alert(alert)

        # Log alert
        logging.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")

    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert (placeholder)"""
        # Implement email sending logic
        pass

    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert (placeholder)"""
        # Implement Slack webhook logic
        pass

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        conn = sqlite3.connect(self.performance_calc.db_path)

        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        query = """
            SELECT * FROM system_alerts
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        """

        cursor = conn.execute(query, (cutoff_time,))
        alerts = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return alerts

class DashboardVisualization:
    """Creates comprehensive dashboard visualizations"""

    def __init__(self, performance_calc: PerformanceCalculator):
        self.performance_calc = performance_calc

    def create_main_dashboard(self, metrics: DashboardMetrics,
                            portfolio_data: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create main dashboard with all visualizations"""

        figures = {}

        # 1. Portfolio Value Chart
        figures['portfolio_chart'] = self._create_portfolio_chart()

        # 2. Performance Metrics
        figures['performance_metrics'] = self._create_performance_metrics(metrics)

        # 3. Position Analysis
        figures['position_analysis'] = self._create_position_analysis(portfolio_data)

        # 4. Strategy Performance
        figures['strategy_performance'] = self._create_strategy_performance()

        # 5. Risk Metrics
        figures['risk_metrics'] = self._create_risk_dashboard(metrics)

        # 6. Real-time Trading Activity
        figures['trading_activity'] = self._create_trading_activity()

        return figures

    def _create_portfolio_chart(self) -> go.Figure:
        """Create portfolio value and drawdown chart"""
        hist_data = self.performance_calc.get_performance_history(90)

        if hist_data.empty:
            return go.Figure().add_annotation(
                text="No historical data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=['Portfolio Value', 'Drawdown'],
            row_heights=[0.7, 0.3]
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=hist_data['date'],
                y=hist_data['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=hist_data['date'],
                y=hist_data['drawdown'] * 100,
                mode='lines',
                name='Drawdown %',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color='red')
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="Portfolio Performance & Risk",
            height=500,
            showlegend=True
        )

        return fig

    def _create_performance_metrics(self, metrics: DashboardMetrics) -> go.Figure:
        """Create performance metrics gauge charts"""

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Daily P&L']
        )

        # Total Return Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics.total_return * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Total Return %"},
                gauge={
                    'axis': {'range': [-50, 100]},
                    'bar': {'color': "green" if metrics.total_return > 0 else "red"},
                    'steps': [
                        {'range': [-50, 0], 'color': "lightred"},
                        {'range': [0, 20], 'color': "lightyellow"},
                        {'range': [20, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ),
            row=1, col=1
        )

        # Sharpe Ratio Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.sharpe_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sharpe Ratio"},
                gauge={
                    'axis': {'range': [-2, 4]},
                    'bar': {'color': "green" if metrics.sharpe_ratio > 1 else "orange" if metrics.sharpe_ratio > 0 else "red"},
                    'steps': [
                        {'range': [-2, 0], 'color': "lightred"},
                        {'range': [0, 1], 'color': "lightyellow"},
                        {'range': [1, 4], 'color': "lightgreen"}
                    ]
                }
            ),
            row=1, col=2
        )

        # Max Drawdown Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=abs(metrics.max_drawdown) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Max Drawdown %"},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': "green" if abs(metrics.max_drawdown) < 0.05 else "orange" if abs(metrics.max_drawdown) < 0.15 else "red"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 15], 'color': "lightyellow"},
                        {'range': [15, 50], 'color': "lightred"}
                    ]
                }
            ),
            row=2, col=1
        )

        # Daily P&L
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.daily_pnl,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Daily P&L"},
                number={'prefix': "$"},
                delta={'reference': 0, 'relative': False}
            ),
            row=2, col=2
        )

        fig.update_layout(height=600, title="Key Performance Indicators")
        return fig

    def _create_position_analysis(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create position analysis charts"""

        positions = portfolio_data.get('positions', {})

        if not positions:
            return go.Figure().add_annotation(
                text="No active positions",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        # Extract position data
        symbols = list(positions.keys())
        values = [pos.get('market_value', 0) for pos in positions.values()]
        pnls = [pos.get('unrealized_pnl', 0) for pos in positions.values()]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=['Position Allocation', 'Unrealized P&L by Position']
        )

        # Position allocation pie chart
        fig.add_trace(
            go.Pie(
                labels=symbols,
                values=values,
                name="Position Allocation"
            ),
            row=1, col=1
        )

        # Unrealized P&L bar chart
        colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=pnls,
                name="Unrealized P&L",
                marker_color=colors
            ),
            row=1, col=2
        )

        fig.update_layout(height=400, title="Position Analysis")
        return fig

    def _create_strategy_performance(self) -> go.Figure:
        """Create strategy performance comparison"""

        # Get strategy performance data
        conn = sqlite3.connect(self.performance_calc.db_path)

        query = """
            SELECT strategy_name, SUM(total_pnl) as total_pnl,
                   AVG(win_rate) as avg_win_rate, AVG(sharpe_ratio) as avg_sharpe
            FROM strategy_performance
            WHERE date >= date('now', '-30 days')
            GROUP BY strategy_name
        """

        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
        except:
            conn.close()
            return go.Figure().add_annotation(
                text="No strategy data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        if df.empty:
            return go.Figure().add_annotation(
                text="No strategy data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=['Strategy P&L (Last 30 Days)', 'Strategy Win Rates']
        )

        # Strategy P&L
        colors = ['green' if pnl >= 0 else 'red' for pnl in df['total_pnl']]
        fig.add_trace(
            go.Bar(
                x=df['strategy_name'],
                y=df['total_pnl'],
                name="Total P&L",
                marker_color=colors
            ),
            row=1, col=1
        )

        # Win rates
        fig.add_trace(
            go.Scatter(
                x=df['strategy_name'],
                y=df['avg_win_rate'] * 100,
                mode='markers+lines',
                name="Win Rate %",
                marker=dict(size=10, color='blue')
            ),
            row=2, col=1
        )

        fig.update_layout(height=500, title="Strategy Performance Comparison")
        return fig

    def _create_risk_dashboard(self, metrics: DashboardMetrics) -> go.Figure:
        """Create comprehensive risk dashboard"""

        hist_data = self.performance_calc.get_performance_history(252)

        if hist_data.empty:
            return go.Figure().add_annotation(
                text="No historical data for risk analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        returns = hist_data['daily_return'].dropna()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Return Distribution', 'Rolling Volatility',
                          'VaR Analysis', 'Correlation Heatmap']
        )

        # Return distribution
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=30,
                name="Daily Returns %",
                opacity=0.7
            ),
            row=1, col=1
        )

        # Rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(
                x=hist_data['date'].iloc[29:],
                y=rolling_vol.iloc[29:],
                mode='lines',
                name="30-Day Rolling Volatility %"
            ),
            row=1, col=2
        )

        # VaR analysis
        if len(returns) > 20:
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100

            fig.add_trace(
                go.Bar(
                    x=['VaR 95%', 'VaR 99%', 'Current Drawdown'],
                    y=[var_95, var_99, metrics.max_drawdown * 100],
                    name="Risk Measures",
                    marker_color=['orange', 'red', 'blue']
                ),
                row=2, col=1
            )

        fig.update_layout(height=600, title="Risk Analysis Dashboard")
        return fig

    def _create_trading_activity(self) -> go.Figure:
        """Create real-time trading activity chart"""

        # Get recent trading activity
        conn = sqlite3.connect(self.performance_calc.db_path)

        query = """
            SELECT date, trades_count, SUM(trades_count) OVER (
                ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as rolling_trades
            FROM daily_performance
            WHERE date >= date('now', '-30 days')
            ORDER BY date
        """

        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
        except:
            conn.close()
            return go.Figure().add_annotation(
                text="No trading activity data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        if df.empty:
            return go.Figure().add_annotation(
                text="No trading activity data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=['Daily Trading Volume', '7-Day Rolling Average'],
            vertical_spacing=0.1
        )

        # Daily trades
        fig.add_trace(
            go.Bar(
                x=pd.to_datetime(df['date']),
                y=df['trades_count'],
                name="Daily Trades",
                opacity=0.7
            ),
            row=1, col=1
        )

        # Rolling average
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(df['date']),
                y=df['rolling_trades'] / 7,
                mode='lines',
                name="7-Day Average",
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )

        fig.update_layout(height=400, title="Trading Activity Analysis")
        return fig

class StreamlitDashboard:
    """Streamlit-based interactive dashboard"""

    def __init__(self):
        self.data_manager = DataStreamManager()
        self.performance_calc = PerformanceCalculator()
        self.alert_config = AlertConfig()
        self.alert_manager = AlertManager(self.alert_config, self.performance_calc)
        self.visualizer = DashboardVisualization(self.performance_calc)

    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="HiveTrading Performance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🚀 HiveTrading Advanced Performance Dashboard")

        # Sidebar configuration
        self._setup_sidebar()

        # Get latest data
        portfolio_data = self._get_sample_data()  # Replace with real data
        metrics = self.performance_calc.calculate_metrics(portfolio_data)

        # Check alerts
        alerts = self.alert_manager.check_alerts(metrics, portfolio_data)

        # Display alerts
        self._display_alerts(alerts)

        # Main dashboard
        self._display_main_dashboard(metrics, portfolio_data)

        # Auto-refresh
        time.sleep(5)
        st.rerun()

    def _setup_sidebar(self):
        """Setup sidebar controls"""
        st.sidebar.header("Dashboard Controls")

        # Refresh rate
        refresh_rate = st.sidebar.selectbox(
            "Refresh Rate",
            [5, 10, 30, 60],
            index=0,
            help="Dashboard refresh rate in seconds"
        )

        # Time range
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["1D", "1W", "1M", "3M", "1Y"],
            index=2
        )

        # Alert settings
        st.sidebar.header("Alert Settings")
        enable_alerts = st.sidebar.checkbox("Enable Alerts", value=True)

        if enable_alerts:
            drawdown_threshold = st.sidebar.slider(
                "Max Drawdown Alert %",
                min_value=1,
                max_value=20,
                value=5
            )

        # Export options
        st.sidebar.header("Export Options")
        if st.sidebar.button("Export Performance Report"):
            self._export_performance_report()

        if st.sidebar.button("Export Trade Log"):
            self._export_trade_log()

    def _display_alerts(self, alerts: List[Dict[str, Any]]):
        """Display system alerts"""
        if alerts:
            st.header("🚨 Active Alerts")

            for alert in alerts:
                if alert['severity'] == 'high':
                    st.error(f"**{alert['type'].upper()}**: {alert['message']}")
                elif alert['severity'] == 'medium':
                    st.warning(f"**{alert['type'].upper()}**: {alert['message']}")
                else:
                    st.info(f"**{alert['type'].upper()}**: {alert['message']}")

        # Recent alerts
        recent_alerts = self.alert_manager.get_recent_alerts(24)
        if recent_alerts:
            with st.expander("Recent Alerts (24h)"):
                for alert in recent_alerts[:10]:
                    st.text(f"{alert['timestamp']}: {alert['message']}")

    def _display_main_dashboard(self, metrics: DashboardMetrics,
                              portfolio_data: Dict[str, Any]):
        """Display main dashboard"""

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Portfolio Value",
                f"${metrics.current_portfolio_value:,.2f}",
                f"${metrics.daily_pnl:,.2f}"
            )

        with col2:
            st.metric(
                "Total Return",
                f"{metrics.total_return:.2%}",
                help="Total return since inception"
            )

        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.sharpe_ratio:.2f}",
                help="Risk-adjusted return measure"
            )

        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics.max_drawdown:.2%}",
                help="Maximum portfolio decline"
            )

        with col5:
            health_color = {
                "Healthy": "🟢",
                "Warning": "🟡",
                "Critical": "🔴"
            }
            st.metric(
                "System Health",
                f"{health_color.get(metrics.system_health, '⚪')} {metrics.system_health}",
                help="Overall system health status"
            )

        # Charts
        figures = self.visualizer.create_main_dashboard(metrics, portfolio_data)

        # Portfolio and Risk Charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(figures['portfolio_chart'], use_container_width=True)

        with col2:
            st.plotly_chart(figures['performance_metrics'], use_container_width=True)

        # Position and Strategy Analysis
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(figures['position_analysis'], use_container_width=True)

        with col2:
            st.plotly_chart(figures['strategy_performance'], use_container_width=True)

        # Risk and Trading Activity
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(figures['risk_metrics'], use_container_width=True)

        with col2:
            st.plotly_chart(figures['trading_activity'], use_container_width=True)

        # Detailed tables
        self._display_detailed_tables(portfolio_data)

    def _display_detailed_tables(self, portfolio_data: Dict[str, Any]):
        """Display detailed data tables"""

        tab1, tab2, tab3 = st.tabs(["Current Positions", "Recent Trades", "Strategy Performance"])

        with tab1:
            positions = portfolio_data.get('positions', {})
            if positions:
                positions_df = pd.DataFrame([
                    {
                        'Symbol': symbol,
                        'Quantity': pos.get('quantity', 0),
                        'Market Value': pos.get('market_value', 0),
                        'Unrealized P&L': pos.get('unrealized_pnl', 0),
                        'Strategy': pos.get('strategy', 'N/A')
                    }
                    for symbol, pos in positions.items()
                ])
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("No active positions")

        with tab2:
            # Recent trades would come from the trading system
            st.info("Recent trades data would be displayed here")

        with tab3:
            # Strategy performance breakdown
            st.info("Strategy performance breakdown would be displayed here")

    def _get_sample_data(self) -> Dict[str, Any]:
        """Get sample data for demonstration"""
        # This would be replaced with real data from the trading system
        return {
            'portfolio_value': 1050000.0,
            'daily_pnl': 2500.0,
            'daily_return': 0.0024,
            'positions_count': 8,
            'active_strategies': 4,
            'trades_count': 12,
            'drawdown': -0.03,
            'volatility': 0.15,
            'sharpe_ratio': 1.2,
            'system_latency': 45,
            'error_rate': 0.001,
            'positions': {
                'AAPL': {'quantity': 100, 'market_value': 15000, 'unrealized_pnl': 500, 'strategy': 'momentum'},
                'GOOGL': {'quantity': 50, 'market_value': 12000, 'unrealized_pnl': -200, 'strategy': 'mean_reversion'},
                'TSLA': {'quantity': 75, 'market_value': 18000, 'unrealized_pnl': 800, 'strategy': 'volatility'},
            }
        }

    def _export_performance_report(self):
        """Export comprehensive performance report"""
        st.success("Performance report exported successfully!")

    def _export_trade_log(self):
        """Export trade log"""
        st.success("Trade log exported successfully!")

def run_dashboard():
    """Run the performance monitoring dashboard"""
    dashboard = StreamlitDashboard()
    dashboard.run()

if __name__ == "__main__":
    # To run the dashboard:
    # streamlit run performance_monitoring_dashboard.py
    run_dashboard()