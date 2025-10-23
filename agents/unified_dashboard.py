"""
Unified Trading System Dashboard

This module provides a comprehensive dashboard that integrates all agent visualizations
and system monitoring into a single interface.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Import our visualization modules
from agents.performance_dashboard import PerformanceDashboard
from agents.agent_visualizers import (
    MomentumAgentVisualizer, 
    MeanReversionAgentVisualizer, 
    PortfolioAllocatorVisualizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTradingDashboard:
    """
    Unified dashboard for the entire trading system
    """
    
    def __init__(self):
        self.root = None
        self.notebook = None
        self.performance_dashboard = None
        self.agent_visualizers = {}
        self.is_running = False
        
        # Sample data for demonstration
        self.sample_data = self._generate_sample_data()
        
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample data for demonstration"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Sample market data
        market_data = [
            {
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                'open': 150 + np.random.normal(0, 1),
                'high': 152 + np.random.normal(0, 1),
                'low': 148 + np.random.normal(0, 1),
                'close': 150 + np.random.normal(0, 1),
                'volume': 1000000 + np.random.randint(-100000, 100000)
            }
            for i in range(30, 0, -1)
        ]
        
        # Sample signals
        signals = [
            {
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                'signal_type': 'buy' if np.random.random() > 0.5 else 'sell',
                'value': np.random.uniform(-1, 1),
                'confidence': np.random.uniform(0.5, 1.0)
            }
            for i in range(30, 0, -1)
        ][::5]  # Every 5th day
        
        # Sample indicator data
        indicator_data = {
            'ema_fast': [150 + np.random.normal(0, 0.5) for _ in range(30)],
            'ema_slow': [150 + np.random.normal(0, 0.3) for _ in range(30)],
            'rsi': [50 + np.random.normal(0, 15) for _ in range(30)],
            'macd_line': [0 + np.random.normal(0, 0.1) for _ in range(30)],
            'signal_line': [0 + np.random.normal(0, 0.05) for _ in range(30)],
            'histogram': [0 + np.random.normal(0, 0.1) for _ in range(30)],
            'upper_band': [155 + np.random.normal(0, 1) for _ in range(30)],
            'middle_band': [150 + np.random.normal(0, 0.5) for _ in range(30)],
            'lower_band': [145 + np.random.normal(0, 1) for _ in range(30)],
            'z_score': [0 + np.random.normal(0, 1) for _ in range(30)],
            'spread': [0 + np.random.normal(0, 0.5) for _ in range(30)],
            'spread_std': [1.0 for _ in range(30)]
        }
        
        # Sample portfolio data
        portfolio_data = {
            'positions': {
                'AAPL': {'quantity': 100, 'market_value': 15000, 'unrealized_pnl': 500},
                'GOOGL': {'quantity': 50, 'market_value': 75000, 'unrealized_pnl': -200},
                'TSLA': {'quantity': 200, 'market_value': 40000, 'unrealized_pnl': 1200}
            },
            'total_value': 130000,
            'cash': 20000,
            'unrealized_pnl': 1500
        }
        
        # Sample allocation history
        allocation_history = [
            {
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                'total_value': 130000 - i * 100,
                'cash': 20000 - i * 50,
                'positions_value': 110000 - i * 50
            }
            for i in range(30, 0, -1)
        ]
        
        return {
            'market_data': market_data,
            'signals': signals,
            'indicator_data': indicator_data,
            'portfolio_data': portfolio_data,
            'allocation_history': allocation_history
        }
        
    def setup_dashboard(self):
        """Setup the unified dashboard"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Unified Trading System Dashboard")
        self.root.geometry("1400x900")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # System Overview Tab
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="System Overview")
        self._create_overview_tab(overview_frame)
        
        # Performance Monitoring Tab
        performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(performance_frame, text="Performance Monitoring")
        self._create_performance_tab(performance_frame)
        
        # Momentum Agent Tab
        momentum_frame = ttk.Frame(self.notebook)
        self.notebook.add(momentum_frame, text="Momentum Agent")
        self._create_momentum_tab(momentum_frame)
        
        # Mean Reversion Agent Tab
        mean_reversion_frame = ttk.Frame(self.notebook)
        self.notebook.add(mean_reversion_frame, text="Mean Reversion Agent")
        self._create_mean_reversion_tab(mean_reversion_frame)
        
        # Portfolio Allocator Tab
        portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(portfolio_frame, text="Portfolio Allocator")
        self._create_portfolio_tab(portfolio_frame)
        
        # System Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(control_frame, text="Start Monitoring", 
                                      command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Monitoring", 
                                     command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready")
        status_bar = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.RIGHT, padx=(10, 0))
        
    def _create_overview_tab(self, parent):
        """Create system overview tab"""
        # Create a grid layout
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)
        
        # System status panel
        status_frame = ttk.LabelFrame(parent, text="System Status", padding=(10, 5))
        status_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))
        
        # Health score
        health_frame = ttk.Frame(status_frame)
        health_frame.pack(fill=tk.X, pady=5)
        ttk.Label(health_frame, text="System Health:").pack(side=tk.LEFT)
        self.health_label = ttk.Label(health_frame, text="95%", foreground="green")
        self.health_label.pack(side=tk.RIGHT)
        
        # Active agents
        agents_frame = ttk.Frame(status_frame)
        agents_frame.pack(fill=tk.X, pady=5)
        ttk.Label(agents_frame, text="Active Agents:").pack(side=tk.LEFT)
        self.agents_label = ttk.Label(agents_frame, text="5/5")
        self.agents_label.pack(side=tk.RIGHT)
        
        # Current P&L
        pnl_frame = ttk.Frame(status_frame)
        pnl_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pnl_frame, text="Today's P&L:").pack(side=tk.LEFT)
        self.pnl_label = ttk.Label(pnl_frame, text="+$1,250.00", foreground="green")
        self.pnl_label.pack(side=tk.RIGHT)
        
        # Active positions
        positions_frame = ttk.Frame(status_frame)
        positions_frame.pack(fill=tk.X, pady=5)
        ttk.Label(positions_frame, text="Active Positions:").pack(side=tk.LEFT)
        self.positions_label = ttk.Label(positions_frame, text="3")
        self.positions_label.pack(side=tk.RIGHT)
        
        # Recent alerts panel
        alerts_frame = ttk.LabelFrame(parent, text="Recent Alerts", padding=(10, 5))
        alerts_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 5))
        
        # Alerts listbox
        self.alerts_listbox = tk.Listbox(alerts_frame, height=8)
        self.alerts_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Add sample alerts
        sample_alerts = [
            "[WARN] WARNING: High latency in signal generation (1250ms)",
            "ℹ️ INFO: New momentum signal generated for AAPL",
            "[OK] SUCCESS: Position updated for GOOGL",
            "[WARN] WARNING: Low confidence in mean reversion signal"
        ]
        for alert in sample_alerts:
            self.alerts_listbox.insert(tk.END, alert)
        
        # Performance summary panel
        perf_frame = ttk.LabelFrame(parent, text="Performance Summary", padding=(10, 5))
        perf_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", 
                       padx=0, pady=(5, 0))
        
        # Create a frame for performance metrics
        metrics_frame = ttk.Frame(perf_frame)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sample performance metrics
        metrics = [
            ("Total Return", "12.5%", "green"),
            ("Sharpe Ratio", "1.8", "blue"),
            ("Max Drawdown", "3.2%", "red"),
            ("Win Rate", "68%", "green"),
            ("Avg Trade", "$245", "blue"),
            ("Trades Today", "24", "black")
        ]
        
        for i, (label, value, color) in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.grid(row=row, column=col, sticky="ew", padx=5, pady=5)
            metrics_frame.grid_columnconfigure(col, weight=1)
            
            ttk.Label(metric_frame, text=label + ":").pack(side=tk.LEFT)
            ttk.Label(metric_frame, text=value, foreground=color).pack(side=tk.RIGHT)
        
    def _create_performance_tab(self, parent):
        """Create performance monitoring tab"""
        # This would integrate with the PerformanceDashboard
        # For now, we'll show a placeholder
        info_label = ttk.Label(parent, text="Performance Monitoring Dashboard\n(Real-time system metrics and alerts)", 
                              font=("Arial", 12))
        info_label.pack(expand=True)
        
        # In a real implementation, this would show:
        # - Real-time system resource usage charts
        # - Latency monitoring graphs
        # - P&L tracking charts
        # - Alert monitoring panel
        ttk.Label(parent, text="This tab would display real-time performance metrics\nwhen connected to the PerformanceMonitoringAgent", 
                  foreground="gray").pack(pady=20)
        
    def _create_momentum_tab(self, parent):
        """Create momentum agent tab"""
        # Create momentum agent visualizer
        visualizer = MomentumAgentVisualizer()
        
        # Generate sample chart
        fig = visualizer.create_technical_chart(
            self.sample_data['market_data'],
            self.sample_data['signals'],
            self.sample_data['indicator_data']
        )
        
        # Add to tkinter canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _create_mean_reversion_tab(self, parent):
        """Create mean reversion agent tab"""
        # Create mean reversion agent visualizer
        visualizer = MeanReversionAgentVisualizer()
        
        # Generate sample chart
        fig = visualizer.create_mean_reversion_chart(
            self.sample_data['market_data'],
            self.sample_data['signals'],
            self.sample_data['indicator_data']
        )
        
        # Add to tkinter canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _create_portfolio_tab(self, parent):
        """Create portfolio allocator tab"""
        # Create portfolio allocator visualizer
        visualizer = PortfolioAllocatorVisualizer()
        
        # Generate sample chart
        fig = visualizer.create_portfolio_chart(
            self.sample_data['portfolio_data'],
            self.sample_data['allocation_history']
        )
        
        # Add to tkinter canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def start_monitoring(self):
        """Start system monitoring"""
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Monitoring Active")
        
        # In a real implementation, this would start the actual monitoring
        # For now, we'll just simulate updates
        self._simulate_updates()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Monitoring Stopped")
        
    def _simulate_updates(self):
        """Simulate real-time updates"""
        if not self.is_running:
            return
            
        # Simulate updating system metrics
        import random
        health_score = random.randint(90, 100)
        self.health_label.config(text=f"{health_score}%")
        self.health_label.config(foreground="green" if health_score > 95 else "orange")
        
        # Update P&L
        pnl = random.uniform(-500, 2000)
        pnl_text = f"${pnl:+.2f}"
        self.pnl_label.config(text=pnl_text)
        self.pnl_label.config(foreground="green" if pnl > 0 else "red")
        
        # Schedule next update
        if self.is_running:
            self.root.after(5000, self._simulate_updates)  # Update every 5 seconds
            
    def run(self):
        """Run the dashboard"""
        self.setup_dashboard()
        self.root.mainloop()

# Web-based dashboard using Dash (alternative implementation)
def create_web_dashboard():
    """
    Create a web-based dashboard using Dash/Plotly
    This is an alternative to the tkinter dashboard
    """
    try:
        import dash
        from dash import dcc, html, Input, Output
        import plotly.graph_objs as go
        import plotly.express as px
        
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("Trading System Dashboard", style={'text-align': 'center'}),
            
            # System overview cards
            html.Div([
                html.Div([
                    html.H3("System Health"),
                    html.H4("95%", id="health-score", style={'color': 'green'})
                ], className="card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.H3("Active Agents"),
                    html.H4("5/5", id="active-agents")
                ], className="card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.H3("Today's P&L"),
                    html.H4("+$1,250.00", id="todays-pnl", style={'color': 'green'})
                ], className="card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.H3("Active Positions"),
                    html.H4("3", id="active-positions")
                ], className="card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'})
            ], style={'text-align': 'center'}),
            
            # Charts section
            html.Div([
                dcc.Graph(id="price-chart"),
                dcc.Graph(id="performance-chart")
            ]),
            
            # Interval for updates
            dcc.Interval(
                id='interval-component',
                interval=5000,  # 5 seconds
                n_intervals=0
            )
        ])
        
        @app.callback(
            Output('health-score', 'children'),
            Output('health-score', 'style'),
            Input('interval-component', 'n_intervals')
        )
        def update_health(n):
            import random
            score = random.randint(90, 100)
            color = 'green' if score > 95 else 'orange'
            return f"{score}%", {'color': color}
        
        return app
        
    except ImportError:
        print("Dash not installed. Install with: pip install dash")
        return None

# Example usage
if __name__ == "__main__":
    # Run the tkinter dashboard
    dashboard = UnifiedTradingDashboard()
    dashboard.run()