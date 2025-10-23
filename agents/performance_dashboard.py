"""
Performance Dashboard for Performance Monitoring Agent

This module provides real-time visualization capabilities for the Performance Monitoring Agent,
including live charts for system metrics, P&L tracking, and latency monitoring.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """
    Real-time performance dashboard for visualizing system metrics
    """
    
    def __init__(self, performance_agent=None):
        self.performance_agent = performance_agent
        self.fig = None
        self.canvas = None
        self.root = None
        self.animation = None
        self.is_running = False
        
        # Data storage for charts
        self.pnl_history = []
        self.latency_history = {}
        self.resource_history = []
        self.alert_history = []
        
        # Initialize the dashboard
        self._setup_dashboard()
        
    def _setup_dashboard(self):
        """Setup the dashboard layout and charts"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Trading System Performance Dashboard")
        self.root.geometry("1200x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # System Health Tab
        system_frame = ttk.Frame(notebook)
        notebook.add(system_frame, text="System Health")
        self._create_system_health_tab(system_frame)
        
        # Performance Metrics Tab
        performance_frame = ttk.Frame(notebook)
        notebook.add(performance_frame, text="Performance Metrics")
        self._create_performance_metrics_tab(performance_frame)
        
        # P&L Tracking Tab
        pnl_frame = ttk.Frame(notebook)
        notebook.add(pnl_frame, text="P&L Tracking")
        self._create_pnl_tracking_tab(pnl_frame)
        
        # Alerts Tab
        alerts_frame = ttk.Frame(notebook)
        notebook.add(alerts_frame, text="Alerts")
        self._create_alerts_tab(alerts_frame)
        
    def _create_system_health_tab(self, parent):
        """Create system health monitoring charts"""
        # Create figure with subplots
        fig = Figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Resource usage chart
        self.resource_ax = fig.add_subplot(gs[0, 0])
        self.resource_ax.set_title("System Resource Usage")
        self.resource_ax.set_ylabel("Percentage (%)")
        
        # Health score chart
        self.health_ax = fig.add_subplot(gs[0, 1])
        self.health_ax.set_title("System Health Score")
        self.health_ax.set_ylabel("Score")
        
        # Process count chart
        self.process_ax = fig.add_subplot(gs[1, 0])
        self.process_ax.set_title("Active Processes")
        self.process_ax.set_ylabel("Count")
        
        # Component status chart
        self.component_ax = fig.add_subplot(gs[1, 1])
        self.component_ax.set_title("Component Status")
        
        # Add figure to tkinter canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.system_fig = fig
        
    def _create_performance_metrics_tab(self, parent):
        """Create performance metrics charts"""
        # Create figure with subplots
        fig = Figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Latency chart
        self.latency_ax = fig.add_subplot(gs[0, :])
        self.latency_ax.set_title("Component Latency (p99)")
        self.latency_ax.set_ylabel("Latency (ms)")
        
        # Throughput chart
        self.throughput_ax = fig.add_subplot(gs[1, 0])
        self.throughput_ax.set_title("System Throughput")
        self.throughput_ax.set_ylabel("Operations/sec")
        
        # Memory usage chart
        self.memory_ax = fig.add_subplot(gs[1, 1])
        self.memory_ax.set_title("Memory Usage")
        self.memory_ax.set_ylabel("Usage (GB)")
        
        # Add figure to tkinter canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.performance_fig = fig
        
    def _create_pnl_tracking_tab(self, parent):
        """Create P&L tracking charts"""
        # Create figure with subplots
        fig = Figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 1, hspace=0.3)
        
        # Cumulative P&L chart
        self.pnl_ax = fig.add_subplot(gs[0, :])
        self.pnl_ax.set_title("Cumulative P&L")
        self.pnl_ax.set_ylabel("P&L ($)")
        self.pnl_ax.set_xlabel("Time")
        
        # Position tracking chart
        self.position_ax = fig.add_subplot(gs[1, :])
        self.position_ax.set_title("Position Tracking")
        self.position_ax.set_ylabel("Position Size")
        self.position_ax.set_xlabel("Time")
        
        # Add figure to tkinter canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.pnl_fig = fig
        
    def _create_alerts_tab(self, parent):
        """Create alerts monitoring"""
        # Create text widget for alerts
        self.alerts_text = tk.Text(parent, wrap=tk.WORD, height=20)
        self.alerts_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.alerts_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.alerts_text.configure(yscrollcommand=scrollbar.set)
        
        # Add alert summary
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.alert_count_label = ttk.Label(summary_frame, text="Total Alerts: 0")
        self.alert_count_label.pack(side=tk.LEFT)
        
        self.critical_alert_label = ttk.Label(summary_frame, text="Critical: 0", foreground="red")
        self.critical_alert_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.warning_alert_label = ttk.Label(summary_frame, text="Warnings: 0", foreground="orange")
        self.warning_alert_label.pack(side=tk.LEFT, padx=(20, 0))
        
    def update_system_health_charts(self):
        """Update system health charts with current data"""
        if not self.performance_agent:
            return
            
        # Get current dashboard data
        dashboard_data = self.performance_agent.get_dashboard_data()
        
        # Update resource usage chart
        self.resource_ax.clear()
        self.resource_ax.set_title("System Resource Usage")
        self.resource_ax.set_ylabel("Percentage (%)")
        
        resources = dashboard_data.resource_usage
        if resources:
            labels = list(resources.keys())
            values = [resources[key] for key in labels if key != 'process_count']
            bars = self.resource_ax.bar(labels[:-1], values, color=['blue', 'green', 'red'])
            self.resource_ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                self.resource_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Update health score chart
        self.health_ax.clear()
        self.health_ax.set_title("System Health Score")
        self.health_ax.set_ylabel("Score")
        
        health_data = [dashboard_data.system_health.get('health_score', 0)]
        bars = self.health_ax.bar(['Health Score'], health_data, 
                                 color='green' if health_data[0] > 80 else 'red')
        self.health_ax.set_ylim(0, 100)
        
        # Add value label
        for bar, value in zip(bars, health_data):
            self.health_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value:.1f}', ha='center', va='bottom')
        
        # Update process count chart
        self.process_ax.clear()
        self.process_ax.set_title("Active Processes")
        self.process_ax.set_ylabel("Count")
        
        process_count = resources.get('process_count', 0)
        bars = self.process_ax.bar(['Processes'], [process_count], color='blue')
        self.process_ax.set_ylim(0, max(100, process_count * 1.2))
        
        # Add value label
        for bar, value in zip(bars, [process_count]):
            self.process_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value}', ha='center', va='bottom')
        
        # Update component status chart
        self.component_ax.clear()
        self.component_ax.set_title("Component Status")
        
        # Mock component status data (in a real implementation, this would come from the agent)
        components = ['Data Ingestion', 'Signal Generation', 'Risk Management', 'Execution']
        status_values = [95, 87, 92, 89]  # Mock health scores
        
        bars = self.component_ax.bar(components, status_values, 
                                   color=['green' if x > 80 else 'orange' if x > 60 else 'red' 
                                         for x in status_values])
        self.component_ax.set_ylabel("Health Score")
        self.component_ax.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, status_values):
            self.component_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                 f'{value}', ha='center', va='bottom')
        
        self.system_fig.canvas.draw()
        
    def update_performance_metrics_charts(self):
        """Update performance metrics charts with current data"""
        if not self.performance_agent:
            return
            
        # Get current dashboard data
        dashboard_data = self.performance_agent.get_dashboard_data()
        
        # Update latency chart
        self.latency_ax.clear()
        self.latency_ax.set_title("Component Latency (p99)")
        self.latency_ax.set_ylabel("Latency (ms)")
        
        latency_data = dashboard_data.latency_summary
        if latency_data:
            components = list(latency_data.keys())
            p99_values = [latency_data[comp].p99 for comp in components]
            
            bars = self.latency_ax.bar(components, p99_values, color='blue')
            self.latency_ax.set_ylim(0, max(100, max(p99_values) * 1.2) if p99_values else 100)
            
            # Add value labels
            for bar, value in zip(bars, p99_values):
                self.latency_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                   f'{value:.1f}ms', ha='center', va='bottom', rotation=45)
        
        # Update throughput chart (mock data)
        self.throughput_ax.clear()
        self.throughput_ax.set_title("System Throughput")
        self.throughput_ax.set_ylabel("Operations/sec")
        
        # Mock throughput data
        operations = ['Orders', 'Signals', 'Data Points']
        throughput_values = [120, 45, 320]  # Mock values
        
        bars = self.throughput_ax.bar(operations, throughput_values, color=['red', 'blue', 'green'])
        self.throughput_ax.set_ylim(0, max(400, max(throughput_values) * 1.2) if throughput_values else 400)
        
        # Add value labels
        for bar, value in zip(bars, throughput_values):
            self.throughput_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                  f'{value}', ha='center', va='bottom')
        
        # Update memory usage chart
        self.memory_ax.clear()
        self.memory_ax.set_title("Memory Usage")
        self.memory_ax.set_ylabel("Usage (GB)")
        
        resources = dashboard_data.resource_usage
        memory_gb = resources.get('memory_used_gb', 0)
        bars = self.memory_ax.bar(['Memory'], [memory_gb], color='green')
        self.memory_ax.set_ylim(0, max(8, memory_gb * 1.2))
        
        # Add value label
        for bar, value in zip(bars, [memory_gb]):
            self.memory_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                              f'{value:.2f}GB', ha='center', va='bottom')
        
        self.performance_fig.canvas.draw()
        
    def update_pnl_tracking_charts(self):
        """Update P&L tracking charts with current data"""
        if not self.performance_agent:
            return
            
        # Get current dashboard data
        dashboard_data = self.performance_agent.get_dashboard_data()
        
        # Update P&L chart
        self.pnl_ax.clear()
        self.pnl_ax.set_title("Cumulative P&L")
        self.pnl_ax.set_ylabel("P&L ($)")
        self.pnl_ax.set_xlabel("Time")
        
        # Add data point to history
        timestamp = datetime.now()
        total_pnl = dashboard_data.pnl_summary.get('total_pnl', 0)
        self.pnl_history.append((timestamp, total_pnl))
        
        # Keep only last 100 data points
        if len(self.pnl_history) > 100:
            self.pnl_history.pop(0)
        
        if self.pnl_history:
            timestamps, pnls = zip(*self.pnl_history)
            self.pnl_ax.plot(timestamps, pnls, marker='o', linewidth=2, markersize=4)
            self.pnl_ax.set_ylim(min(pnls) * 1.1, max(pnls) * 1.1 if max(pnls) > 0 else 0)
            
            # Format x-axis as time
            self.pnl_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            self.pnl_ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            plt.setp(self.pnl_ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Update position tracking chart
        self.position_ax.clear()
        self.position_ax.set_title("Position Tracking")
        self.position_ax.set_ylabel("Position Size")
        self.position_ax.set_xlabel("Time")
        
        # Mock position data (in a real implementation, this would come from positions)
        positions = dashboard_data.pnl_summary.get('positions', {})
        if positions:
            symbols = list(positions.keys())
            position_sizes = [abs(pos.get('quantity', 0)) for pos in positions.values()]
            
            bars = self.position_ax.bar(symbols, position_sizes, color='blue')
            self.position_ax.set_ylim(0, max(1000, max(position_sizes) * 1.2) if position_sizes else 1000)
            
            # Add value labels
            for bar, value in zip(bars, position_sizes):
                self.position_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                    f'{value}', ha='center', va='bottom')
        else:
            # Show message if no positions
            self.position_ax.text(0.5, 0.5, 'No active positions', 
                                horizontalalignment='center', verticalalignment='center',
                                transform=self.position_ax.transAxes, fontsize=12)
        
        self.pnl_fig.canvas.draw()
        
    def update_alerts_tab(self):
        """Update alerts tab with current alerts"""
        if not self.performance_agent:
            return
            
        # Get current dashboard data
        dashboard_data = self.performance_agent.get_dashboard_data()
        
        # Clear existing text
        self.alerts_text.delete(1.0, tk.END)
        
        # Add new alerts
        for alert in dashboard_data.alerts:
            alert_text = f"[{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            if alert.severity == "critical":
                alert_text += "[RED] CRITICAL: "
            elif alert.severity == "error":
                alert_text += "[X] ERROR: "
            elif alert.severity == "warning":
                alert_text += "[WARN] WARNING: "
            else:
                alert_text += "ℹ️ INFO: "
            
            alert_text += f"{alert.message}\n"
            alert_text += f"   Component: {alert.component} | Value: {alert.value:.2f} | Threshold: {alert.threshold:.2f}\n\n"
            
            self.alerts_text.insert(tk.END, alert_text)
            
            # Color code based on severity
            if alert.severity == "critical":
                start_idx = self.alerts_text.index("end-2c linestart")
                end_idx = self.alerts_text.index("end-2c")
                self.alerts_text.tag_add("critical", start_idx, end_idx)
                self.alerts_text.tag_config("critical", foreground="red")
            elif alert.severity == "error":
                start_idx = self.alerts_text.index("end-2c linestart")
                end_idx = self.alerts_text.index("end-2c")
                self.alerts_text.tag_add("error", start_idx, end_idx)
                self.alerts_text.tag_config("error", foreground="darkred")
            elif alert.severity == "warning":
                start_idx = self.alerts_text.index("end-2c linestart")
                end_idx = self.alerts_text.index("end-2c")
                self.alerts_text.tag_add("warning", start_idx, end_idx)
                self.alerts_text.tag_config("warning", foreground="orange")
        
        # Scroll to end
        self.alerts_text.see(tk.END)
        
        # Update alert summary
        alert_summary = self.performance_agent.alert_manager.get_alert_summary()
        self.alert_count_label.config(text=f"Total Alerts: {alert_summary.get('total_alerts', 0)}")
        self.critical_alert_label.config(text=f"Critical: {alert_summary.get('severity_breakdown', {}).get('critical', 0)}")
        self.warning_alert_label.config(text=f"Warnings: {alert_summary.get('severity_breakdown', {}).get('warning', 0)}")
        
    def animate(self, frame):
        """Animation function to update all charts"""
        try:
            self.update_system_health_charts()
            self.update_performance_metrics_charts()
            self.update_pnl_tracking_charts()
            self.update_alerts_tab()
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            
    def start_dashboard(self):
        """Start the dashboard animation"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start animation
        self.animation = FuncAnimation(self.system_fig, self.animate, interval=5000, blit=False)
        
        # Start tkinter event loop
        self.root.mainloop()
        
    def stop_dashboard(self):
        """Stop the dashboard"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        if self.root:
            self.root.quit()

# Example usage
if __name__ == "__main__":
    # This would normally be connected to a real PerformanceMonitoringAgent
    dashboard = PerformanceDashboard()
    dashboard.start_dashboard()