# Trading System Visualization Components

This directory contains various visualization components for the trading system:

## Components

### 1. Performance Dashboard (`performance_dashboard.py`)
- Real-time system monitoring dashboard using Tkinter
- Displays system health metrics, resource usage, and alerts
- Live updating charts for performance metrics

### 2. Agent Visualizers (`agent_visualizers.py`)
- Individual visualization panels for each trading agent:
  - Momentum Agent: Technical indicators (EMA, RSI, MACD)
  - Mean Reversion Agent: Bollinger Bands, Z-score analysis
  - Portfolio Allocator: Portfolio allocation and risk metrics

### 3. Unified Dashboard (`unified_dashboard.py`)
- Comprehensive dashboard integrating all system components
- Tabbed interface for different system aspects
- Both Tkinter and web-based implementations

### 4. Web Dashboard (`web_dashboard.py`)
- Flask-based web dashboard with Plotly charts
- REST API endpoints for real-time data
- Responsive HTML interface

## Installation

The required visualization libraries are included in `pyproject.toml`:

```toml
# Visualization
matplotlib = "^3.9.0"
seaborn = "^0.13.0"
plotly = "^5.22.0"
# Web Framework
flask = "^3.0.0"
# GUI
tk = "^0.1.0"
```

## Usage

### Running the Tkinter Dashboards

```bash
# Run the unified dashboard
python agents/unified_dashboard.py

# Run the performance monitoring dashboard
python agents/performance_dashboard.py
```

### Running the Web Dashboard

```bash
# Run the web dashboard
python agents/web_dashboard.py
```

Then open your browser to `http://localhost:5000`

### Demo Scripts

```bash
# Run the visualization demo
python examples/visualization_demo.py
```

## Features

### Performance Monitoring
- Real-time system resource monitoring (CPU, memory, disk)
- Latency tracking for system components
- Alert monitoring and management
- P&L tracking and portfolio monitoring

### Agent-Specific Visualizations
- Technical indicator charts for momentum strategies
- Statistical analysis charts for mean reversion strategies
- Portfolio allocation and risk metrics visualization

### Interactive Elements
- Live updating charts
- Color-coded health indicators
- Interactive Plotly charts in web dashboard
- Tabbed interface for different system components

## Customization

To customize the visualizations:

1. Modify the chart styles in each visualizer class
2. Add new metrics to the dashboard components
3. Extend the web API endpoints for additional data
4. Customize the color schemes and layouts

## Integration

To integrate with the live trading system:

1. Connect the `PerformanceMonitoringAgent` to the `PerformanceDashboard`
2. Feed real market data to the agent visualizers
3. Update the web dashboard API endpoints with live data sources
4. Add real-time data streaming for live updates