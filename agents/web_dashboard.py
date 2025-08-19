"""
Web-based Trading Dashboard using Flask

This module provides a web-based dashboard for the trading system using Flask and Plotly.
"""

from flask import Flask, render_template, jsonify
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

def generate_sample_data():
    """Generate sample data for the dashboard"""
    # Generate time series data
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    
    # Sample price data
    prices = [150 + np.cumsum(np.random.normal(0, 1, 30))]
    
    # Sample P&L data
    pnl_data = np.cumsum(np.random.normal(0, 100, 30))
    
    # Sample latency data
    latency_data = np.random.exponential(50, 30)  # Average 50ms
    
    return {
        'dates': dates,
        'prices': prices[0].tolist(),
        'pnl': pnl_data.tolist(),
        'latency': latency_data.tolist()
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/system_health')
def system_health():
    """API endpoint for system health data"""
    # Mock system health data
    health_data = {
        'cpu_percent': np.random.uniform(20, 80),
        'memory_percent': np.random.uniform(30, 70),
        'disk_percent': np.random.uniform(20, 60),
        'network_io': np.random.uniform(100, 1000),
        'health_score': np.random.uniform(85, 100)
    }
    return jsonify(health_data)

@app.route('/api/portfolio')
def portfolio_data():
    """API endpoint for portfolio data"""
    # Mock portfolio data
    portfolio = {
        'total_value': 130000 + np.random.uniform(-5000, 5000),
        'cash': 20000 + np.random.uniform(-2000, 2000),
        'positions': [
            {'symbol': 'AAPL', 'value': 15000, 'pnl': 500},
            {'symbol': 'GOOGL', 'value': 75000, 'pnl': -200},
            {'symbol': 'TSLA', 'value': 40000, 'pnl': 1200}
        ],
        'daily_pnl': np.random.uniform(-500, 1500)
    }
    return jsonify(portfolio)

@app.route('/api/charts/price')
def price_chart():
    """API endpoint for price chart data"""
    sample_data = generate_sample_data()
    
    # Create price chart
    trace = go.Scatter(
        x=sample_data['dates'],
        y=sample_data['prices'],
        mode='lines+markers',
        name='Price',
        line=dict(color='blue')
    )
    
    layout = go.Layout(
        title='Asset Price Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price ($)'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    return graphJSON

@app.route('/api/charts/pnl')
def pnl_chart():
    """API endpoint for P&L chart data"""
    sample_data = generate_sample_data()
    
    # Create P&L chart
    trace = go.Scatter(
        x=sample_data['dates'],
        y=sample_data['pnl'],
        mode='lines+markers',
        name='P&L',
        line=dict(color='green' if sample_data['pnl'][-1] > 0 else 'red')
    )
    
    layout = go.Layout(
        title='Portfolio P&L Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='P&L ($)'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    return graphJSON

@app.route('/api/charts/latency')
def latency_chart():
    """API endpoint for latency chart data"""
    sample_data = generate_sample_data()
    
    # Create latency chart
    trace = go.Scatter(
        x=sample_data['dates'],
        y=sample_data['latency'],
        mode='lines+markers',
        name='Latency',
        line=dict(color='orange')
    )
    
    layout = go.Layout(
        title='System Latency Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Latency (ms)'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    return graphJSON

if __name__ == '__main__':
    app.run(debug=True, port=5000)