"""
Advanced Trading Dashboard
Real-time monitoring and visualization using Dash and Plotly
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json

# Dashboard libraries
try:
    import dash
    from dash import dcc, html, Input, Output, callback
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    # Create dummy objects to prevent NameError
    class DummyGo:
        Figure = object
    class DummyPx:
        pass
    class DummyHtml:
        Div = object
        Table = object
        Tr = object
        Td = object
        Th = object
        Thead = object
        Tbody = object
        H1 = object
        H2 = object
        H3 = object
        P = object
        Span = object
        A = object
        Button = object
    class DummyDcc:
        Graph = object
        Dropdown = object
        Interval = object
        Store = object
        Location = object
    go = DummyGo()
    px = DummyPx()
    html = DummyHtml()
    dcc = DummyDcc()
    def callback(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    Input = object
    Output = object

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False

from config.logging_config import get_logger

logger = get_logger(__name__)

class TradingDashboard:
    """Advanced trading dashboard for real-time monitoring"""
    
    def __init__(self, port: int = 8050):
        self.port = port
        self.app = None
        self.data_sources = {}
        self.update_interval = 30000  # 30 seconds
        
        if DASH_AVAILABLE:
            self._initialize_dashboard()
    
    def _initialize_dashboard(self):
        """Initialize the Dash application"""
        
        self.app = dash.Dash(__name__)
        
        # Define the layout
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("OPTIONS_BOT Trading Dashboard", 
                       style={'textAlign': 'center', 'color': '#2E86AB'}),
                html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       id='last-update', style={'textAlign': 'center'})
            ], style={'backgroundColor': '#F8F9FA', 'padding': '20px'}),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Symbol:"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[
                            {'label': 'SPY', 'value': 'SPY'},
                            {'label': 'QQQ', 'value': 'QQQ'},
                            {'label': 'AAPL', 'value': 'AAPL'},
                            {'label': 'MSFT', 'value': 'MSFT'},
                            {'label': 'TSLA', 'value': 'TSLA'}
                        ],
                        value='SPY'
                    )
                ], className='four columns'),
                
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id='timerange-dropdown',
                        options=[
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '5 Days', 'value': '5d'},
                            {'label': '1 Month', 'value': '1mo'},
                            {'label': '3 Months', 'value': '3mo'}
                        ],
                        value='5d'
                    )
                ], className='four columns'),
                
                html.Div([
                    html.Button('Refresh Data', id='refresh-btn', n_clicks=0,
                               style={'backgroundColor': '#28A745', 'color': 'white'})
                ], className='four columns')
            ], className='row', style={'padding': '20px'}),
            
            # Key Metrics Cards
            html.Div([
                html.Div([
                    html.H4("Portfolio Value", style={'textAlign': 'center'}),
                    html.H2("$100,000", id='portfolio-value', style={'textAlign': 'center', 'color': '#28A745'})
                ], className='three columns', style={'backgroundColor': '#F8F9FA', 'padding': '20px', 'margin': '10px'}),
                
                html.Div([
                    html.H4("Daily P&L", style={'textAlign': 'center'}),
                    html.H2("+$1,250", id='daily-pnl', style={'textAlign': 'center', 'color': '#28A745'})
                ], className='three columns', style={'backgroundColor': '#F8F9FA', 'padding': '20px', 'margin': '10px'}),
                
                html.Div([
                    html.H4("Win Rate", style={'textAlign': 'center'}),
                    html.H2("68%", id='win-rate', style={'textAlign': 'center', 'color': '#17A2B8'})
                ], className='three columns', style={'backgroundColor': '#F8F9FA', 'padding': '20px', 'margin': '10px'}),
                
                html.Div([
                    html.H4("Sharpe Ratio", style={'textAlign': 'center'}),
                    html.H2("1.45", id='sharpe-ratio', style={'textAlign': 'center', 'color': '#6F42C1'})
                ], className='three columns', style={'backgroundColor': '#F8F9FA', 'padding': '20px', 'margin': '10px'})
            ], className='row'),
            
            # Main Charts Row
            html.Div([
                # Price Chart
                html.Div([
                    dcc.Graph(id='price-chart')
                ], className='eight columns'),
                
                # Technical Indicators
                html.Div([
                    dcc.Graph(id='technical-indicators')
                ], className='four columns')
            ], className='row'),
            
            # Secondary Charts Row
            html.Div([
                # Volatility Chart
                html.Div([
                    dcc.Graph(id='volatility-chart')
                ], className='six columns'),
                
                # Risk Metrics
                html.Div([
                    dcc.Graph(id='risk-metrics')
                ], className='six columns')
            ], className='row'),
            
            # ML Predictions and Signals
            html.Div([
                # ML Predictions
                html.Div([
                    html.H3("ML Predictions"),
                    dcc.Graph(id='ml-predictions')
                ], className='six columns'),
                
                # Trading Signals
                html.Div([
                    html.H3("Current Signals"),
                    html.Div(id='trading-signals')
                ], className='six columns')
            ], className='row'),
            
            # Positions Table
            html.Div([
                html.H3("Current Positions"),
                html.Div(id='positions-table')
            ], style={'padding': '20px'}),
            
            # Risk Alerts
            html.Div([
                html.H3("Risk Alerts"),
                html.Div(id='risk-alerts')
            ], style={'padding': '20px'}),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
            
            # CSS Styles
            html.Link(
                rel='stylesheet',
                href='https://codepen.io/chriddyp/pen/bWLwgP.css'
            )
        ])
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks"""
        
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('technical-indicators', 'figure'),
             Output('volatility-chart', 'figure'),
             Output('risk-metrics', 'figure'),
             Output('ml-predictions', 'figure'),
             Output('trading-signals', 'children'),
             Output('positions-table', 'children'),
             Output('risk-alerts', 'children'),
             Output('portfolio-value', 'children'),
             Output('daily-pnl', 'children'),
             Output('win-rate', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('last-update', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-btn', 'n_clicks'),
             Input('symbol-dropdown', 'value'),
             Input('timerange-dropdown', 'value')]
        )
        def update_dashboard(n_intervals, n_clicks, symbol, timerange):
            """Update all dashboard components"""
            
            try:
                # Get fresh data
                market_data = self._get_market_data(symbol, timerange)
                portfolio_data = self._get_portfolio_data()
                risk_data = self._get_risk_data()
                ml_data = self._get_ml_data(symbol)
                
                # Create charts
                price_chart = self._create_price_chart(market_data, symbol)
                tech_chart = self._create_technical_chart(market_data)
                vol_chart = self._create_volatility_chart(market_data)
                risk_chart = self._create_risk_chart(risk_data)
                ml_chart = self._create_ml_chart(ml_data)
                
                # Create components
                signals = self._create_signals_component(market_data)
                positions = self._create_positions_table(portfolio_data)
                alerts = self._create_alerts_component(risk_data)
                
                # Update metrics
                portfolio_value = f"${portfolio_data.get('total_value', 100000):,.0f}"
                daily_pnl = f"{portfolio_data.get('daily_pnl', 1250):+,.0f}"
                win_rate = f"{portfolio_data.get('win_rate', 68):.0f}%"
                sharpe = f"{portfolio_data.get('sharpe_ratio', 1.45):.2f}"
                
                last_update = f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                return (price_chart, tech_chart, vol_chart, risk_chart, ml_chart,
                       signals, positions, alerts, portfolio_value, daily_pnl, 
                       win_rate, sharpe, last_update)
                
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                return self._get_error_components()
    
    def _create_price_chart(self, data: Dict, symbol: str) -> go.Figure:
        """Create price chart with volume"""
        
        try:
            df = data.get('price_data', pd.DataFrame())
            
            if df.empty:
                return go.Figure().add_annotation(text="No data available", 
                                                 x=0.5, y=0.5, showarrow=False)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f'{symbol} Price', 'Volume'],
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Moving averages
            if 'SMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange', width=2)
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_50'],
                        name='SMA 50',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f'{symbol} Price Analysis',
                xaxis_rangeslider_visible=False,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Price chart error: {e}")
            return go.Figure().add_annotation(text="Error creating chart", 
                                             x=0.5, y=0.5, showarrow=False)
    
    def _create_technical_chart(self, data: Dict) -> go.Figure:
        """Create technical indicators chart"""
        
        try:
            indicators = data.get('technical_indicators', {})
            
            fig = go.Figure()
            
            # RSI
            if 'rsi_history' in data:
                rsi_data = data['rsi_history']
                fig.add_trace(
                    go.Scatter(
                        y=rsi_data,
                        name='RSI',
                        line=dict(color='purple')
                    )
                )
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            
            fig.update_layout(
                title='Technical Indicators',
                yaxis_title='RSI',
                height=300
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Technical chart error: {e}")
            return go.Figure()
    
    def _create_volatility_chart(self, data: Dict) -> go.Figure:
        """Create volatility analysis chart"""
        
        try:
            fig = go.Figure()
            
            # Realized volatility
            if 'volatility_history' in data:
                vol_data = data['volatility_history']
                fig.add_trace(
                    go.Scatter(
                        y=vol_data,
                        name='Realized Vol',
                        line=dict(color='red')
                    )
                )
            
            # VIX if available
            if 'vix_data' in data:
                vix_data = data['vix_data']
                fig.add_trace(
                    go.Scatter(
                        y=vix_data,
                        name='VIX',
                        line=dict(color='orange')
                    )
                )
            
            fig.update_layout(
                title='Volatility Analysis',
                yaxis_title='Volatility (%)',
                height=300
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Volatility chart error: {e}")
            return go.Figure()
    
    def _create_risk_chart(self, data: Dict) -> go.Figure:
        """Create risk metrics chart"""
        
        try:
            risk_metrics = data.get('risk_metrics', {})
            
            # Create gauge chart for risk level
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_metrics.get('overall_risk_score', 50),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Level"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(height=300)
            
            return fig
            
        except Exception as e:
            logger.error(f"Risk chart error: {e}")
            return go.Figure()
    
    def _create_ml_chart(self, data: Dict) -> go.Figure:
        """Create ML predictions chart"""
        
        try:
            fig = go.Figure()
            
            predictions = data.get('predictions', {})
            
            if 'price_prediction' in predictions:
                current_price = predictions.get('current_price', 100)
                predicted_price = predictions.get('price_prediction', current_price)
                confidence = predictions.get('confidence', 0.5)
                
                # Create prediction bar chart
                fig.add_trace(
                    go.Bar(
                        x=['Current', 'Predicted'],
                        y=[current_price, predicted_price],
                        name='Price Prediction',
                        marker_color=['blue', 'green' if predicted_price > current_price else 'red']
                    )
                )
                
                fig.update_layout(
                    title=f'ML Price Prediction (Confidence: {confidence:.1%})',
                    yaxis_title='Price ($)',
                    height=300
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"ML chart error: {e}")
            return go.Figure()
    
    def _create_signals_component(self, data: Dict) -> html.Div:
        """Create trading signals component"""
        
        try:
            signals = data.get('signals', {})
            
            signal_cards = []
            
            for signal_name, signal_data in signals.items():
                signal_value = signal_data.get('signal', 'NEUTRAL')
                confidence = signal_data.get('confidence', 0.5)
                
                color = {'BULLISH': 'green', 'BEARISH': 'red', 'NEUTRAL': 'gray'}.get(signal_value, 'gray')
                
                card = html.Div([
                    html.H5(signal_name.replace('_', ' ').title()),
                    html.P(signal_value, style={'color': color, 'fontWeight': 'bold'}),
                    html.P(f"Confidence: {confidence:.1%}")
                ], style={'backgroundColor': '#F8F9FA', 'padding': '10px', 'margin': '5px'})
                
                signal_cards.append(card)
            
            return html.Div(signal_cards)
            
        except Exception as e:
            logger.error(f"Signals component error: {e}")
            return html.Div("Error loading signals")
    
    def _create_positions_table(self, data: Dict) -> html.Table:
        """Create positions table"""
        
        try:
            positions = data.get('positions', [])
            
            if not positions:
                return html.P("No open positions")
            
            headers = ['Symbol', 'Quantity', 'Entry Price', 'Current Price', 'P&L', 'P&L %']
            
            table_rows = []
            
            # Header row
            table_rows.append(html.Tr([html.Th(header) for header in headers]))
            
            # Position rows
            for pos in positions:
                pnl_color = 'green' if pos.get('pnl', 0) > 0 else 'red'
                
                row = html.Tr([
                    html.Td(pos.get('symbol', 'N/A')),
                    html.Td(pos.get('quantity', 0)),
                    html.Td(f"${pos.get('entry_price', 0):.2f}"),
                    html.Td(f"${pos.get('current_price', 0):.2f}"),
                    html.Td(f"${pos.get('pnl', 0):.2f}", style={'color': pnl_color}),
                    html.Td(f"{pos.get('pnl_percent', 0):.1f}%", style={'color': pnl_color})
                ])
                
                table_rows.append(row)
            
            return html.Table(table_rows, style={'width': '100%'})
            
        except Exception as e:
            logger.error(f"Positions table error: {e}")
            return html.P("Error loading positions")
    
    def _create_alerts_component(self, data: Dict) -> html.Div:
        """Create risk alerts component"""
        
        try:
            alerts = data.get('alerts', [])
            
            if not alerts:
                return html.P("No active alerts", style={'color': 'green'})
            
            alert_items = []
            
            for alert in alerts:
                severity = alert.get('severity', 'INFO')
                message = alert.get('message', 'Unknown alert')
                
                color = {'CRITICAL': 'red', 'WARNING': 'orange', 'INFO': 'blue'}.get(severity, 'blue')
                
                alert_item = html.Div([
                    html.Strong(f"{severity}: ", style={'color': color}),
                    html.Span(message)
                ], style={'padding': '5px', 'margin': '5px', 'border': f'1px solid {color}'})
                
                alert_items.append(alert_item)
            
            return html.Div(alert_items)
            
        except Exception as e:
            logger.error(f"Alerts component error: {e}")
            return html.P("Error loading alerts")
    
    def _get_market_data(self, symbol: str, timerange: str) -> Dict:
        """Get market data for dashboard"""
        
        try:
            # This would integrate with your existing data sources
            # For now, return simulated data
            
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=timerange)
            
            # Add technical indicators
            hist['SMA_20'] = hist['Close'].rolling(20).mean()
            hist['SMA_50'] = hist['Close'].rolling(50).mean()
            
            return {
                'price_data': hist,
                'technical_indicators': {
                    'rsi': 65.0,
                    'macd': 0.5,
                    'bb_position': 0.7
                },
                'rsi_history': [60, 62, 65, 68, 65, 63, 61],
                'volatility_history': [20, 22, 25, 28, 26, 24, 22],
                'vix_data': [18, 20, 22, 25, 23, 21, 19],
                'signals': {
                    'technical_signal': {'signal': 'BULLISH', 'confidence': 0.75},
                    'ml_signal': {'signal': 'NEUTRAL', 'confidence': 0.60},
                    'economic_signal': {'signal': 'BEARISH', 'confidence': 0.80}
                }
            }
            
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return {}
    
    def _get_portfolio_data(self) -> Dict:
        """Get portfolio data for dashboard"""
        
        # This would integrate with your broker/portfolio system
        return {
            'total_value': 102500,
            'daily_pnl': 1250,
            'win_rate': 68,
            'sharpe_ratio': 1.45,
            'positions': [
                {
                    'symbol': 'SPY',
                    'quantity': 10,
                    'entry_price': 450.00,
                    'current_price': 455.00,
                    'pnl': 500.00,
                    'pnl_percent': 1.1
                },
                {
                    'symbol': 'AAPL',
                    'quantity': 5,
                    'entry_price': 180.00,
                    'current_price': 185.00,
                    'pnl': 250.00,
                    'pnl_percent': 2.8
                }
            ]
        }
    
    def _get_risk_data(self) -> Dict:
        """Get risk data for dashboard"""
        
        return {
            'risk_metrics': {
                'overall_risk_score': 35,
                'var_95': -0.025,
                'max_drawdown': -0.08
            },
            'alerts': [
                {
                    'severity': 'WARNING',
                    'message': 'Portfolio volatility elevated above normal levels'
                }
            ]
        }
    
    def _get_ml_data(self, symbol: str) -> Dict:
        """Get ML prediction data"""
        
        return {
            'predictions': {
                'current_price': 455.00,
                'price_prediction': 460.00,
                'confidence': 0.72,
                'volatility_prediction': 0.22
            }
        }
    
    def _get_error_components(self):
        """Return error components when dashboard update fails"""
        
        error_fig = go.Figure().add_annotation(text="Error loading data", 
                                              x=0.5, y=0.5, showarrow=False)
        
        return (error_fig, error_fig, error_fig, error_fig, error_fig,
               html.P("Error"), html.P("Error"), html.P("Error"),
               "$0", "$0", "0%", "0.0", "Error")
    
    def run_dashboard(self, debug: bool = False):
        """Run the dashboard server"""
        
        if not DASH_AVAILABLE:
            logger.error("Dash not available - cannot run dashboard")
            return
        
        if self.app is None:
            logger.error("Dashboard not initialized")
            return
        
        try:
            logger.info(f"Starting dashboard on port {self.port}")
            self.app.run_server(host='127.0.0.1', port=self.port, debug=debug)
            
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
    
    async def start_background_dashboard(self):
        """Start dashboard in background"""
        
        if not DASH_AVAILABLE:
            logger.warning("Dashboard not available - Dash not installed")
            return
        
        try:
            import threading
            
            def run_dash():
                self.run_dashboard(debug=False)
            
            dashboard_thread = threading.Thread(target=run_dash, daemon=True)
            dashboard_thread.start()
            
            logger.info(f"Dashboard started in background on http://127.0.0.1:{self.port}")
            
        except Exception as e:
            logger.error(f"Background dashboard error: {e}")

# Singleton instance
trading_dashboard = TradingDashboard()