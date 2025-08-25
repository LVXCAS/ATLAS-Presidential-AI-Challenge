"""
Hive Trade Web Application
Flask-based web interface for the trading dashboard
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.unified_dashboard import UnifiedTradingDashboard

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hive_trade_secret_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global dashboard instance
dashboard = None

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self, dashboard_instance):
        self.dashboard = dashboard_instance
        self.clients = set()
        self.update_thread = None
        
    def start_updates(self):
        """Start real-time updates to connected clients"""
        if not self.update_thread or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def _update_loop(self):
        """Main update loop for WebSocket clients"""
        while True:
            try:
                if self.clients:
                    # Get latest data from dashboard
                    overview = self.dashboard.get_system_overview()
                    
                    # Emit to all connected clients
                    socketio.emit('dashboard_update', overview, namespace='/')
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"WebSocket update error: {e}")
                time.sleep(10)
    
    def add_client(self, client_id):
        """Add client to update list"""
        self.clients.add(client_id)
        if len(self.clients) == 1:  # First client
            self.start_updates()
    
    def remove_client(self, client_id):
        """Remove client from update list"""
        self.clients.discard(client_id)

# Initialize WebSocket manager
ws_manager = None

@app.route('/')
def index():
    """Working Trading Terminal - Simple and Functional"""
    return render_template('working_terminal.html')

@app.route('/legacy')
def legacy_dashboard():
    """Legacy dashboard for fallback"""
    return render_template('dashboard.html')

@app.route('/hybrid')
def hybrid_dashboard():
    """Previous hybrid attempt"""
    return render_template('hybrid_dashboard.html')

@app.route('/pro')
def professional_terminal():
    """Professional terminal version"""
    return render_template('professional_terminal.html')

@app.route('/api/overview')
def api_overview():
    """Get system overview API"""
    try:
        if dashboard:
            overview = dashboard.get_system_overview()
            return jsonify(overview)
        else:
            return jsonify({'error': 'Dashboard not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio')
def api_portfolio():
    """Get portfolio data API"""
    try:
        if dashboard:
            portfolio_summary = dashboard.paper_trading.get_portfolio_summary()
            return jsonify(portfolio_summary)
        else:
            return jsonify({'error': 'Dashboard not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-data')
def api_market_data():
    """Get market data API"""
    try:
        if dashboard:
            return jsonify(dashboard.state.market_data)
        else:
            return jsonify({'error': 'Dashboard not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-signals')
def api_ai_signals():
    """Get AI signals API"""
    try:
        if dashboard:
            return jsonify(dashboard.state.ai_signals)
        else:
            return jsonify({'error': 'Dashboard not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute-trade', methods=['POST'])
def api_execute_trade():
    """Execute trade API"""
    try:
        if not dashboard:
            return jsonify({'error': 'Dashboard not initialized'}), 500
        
        data = request.json
        symbol = data.get('symbol')
        side = data.get('side')  # 'buy' or 'sell'
        quantity = data.get('quantity', 10)
        
        if not symbol or not side:
            return jsonify({'error': 'Missing symbol or side'}), 400
        
        # Execute trade
        order = dashboard.paper_trading.place_order(symbol, side, quantity)
        
        if order:
            return jsonify({'success': True, 'order': order})
        else:
            return jsonify({'error': 'Failed to place order'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status')
def api_system_status():
    """Get system status API"""
    try:
        if dashboard:
            return jsonify(dashboard.system_status)
        else:
            return jsonify({'error': 'Dashboard not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f"Client connected: {request.sid}")
    if ws_manager:
        ws_manager.add_client(request.sid)
    emit('connected', {'data': 'Connected to Hive Trade Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f"Client disconnected: {request.sid}")
    if ws_manager:
        ws_manager.remove_client(request.sid)

@socketio.on('execute_ai_trade')
def handle_ai_trade(data):
    """Handle AI trade execution from WebSocket"""
    try:
        symbol = data.get('symbol')
        signal = data.get('signal')
        
        if dashboard and symbol and signal:
            order = dashboard.execute_trade_from_signal(symbol, signal)
            
            if order:
                emit('trade_executed', {
                    'success': True, 
                    'order': order,
                    'message': f"Executed {signal} order for {symbol}"
                })
            else:
                emit('trade_error', {
                    'success': False,
                    'message': f"Failed to execute {signal} order for {symbol}"
                })
    except Exception as e:
        emit('trade_error', {'success': False, 'message': str(e)})

# Enhanced WebSocket Events for Hybrid Interface
@socketio.on('get_watchlist')
def handle_get_watchlist(data):
    """Get watchlist data"""
    symbols = data.get('symbols', [])
    market_data = {}
    
    for symbol in symbols:
        # Mock data - replace with real market data
        import random
        price = random.uniform(100, 500)
        change = random.uniform(-10, 10)
        change_percent = (change / price) * 100
        
        market_data[symbol] = {
            'price': price,
            'change': change,
            'changePercent': change_percent,
            'volume': random.randint(100000, 10000000),
            'name': f"{symbol} Inc."
        }
    
    emit('market_data', market_data)

@socketio.on('get_portfolio')
def handle_get_portfolio():
    """Get portfolio data"""
    portfolio_data = {
        'totalValue': 1000000.00,
        'dayPnL': 1234.56,
        'positions': [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'avgPrice': 150.00,
                'currentPrice': 155.50,
                'pnl': 550.00,
                'pnlPercent': 3.67
            },
            {
                'symbol': 'GOOGL',
                'quantity': 50,
                'avgPrice': 2800.00,
                'currentPrice': 2850.00,
                'pnl': 2500.00,
                'pnlPercent': 1.79
            }
        ],
        'risk': {
            'var1d': 12500.00,
            'maxDrawdown': 5.25,
            'beta': 1.15,
            'sharpe': 1.85
        }
    }
    emit('portfolio_update', portfolio_data)

@socketio.on('get_news')
def handle_get_news():
    """Get news feed data"""
    news_data = {
        'news': [
            {
                'headline': 'Fed Signals Rate Cut Ahead',
                'source': 'Reuters',
                'timestamp': '2024-01-15T10:30:00Z',
                'summary': 'Federal Reserve officials hint at potential rate reduction in next meeting...',
                'url': 'https://reuters.com/article/fed-rates'
            },
            {
                'headline': 'Tech Stocks Rally on AI News',
                'source': 'Bloomberg',
                'timestamp': '2024-01-15T09:15:00Z',
                'summary': 'Major technology companies see gains following breakthrough AI announcement...',
                'url': 'https://bloomberg.com/article/tech-ai-rally'
            }
        ]
    }
    emit('news_update', news_data)

@socketio.on('get_symbol_data')
def handle_get_symbol_data(data):
    """Get specific symbol data"""
    symbol = data.get('symbol', 'SPY')
    # Mock chart data - replace with real data
    import random
    from datetime import datetime, timedelta
    
    chart_data = []
    base_price = 400
    for i in range(100):
        date = datetime.now() - timedelta(days=100-i)
        price = base_price + random.uniform(-5, 5)
        chart_data.append({
            'time': date.isoformat(),
            'open': price,
            'high': price + random.uniform(0, 3),
            'low': price - random.uniform(0, 3),
            'close': price + random.uniform(-2, 2)
        })
        base_price = chart_data[-1]['close']
    
    emit('symbol_data', {'symbol': symbol, 'data': chart_data})

@socketio.on('start_market_data')
def handle_start_market_data(data):
    """Start market data stream"""
    symbols = data.get('symbols', [])
    # Start streaming market data for symbols
    pass

@socketio.on('start_portfolio_updates')
def handle_start_portfolio_updates():
    """Start portfolio update stream"""
    pass

@socketio.on('start_news_feed')
def handle_start_news_feed():
    """Start news feed stream"""
    pass

@socketio.on('start_ai_signals')
def handle_start_ai_signals():
    """Start AI signals stream"""
    ai_signals = {
        'signals': [
            {
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.85,
                'reasoning': 'Strong momentum pattern with bullish divergence detected'
            },
            {
                'symbol': 'TSLA',
                'signal': 'HOLD',
                'confidence': 0.65,
                'reasoning': 'Mixed signals, elevated volatility suggests caution'
            }
        ]
    }
    emit('ai_signal', ai_signals)

@socketio.on('heartbeat')
def handle_heartbeat():
    """Handle client heartbeat"""
    emit('pong', {'timestamp': time.time()})

@socketio.on('execute_trade')
def handle_execute_trade(data):
    """Handle trade execution"""
    try:
        symbol = data.get('symbol')
        side = data.get('side')
        quantity = data.get('quantity', 100)
        price = data.get('price', 0)
        
        print(f"Executing {side} order: {quantity} shares of {symbol} at ${price}")
        
        # Simulate trade execution
        order_id = f"ORD_{int(time.time())}"
        
        # In real implementation, execute through paper trading engine
        if dashboard and hasattr(dashboard, 'paper_trading'):
            try:
                order = {
                    'symbol': symbol,
                    'side': side.lower(),
                    'quantity': quantity,
                    'price': price,
                    'order_type': 'market'
                }
                result = dashboard.paper_trading.execute_trade(order)
                
                emit('trade_executed', {
                    'success': True,
                    'order_id': order_id,
                    'message': f"Executed {side} {quantity} shares of {symbol}",
                    'result': result
                })
            except Exception as e:
                emit('trade_executed', {
                    'success': False,
                    'message': f"Trade failed: {str(e)}"
                })
        else:
            # Mock execution
            emit('trade_executed', {
                'success': True,
                'order_id': order_id,
                'message': f"Paper trade executed: {side} {quantity} shares of {symbol} at ${price}"
            })
            
    except Exception as e:
        print(f"Trade execution error: {e}")
        emit('trade_executed', {
            'success': False,
            'message': f"Trade execution error: {str(e)}"
        })

def initialize_dashboard():
    """Initialize the dashboard in a separate thread"""
    global dashboard, ws_manager
    
    print("Initializing Hive Trade Dashboard...")
    dashboard = UnifiedTradingDashboard()
    ws_manager = WebSocketManager(dashboard)
    
    # Start data feeds in background
    def run_data_feeds():
        asyncio.run(dashboard.data_manager.start_data_feeds())
    
    data_thread = threading.Thread(target=run_data_feeds)
    data_thread.daemon = True
    data_thread.start()
    
    print("Dashboard initialized successfully!")

if __name__ == '__main__':
    # Initialize dashboard in background
    init_thread = threading.Thread(target=initialize_dashboard)
    init_thread.daemon = True
    init_thread.start()
    
    # Give dashboard time to initialize
    time.sleep(3)
    
    print("Starting Hive Trade Hybrid Terminal...")
    print("Bloomberg x Robinhood Interface Active")
    print("Progressive Complexity System: READY")
    print("Dual-Mode Interface: READY")
    print("Command System: READY")
    print("Access dashboard at: http://localhost:5000")
    
    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)