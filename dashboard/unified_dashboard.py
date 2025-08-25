"""
Hive Trade Unified Integration Dashboard
Comprehensive real-time trading dashboard integrating all systems
"""

import os
import sys
import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Import all our trading systems
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.options_strategies import AdvancedOptionsAnalyzer
from trading.crypto_trading import AdvancedCryptoTrader
from trading.futures_trading import AdvancedFuturesTrader
from trading.international_markets import InternationalMarketsAnalyzer
from risk_management.portfolio_heatmaps import PortfolioHeatMapSystem
from risk_management.correlation_position_sizing import CorrelationBasedPositionSizer
from ai.enhanced_models import EnhancedAISystem

@dataclass
class DashboardState:
    """Central dashboard state management"""
    last_update: datetime
    portfolio_value: float
    daily_pnl: float
    daily_return_pct: float
    positions: Dict[str, Any]
    market_data: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    ai_signals: Dict[str, Any]
    system_status: Dict[str, bool]
    alerts: List[Dict[str, Any]]

class RealTimeDataManager:
    """Manage real-time data feeds and updates"""
    
    def __init__(self):
        self.data_feeds = {}
        self.update_callbacks = []
        self.is_running = False
        
    def add_data_feed(self, name: str, update_function):
        """Add a data feed with update function"""
        self.data_feeds[name] = {
            'function': update_function,
            'last_update': None,
            'data': None
        }
    
    def add_update_callback(self, callback):
        """Add callback for data updates"""
        self.update_callbacks.append(callback)
    
    async def start_data_feeds(self):
        """Start all data feeds"""
        self.is_running = True
        print("Starting real-time data feeds...")
        
        while self.is_running:
            try:
                # Update all data feeds
                for name, feed in self.data_feeds.items():
                    try:
                        # Call update function
                        new_data = await self._call_update_function(feed['function'])
                        
                        if new_data:
                            feed['data'] = new_data
                            feed['last_update'] = datetime.now()
                            
                            # Notify callbacks
                            for callback in self.update_callbacks:
                                try:
                                    await callback(name, new_data)
                                except Exception as e:
                                    print(f"Callback error for {name}: {e}")
                    
                    except Exception as e:
                        print(f"Data feed error for {name}: {e}")
                
                # Wait before next update cycle
                await asyncio.sleep(5)  # 5-second updates
                
            except Exception as e:
                print(f"Data manager error: {e}")
                await asyncio.sleep(10)
    
    async def _call_update_function(self, func):
        """Call update function safely"""
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func)
    
    def stop(self):
        """Stop data feeds"""
        self.is_running = False

class PaperTradingEngine:
    """Paper trading engine for live testing"""
    
    def __init__(self, initial_balance: float = 1000000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        self.performance_metrics = {}
        
    def place_order(self, symbol: str, side: str, quantity: float, 
                   price: Optional[float] = None, order_type: str = 'market') -> Dict[str, Any]:
        """Place paper trading order"""
        
        order_id = f"paper_{int(time.time() * 1000)}"
        
        # Mock execution price
        if price is None:
            price = self._get_mock_price(symbol)
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'type': order_type,
            'status': 'filled',
            'timestamp': datetime.now().isoformat(),
            'value': quantity * price
        }
        
        # Update positions
        self._update_position(order)
        
        # Add to trade history
        self.trade_history.append(order)
        self.orders[order_id] = order
        
        print(f"Paper Trade Executed: {side.upper()} {quantity} {symbol} @ ${price:.2f}")
        
        return order
    
    def _get_mock_price(self, symbol: str) -> float:
        """Get mock market price"""
        import random
        
        base_prices = {
            'AAPL': 180, 'GOOGL': 140, 'MSFT': 350, 'TSLA': 250, 'AMZN': 150,
            'NVDA': 500, 'JPM': 150, 'BAC': 30, 'JNJ': 160, 'PFE': 30,
            'BTC': 45000, 'ETH': 3000, 'EURUSD': 1.08, 'GOLD': 2000,
            'CRUDE': 80, 'SP500': 4500
        }
        
        base_price = base_prices.get(symbol, 100)
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
        return base_price * (1 + variation)
    
    def _update_position(self, order: Dict[str, Any]):
        """Update position based on order"""
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        price = order['price']
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'market_value': 0,
                'unrealized_pnl': 0
            }
        
        position = self.positions[symbol]
        
        if side.lower() == 'buy':
            # Update average price
            total_cost = position['quantity'] * position['avg_price'] + quantity * price
            total_quantity = position['quantity'] + quantity
            
            position['avg_price'] = total_cost / total_quantity if total_quantity > 0 else price
            position['quantity'] = total_quantity
            
            # Update balance
            self.current_balance -= quantity * price
            
        elif side.lower() == 'sell':
            position['quantity'] -= quantity
            
            # Realize P&L
            realized_pnl = quantity * (price - position['avg_price'])
            self.current_balance += quantity * price
            
            if position['quantity'] <= 0:
                position['quantity'] = 0
                position['avg_price'] = 0
    
    def update_market_values(self, market_data: Dict[str, float]):
        """Update market values for all positions"""
        total_portfolio_value = self.current_balance
        
        for symbol, position in self.positions.items():
            if position['quantity'] > 0:
                current_price = market_data.get(symbol, position['avg_price'])
                position['market_value'] = position['quantity'] * current_price
                position['unrealized_pnl'] = position['quantity'] * (current_price - position['avg_price'])
                
                total_portfolio_value += position['market_value']
        
        # Update performance metrics
        self.performance_metrics = {
            'total_value': total_portfolio_value,
            'cash_balance': self.current_balance,
            'invested_value': total_portfolio_value - self.current_balance,
            'total_return': total_portfolio_value - self.initial_balance,
            'total_return_pct': (total_portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'num_positions': len([p for p in self.positions.values() if p['quantity'] > 0]),
            'num_trades': len(self.trade_history)
        }
        
        return self.performance_metrics
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            'performance_metrics': self.performance_metrics,
            'positions': {k: v for k, v in self.positions.items() if v['quantity'] > 0},
            'recent_trades': self.trade_history[-10:],  # Last 10 trades
            'cash_balance': self.current_balance
        }

class UnifiedTradingDashboard:
    """Main unified trading dashboard"""
    
    def __init__(self):
        self.state = DashboardState(
            last_update=datetime.now(),
            portfolio_value=1000000,
            daily_pnl=0,
            daily_return_pct=0,
            positions={},
            market_data={},
            risk_metrics={},
            ai_signals={},
            system_status={},
            alerts=[]
        )
        
        # Initialize systems
        self.options_analyzer = AdvancedOptionsAnalyzer(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        self.crypto_trader = AdvancedCryptoTrader()
        self.futures_trader = AdvancedFuturesTrader()
        self.international_analyzer = InternationalMarketsAnalyzer()
        self.portfolio_heatmap = PortfolioHeatMapSystem()
        self.position_sizer = CorrelationBasedPositionSizer()
        
        # Paper trading engine
        self.paper_trading = PaperTradingEngine()
        
        # Real-time data manager
        self.data_manager = RealTimeDataManager()
        self.setup_data_feeds()
        
        # System status tracking
        self.system_status = {
            'options_system': True,
            'crypto_system': True,
            'futures_system': True,
            'international_markets': True,
            'risk_management': True,
            'ai_models': True,
            'paper_trading': True,
            'data_feeds': True
        }
    
    def setup_data_feeds(self):
        """Setup real-time data feeds"""
        
        # Add data feeds for different systems
        self.data_manager.add_data_feed('market_data', self.update_market_data)
        self.data_manager.add_data_feed('portfolio_data', self.update_portfolio_data)
        self.data_manager.add_data_feed('risk_metrics', self.update_risk_metrics)
        self.data_manager.add_data_feed('ai_signals', self.update_ai_signals)
        
        # Add update callback
        self.data_manager.add_update_callback(self.on_data_update)
    
    async def update_market_data(self) -> Dict[str, Any]:
        """Update market data from all sources"""
        try:
            market_data = {}
            
            # Get crypto data
            crypto_symbols = ['BTCUSDT', 'ETHUSDT']
            for symbol in crypto_symbols:
                if hasattr(self.crypto_trader, 'exchanges') and self.crypto_trader.exchanges:
                    exchange = self.crypto_trader.exchanges[0]
                    data = exchange.get_market_data(symbol)
                    market_data[symbol] = data
            
            # Get forex data
            forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
            for pair in forex_pairs:
                price_data = self.international_analyzer.forex_market.get_forex_price(pair)
                market_data[pair] = price_data
            
            # Get commodity data
            commodities = ['GOLD', 'CRUDE_OIL', 'SILVER']
            for commodity in commodities:
                price_data = self.international_analyzer.commodities_market.get_commodity_price(commodity)
                market_data[commodity] = price_data
            
            # Mock stock data
            stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
            for stock in stocks:
                market_data[stock] = {
                    'symbol': stock,
                    'price': self.paper_trading._get_mock_price(stock),
                    'change_24h': (hash(stock + str(int(time.time() / 60))) % 1000) / 100 - 5,  # -5% to +5%
                    'volume': (hash(stock + str(int(time.time() / 30))) % 1000000) + 100000,
                    'timestamp': datetime.now().isoformat()
                }
            
            return market_data
            
        except Exception as e:
            print(f"Error updating market data: {e}")
            return {}
    
    async def update_portfolio_data(self) -> Dict[str, Any]:
        """Update portfolio data"""
        try:
            # Update paper trading portfolio
            market_prices = {}
            if hasattr(self.state, 'market_data'):
                for symbol, data in self.state.market_data.items():
                    if isinstance(data, dict) and 'price' in data:
                        market_prices[symbol] = data['price']
            
            performance = self.paper_trading.update_market_values(market_prices)
            portfolio_summary = self.paper_trading.get_portfolio_summary()
            
            return {
                'performance': performance,
                'portfolio': portfolio_summary
            }
            
        except Exception as e:
            print(f"Error updating portfolio data: {e}")
            return {}
    
    async def update_risk_metrics(self) -> Dict[str, Any]:
        """Update risk metrics"""
        try:
            # Create sample positions for risk analysis
            sample_positions = self.portfolio_heatmap.create_sample_portfolio()
            
            # Run risk analysis
            risk_analysis = self.portfolio_heatmap.risk_calculator.calculate_portfolio_var(sample_positions)
            
            risk_metrics = {
                'portfolio_var_95': risk_analysis,
                'var_as_pct_portfolio': (risk_analysis / 1000000) * 100,  # Assuming $1M portfolio
                'diversification_ratio': 1.2 + (hash(str(int(time.time() / 60))) % 100) / 500,  # Mock diversification
                'concentration_risk': 'Medium',
                'stress_test_result': 'Pass',
                'last_update': datetime.now().isoformat()
            }
            
            return risk_metrics
            
        except Exception as e:
            print(f"Error updating risk metrics: {e}")
            return {}
    
    async def update_ai_signals(self) -> Dict[str, Any]:
        """Update AI trading signals"""
        try:
            # Generate mock AI signals (in real system would use trained models)
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'BTCUSDT', 'ETHUSDT']
            
            signals = {}
            for symbol in symbols:
                # Mock signal generation
                signal_strength = (hash(symbol + str(int(time.time() / 300))) % 100) - 50  # -50 to +50
                
                if signal_strength > 20:
                    signal = 'BUY'
                    confidence = min((signal_strength - 20) / 30, 1.0)
                elif signal_strength < -20:
                    signal = 'SELL'
                    confidence = min((-signal_strength - 20) / 30, 1.0)
                else:
                    signal = 'HOLD'
                    confidence = 0.5
                
                signals[symbol] = {
                    'signal': signal,
                    'confidence': confidence,
                    'strength': signal_strength,
                    'reasoning': f"AI model indicates {signal} with {confidence:.1%} confidence",
                    'timestamp': datetime.now().isoformat()
                }
            
            return signals
            
        except Exception as e:
            print(f"Error updating AI signals: {e}")
            return {}
    
    async def on_data_update(self, feed_name: str, data: Any):
        """Handle data updates"""
        try:
            if feed_name == 'market_data':
                self.state.market_data = data
            elif feed_name == 'portfolio_data':
                if isinstance(data, dict) and 'performance' in data:
                    perf = data['performance']
                    self.state.portfolio_value = perf.get('total_value', self.state.portfolio_value)
                    
                    # Calculate daily P&L (simplified)
                    daily_pnl = perf.get('total_return', 0) * 0.1  # Mock daily component
                    self.state.daily_pnl = daily_pnl
                    self.state.daily_return_pct = daily_pnl / self.state.portfolio_value * 100
                
                self.state.positions = data.get('portfolio', {}).get('positions', {})
            
            elif feed_name == 'risk_metrics':
                self.state.risk_metrics = data
                
            elif feed_name == 'ai_signals':
                self.state.ai_signals = data
                
                # Generate alerts based on strong signals
                self.check_ai_signal_alerts(data)
            
            self.state.last_update = datetime.now()
            
        except Exception as e:
            print(f"Error handling data update for {feed_name}: {e}")
    
    def check_ai_signal_alerts(self, signals: Dict[str, Any]):
        """Check for AI signal alerts"""
        try:
            for symbol, signal_data in signals.items():
                if signal_data.get('confidence', 0) > 0.8:  # High confidence signals
                    alert = {
                        'type': 'AI_SIGNAL',
                        'level': 'HIGH',
                        'symbol': symbol,
                        'signal': signal_data.get('signal'),
                        'confidence': signal_data.get('confidence'),
                        'message': f"Strong {signal_data.get('signal')} signal for {symbol} "
                                 f"({signal_data.get('confidence', 0):.1%} confidence)",
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Add to alerts (keep only recent alerts)
                    self.state.alerts.append(alert)
                    self.state.alerts = self.state.alerts[-20:]  # Keep last 20 alerts
                    
        except Exception as e:
            print(f"Error checking AI signal alerts: {e}")
    
    def execute_trade_from_signal(self, symbol: str, signal: str, quantity: float = None):
        """Execute trade based on AI signal"""
        try:
            if quantity is None:
                # Default position size (1% of portfolio)
                quantity = self.state.portfolio_value * 0.01 / self.paper_trading._get_mock_price(symbol)
            
            if signal == 'BUY':
                order = self.paper_trading.place_order(symbol, 'buy', quantity)
            elif signal == 'SELL':
                order = self.paper_trading.place_order(symbol, 'sell', quantity)
            else:
                return None
            
            # Create trade alert
            alert = {
                'type': 'TRADE_EXECUTED',
                'level': 'INFO',
                'symbol': symbol,
                'message': f"Executed {signal} order for {quantity:.2f} shares of {symbol}",
                'order_id': order.get('id'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.state.alerts.append(alert)
            
            return order
            
        except Exception as e:
            print(f"Error executing trade for {symbol}: {e}")
            return None
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        
        # Calculate uptime
        uptime = datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # System health check
        healthy_systems = sum(1 for status in self.system_status.values() if status)
        total_systems = len(self.system_status)
        health_percentage = (healthy_systems / total_systems) * 100
        
        overview = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'overall_health': health_percentage,
                'healthy_systems': healthy_systems,
                'total_systems': total_systems,
                'uptime_hours': uptime.total_seconds() / 3600,
                'status_details': self.system_status
            },
            'portfolio_summary': {
                'total_value': self.state.portfolio_value,
                'daily_pnl': self.state.daily_pnl,
                'daily_return_pct': self.state.daily_return_pct,
                'num_positions': len(self.state.positions),
                'cash_balance': self.paper_trading.current_balance
            },
            'market_overview': {
                'num_markets_tracked': len(self.state.market_data),
                'data_feeds_active': len(self.data_manager.data_feeds),
                'last_market_update': self.state.last_update.isoformat() if self.state.last_update else None
            },
            'ai_signals_summary': {
                'total_signals': len(self.state.ai_signals),
                'buy_signals': sum(1 for s in self.state.ai_signals.values() 
                                 if s.get('signal') == 'BUY'),
                'sell_signals': sum(1 for s in self.state.ai_signals.values() 
                                  if s.get('signal') == 'SELL'),
                'high_confidence_signals': sum(1 for s in self.state.ai_signals.values() 
                                             if s.get('confidence', 0) > 0.7)
            },
            'risk_overview': {
                'portfolio_var': self.state.risk_metrics.get('portfolio_var_95', 0),
                'var_percentage': self.state.risk_metrics.get('var_as_pct_portfolio', 0),
                'risk_level': 'Medium' if self.state.risk_metrics.get('var_as_pct_portfolio', 0) < 20 else 'High'
            },
            'recent_alerts': self.state.alerts[-5:] if self.state.alerts else [],
            'performance_metrics': self.paper_trading.performance_metrics
        }
        
        return overview
    
    async def run_dashboard(self):
        """Run the main dashboard"""
        print("HIVE TRADE UNIFIED DASHBOARD STARTING...")
        print("="*45)
        
        # Start data feeds
        data_task = asyncio.create_task(self.data_manager.start_data_feeds())
        
        try:
            # Main dashboard loop
            while True:
                # Print dashboard update
                self.print_dashboard_update()
                
                # Wait for next update
                await asyncio.sleep(30)  # 30-second dashboard updates
                
        except KeyboardInterrupt:
            print("\nShutting down dashboard...")
            self.data_manager.stop()
            data_task.cancel()
            
        except Exception as e:
            print(f"Dashboard error: {e}")
            self.data_manager.stop()
    
    def print_dashboard_update(self):
        """Print real-time dashboard update"""
        
        overview = self.get_system_overview()
        
        print(f"\n{'='*60}")
        print(f"HIVE TRADE UNIFIED DASHBOARD - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # System Health
        health = overview['system_health']
        print(f"SYSTEM HEALTH: {health['overall_health']:.0f}% "
              f"({health['healthy_systems']}/{health['total_systems']} systems)")
        
        # Portfolio Summary
        portfolio = overview['portfolio_summary']
        print(f"PORTFOLIO: ${portfolio['total_value']:,.2f} "
              f"(Daily: {portfolio['daily_return_pct']:+.2f}%, "
              f"${portfolio['daily_pnl']:+,.2f})")
        
        # Market Data
        market = overview['market_overview']
        print(f"MARKETS: {market['num_markets_tracked']} tracked, "
              f"{market['data_feeds_active']} feeds active")
        
        # AI Signals
        ai = overview['ai_signals_summary']
        if ai['total_signals'] > 0:
            print(f"AI SIGNALS: {ai['buy_signals']} BUY, {ai['sell_signals']} SELL, "
                  f"{ai['high_confidence_signals']} high confidence")
        
        # Risk Metrics
        risk = overview['risk_overview']
        print(f"RISK: VaR ${risk['portfolio_var']:,.0f} ({risk['var_percentage']:.1f}%), "
              f"Level: {risk['risk_level']}")
        
        # Recent Alerts
        if overview['recent_alerts']:
            print(f"\nRECENT ALERTS:")
            for alert in overview['recent_alerts'][-3:]:
                timestamp = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M')
                print(f"  [{timestamp}] {alert['level']}: {alert['message']}")
        
        # Top Positions
        if self.state.positions:
            print(f"\nTOP POSITIONS:")
            sorted_positions = sorted(
                self.state.positions.items(), 
                key=lambda x: x[1].get('market_value', 0), 
                reverse=True
            )[:3]
            
            for symbol, pos in sorted_positions:
                pnl = pos.get('unrealized_pnl', 0)
                value = pos.get('market_value', 0)
                print(f"  {symbol}: ${value:,.2f} (P&L: ${pnl:+,.2f})")
        
        print(f"{'='*60}")

def main():
    """Run the unified dashboard"""
    
    dashboard = UnifiedTradingDashboard()
    
    # Run the dashboard
    try:
        asyncio.run(dashboard.run_dashboard())
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()