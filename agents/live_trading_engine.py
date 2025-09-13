#!/usr/bin/env python3
"""
Live Trading Engine - Execute trades based on real-time signals
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import importlib.util
import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    from .live_data_manager import live_data_manager
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from live_data_manager import live_data_manager
        DATA_MANAGER_AVAILABLE = True
    except ImportError:
        DATA_MANAGER_AVAILABLE = False

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

class LiveTradingEngine:
    """Execute live trades based on strategy signals"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategies = {}
        self.positions = {}
        self.orders = {}
        self.performance = {}
        self.is_trading = False
        self.trading_mode = 'paper'  # 'paper' or 'live'
        
        # Risk management
        self.risk_limits = {
            'max_portfolio_risk': 0.02,     # 2% max portfolio risk per day
            'max_position_size': 0.05,      # 5% max position size
            'max_daily_trades': 50,         # Max trades per day
            'max_total_exposure': 1.0,      # 100% max total exposure
            'stop_loss_pct': 0.02,          # 2% stop loss
            'take_profit_pct': 0.06         # 6% take profit
        }
        
        # Trading state
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.total_exposure = 0.0
        self.account_value = 100000.0  # Starting capital
        
        # Initialize broker connection
        self._setup_broker()
    
    def _setup_broker(self):
        """Setup broker connection"""
        try:
            if self.config.get('alpaca_key') and ALPACA_AVAILABLE:
                self.broker = tradeapi.REST(
                    self.config['alpaca_key'],
                    self.config['alpaca_secret'],
                    self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets'),
                    api_version='v2'
                )
                self.broker_type = 'alpaca'
                print("+ Alpaca broker connected")
            else:
                self.broker = None
                self.broker_type = 'simulation'
                print("- Using simulation mode (no broker connected)")
                
        except Exception as e:
            print(f"Broker setup error: {e}")
            self.broker = None
            self.broker_type = 'simulation'
    
    async def load_strategies(self, strategy_dir: str = "deployed_strategies"):
        """Load deployed trading strategies"""
        print(f"\nLOADING TRADING STRATEGIES")
        print("=" * 35)
        
        if not os.path.exists(strategy_dir):
            print(f"Strategy directory not found: {strategy_dir}")
            return False
        
        loaded_count = 0
        
        for filename in os.listdir(strategy_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                strategy_name = filename[:-3]  # Remove .py extension
                strategy_path = os.path.join(strategy_dir, filename)
                
                try:
                    # Load strategy module
                    spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Get strategy instance
                    if hasattr(module, 'get_trading_signal'):
                        self.strategies[strategy_name] = {
                            'module': module,
                            'signal_func': module.get_trading_signal,
                            'instance': getattr(module, 'strategy_instance', None),
                            'loaded_at': datetime.now(),
                            'active': True
                        }
                        
                        loaded_count += 1
                        print(f"+ Loaded: {strategy_name}")
                        
                        # Initialize position tracking
                        if hasattr(module, 'strategy_instance'):
                            symbol = module.strategy_instance.symbol
                            self.positions[symbol] = {
                                'quantity': 0,
                                'avg_price': 0,
                                'market_value': 0,
                                'unrealized_pnl': 0,
                                'strategy': strategy_name
                            }
                    
                except Exception as e:
                    print(f"X Failed to load {strategy_name}: {e}")
        
        print(f"\nLoaded {loaded_count} strategies")
        return loaded_count > 0
    
    async def start_trading(self, symbols: List[str] = None):
        """Start live trading engine"""
        print(f"\nSTARTING LIVE TRADING ENGINE")
        print(f"Mode: {self.trading_mode.upper()}")
        print(f"Broker: {self.broker_type}")
        print("=" * 40)
        
        if not self.strategies:
            print("No strategies loaded! Load strategies first.")
            return False
        
        # Get symbols from strategies if not provided
        if not symbols:
            symbols = []
            for strategy_name, strategy_info in self.strategies.items():
                if strategy_info['instance']:
                    symbol = strategy_info['instance'].symbol
                    if symbol not in symbols:
                        symbols.append(symbol)
        
        if not symbols:
            print("No symbols to trade!")
            return False
        
        print(f"Trading symbols: {', '.join(symbols)}")
        self.is_trading = True
        
        # Start data feed with trading callback
        if DATA_MANAGER_AVAILABLE:
            live_data_manager.register_callback('trading_engine', self._on_market_data)
            
            # Start market data feed and trading logic concurrently
            tasks = [
                live_data_manager.start_live_feed(symbols),
                self._trading_loop(),
                self._risk_monitor(),
                self._performance_tracker()
            ]
            
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                print("\nStopping trading engine...")
                await self.stop_trading()
        else:
            print("Live data manager not available!")
            return False
        
        return True
    
    async def _on_market_data(self, market_data: Dict):
        """Handle incoming market data"""
        try:
            symbol = market_data['symbol']
            price = market_data['price']
            
            # Update position values
            if symbol in self.positions:
                position = self.positions[symbol]
                if position['quantity'] != 0:
                    position['market_value'] = position['quantity'] * price
                    position['unrealized_pnl'] = position['market_value'] - (position['quantity'] * position['avg_price'])
            
            # Generate trading signals
            await self._process_signals(market_data)
            
        except Exception as e:
            print(f"Error processing market data: {e}")
    
    async def _process_signals(self, market_data: Dict):
        """Process trading signals from strategies"""
        try:
            symbol = market_data['symbol']
            
            # Find strategies for this symbol
            relevant_strategies = []
            for strategy_name, strategy_info in self.strategies.items():
                if (strategy_info['active'] and 
                    strategy_info['instance'] and 
                    strategy_info['instance'].symbol == symbol):
                    relevant_strategies.append((strategy_name, strategy_info))
            
            if not relevant_strategies:
                return
            
            # Get signals from all relevant strategies
            signals = []
            for strategy_name, strategy_info in relevant_strategies:
                try:
                    signal_result = await strategy_info['signal_func'](market_data)
                    if signal_result and signal_result.get('signal', 0) != 0:
                        signals.append({
                            'strategy': strategy_name,
                            'signal': signal_result['signal'],
                            'confidence': signal_result.get('confidence', 0),
                            'symbol': symbol,
                            'price': market_data['price']
                        })
                        
                        print(f"Signal: {strategy_name} -> {signal_result['signal']} "
                              f"for {symbol} @ ${market_data['price']:.2f} "
                              f"(Confidence: {signal_result.get('confidence', 0):.1%})")
                
                except Exception as e:
                    print(f"Error getting signal from {strategy_name}: {e}")
            
            # Execute trades based on signals
            if signals:
                await self._execute_signals(signals)
                
        except Exception as e:
            print(f"Error processing signals: {e}")
    
    async def _execute_signals(self, signals: List[Dict]):
        """Execute trades based on signals"""
        try:
            # Aggregate signals by symbol
            symbol_signals = {}
            for signal in signals:
                symbol = signal['symbol']
                if symbol not in symbol_signals:
                    symbol_signals[symbol] = []
                symbol_signals[symbol].append(signal)
            
            # Process each symbol
            for symbol, symbol_signal_list in symbol_signals.items():
                # Calculate weighted signal
                total_weight = sum(s['confidence'] for s in symbol_signal_list)
                if total_weight == 0:
                    continue
                
                weighted_signal = sum(s['signal'] * s['confidence'] for s in symbol_signal_list) / total_weight
                avg_price = sum(s['price'] for s in symbol_signal_list) / len(symbol_signal_list)
                
                # Risk checks
                if not self._check_risk_limits(symbol, weighted_signal):
                    continue
                
                # Calculate position size
                position_size = self._calculate_position_size(symbol, weighted_signal, avg_price)
                
                if abs(position_size) < 1:  # Less than 1 share
                    continue
                
                # Execute trade
                await self._place_order(symbol, position_size, avg_price, symbol_signal_list)
                
        except Exception as e:
            print(f"Error executing signals: {e}")
    
    def _check_risk_limits(self, symbol: str, signal: float) -> bool:
        """Check if trade passes risk limits"""
        try:
            # Check daily trade limit
            if self.daily_trades >= self.risk_limits['max_daily_trades']:
                return False
            
            # Check daily PnL limit
            if self.daily_pnl < -self.risk_limits['max_portfolio_risk'] * self.account_value:
                return False
            
            # Check total exposure
            if self.total_exposure >= self.risk_limits['max_total_exposure']:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking risk limits: {e}")
            return False
    
    def _calculate_position_size(self, symbol: str, signal: float, price: float) -> float:
        """Calculate optimal position size"""
        try:
            # Base position size as percentage of account
            base_size_pct = self.risk_limits['max_position_size']
            
            # Adjust for signal strength
            adjusted_size_pct = base_size_pct * abs(signal)
            
            # Calculate dollar amount
            dollar_amount = self.account_value * adjusted_size_pct
            
            # Convert to shares
            shares = int(dollar_amount / price)
            
            # Apply signal direction
            return shares * (1 if signal > 0 else -1)
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0
    
    async def _place_order(self, symbol: str, quantity: float, price: float, strategies: List[Dict]):
        """Place trading order"""
        try:
            # Get current position
            current_position = self.positions.get(symbol, {}).get('quantity', 0)
            net_quantity = quantity
            
            # Skip if no position change
            if abs(net_quantity) < 1:
                return
            
            order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': net_quantity,
                'price': price,
                'side': 'buy' if net_quantity > 0 else 'sell',
                'strategies': [s['strategy'] for s in strategies],
                'timestamp': datetime.now(),
                'status': 'pending'
            }
            
            print(f"\nORDER: {order['side'].upper()} {abs(net_quantity)} {symbol} @ ${price:.2f}")
            print(f"Strategies: {', '.join(order['strategies'])}")
            
            # Execute order
            if self.broker_type == 'alpaca' and self.broker:
                success = await self._execute_alpaca_order(order)
            else:
                success = await self._execute_simulated_order(order)
            
            if success:
                self.orders[order_id] = order
                self.daily_trades += 1
                
                # Update position
                self._update_position(symbol, net_quantity, price)
                
            return success
            
        except Exception as e:
            print(f"Error placing order: {e}")
            return False
    
    async def _execute_alpaca_order(self, order: Dict) -> bool:
        """Execute order through Alpaca"""
        try:
            side = 'buy' if order['quantity'] > 0 else 'sell'
            qty = abs(order['quantity'])
            
            # Place market order
            alpaca_order = self.broker.submit_order(
                symbol=order['symbol'],
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            order['alpaca_order_id'] = alpaca_order.id
            order['status'] = 'submitted'
            
            print(f"+ Alpaca order submitted: {alpaca_order.id}")
            return True
            
        except Exception as e:
            print(f"Alpaca order error: {e}")
            order['status'] = 'failed'
            order['error'] = str(e)
            return False
    
    async def _execute_simulated_order(self, order: Dict) -> bool:
        """Execute simulated order"""
        try:
            # Simulate order execution
            order['status'] = 'filled'
            order['fill_price'] = order['price']
            order['fill_time'] = datetime.now()
            
            print(f"+ Simulated order filled @ ${order['price']:.2f}")
            return True
            
        except Exception as e:
            print(f"Simulated order error: {e}")
            return False
    
    def _update_position(self, symbol: str, quantity: float, price: float):
        """Update position tracking"""
        try:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'market_value': 0,
                    'unrealized_pnl': 0
                }
            
            position = self.positions[symbol]
            old_quantity = position['quantity']
            old_value = old_quantity * position['avg_price']
            
            # Calculate new position
            new_quantity = old_quantity + quantity
            new_value = old_value + (quantity * price)
            
            if new_quantity != 0:
                position['avg_price'] = new_value / new_quantity
            else:
                position['avg_price'] = 0
            
            position['quantity'] = new_quantity
            position['market_value'] = new_quantity * price
            
            # Calculate realized PnL for position closures
            if old_quantity != 0 and new_quantity != 0 and np.sign(old_quantity) != np.sign(new_quantity):
                # Position flip - realize some PnL
                realized_pnl = min(abs(old_quantity), abs(quantity)) * (price - position['avg_price'])
                self.daily_pnl += realized_pnl
                print(f"Realized PnL: ${realized_pnl:.2f}")
            
            print(f"Position: {symbol} = {new_quantity} shares @ ${position['avg_price']:.2f}")
            
        except Exception as e:
            print(f"Error updating position: {e}")
    
    async def _trading_loop(self):
        """Main trading loop"""
        while self.is_trading:
            try:
                # Check for order updates
                await self._update_orders()
                
                # Update account value
                await self._update_account_value()
                
                # Check stop losses and take profits
                await self._check_exit_conditions()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Trading loop error: {e}")
                await asyncio.sleep(10)
    
    async def _risk_monitor(self):
        """Monitor risk limits"""
        while self.is_trading:
            try:
                # Check daily loss limit
                max_loss = self.risk_limits['max_portfolio_risk'] * self.account_value
                if self.daily_pnl < -max_loss:
                    print(f"! DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f}")
                    await self._close_all_positions("Daily loss limit exceeded")
                
                # Check total exposure
                total_exposure = sum(abs(pos['market_value']) for pos in self.positions.values())
                self.total_exposure = total_exposure / self.account_value
                
                if self.total_exposure > self.risk_limits['max_total_exposure']:
                    print(f"! EXPOSURE LIMIT HIT: {self.total_exposure:.1%}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Risk monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracker(self):
        """Track performance metrics"""
        while self.is_trading:
            try:
                # Calculate portfolio metrics
                total_unrealized = sum(pos['unrealized_pnl'] for pos in self.positions.values())
                total_pnl = self.daily_pnl + total_unrealized
                
                # Update performance
                self.performance = {
                    'account_value': self.account_value,
                    'daily_pnl': self.daily_pnl,
                    'unrealized_pnl': total_unrealized,
                    'total_pnl': total_pnl,
                    'daily_trades': self.daily_trades,
                    'total_exposure': self.total_exposure,
                    'active_positions': len([p for p in self.positions.values() if p['quantity'] != 0]),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Print status every 5 minutes
                if datetime.now().second == 0 and datetime.now().minute % 5 == 0:
                    print(f"\n📊 Performance: PnL: ${total_pnl:+.2f} | "
                          f"Exposure: {self.total_exposure:.1%} | "
                          f"Trades: {self.daily_trades}")
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                print(f"Performance tracker error: {e}")
                await asyncio.sleep(60)
    
    async def _update_orders(self):
        """Update order statuses"""
        try:
            if self.broker_type == 'alpaca' and self.broker:
                for order_id, order in self.orders.items():
                    if order['status'] in ['pending', 'submitted']:
                        alpaca_order = self.broker.get_order(order['alpaca_order_id'])
                        order['status'] = alpaca_order.status
                        
                        if alpaca_order.status == 'filled':
                            order['fill_price'] = float(alpaca_order.filled_avg_price)
                            order['fill_time'] = alpaca_order.filled_at
        
        except Exception as e:
            print(f"Error updating orders: {e}")
    
    async def _update_account_value(self):
        """Update account value"""
        try:
            if self.broker_type == 'alpaca' and self.broker:
                account = self.broker.get_account()
                self.account_value = float(account.portfolio_value)
            else:
                # Simulate account value
                total_position_value = sum(pos['market_value'] for pos in self.positions.values())
                self.account_value = 100000 + self.daily_pnl + total_position_value
                
        except Exception as e:
            print(f"Error updating account value: {e}")
    
    async def _check_exit_conditions(self):
        """Check stop loss and take profit conditions"""
        try:
            for symbol, position in self.positions.items():
                if position['quantity'] == 0:
                    continue
                
                current_price = live_data_manager.get_latest_price(symbol)
                if not current_price:
                    continue
                
                price = current_price['price']
                entry_price = position['avg_price']
                quantity = position['quantity']
                
                pnl_pct = (price - entry_price) / entry_price
                
                # Adjust for short positions
                if quantity < 0:
                    pnl_pct = -pnl_pct
                
                # Check stop loss
                if pnl_pct < -self.risk_limits['stop_loss_pct']:
                    print(f"🛑 Stop loss triggered for {symbol}: {pnl_pct:.1%}")
                    await self._close_position(symbol, "Stop loss")
                
                # Check take profit
                elif pnl_pct > self.risk_limits['take_profit_pct']:
                    print(f"💰 Take profit triggered for {symbol}: {pnl_pct:.1%}")
                    await self._close_position(symbol, "Take profit")
        
        except Exception as e:
            print(f"Error checking exit conditions: {e}")
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a specific position"""
        try:
            position = self.positions.get(symbol)
            if not position or position['quantity'] == 0:
                return
            
            # Close position with market order
            close_quantity = -position['quantity']
            current_price = live_data_manager.get_latest_price(symbol)
            price = current_price['price'] if current_price else position['avg_price']
            
            await self._place_order(symbol, close_quantity, price, [{'strategy': f'exit_{reason}'}])
            
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")
    
    async def _close_all_positions(self, reason: str):
        """Close all open positions"""
        print(f"🚨 CLOSING ALL POSITIONS: {reason}")
        
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol, reason)
    
    async def stop_trading(self):
        """Stop the trading engine"""
        print("Stopping trading engine...")
        self.is_trading = False
        
        # Save final performance
        self._save_performance_log()
    
    def _save_performance_log(self):
        """Save performance log"""
        try:
            log_data = {
                'session_end': datetime.now().isoformat(),
                'performance': self.performance,
                'positions': self.positions,
                'orders': {k: {**v, 'timestamp': v['timestamp'].isoformat()} for k, v in self.orders.items()},
                'final_stats': {
                    'total_trades': self.daily_trades,
                    'daily_pnl': self.daily_pnl,
                    'account_value': self.account_value
                }
            }
            
            filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            print(f"Performance log saved: {filename}")
            
        except Exception as e:
            print(f"Error saving performance log: {e}")
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        return {
            'is_trading': self.is_trading,
            'trading_mode': self.trading_mode,
            'broker_type': self.broker_type,
            'strategies_loaded': len(self.strategies),
            'active_positions': len([p for p in self.positions.values() if p['quantity'] != 0]),
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'account_value': self.account_value,
            'total_exposure': self.total_exposure,
            'performance': self.performance
        }

# Global instance
live_trading_engine = LiveTradingEngine()

async def start_live_trading(config: Dict = None, symbols: List[str] = None):
    """Start live trading"""
    global live_trading_engine
    if config:
        live_trading_engine = LiveTradingEngine(config)
    
    # Load strategies
    await live_trading_engine.load_strategies()
    
    # Start trading
    return await live_trading_engine.start_trading(symbols)

if __name__ == "__main__":
    async def test_live_trading():
        # Test configuration
        config = {
            'trading_mode': 'paper',
            # Add broker API keys for live trading
            # 'alpaca_key': 'your_alpaca_key',
            # 'alpaca_secret': 'your_alpaca_secret',
        }
        
        print("Starting live trading engine test...")
        print("This will run in simulation mode.")
        print("Press Ctrl+C to stop...")
        
        try:
            await start_live_trading(config)
        except KeyboardInterrupt:
            print("\nStopping live trading...")
            await live_trading_engine.stop_trading()
    
    # Run test
    import asyncio
    asyncio.run(test_live_trading())