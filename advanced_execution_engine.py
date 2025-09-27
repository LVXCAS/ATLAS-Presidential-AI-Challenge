#!/usr/bin/env python3
"""
ADVANCED EXECUTION ENGINE WITH REAL-TIME LEARNING
Executes trades with intelligent order management and learns from every outcome
Optimizes execution quality and adapts to market conditions in real-time
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - EXECUTE - %(message)s',
    handlers=[
        logging.FileHandler('advanced_execution_engine.log'),
        logging.StreamHandler()
    ]
)

class AdvancedExecutionEngine:
    def __init__(self):
        self.execution_history = []
        self.live_positions = {}
        self.pending_orders = {}
        self.execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'average_slippage': 0.0,
            'average_fill_time': 0.0,
            'win_rate': 0.0,
            'total_pnl': 0.0
        }

        # Learning models for execution optimization
        self.slippage_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.timing_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()

        # Real-time execution monitoring
        self.execution_queue = deque()
        self.monitoring_active = False
        self.learning_buffer = []

        # Initialize Alpaca
        self.alpaca = None
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if api_key and secret_key:
                self.alpaca = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
                logging.info("EXECUTE: Connected to Alpaca for advanced execution")
        except Exception as e:
            logging.error(f"EXECUTE: Alpaca connection failed: {e}")

        # Execution configuration
        self.config = {
            'max_slippage_tolerance': 0.005,  # 0.5% max slippage
            'order_timeout': 300,  # 5 minutes
            'partial_fill_threshold': 0.8,  # Accept 80%+ fills
            'adaptive_sizing': True,
            'smart_routing': True,
            'market_impact_protection': True
        }

        logging.info("EXECUTE: Advanced Execution Engine initialized")
        logging.info("EXECUTE: Real-time learning and adaptive execution enabled")

    async def execute_strategy(self, strategy_data, market_conditions=None):
        """Execute a complete strategy with intelligent order management"""
        try:
            strategy_id = strategy_data.get('id', 'unknown')
            symbols = strategy_data.get('symbols', [])

            logging.info(f"EXECUTE: Executing strategy {strategy_id} with {len(symbols)} symbols")

            execution_results = []

            for symbol in symbols:
                # Determine optimal execution parameters
                execution_params = await self._optimize_execution_parameters(
                    symbol, strategy_data, market_conditions
                )

                # Execute position
                result = await self._execute_position(symbol, execution_params, strategy_data)

                if result:
                    execution_results.append(result)

                    # Learn from execution immediately
                    await self._learn_from_execution(result)

            # Update strategy execution metrics
            strategy_metrics = self._calculate_strategy_metrics(execution_results)

            logging.info(f"EXECUTE: Strategy {strategy_id} execution complete")
            logging.info(f"EXECUTE: {len(execution_results)}/{len(symbols)} positions executed successfully")

            return {
                'strategy_id': strategy_id,
                'execution_results': execution_results,
                'strategy_metrics': strategy_metrics,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"EXECUTE: Strategy execution error: {e}")
            return None

    async def _optimize_execution_parameters(self, symbol, strategy_data, market_conditions):
        """Optimize execution parameters for specific symbol and conditions"""
        try:
            # Base parameters
            params = {
                'symbol': symbol,
                'order_type': 'market',
                'time_in_force': 'day',
                'position_size': strategy_data.get('position_sizing', {}).get('max_position_size', 0.05),
                'urgency': 'normal'
            }

            # Get real-time market data for optimization
            market_data = await self._get_realtime_market_data(symbol)

            if market_data:
                # Analyze market conditions
                spread = market_data.get('spread', 0.01)
                volume = market_data.get('volume', 0)
                volatility = market_data.get('volatility', 0.02)

                # Optimize order type based on conditions
                if spread > 0.005:  # Wide spread
                    params['order_type'] = 'limit'
                    params['limit_offset'] = spread * 0.3  # Place inside spread

                if volume < 1000:  # Low volume
                    params['position_size'] *= 0.5  # Reduce size
                    params['order_type'] = 'limit'

                if volatility > 0.05:  # High volatility
                    params['order_type'] = 'limit'
                    params['urgency'] = 'high'

                # Use ML model to predict optimal execution approach
                if len(self.learning_buffer) > 10:
                    predicted_approach = self._predict_optimal_approach(symbol, market_data)
                    if predicted_approach:
                        params.update(predicted_approach)

            logging.info(f"EXECUTE: Optimized parameters for {symbol}: {params['order_type']} order")

            return params

        except Exception as e:
            logging.error(f"EXECUTE: Parameter optimization error: {e}")
            return {'symbol': symbol, 'order_type': 'market', 'time_in_force': 'day'}

    async def _execute_position(self, symbol, params, strategy_data):
        """Execute individual position with advanced order management"""
        try:
            logging.info(f"EXECUTE: Executing {symbol} position with {params['order_type']} order")

            # Calculate position size in shares
            account = self.alpaca.get_account() if self.alpaca else None

            if account:
                buying_power = float(account.buying_power)
                position_value = buying_power * params['position_size']
            else:
                position_value = 50000 * params['position_size']  # Fallback

            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None

            quantity = max(1, int(position_value / current_price))

            # Record execution start time
            execution_start = time.time()

            if self.alpaca and not symbol.endswith('-USD'):
                try:
                    # Determine order side (buy for most strategies)
                    side = 'buy'  # Can be enhanced based on strategy type

                    # Submit main order
                    if params['order_type'] == 'limit':
                        limit_price = current_price * (1 + params.get('limit_offset', 0.001))
                        order = self.alpaca.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side=side,
                            type='limit',
                            limit_price=round(limit_price, 2),
                            time_in_force=params['time_in_force']
                        )
                    else:
                        order = self.alpaca.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side=side,
                            type='market',
                            time_in_force=params['time_in_force']
                        )

                    # Monitor order execution
                    fill_result = await self._monitor_order_execution(order.id, params)

                    if fill_result['status'] == 'filled':
                        execution_time = time.time() - execution_start

                        # Set up exit orders (profit target and stop loss)
                        await self._setup_exit_orders(symbol, quantity, fill_result['fill_price'], strategy_data)

                        result = {
                            'symbol': symbol,
                            'order_id': order.id,
                            'status': 'executed',
                            'side': side,
                            'quantity': quantity,
                            'entry_price': fill_result['fill_price'],
                            'execution_time': execution_time,
                            'slippage': fill_result.get('slippage', 0.0),
                            'timestamp': datetime.now().isoformat(),
                            'strategy_id': strategy_data.get('id'),
                            'params_used': params
                        }

                        logging.info(f"EXECUTE: ✅ {symbol} executed - {quantity} @ ${fill_result['fill_price']:.2f}")
                        return result

                    else:
                        logging.warning(f"EXECUTE: ⚠️ {symbol} execution failed - {fill_result['status']}")
                        return None

                except Exception as e:
                    logging.error(f"EXECUTE: Order execution failed for {symbol}: {e}")
                    return None
            else:
                # Simulate execution for crypto or when Alpaca unavailable
                simulated_fill_price = current_price * (1 + np.random.normal(0, 0.001))
                execution_time = np.random.uniform(0.5, 2.0)

                result = {
                    'symbol': symbol,
                    'order_id': f'SIM_{symbol}_{int(time.time())}',
                    'status': 'simulated',
                    'side': 'buy',
                    'quantity': quantity,
                    'entry_price': simulated_fill_price,
                    'execution_time': execution_time,
                    'slippage': abs(simulated_fill_price - current_price) / current_price,
                    'timestamp': datetime.now().isoformat(),
                    'strategy_id': strategy_data.get('id'),
                    'params_used': params
                }

                logging.info(f"EXECUTE: 🔄 {symbol} simulated - {quantity} @ ${simulated_fill_price:.2f}")
                return result

        except Exception as e:
            logging.error(f"EXECUTE: Position execution error for {symbol}: {e}")
            return None

    async def _monitor_order_execution(self, order_id, params):
        """Monitor order execution and handle partial fills"""
        try:
            max_wait_time = params.get('order_timeout', 300)
            check_interval = 2  # Check every 2 seconds
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                order = self.alpaca.get_order(order_id)

                if order.status == 'filled':
                    fill_price = float(order.filled_avg_price)
                    return {
                        'status': 'filled',
                        'fill_price': fill_price,
                        'filled_qty': int(order.filled_qty)
                    }

                elif order.status in ['canceled', 'rejected', 'expired']:
                    return {'status': order.status}

                elif order.status == 'partially_filled':
                    filled_qty = int(order.filled_qty)
                    total_qty = int(order.qty)
                    fill_percentage = filled_qty / total_qty

                    if fill_percentage >= self.config['partial_fill_threshold']:
                        # Accept partial fill
                        self.alpaca.cancel_order(order_id)
                        return {
                            'status': 'filled',
                            'fill_price': float(order.filled_avg_price),
                            'filled_qty': filled_qty
                        }

                await asyncio.sleep(check_interval)
                elapsed_time += check_interval

            # Timeout reached
            return {'status': 'timeout'}

        except Exception as e:
            logging.error(f"EXECUTE: Order monitoring error: {e}")
            return {'status': 'error'}

    async def _setup_exit_orders(self, symbol, quantity, entry_price, strategy_data):
        """Set up profit target and stop loss orders"""
        try:
            risk_mgmt = strategy_data.get('risk_management', {})
            profit_target_pct = risk_mgmt.get('profit_target', 0.06)  # 6% default
            stop_loss_pct = risk_mgmt.get('stop_loss', 0.03)  # 3% default

            profit_price = entry_price * (1 + profit_target_pct)
            stop_price = entry_price * (1 - stop_loss_pct)

            if self.alpaca:
                # Submit profit target order
                profit_order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='limit',
                    limit_price=round(profit_price, 2),
                    time_in_force='gtc'
                )

                # Submit stop loss order
                stop_order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='stop',
                    stop_price=round(stop_price, 2),
                    time_in_force='gtc'
                )

                logging.info(f"EXECUTE: Exit orders set for {symbol}: Profit @ ${profit_price:.2f}, Stop @ ${stop_price:.2f}")

                return {
                    'profit_order_id': profit_order.id,
                    'stop_order_id': stop_order.id
                }

        except Exception as e:
            logging.error(f"EXECUTE: Exit order setup error for {symbol}: {e}")
            return None

    async def _learn_from_execution(self, execution_result):
        """Learn from execution outcome in real-time"""
        try:
            # Add to learning buffer
            self.learning_buffer.append(execution_result)

            # Keep buffer manageable
            if len(self.learning_buffer) > 1000:
                self.learning_buffer = self.learning_buffer[-500:]

            # Update execution metrics
            self.execution_metrics['total_executions'] += 1

            if execution_result['status'] in ['executed', 'simulated']:
                self.execution_metrics['successful_executions'] += 1

                # Update slippage tracking
                slippage = execution_result.get('slippage', 0.0)
                current_avg_slippage = self.execution_metrics['average_slippage']
                total_execs = self.execution_metrics['total_executions']

                self.execution_metrics['average_slippage'] = (
                    (current_avg_slippage * (total_execs - 1) + slippage) / total_execs
                )

                # Update timing metrics
                exec_time = execution_result.get('execution_time', 0.0)
                current_avg_time = self.execution_metrics['average_fill_time']

                self.execution_metrics['average_fill_time'] = (
                    (current_avg_time * (total_execs - 1) + exec_time) / total_execs
                )

            # Retrain models periodically
            if len(self.learning_buffer) % 25 == 0 and len(self.learning_buffer) >= 25:
                await self._retrain_execution_models()

            logging.info(f"EXECUTE: Learning from execution - Total: {self.execution_metrics['total_executions']}, Success Rate: {self.execution_metrics['successful_executions']/max(self.execution_metrics['total_executions'],1):.2%}")

        except Exception as e:
            logging.error(f"EXECUTE: Learning error: {e}")

    async def _retrain_execution_models(self):
        """Retrain ML models based on execution history"""
        try:
            logging.info("EXECUTE: Retraining execution optimization models")

            if len(self.learning_buffer) < 10:
                return

            # Prepare training data
            X_slippage = []
            y_slippage = []
            X_timing = []
            y_timing = []

            for result in self.learning_buffer[-100:]:  # Use last 100 executions
                if result['status'] in ['executed', 'simulated']:
                    # Features: symbol type, market conditions, order type, etc.
                    features = [
                        1 if result['symbol'].endswith('-USD') else 0,  # Is crypto
                        1 if result['params_used']['order_type'] == 'limit' else 0,  # Order type
                        result.get('quantity', 0) / 100,  # Normalized quantity
                        result.get('entry_price', 100) / 100  # Normalized price
                    ]

                    # Slippage prediction
                    X_slippage.append(features)
                    y_slippage.append(result.get('slippage', 0.0))

                    # Timing prediction
                    X_timing.append(features)
                    y_timing.append(result.get('execution_time', 1.0))

            if len(X_slippage) >= 10:
                # Train slippage model
                self.slippage_model.fit(X_slippage, y_slippage)

                # Train timing model
                self.timing_model.fit(X_timing, y_timing)

                logging.info("EXECUTE: Models retrained successfully")

        except Exception as e:
            logging.error(f"EXECUTE: Model retraining error: {e}")

    def _predict_optimal_approach(self, symbol, market_data):
        """Predict optimal execution approach using ML models"""
        try:
            features = [
                1 if symbol.endswith('-USD') else 0,
                market_data.get('spread', 0.01),
                market_data.get('volume', 1000) / 10000,
                market_data.get('volatility', 0.02)
            ]

            # Predict slippage for different approaches
            limit_features = features + [1]  # Limit order
            market_features = features + [0]  # Market order

            predicted_limit_slippage = self.slippage_model.predict([limit_features])[0]
            predicted_market_slippage = self.slippage_model.predict([market_features])[0]

            # Choose approach with lower predicted slippage
            if predicted_limit_slippage < predicted_market_slippage:
                return {'order_type': 'limit', 'predicted_slippage': predicted_limit_slippage}
            else:
                return {'order_type': 'market', 'predicted_slippage': predicted_market_slippage}

        except Exception as e:
            logging.error(f"EXECUTE: Prediction error: {e}")
            return None

    async def _get_realtime_market_data(self, symbol):
        """Get real-time market data for execution optimization"""
        try:
            # Simplified market data (in production, would use real-time feeds)
            return {
                'spread': np.random.uniform(0.001, 0.01),
                'volume': np.random.randint(1000, 50000),
                'volatility': np.random.uniform(0.01, 0.08),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"EXECUTE: Market data error: {e}")
            return None

    async def _get_current_price(self, symbol):
        """Get current market price for symbol"""
        try:
            if self.alpaca and not symbol.endswith('-USD'):
                # Get latest trade
                latest_trade = self.alpaca.get_latest_trade(symbol)
                return float(latest_trade.price)
            else:
                # Fallback to random price for demo
                return np.random.uniform(100, 500)

        except Exception as e:
            logging.error(f"EXECUTE: Price lookup error: {e}")
            return None

    def _calculate_strategy_metrics(self, execution_results):
        """Calculate metrics for strategy execution"""
        if not execution_results:
            return {}

        successful_executions = [r for r in execution_results if r['status'] in ['executed', 'simulated']]

        metrics = {
            'total_positions': len(execution_results),
            'successful_positions': len(successful_executions),
            'success_rate': len(successful_executions) / len(execution_results),
            'average_slippage': np.mean([r.get('slippage', 0) for r in successful_executions]) if successful_executions else 0,
            'average_execution_time': np.mean([r.get('execution_time', 0) for r in successful_executions]) if successful_executions else 0,
            'total_capital_deployed': sum([r.get('quantity', 0) * r.get('entry_price', 0) for r in successful_executions])
        }

        return metrics

    async def get_execution_report(self):
        """Generate comprehensive execution performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_metrics': self.execution_metrics,
                'recent_executions': self.learning_buffer[-20:] if self.learning_buffer else [],
                'model_status': {
                    'slippage_model_trained': hasattr(self.slippage_model, 'feature_importances_'),
                    'timing_model_trained': hasattr(self.timing_model, 'feature_importances_'),
                    'learning_buffer_size': len(self.learning_buffer)
                },
                'configuration': self.config
            }

            filename = f'execution_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logging.info(f"EXECUTE: Execution report saved to {filename}")
            return report

        except Exception as e:
            logging.error(f"EXECUTE: Report generation error: {e}")
            return None

async def main():
    """Test the advanced execution engine"""
    logging.info("=" * 60)
    logging.info("ADVANCED EXECUTION ENGINE TEST")
    logging.info("=" * 60)

    engine = AdvancedExecutionEngine()

    # Test strategy
    test_strategy = {
        'id': 'TEST_STRATEGY_001',
        'symbols': ['AAPL', 'TSLA'],
        'position_sizing': {'max_position_size': 0.05},
        'risk_management': {
            'profit_target': 0.06,
            'stop_loss': 0.03
        }
    }

    # Execute strategy
    result = await engine.execute_strategy(test_strategy)

    if result:
        logging.info(f"Strategy execution completed: {result['strategy_metrics']}")

    # Generate report
    await engine.get_execution_report()

if __name__ == "__main__":
    asyncio.run(main())