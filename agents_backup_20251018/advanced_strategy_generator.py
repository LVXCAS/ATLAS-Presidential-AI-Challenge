#!/usr/bin/env python3
"""
Advanced Strategy Generator - Creates sophisticated trading strategies using AI/ML
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from itertools import product, combinations
import json
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

class AdvancedStrategyGenerator:
    """Generate sophisticated trading strategies using AI/ML"""
    
    def __init__(self):
        self.strategies = {}
        self.market_regimes = {}
        self.strategy_templates = self._initialize_strategy_templates()
        self.performance_cache = {}
        
    def _initialize_strategy_templates(self) -> Dict:
        """Initialize sophisticated strategy templates"""
        return {
            'adaptive_momentum': {
                'description': 'Adaptive momentum strategy with regime detection',
                'parameters': {
                    'fast_period': [5, 8, 12, 20],
                    'slow_period': [20, 50, 100, 200],
                    'volatility_lookback': [10, 20, 30],
                    'regime_threshold': [0.5, 0.75, 1.0, 1.5]
                },
                'signals': self._adaptive_momentum_signals
            },
            'mean_reversion_ml': {
                'description': 'ML-enhanced mean reversion with volatility clustering',
                'parameters': {
                    'lookback_period': [10, 20, 30, 50],
                    'threshold_multiplier': [1.0, 1.5, 2.0, 2.5],
                    'volatility_window': [5, 10, 20],
                    'ml_confidence_threshold': [0.6, 0.7, 0.8, 0.9]
                },
                'signals': self._mean_reversion_ml_signals
            },
            'breakout_confirmation': {
                'description': 'Multi-timeframe breakout with volume confirmation',
                'parameters': {
                    'breakout_period': [10, 20, 50],
                    'volume_threshold': [1.2, 1.5, 2.0, 3.0],
                    'confirmation_period': [2, 3, 5],
                    'atr_multiplier': [1.0, 1.5, 2.0, 2.5]
                },
                'signals': self._breakout_confirmation_signals
            },
            'statistical_arbitrage': {
                'description': 'Statistical arbitrage with cointegration',
                'parameters': {
                    'pairs_lookback': [60, 120, 252],
                    'entry_z_score': [1.5, 2.0, 2.5],
                    'exit_z_score': [0.5, 0.75, 1.0],
                    'half_life_threshold': [5, 10, 20]
                },
                'signals': self._statistical_arbitrage_signals
            },
            'volatility_targeting': {
                'description': 'Volatility targeting with regime switching',
                'parameters': {
                    'target_volatility': [0.10, 0.15, 0.20, 0.25],
                    'lookback_window': [20, 50, 100],
                    'rebalance_frequency': [1, 5, 10],
                    'max_leverage': [1.5, 2.0, 2.5, 3.0]
                },
                'signals': self._volatility_targeting_signals
            },
            'options_flow': {
                'description': 'Options flow sentiment strategy',
                'parameters': {
                    'flow_threshold': [0.6, 0.7, 0.8, 0.9],
                    'volume_percentile': [80, 90, 95],
                    'iv_rank_threshold': [30, 50, 70],
                    'gamma_exposure_limit': [0.1, 0.15, 0.2]
                },
                'signals': self._options_flow_signals
            }
        }
    
    async def generate_strategies(self, symbols: List[str], max_strategies: int = 50) -> Dict:
        """Generate optimized strategies for given symbols"""
        print("ADVANCED STRATEGY GENERATOR")
        print("=" * 50)
        print(f"Generating strategies for {len(symbols)} symbols...")
        
        all_strategies = {}
        generation_stats = {
            'total_combinations': 0,
            'tested_strategies': 0,
            'profitable_strategies': 0,
            'best_sharpe': 0,
            'generation_time': 0
        }
        
        start_time = datetime.now()
        
        # Generate strategies for each symbol
        for symbol in symbols:
            print(f"\nGenerating strategies for {symbol}...")
            
            # Get market data
            market_data = await self._get_market_data(symbol)
            if market_data is None:
                continue
            
            # Detect market regime
            regime = self._detect_market_regime(market_data)
            
            # Generate strategies for this symbol
            symbol_strategies = await self._generate_symbol_strategies(
                symbol, market_data, regime, max_strategies // len(symbols)
            )
            
            all_strategies[symbol] = symbol_strategies
            
            # Update stats
            for strategy in symbol_strategies:
                generation_stats['tested_strategies'] += 1
                if strategy['performance']['total_return'] > 0:
                    generation_stats['profitable_strategies'] += 1
                if strategy['performance']['sharpe_ratio'] > generation_stats['best_sharpe']:
                    generation_stats['best_sharpe'] = strategy['performance']['sharpe_ratio']
        
        generation_stats['generation_time'] = (datetime.now() - start_time).total_seconds()
        
        # Find top strategies across all symbols
        top_strategies = self._rank_strategies(all_strategies, max_strategies // 2)
        
        print(f"\nSTRATEGY GENERATION COMPLETE!")
        print(f"- Generated {generation_stats['tested_strategies']} strategies")
        print(f"- {generation_stats['profitable_strategies']} profitable strategies")
        print(f"- Best Sharpe ratio: {generation_stats['best_sharpe']:.2f}")
        print(f"- Generation time: {generation_stats['generation_time']:.1f} seconds")
        
        return {
            'strategies': all_strategies,
            'top_strategies': top_strategies,
            'stats': generation_stats
        }
    
    async def _get_market_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get market data for strategy generation"""
        try:
            if not YFINANCE_AVAILABLE:
                print(f"yfinance not available, using synthetic data for {symbol}")
                return self._generate_synthetic_data()
            
            print(f"Downloading {symbol} data...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"No data returned for {symbol}, using synthetic data")
                return self._generate_synthetic_data()
            
            # Fix MultiIndex columns if needed
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]
            
            # Calculate additional features
            data['Return'] = data['Close'].pct_change()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['Volatility'] = data['Return'].rolling(20).std() * np.sqrt(252)
            data['ATR'] = self._calculate_atr(data)
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            print(f"Successfully loaded {len(data)} days of {symbol} data")
            return data.dropna()
            
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            print(f"Using synthetic data for {symbol}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        # Generate realistic price series
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [100]  # Starting price
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Generate other OHLCV data
        data = pd.DataFrame(index=dates[:len(prices)])
        data['Close'] = prices
        data['Open'] = data['Close'] * np.random.uniform(0.98, 1.02, len(data))
        data['High'] = np.maximum(data['Open'], data['Close']) * np.random.uniform(1.0, 1.05, len(data))
        data['Low'] = np.minimum(data['Open'], data['Close']) * np.random.uniform(0.95, 1.0, len(data))
        data['Volume'] = np.random.randint(1000000, 10000000, len(data))
        
        # Calculate features
        data['Return'] = data['Close'].pct_change()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['Volatility'] = data['Return'].rolling(20).std() * np.sqrt(252)
        data['ATR'] = self._calculate_atr(data)
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        return data.dropna()
    
    def _detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """Detect current market regime"""
        try:
            recent_data = data.tail(50)
            
            # Volatility regime
            current_vol = recent_data['Volatility'].iloc[-1]
            historical_vol = data['Volatility'].median()
            vol_regime = 'high' if current_vol > historical_vol * 1.5 else 'normal'
            
            # Trend regime
            sma_20 = recent_data['SMA_20'].iloc[-1]
            sma_50 = recent_data['SMA_50'].iloc[-1]
            current_price = recent_data['Close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                trend_regime = 'bull'
            elif current_price < sma_20 < sma_50:
                trend_regime = 'bear'
            else:
                trend_regime = 'sideways'
            
            # Momentum regime
            momentum = recent_data['Return'].rolling(10).mean().iloc[-1]
            momentum_regime = 'strong' if abs(momentum) > 0.01 else 'weak'
            
            return {
                'volatility': vol_regime,
                'trend': trend_regime,
                'momentum': momentum_regime
            }
            
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return {'volatility': 'normal', 'trend': 'sideways', 'momentum': 'weak'}
    
    async def _generate_symbol_strategies(self, symbol: str, data: pd.DataFrame, 
                                        regime: Dict, max_strategies: int) -> List[Dict]:
        """Generate strategies for a specific symbol"""
        strategies = []
        
        # Filter strategy templates based on regime
        suitable_templates = self._filter_templates_by_regime(regime)
        
        for template_name, template in suitable_templates.items():
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(
                template['parameters'], max_strategies // len(suitable_templates)
            )
            
            for params in param_combinations:
                try:
                    # Generate signals
                    signals = template['signals'](data, params)
                    
                    # Calculate performance
                    performance = self._calculate_strategy_performance(data, signals)
                    
                    # Only keep reasonable strategies (relaxed criteria for demo)
                    if (performance['total_return'] > -0.1 and 
                        performance['sharpe_ratio'] > -0.5 and
                        performance['max_drawdown'] < 0.8):
                        
                        strategy = {
                            'symbol': symbol,
                            'name': f"{template_name}_{symbol}",
                            'template': template_name,
                            'description': template['description'],
                            'parameters': params,
                            'performance': performance,
                            'regime_suitability': self._calculate_regime_suitability(template_name, regime),
                            'signals': signals.tail(252).to_dict()  # Keep last year of signals
                        }
                        
                        strategies.append(strategy)
                        
                except Exception as e:
                    continue  # Skip failed strategies
        
        # Sort by risk-adjusted return
        strategies.sort(key=lambda x: x['performance']['sharpe_ratio'], reverse=True)
        
        return strategies[:max_strategies]
    
    def _filter_templates_by_regime(self, regime: Dict) -> Dict:
        """Filter strategy templates based on market regime"""
        suitable_templates = {}
        
        # All templates are suitable for normal conditions
        if regime['volatility'] == 'normal' and regime['trend'] == 'sideways':
            suitable_templates = self.strategy_templates.copy()
        else:
            # High volatility - favor mean reversion and volatility strategies
            if regime['volatility'] == 'high':
                suitable_templates.update({
                    'mean_reversion_ml': self.strategy_templates['mean_reversion_ml'],
                    'volatility_targeting': self.strategy_templates['volatility_targeting'],
                    'options_flow': self.strategy_templates['options_flow']
                })
            
            # Strong trends - favor momentum and breakout
            if regime['trend'] in ['bull', 'bear'] and regime['momentum'] == 'strong':
                suitable_templates.update({
                    'adaptive_momentum': self.strategy_templates['adaptive_momentum'],
                    'breakout_confirmation': self.strategy_templates['breakout_confirmation']
                })
            
            # Sideways markets - favor mean reversion and statistical arbitrage
            if regime['trend'] == 'sideways':
                suitable_templates.update({
                    'mean_reversion_ml': self.strategy_templates['mean_reversion_ml'],
                    'statistical_arbitrage': self.strategy_templates['statistical_arbitrage']
                })
        
        return suitable_templates if suitable_templates else self.strategy_templates
    
    def _generate_parameter_combinations(self, parameters: Dict, max_combinations: int) -> List[Dict]:
        """Generate parameter combinations with smart sampling"""
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        
        # Calculate total possible combinations
        total_combinations = np.prod([len(values) for values in param_values])
        
        if total_combinations <= max_combinations:
            # Use all combinations
            combinations_list = list(product(*param_values))
        else:
            # Smart sampling - use Latin hypercube sampling approximation
            combinations_list = []
            np.random.seed(42)  # For reproducibility
            
            for _ in range(max_combinations):
                combination = []
                for values in param_values:
                    combination.append(np.random.choice(values))
                combinations_list.append(tuple(combination))
            
            # Remove duplicates
            combinations_list = list(set(combinations_list))
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations_list:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        return param_combinations[:max_combinations]
    
    def _adaptive_momentum_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Generate adaptive momentum signals"""
        try:
            fast_sma = data['Close'].rolling(params['fast_period']).mean()
            slow_sma = data['Close'].rolling(params['slow_period']).mean()
            volatility = data['Return'].rolling(params['volatility_lookback']).std()
            
            # Adaptive threshold based on volatility regime
            threshold = volatility * params['regime_threshold']
            
            # Generate signals
            momentum_score = (fast_sma - slow_sma) / slow_sma
            signals = pd.Series(0, index=data.index)
            
            # Buy when momentum is strong positive
            signals[momentum_score > threshold] = 1
            # Sell when momentum is strong negative
            signals[momentum_score < -threshold] = -1
            
            return signals
            
        except Exception as e:
            return pd.Series(0, index=data.index)
    
    def _mean_reversion_ml_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Generate ML-enhanced mean reversion signals"""
        try:
            # Calculate z-score
            price_mean = data['Close'].rolling(params['lookback_period']).mean()
            price_std = data['Close'].rolling(params['lookback_period']).std()
            z_score = (data['Close'] - price_mean) / price_std
            
            # Volatility adjustment
            vol_adjustment = data['Volatility'].rolling(params['volatility_window']).mean()
            adjusted_threshold = params['threshold_multiplier'] * vol_adjustment
            
            # Generate base signals
            signals = pd.Series(0, index=data.index)
            signals[z_score < -adjusted_threshold] = 1  # Buy when oversold
            signals[z_score > adjusted_threshold] = -1   # Sell when overbought
            
            # ML confidence filter (simplified)
            if SKLEARN_AVAILABLE:
                confidence = np.abs(z_score) / 2.0  # Simplified confidence measure
                signals[confidence < params['ml_confidence_threshold']] = 0
            
            return signals
            
        except Exception as e:
            return pd.Series(0, index=data.index)
    
    def _breakout_confirmation_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Generate breakout confirmation signals"""
        try:
            # Calculate breakout levels
            high_breakout = data['High'].rolling(params['breakout_period']).max()
            low_breakout = data['Low'].rolling(params['breakout_period']).min()
            
            # Volume confirmation
            avg_volume = data['Volume'].rolling(params['breakout_period']).mean()
            volume_spike = data['Volume'] > avg_volume * params['volume_threshold']
            
            # ATR for dynamic stops
            atr = data['ATR'] * params['atr_multiplier']
            
            signals = pd.Series(0, index=data.index)
            
            # Upward breakout with volume confirmation
            upward_breakout = (data['Close'] > high_breakout.shift(1)) & volume_spike
            signals[upward_breakout] = 1
            
            # Downward breakout with volume confirmation
            downward_breakout = (data['Close'] < low_breakout.shift(1)) & volume_spike
            signals[downward_breakout] = -1
            
            return signals
            
        except Exception as e:
            return pd.Series(0, index=data.index)
    
    def _statistical_arbitrage_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Generate statistical arbitrage signals (simplified for single asset)"""
        try:
            # Use price vs moving average as proxy for pairs trading
            long_ma = data['Close'].rolling(params['pairs_lookback']).mean()
            spread = data['Close'] - long_ma
            spread_mean = spread.rolling(params['pairs_lookback']).mean()
            spread_std = spread.rolling(params['pairs_lookback']).std()
            
            z_score = (spread - spread_mean) / spread_std
            
            signals = pd.Series(0, index=data.index)
            
            # Enter when z-score exceeds threshold
            signals[z_score > params['entry_z_score']] = -1  # Short when above mean
            signals[z_score < -params['entry_z_score']] = 1   # Long when below mean
            
            # Exit when z-score approaches zero
            exit_condition = np.abs(z_score) < params['exit_z_score']
            signals[exit_condition] = 0
            
            return signals
            
        except Exception as e:
            return pd.Series(0, index=data.index)
    
    def _volatility_targeting_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Generate volatility targeting signals"""
        try:
            # Calculate realized volatility
            realized_vol = data['Return'].rolling(params['lookback_window']).std() * np.sqrt(252)
            
            # Calculate position size based on volatility targeting
            target_vol = params['target_volatility']
            position_size = np.minimum(
                target_vol / realized_vol,
                params['max_leverage']
            )
            
            # Simple momentum signal
            momentum = data['Close'].pct_change(params['rebalance_frequency'])
            base_signal = np.where(momentum > 0, 1, -1)
            
            # Scale by volatility target
            signals = pd.Series(base_signal * position_size, index=data.index)
            signals = signals.fillna(0)
            
            return signals
            
        except Exception as e:
            return pd.Series(0, index=data.index)
    
    def _options_flow_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Generate options flow signals (simplified - would need real options data)"""
        try:
            # Simulate options flow using volume and price action
            volume_percentile = data['Volume'].rolling(50).rank(pct=True) * 100
            price_momentum = data['Close'].pct_change(5)
            
            # Simulate IV rank using volatility
            vol_rank = data['Volatility'].rolling(252).rank(pct=True) * 100
            
            # Generate signals based on simulated flow
            signals = pd.Series(0, index=data.index)
            
            # Bullish flow conditions
            bullish_flow = (
                (volume_percentile > params['volume_percentile']) &
                (price_momentum > 0) &
                (vol_rank > params['iv_rank_threshold'])
            )
            
            # Bearish flow conditions  
            bearish_flow = (
                (volume_percentile > params['volume_percentile']) &
                (price_momentum < 0) &
                (vol_rank > params['iv_rank_threshold'])
            )
            
            signals[bullish_flow] = 1
            signals[bearish_flow] = -1
            
            return signals
            
        except Exception as e:
            return pd.Series(0, index=data.index)
    
    def _calculate_strategy_performance(self, data: pd.DataFrame, signals: pd.Series) -> Dict:
        """Calculate comprehensive strategy performance metrics"""
        try:
            # Align signals with returns
            returns = data['Return'].fillna(0)
            signals = signals.fillna(0).shift(1)  # Use previous day signal
            
            # Calculate strategy returns
            strategy_returns = returns * signals
            strategy_returns = strategy_returns.fillna(0)
            
            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            annual_vol = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Win rate
            winning_trades = (strategy_returns > 0).sum()
            total_trades = (signals != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'annual_volatility': float(annual_vol),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'calmar_ratio': float(calmar_ratio),
                'win_rate': float(win_rate),
                'total_trades': int(total_trades)
            }
            
        except Exception as e:
            print(f"Performance calculation error: {e}")
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }
    
    def _calculate_regime_suitability(self, template_name: str, regime: Dict) -> float:
        """Calculate how suitable a strategy is for current regime"""
        suitability_map = {
            'adaptive_momentum': {
                'bull': 0.9, 'bear': 0.7, 'sideways': 0.5,
                'high': 0.6, 'normal': 0.8,
                'strong': 0.9, 'weak': 0.4
            },
            'mean_reversion_ml': {
                'bull': 0.6, 'bear': 0.6, 'sideways': 0.9,
                'high': 0.8, 'normal': 0.7,
                'strong': 0.4, 'weak': 0.8
            },
            'breakout_confirmation': {
                'bull': 0.9, 'bear': 0.8, 'sideways': 0.4,
                'high': 0.7, 'normal': 0.8,
                'strong': 0.9, 'weak': 0.5
            },
            'statistical_arbitrage': {
                'bull': 0.5, 'bear': 0.5, 'sideways': 0.9,
                'high': 0.6, 'normal': 0.8,
                'strong': 0.3, 'weak': 0.9
            },
            'volatility_targeting': {
                'bull': 0.7, 'bear': 0.7, 'sideways': 0.7,
                'high': 0.9, 'normal': 0.6,
                'strong': 0.6, 'weak': 0.6
            },
            'options_flow': {
                'bull': 0.8, 'bear': 0.8, 'sideways': 0.6,
                'high': 0.9, 'normal': 0.7,
                'strong': 0.8, 'weak': 0.5
            }
        }
        
        if template_name not in suitability_map:
            return 0.5
        
        template_scores = suitability_map[template_name]
        
        # Calculate weighted average suitability
        suitability = (
            template_scores.get(regime['trend'], 0.5) * 0.4 +
            template_scores.get(regime['volatility'], 0.5) * 0.4 +
            template_scores.get(regime['momentum'], 0.5) * 0.2
        )
        
        return suitability
    
    def _rank_strategies(self, all_strategies: Dict, top_n: int) -> List[Dict]:
        """Rank strategies across all symbols"""
        all_strategy_list = []
        
        for symbol, strategies in all_strategies.items():
            all_strategy_list.extend(strategies)
        
        # Multi-criteria ranking
        for strategy in all_strategy_list:
            perf = strategy['performance']
            regime_suit = strategy['regime_suitability']
            
            # Composite score combining multiple factors
            composite_score = (
                perf['sharpe_ratio'] * 0.3 +
                perf['calmar_ratio'] * 0.2 +
                perf['annual_return'] * 0.2 +
                (1 - perf['max_drawdown']) * 0.15 +
                perf['win_rate'] * 0.1 +
                regime_suit * 0.05
            )
            
            strategy['composite_score'] = composite_score
        
        # Sort by composite score
        all_strategy_list.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return all_strategy_list[:top_n]
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = tr.rolling(period).mean()
            
            return atr
        except Exception:
            return pd.Series(0, index=data.index)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception:
            return pd.Series(50, index=prices.index)
    
    def save_strategies(self, strategies: Dict, filename: str = None):
        """Save generated strategies to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_strategies_{timestamp}.json"
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj
            
            # Deep convert all numpy types
            import json
            strategies_json = json.loads(json.dumps(strategies, default=convert_numpy))
            
            with open(filename, 'w') as f:
                json.dump(strategies_json, f, indent=2)
            
            print(f"Strategies saved to {filename}")
            
        except Exception as e:
            print(f"Error saving strategies: {e}")

# Global instance
advanced_strategy_generator = AdvancedStrategyGenerator()

async def generate_advanced_strategies(symbols: List[str], max_strategies: int = 50) -> Dict:
    """Generate advanced strategies for given symbols"""
    return await advanced_strategy_generator.generate_strategies(symbols, max_strategies)

if __name__ == "__main__":
    async def test_strategy_generator():
        symbols = ['SPY', 'QQQ', 'AAPL']
        results = await generate_advanced_strategies(symbols, max_strategies=30)
        
        print("\nTOP STRATEGIES:")
        print("-" * 60)
        for i, strategy in enumerate(results['top_strategies'][:10]):
            perf = strategy['performance']
            print(f"{i+1:2d}. {strategy['name']}")
            print(f"    Template: {strategy['template']}")
            print(f"    Sharpe: {perf['sharpe_ratio']:.2f} | Return: {perf['annual_return']:.1%}")
            print(f"    Max DD: {perf['max_drawdown']:.1%} | Win Rate: {perf['win_rate']:.1%}")
            print(f"    Regime Suit: {strategy['regime_suitability']:.2f}")
            print()
        
        # Save strategies
        advanced_strategy_generator.save_strategies(results)
    
    # Run test
    import asyncio
    asyncio.run(test_strategy_generator())