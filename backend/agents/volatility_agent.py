"""
Volatility Agent for Bloomberg Terminal
Advanced volatility trading and modeling agent.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import math

from agents.base_agent import BaseAgent, TradingSignal, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class VolatilityAgent(BaseAgent):
    """
    Volatility trading agent specializing in:
    - Volatility surface modeling and arbitrage
    - VIX and volatility index trading
    - Volatility risk premium strategies
    - GARCH-based volatility forecasting
    - Options volatility smile analysis
    - Term structure volatility trading
    - Stochastic volatility modeling
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        default_config = {
            'volatility_lookback': 30,
            'realized_vol_window': 20,
            'vol_forecast_horizon': 5,
            'garch_alpha': 0.1,
            'garch_beta': 0.85,
            'vol_risk_premium_threshold': 0.05,
            'min_vol_level': 0.1,
            'max_vol_level': 0.8,
            'vol_regime_threshold': 0.3,
            'term_structure_periods': [7, 14, 30, 60, 90],
            'volatility_smile_points': [-2, -1, 0, 1, 2],  # Moneyness points
            'implied_vol_threshold': 0.02,
            'vol_clustering_lookback': 10,
            'jump_detection_threshold': 3.0,  # Standard deviations
            'regime_change_sensitivity': 0.8,
            'vol_momentum_period': 5
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            name="VolatilityAgent",
            symbols=symbols,
            config=default_config
        )
        
        # Agent-specific state
        self.realized_vol_history: Dict[str, List[float]] = {}
        self.implied_vol_history: Dict[str, List[float]] = {}
        self.vol_models: Dict[str, Dict] = {}
        self.volatility_regimes: Dict[str, str] = {}  # low, medium, high
        self.vol_surface_cache: Dict[str, Dict] = {}
        self.jump_events: Dict[str, List] = {}
        
    async def initialize(self) -> None:
        """Initialize volatility-specific components."""
        logger.info(f"Initializing {self.name} for volatility modeling")
        
        # Initialize volatility tracking for all symbols
        for symbol in self.symbols:
            self.realized_vol_history[symbol] = []
            self.implied_vol_history[symbol] = []
            self.vol_models[symbol] = {
                'garch_params': {'omega': 0.0001, 'alpha': 0.1, 'beta': 0.85},
                'last_forecast': 0.0,
                'forecast_accuracy': 0.0
            }
            self.volatility_regimes[symbol] = 'medium'
            self.jump_events[symbol] = []
        
        # Initialize historical volatility
        await self._initialize_volatility_history()
        
        logger.info(f"{self.name} initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup volatility-specific resources."""
        self.realized_vol_history.clear()
        self.implied_vol_history.clear()
        self.vol_models.clear()
        self.volatility_regimes.clear()
        self.vol_surface_cache.clear()
        self.jump_events.clear()
        logger.info(f"{self.name} cleanup completed")
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate volatility-based trading signal.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None if no volatility opportunity
        """
        try:
            # Calculate volatility features
            features = await self.calculate_features(symbol)
            if not features:
                return None
            
            # Analyze volatility patterns
            vol_analysis = await self._analyze_volatility_patterns(symbol, features)
            if not vol_analysis:
                return None
            
            # Generate signal based on analysis
            signal = await self._generate_volatility_signal(symbol, vol_analysis, features)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating volatility signal for {symbol}: {e}")
            return None
    
    async def calculate_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate volatility-specific features.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of feature names to values
        """
        try:
            # Check cache first
            cache_key = f"volatility_features:{symbol}"
            cached_features = await self.get_cached_feature(cache_key, ttl=300)
            if cached_features:
                return cached_features
            
            features = {}
            
            # Update volatility histories
            await self._update_volatility_metrics(symbol)
            
            # Realized volatility features
            realized_vol_features = await self._calculate_realized_vol_features(symbol)
            features.update(realized_vol_features)
            
            # GARCH model features
            garch_features = await self._calculate_garch_features(symbol)
            features.update(garch_features)
            
            # Volatility regime features
            regime_features = await self._calculate_regime_features(symbol)
            features.update(regime_features)
            
            # Jump detection features
            jump_features = await self._calculate_jump_features(symbol)
            features.update(jump_features)
            
            # Volatility term structure features
            term_structure_features = await self._calculate_term_structure_features(symbol)
            features.update(term_structure_features)
            
            # Volatility risk premium features
            risk_premium_features = await self._calculate_risk_premium_features(symbol)
            features.update(risk_premium_features)
            
            # Cache the features
            await self.cache_feature(cache_key, features, ttl=300)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating volatility features for {symbol}: {e}")
            return {}
    
    async def _initialize_volatility_history(self) -> None:
        """Initialize historical volatility data."""
        try:
            for symbol in self.symbols:
                # Get historical data
                df = await self.get_market_data(symbol, self.config['volatility_lookback'])
                if df is None or df.empty or 'price' not in df.columns:
                    continue
                
                # Calculate historical realized volatility
                returns = df['price'].pct_change().dropna()
                
                # Rolling volatility calculation
                for window in [5, 10, 20, 30]:
                    if len(returns) >= window:
                        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                        self.realized_vol_history[symbol].extend(rolling_vol.dropna().tolist()[-10:])
                
                # Initialize implied volatility (placeholder - would get from options data)
                implied_vols = np.random.normal(0.25, 0.05, 10).clip(0.1, 0.8)
                self.implied_vol_history[symbol].extend(implied_vols.tolist())
                
        except Exception as e:
            logger.error(f"Error initializing volatility history: {e}")
    
    async def _update_volatility_metrics(self, symbol: str) -> None:
        """Update current volatility metrics."""
        try:
            # Get recent data
            df = await self.get_market_data(symbol, self.config['realized_vol_window'])
            if df is None or df.empty or 'price' not in df.columns:
                return
            
            # Calculate current realized volatility
            returns = df['price'].pct_change().dropna()
            if len(returns) >= 10:
                current_realized_vol = returns.std() * np.sqrt(252)
                
                # Update history
                self.realized_vol_history[symbol].append(current_realized_vol)
                if len(self.realized_vol_history[symbol]) > 50:
                    self.realized_vol_history[symbol] = self.realized_vol_history[symbol][-50:]
                
                # Update GARCH model
                self._update_garch_model(symbol, returns)
                
                # Update volatility regime
                self._update_volatility_regime(symbol, current_realized_vol)
                
                # Check for jumps
                self._detect_jumps(symbol, returns)
            
        except Exception as e:
            logger.error(f"Error updating volatility metrics: {e}")
    
    def _update_garch_model(self, symbol: str, returns: pd.Series) -> None:
        """Update GARCH model parameters."""
        try:
            if len(returns) < 20:
                return
            
            # Simple GARCH(1,1) update (simplified implementation)
            model = self.vol_models[symbol]
            
            # Calculate squared returns
            squared_returns = returns.pow(2)
            
            # Update variance forecast using GARCH(1,1)
            omega = model['garch_params']['omega']
            alpha = model['garch_params']['alpha']
            beta = model['garch_params']['beta']
            
            # Get last variance estimate
            last_return_sq = squared_returns.iloc[-1]
            prev_variance = (returns.iloc[-2] ** 2) if len(returns) > 1 else squared_returns.mean()
            
            # GARCH forecast
            new_variance = omega + alpha * last_return_sq + beta * prev_variance
            model['last_forecast'] = math.sqrt(new_variance) * np.sqrt(252)  # Annualized
            
            # Simple parameter update (in practice would use MLE)
            if len(squared_returns) > 30:
                # Update omega as long-term variance
                long_term_var = squared_returns.tail(30).mean()
                model['garch_params']['omega'] = long_term_var * (1 - alpha - beta)
            
        except Exception as e:
            logger.error(f"Error updating GARCH model: {e}")
    
    def _update_volatility_regime(self, symbol: str, current_vol: float) -> None:
        """Update volatility regime classification."""
        try:
            vol_history = self.realized_vol_history.get(symbol, [])
            if len(vol_history) < 10:
                return
            
            # Calculate percentiles
            vol_array = np.array(vol_history)
            low_threshold = np.percentile(vol_array, 33)
            high_threshold = np.percentile(vol_array, 67)
            
            # Classify regime
            if current_vol < low_threshold:
                self.volatility_regimes[symbol] = 'low'
            elif current_vol > high_threshold:
                self.volatility_regimes[symbol] = 'high'
            else:
                self.volatility_regimes[symbol] = 'medium'
                
        except Exception as e:
            logger.error(f"Error updating volatility regime: {e}")
    
    def _detect_jumps(self, symbol: str, returns: pd.Series) -> None:
        """Detect price jumps in return series."""
        try:
            if len(returns) < 10:
                return
            
            # Calculate rolling standard deviation
            rolling_std = returns.rolling(10).std()
            
            # Detect jumps as returns exceeding threshold
            threshold = self.config['jump_detection_threshold']
            
            for i, (ret, std) in enumerate(zip(returns, rolling_std)):
                if pd.notna(std) and std > 0:
                    if abs(ret) > threshold * std:
                        jump_event = {
                            'timestamp': datetime.now(),
                            'return': ret,
                            'magnitude': abs(ret) / std,
                            'direction': 'up' if ret > 0 else 'down'
                        }
                        
                        self.jump_events[symbol].append(jump_event)
                        
                        # Keep only recent jumps
                        cutoff_time = datetime.now() - timedelta(days=30)
                        self.jump_events[symbol] = [
                            event for event in self.jump_events[symbol]
                            if event['timestamp'] > cutoff_time
                        ]
            
        except Exception as e:
            logger.error(f"Error detecting jumps: {e}")
    
    async def _calculate_realized_vol_features(self, symbol: str) -> Dict[str, float]:
        """Calculate realized volatility features."""
        features = {}
        
        try:
            vol_history = self.realized_vol_history.get(symbol, [])
            if not vol_history:
                return {
                    'current_realized_vol': 0.0,
                    'vol_percentile': 0.5,
                    'vol_momentum': 0.0,
                    'vol_mean_reversion': 0.0
                }
            
            current_vol = vol_history[-1] if vol_history else 0.0
            features['current_realized_vol'] = current_vol
            
            # Volatility percentile
            if len(vol_history) > 10:
                features['vol_percentile'] = stats.percentileofscore(vol_history, current_vol) / 100.0
            else:
                features['vol_percentile'] = 0.5
            
            # Volatility momentum
            if len(vol_history) >= self.config['vol_momentum_period']:
                recent_avg = np.mean(vol_history[-self.config['vol_momentum_period']:])
                older_avg = np.mean(vol_history[-2*self.config['vol_momentum_period']:-self.config['vol_momentum_period']])
                features['vol_momentum'] = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
            else:
                features['vol_momentum'] = 0.0
            
            # Volatility mean reversion signal
            if len(vol_history) > 20:
                long_term_mean = np.mean(vol_history[-20:])
                features['vol_mean_reversion'] = (long_term_mean - current_vol) / long_term_mean if long_term_mean > 0 else 0.0
            else:
                features['vol_mean_reversion'] = 0.0
            
            # Volatility clustering measure
            if len(vol_history) >= self.config['vol_clustering_lookback']:
                recent_vols = vol_history[-self.config['vol_clustering_lookback']:]
                vol_changes = np.diff(recent_vols)
                features['vol_clustering'] = 1.0 - abs(np.corrcoef(vol_changes[:-1], vol_changes[1:])[0, 1]) if len(vol_changes) > 2 else 0.0
            else:
                features['vol_clustering'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating realized vol features: {e}")
            features = {
                'current_realized_vol': 0.0,
                'vol_percentile': 0.5,
                'vol_momentum': 0.0,
                'vol_mean_reversion': 0.0,
                'vol_clustering': 0.0
            }
        
        return features
    
    async def _calculate_garch_features(self, symbol: str) -> Dict[str, float]:
        """Calculate GARCH model features."""
        features = {}
        
        try:
            model = self.vol_models.get(symbol, {})
            garch_params = model.get('garch_params', {})
            
            features['garch_forecast'] = model.get('last_forecast', 0.0)
            features['garch_alpha'] = garch_params.get('alpha', 0.0)
            features['garch_beta'] = garch_params.get('beta', 0.0)
            features['garch_persistence'] = features['garch_alpha'] + features['garch_beta']
            
            # Compare GARCH forecast to realized volatility
            current_realized = self.realized_vol_history.get(symbol, [0.0])[-1]
            garch_forecast = features['garch_forecast']
            
            if garch_forecast > 0:
                features['garch_vs_realized'] = (garch_forecast - current_realized) / garch_forecast
            else:
                features['garch_vs_realized'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating GARCH features: {e}")
            features = {
                'garch_forecast': 0.0,
                'garch_alpha': 0.0,
                'garch_beta': 0.0,
                'garch_persistence': 0.0,
                'garch_vs_realized': 0.0
            }
        
        return features
    
    async def _calculate_regime_features(self, symbol: str) -> Dict[str, float]:
        """Calculate volatility regime features."""
        features = {}
        
        try:
            current_regime = self.volatility_regimes.get(symbol, 'medium')
            
            # Encode regime as numbers
            regime_encoding = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
            features['volatility_regime'] = regime_encoding[current_regime]
            
            # Regime stability (how long in current regime)
            vol_history = self.realized_vol_history.get(symbol, [])
            if len(vol_history) > 10:
                # Simple regime stability measure
                recent_vols = vol_history[-10:]
                vol_std = np.std(recent_vols)
                vol_mean = np.mean(recent_vols)
                features['regime_stability'] = 1.0 - (vol_std / vol_mean) if vol_mean > 0 else 0.0
            else:
                features['regime_stability'] = 0.5
            
            # Regime transition probability (simplified)
            if len(vol_history) > 20:
                regime_changes = 0
                for i in range(1, min(20, len(vol_history))):
                    prev_vol = vol_history[-(i+1)]
                    curr_vol = vol_history[-i]
                    
                    # Count regime changes (simplified threshold)
                    vol_change_pct = abs(curr_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
                    if vol_change_pct > 0.2:  # 20% change indicates regime shift
                        regime_changes += 1
                
                features['regime_transition_prob'] = regime_changes / 20.0
            else:
                features['regime_transition_prob'] = 0.1
            
        except Exception as e:
            logger.error(f"Error calculating regime features: {e}")
            features = {
                'volatility_regime': 0.5,
                'regime_stability': 0.5,
                'regime_transition_prob': 0.1
            }
        
        return features
    
    async def _calculate_jump_features(self, symbol: str) -> Dict[str, float]:
        """Calculate jump-related features."""
        features = {}
        
        try:
            recent_jumps = self.jump_events.get(symbol, [])
            
            # Jump frequency (jumps per day over last 30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            recent_jumps_30d = [j for j in recent_jumps if j['timestamp'] > cutoff_time]
            features['jump_frequency'] = len(recent_jumps_30d) / 30.0
            
            # Average jump magnitude
            if recent_jumps_30d:
                features['avg_jump_magnitude'] = np.mean([j['magnitude'] for j in recent_jumps_30d])
                
                # Jump asymmetry (more up or down jumps)
                up_jumps = len([j for j in recent_jumps_30d if j['direction'] == 'up'])
                total_jumps = len(recent_jumps_30d)
                features['jump_asymmetry'] = (up_jumps / total_jumps - 0.5) * 2  # -1 to 1
            else:
                features['avg_jump_magnitude'] = 0.0
                features['jump_asymmetry'] = 0.0
            
            # Time since last jump
            if recent_jumps:
                last_jump = max(recent_jumps, key=lambda x: x['timestamp'])
                hours_since_jump = (datetime.now() - last_jump['timestamp']).total_seconds() / 3600
                features['hours_since_last_jump'] = min(hours_since_jump, 720)  # Cap at 30 days
            else:
                features['hours_since_last_jump'] = 720
            
        except Exception as e:
            logger.error(f"Error calculating jump features: {e}")
            features = {
                'jump_frequency': 0.0,
                'avg_jump_magnitude': 0.0,
                'jump_asymmetry': 0.0,
                'hours_since_last_jump': 720
            }
        
        return features
    
    async def _calculate_term_structure_features(self, symbol: str) -> Dict[str, float]:
        """Calculate volatility term structure features."""
        features = {}
        
        try:
            # Get market data for different periods
            term_vols = {}
            
            for period in self.config['term_structure_periods']:
                df = await self.get_market_data(symbol, period)
                if df is not None and not df.empty and 'price' in df.columns:
                    returns = df['price'].pct_change().dropna()
                    if len(returns) > 10:
                        vol = returns.std() * np.sqrt(252)
                        term_vols[period] = vol
            
            if len(term_vols) >= 3:
                periods = sorted(term_vols.keys())
                vols = [term_vols[p] for p in periods]
                
                # Term structure slope
                features['term_structure_slope'] = (vols[-1] - vols[0]) / (periods[-1] - periods[0])
                
                # Term structure curvature (second derivative)
                if len(vols) >= 3:
                    mid_idx = len(vols) // 2
                    curvature = vols[0] - 2 * vols[mid_idx] + vols[-1]
                    features['term_structure_curvature'] = curvature
                else:
                    features['term_structure_curvature'] = 0.0
                
                # Backwardation/contango indicator
                short_term_vol = vols[0]
                long_term_vol = vols[-1]
                features['vol_backwardation'] = (short_term_vol - long_term_vol) / long_term_vol if long_term_vol > 0 else 0.0
            else:
                features.update({
                    'term_structure_slope': 0.0,
                    'term_structure_curvature': 0.0,
                    'vol_backwardation': 0.0
                })
            
        except Exception as e:
            logger.error(f"Error calculating term structure features: {e}")
            features = {
                'term_structure_slope': 0.0,
                'term_structure_curvature': 0.0,
                'vol_backwardation': 0.0
            }
        
        return features
    
    async def _calculate_risk_premium_features(self, symbol: str) -> Dict[str, float]:
        """Calculate volatility risk premium features."""
        features = {}
        
        try:
            # Get current realized and implied volatility
            realized_vol = self.realized_vol_history.get(symbol, [0.0])[-1] if self.realized_vol_history.get(symbol) else 0.0
            implied_vol = self.implied_vol_history.get(symbol, [0.0])[-1] if self.implied_vol_history.get(symbol) else 0.0
            
            # Volatility risk premium
            if realized_vol > 0:
                features['vol_risk_premium'] = (implied_vol - realized_vol) / realized_vol
            else:
                features['vol_risk_premium'] = 0.0
            
            # Historical risk premium average
            realized_hist = self.realized_vol_history.get(symbol, [])
            implied_hist = self.implied_vol_history.get(symbol, [])
            
            if len(realized_hist) > 10 and len(implied_hist) > 10:
                min_len = min(len(realized_hist), len(implied_hist))
                realized_recent = realized_hist[-min_len:]
                implied_recent = implied_hist[-min_len:]
                
                risk_premiums = [(imp - real) / real if real > 0 else 0 
                               for real, imp in zip(realized_recent, implied_recent)]
                
                features['avg_vol_risk_premium'] = np.mean(risk_premiums)
                features['vol_risk_premium_std'] = np.std(risk_premiums)
                
                # Risk premium percentile
                if risk_premiums:
                    current_premium = features['vol_risk_premium']
                    features['risk_premium_percentile'] = stats.percentileofscore(risk_premiums, current_premium) / 100.0
                else:
                    features['risk_premium_percentile'] = 0.5
            else:
                features.update({
                    'avg_vol_risk_premium': 0.0,
                    'vol_risk_premium_std': 0.0,
                    'risk_premium_percentile': 0.5
                })
            
        except Exception as e:
            logger.error(f"Error calculating risk premium features: {e}")
            features = {
                'vol_risk_premium': 0.0,
                'avg_vol_risk_premium': 0.0,
                'vol_risk_premium_std': 0.0,
                'risk_premium_percentile': 0.5
            }
        
        return features
    
    async def _analyze_volatility_patterns(self, symbol: str, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Analyze volatility patterns for trading opportunities.
        
        Args:
            symbol: Trading symbol
            features: Calculated features
            
        Returns:
            Volatility analysis results
        """
        try:
            analysis = {
                'signal_type': 'HOLD',
                'confidence_factors': [],
                'risk_factors': [],
                'volatility_forecast': 'UNCHANGED',
                'regime': features.get('volatility_regime', 0.5),
                'opportunity_type': None
            }
            
            current_vol = features.get('current_realized_vol', 0.0)
            vol_percentile = features.get('vol_percentile', 0.5)
            garch_forecast = features.get('garch_forecast', 0.0)
            vol_risk_premium = features.get('vol_risk_premium', 0.0)
            
            # Analyze volatility level
            if vol_percentile < 0.2:
                analysis['volatility_forecast'] = 'RISING'
                analysis['confidence_factors'].append("Volatility at low historical levels")
            elif vol_percentile > 0.8:
                analysis['volatility_forecast'] = 'FALLING' 
                analysis['confidence_factors'].append("Volatility at high historical levels")
            
            # GARCH model analysis
            garch_vs_realized = features.get('garch_vs_realized', 0.0)
            if abs(garch_vs_realized) > 0.1:
                if garch_vs_realized > 0:
                    analysis['confidence_factors'].append("GARCH model predicts higher volatility")
                else:
                    analysis['confidence_factors'].append("GARCH model predicts lower volatility")
            
            # Volatility risk premium analysis
            if abs(vol_risk_premium) > self.config['vol_risk_premium_threshold']:
                if vol_risk_premium > 0:
                    analysis['opportunity_type'] = 'SHORT_VOLATILITY'
                    analysis['confidence_factors'].append("Positive volatility risk premium")
                else:
                    analysis['opportunity_type'] = 'LONG_VOLATILITY'
                    analysis['confidence_factors'].append("Negative volatility risk premium")
            
            # Jump analysis
            jump_frequency = features.get('jump_frequency', 0.0)
            if jump_frequency > 0.1:  # More than 3 jumps per month
                analysis['risk_factors'].append("High jump frequency detected")
            
            # Regime analysis
            regime_stability = features.get('regime_stability', 0.5)
            if regime_stability < 0.3:
                analysis['risk_factors'].append("Low regime stability")
            
            # Term structure analysis
            vol_backwardation = features.get('vol_backwardation', 0.0)
            if abs(vol_backwardation) > 0.1:
                if vol_backwardation > 0:
                    analysis['confidence_factors'].append("Volatility term structure in backwardation")
                else:
                    analysis['confidence_factors'].append("Volatility term structure in contango")
            
            # Overall signal determination
            if analysis['opportunity_type'] == 'LONG_VOLATILITY' and analysis['volatility_forecast'] == 'RISING':
                analysis['signal_type'] = 'BUY_VOLATILITY'
            elif analysis['opportunity_type'] == 'SHORT_VOLATILITY' and analysis['volatility_forecast'] == 'FALLING':
                analysis['signal_type'] = 'SELL_VOLATILITY'
            elif vol_percentile < 0.15 and len(analysis['confidence_factors']) > 1:
                analysis['signal_type'] = 'BUY_VOLATILITY'
            elif vol_percentile > 0.85 and len(analysis['confidence_factors']) > 1:
                analysis['signal_type'] = 'SELL_VOLATILITY'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volatility patterns: {e}")
            return None
    
    async def _generate_volatility_signal(
        self, 
        symbol: str, 
        analysis: Dict[str, Any], 
        features: Dict[str, float]
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on volatility analysis."""
        try:
            signal_type_str = analysis['signal_type']
            
            if signal_type_str == 'HOLD':
                return None
            
            # Map volatility signals to trading signals
            if signal_type_str == 'BUY_VOLATILITY':
                # In practice, would buy volatility through options strategies
                signal_type = SignalType.BUY
            elif signal_type_str == 'SELL_VOLATILITY':
                # In practice, would sell volatility through options strategies  
                signal_type = SignalType.SELL
            else:
                return None
            
            # Calculate confidence
            confidence_boost = len(analysis['confidence_factors']) * 0.1
            risk_penalty = len(analysis['risk_factors']) * 0.05
            base_confidence = 0.5 + confidence_boost - risk_penalty
            
            # Boost confidence for extreme volatility levels
            vol_percentile = features.get('vol_percentile', 0.5)
            if vol_percentile < 0.1 or vol_percentile > 0.9:
                base_confidence += 0.1
            
            confidence = max(0.3, min(base_confidence, 0.85))
            
            # Skip if confidence too low
            if confidence < 0.5:
                return None
            
            # Calculate expected move based on volatility
            current_vol = features.get('current_realized_vol', 0.0)
            vol_forecast = features.get('garch_forecast', current_vol)
            
            # Expected price movement (simplified)
            daily_vol = vol_forecast / np.sqrt(252)
            expected_move = daily_vol * 2  # 2 standard deviation move
            
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None
            
            if signal_type == SignalType.BUY:
                target_price = current_price * (1 + expected_move)
                stop_loss = current_price * (1 - expected_move * 0.5)
            else:
                target_price = current_price * (1 - expected_move)
                stop_loss = current_price * (1 + expected_move * 0.5)
            
            # Create signal
            signal = TradingSignal(
                id=str(uuid.uuid4()),
                agent_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type=signal_type,
                confidence=confidence,
                strength=min(abs(features.get('vol_percentile', 0.5) - 0.5) * 2, 1.0),
                reasoning={
                    'analysis': analysis,
                    'volatility_metrics': {
                        'current_vol': current_vol,
                        'vol_percentile': vol_percentile,
                        'vol_forecast': vol_forecast,
                        'risk_premium': features.get('vol_risk_premium', 0.0)
                    }
                },
                features_used=features,
                prediction_horizon=self.config['vol_forecast_horizon'] * 24,  # Convert days to hours
                target_price=target_price,
                stop_loss=stop_loss,
                risk_score=len(analysis['risk_factors']) / 5.0,
                expected_return=expected_move
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating volatility signal: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            if self.market_data_service:
                return await self.market_data_service.get_latest_price(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None


# Convenience function for creating volatility agent
def create_volatility_agent(symbols: List[str], **kwargs) -> VolatilityAgent:
    """Create a volatility agent with default configuration."""
    return VolatilityAgent(symbols, kwargs)