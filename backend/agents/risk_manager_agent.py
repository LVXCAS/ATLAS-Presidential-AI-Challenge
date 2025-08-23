"""
Risk Manager Agent for Bloomberg Terminal
Comprehensive risk management and position sizing agent.
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


class RiskManagerAgent(BaseAgent):
    """
    Risk management agent responsible for:
    - Portfolio risk monitoring and alerting
    - Position sizing optimization
    - Value at Risk (VaR) calculations
    - Stress testing and scenario analysis
    - Correlation risk management
    - Maximum drawdown control
    - Liquidity risk assessment
    - Counterparty risk monitoring
    - Dynamic hedge ratio calculation
    - Risk-adjusted return optimization
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        default_config = {
            'max_portfolio_var': 0.05,  # 5% daily VaR limit
            'max_single_position_weight': 0.15,  # 15% max per position
            'max_sector_concentration': 0.25,  # 25% max per sector
            'max_correlation_exposure': 0.8,  # Max correlation between positions
            'var_confidence_level': 0.95,  # 95% confidence for VaR
            'var_lookback_days': 252,  # 1 year lookback for VaR
            'stress_test_scenarios': 10,  # Number of stress scenarios
            'rebalance_threshold': 0.05,  # 5% deviation triggers rebalance
            'max_leverage': 2.0,  # Maximum portfolio leverage
            'min_liquidity_score': 0.6,  # Minimum liquidity requirement
            'max_drawdown_limit': 0.15,  # 15% maximum drawdown
            'correlation_lookback': 60,  # Days for correlation calculation
            'volatility_target': 0.15,  # 15% annual volatility target
            'kelly_fraction': 0.25,  # Kelly criterion position sizing
            'risk_budget_allocation': 'equal',  # equal, vol_weighted, or custom
            'emergency_stop_loss': 0.08,  # 8% emergency stop loss
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            name="RiskManagerAgent",
            symbols=symbols,
            config=default_config
        )
        
        # Risk management state
        self.portfolio_positions: Dict[str, Dict] = {}
        self.risk_metrics_history: List[Dict] = []
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.var_history: List[float] = []
        self.stress_test_results: Dict[str, Any] = {}
        self.liquidity_scores: Dict[str, float] = {}
        self.sector_exposures: Dict[str, float] = {}
        self.risk_alerts: List[Dict] = []
        
    async def initialize(self) -> None:
        """Initialize risk management components."""
        logger.info(f"Initializing {self.name} for portfolio risk management")
        
        # Initialize position tracking
        for symbol in self.symbols:
            self.portfolio_positions[symbol] = {
                'weight': 0.0,
                'notional': 0.0,
                'var_contribution': 0.0,
                'liquidity_score': 0.0,
                'last_updated': datetime.now()
            }
            self.liquidity_scores[symbol] = 0.5  # Default liquidity score
        
        # Initialize correlation matrix
        await self._initialize_correlation_matrix()
        
        # Run initial risk assessment
        await self._calculate_portfolio_risk_metrics()
        
        logger.info(f"{self.name} initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup risk management resources."""
        self.portfolio_positions.clear()
        self.risk_metrics_history.clear()
        self.correlation_matrix = pd.DataFrame()
        self.var_history.clear()
        self.stress_test_results.clear()
        self.liquidity_scores.clear()
        self.sector_exposures.clear()
        self.risk_alerts.clear()
        logger.info(f"{self.name} cleanup completed")
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate risk management signal - typically position sizing or hedge recommendations.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None if no risk management action needed
        """
        try:
            # Calculate current risk features
            features = await self.calculate_features(symbol)
            if not features:
                return None
            
            # Analyze risk exposure
            risk_analysis = await self._analyze_risk_exposure(symbol, features)
            if not risk_analysis:
                return None
            
            # Generate risk management signal
            signal = await self._generate_risk_management_signal(symbol, risk_analysis, features)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating risk management signal for {symbol}: {e}")
            return None
    
    async def calculate_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate risk management features.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of risk feature names to values
        """
        try:
            # Check cache first
            cache_key = f"risk_features:{symbol}"
            cached_features = await self.get_cached_feature(cache_key, ttl=300)
            if cached_features:
                return cached_features
            
            features = {}
            
            # Update portfolio metrics
            await self._update_portfolio_positions()
            await self._calculate_portfolio_risk_metrics()
            
            # Position-specific risk features
            position_features = await self._calculate_position_risk_features(symbol)
            features.update(position_features)
            
            # Portfolio-level risk features
            portfolio_features = await self._calculate_portfolio_risk_features(symbol)
            features.update(portfolio_features)
            
            # Liquidity risk features
            liquidity_features = await self._calculate_liquidity_risk_features(symbol)
            features.update(liquidity_features)
            
            # Correlation risk features
            correlation_features = await self._calculate_correlation_risk_features(symbol)
            features.update(correlation_features)
            
            # VaR and stress test features
            var_features = await self._calculate_var_features(symbol)
            features.update(var_features)
            
            # Cache the features
            await self.cache_feature(cache_key, features, ttl=300)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating risk features for {symbol}: {e}")
            return {}
    
    async def _initialize_correlation_matrix(self) -> None:
        """Initialize correlation matrix from historical data."""
        try:
            # Get historical data for all symbols
            returns_data = {}
            
            for symbol in self.symbols:
                df = await self.get_market_data(symbol, self.config['correlation_lookback'])
                if df is not None and not df.empty and 'price' in df.columns:
                    returns = df['price'].pct_change().dropna()
                    if len(returns) > 20:
                        returns_data[symbol] = returns.tail(min(len(returns), 60))
            
            if len(returns_data) > 1:
                # Align all return series
                aligned_returns = pd.DataFrame(returns_data)
                aligned_returns = aligned_returns.dropna()
                
                if not aligned_returns.empty:
                    self.correlation_matrix = aligned_returns.corr()
            
        except Exception as e:
            logger.error(f"Error initializing correlation matrix: {e}")
    
    async def _update_portfolio_positions(self) -> None:
        """Update current portfolio positions (placeholder - would get from broker)."""
        try:
            # In a real implementation, this would query the broker for current positions
            # For now, simulate some positions
            total_portfolio_value = 1000000  # $1M portfolio
            
            for i, symbol in enumerate(self.symbols):
                # Simulate position weights (would be actual positions in production)
                if i < len(self.symbols) // 2:
                    weight = np.random.uniform(0.05, 0.12)
                    notional = weight * total_portfolio_value
                else:
                    weight = 0.0
                    notional = 0.0
                
                self.portfolio_positions[symbol].update({
                    'weight': weight,
                    'notional': notional,
                    'last_updated': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Error updating portfolio positions: {e}")
    
    async def _calculate_portfolio_risk_metrics(self) -> None:
        """Calculate portfolio-level risk metrics."""
        try:
            # Calculate portfolio VaR
            portfolio_var = await self._calculate_portfolio_var()
            
            # Calculate portfolio volatility
            portfolio_vol = await self._calculate_portfolio_volatility()
            
            # Calculate max drawdown
            max_drawdown = await self._calculate_max_drawdown()
            
            # Store metrics
            risk_metrics = {
                'timestamp': datetime.now(),
                'portfolio_var': portfolio_var,
                'portfolio_volatility': portfolio_vol,
                'max_drawdown': max_drawdown,
                'total_exposure': sum(pos['notional'] for pos in self.portfolio_positions.values()),
                'num_positions': sum(1 for pos in self.portfolio_positions.values() if pos['weight'] > 0.01)
            }
            
            self.risk_metrics_history.append(risk_metrics)
            
            # Keep only recent history
            if len(self.risk_metrics_history) > 100:
                self.risk_metrics_history = self.risk_metrics_history[-100:]
            
            # Update VaR history
            self.var_history.append(portfolio_var)
            if len(self.var_history) > 252:  # Keep 1 year
                self.var_history = self.var_history[-252:]
            
            # Check for risk alerts
            await self._check_risk_alerts(risk_metrics)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
    
    async def _calculate_portfolio_var(self) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            if self.correlation_matrix.empty:
                return 0.0
            
            # Get position weights and volatilities
            weights = []
            volatilities = []
            symbols_with_positions = []
            
            for symbol, position in self.portfolio_positions.items():
                if position['weight'] > 0.001 and symbol in self.correlation_matrix.columns:
                    weights.append(position['weight'])
                    
                    # Get historical volatility
                    df = await self.get_market_data(symbol, 30)
                    if df is not None and not df.empty and 'price' in df.columns:
                        returns = df['price'].pct_change().dropna()
                        if len(returns) > 10:
                            vol = returns.std()
                            volatilities.append(vol)
                            symbols_with_positions.append(symbol)
                        else:
                            volatilities.append(0.02)  # Default 2% daily vol
                            symbols_with_positions.append(symbol)
            
            if len(weights) < 2:
                return 0.0
            
            # Create covariance matrix
            weights = np.array(weights)
            volatilities = np.array(volatilities)
            
            # Get correlation submatrix for positions
            corr_subset = self.correlation_matrix.loc[symbols_with_positions, symbols_with_positions]
            
            # Calculate covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * corr_subset.values
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)
            
            # VaR calculation (normal distribution assumption)
            confidence_level = self.config['var_confidence_level']
            z_score = stats.norm.ppf(confidence_level)
            var = portfolio_vol * z_score
            
            return var
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    async def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility."""
        try:
            # Similar to VaR calculation but return volatility directly
            if self.correlation_matrix.empty:
                return 0.0
            
            weights = []
            volatilities = []
            symbols_with_positions = []
            
            for symbol, position in self.portfolio_positions.items():
                if position['weight'] > 0.001 and symbol in self.correlation_matrix.columns:
                    weights.append(position['weight'])
                    
                    df = await self.get_market_data(symbol, 30)
                    if df is not None and not df.empty and 'price' in df.columns:
                        returns = df['price'].pct_change().dropna()
                        if len(returns) > 10:
                            vol = returns.std()
                            volatilities.append(vol)
                            symbols_with_positions.append(symbol)
            
            if len(weights) < 2:
                return 0.0
            
            weights = np.array(weights)
            volatilities = np.array(volatilities)
            
            corr_subset = self.correlation_matrix.loc[symbols_with_positions, symbols_with_positions]
            cov_matrix = np.outer(volatilities, volatilities) * corr_subset.values
            
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return np.sqrt(portfolio_variance)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown (simplified)."""
        try:
            if len(self.risk_metrics_history) < 10:
                return 0.0
            
            # Simulate portfolio returns based on positions
            recent_metrics = self.risk_metrics_history[-30:]  # Last 30 periods
            
            # Simple drawdown calculation
            peak = 1.0
            max_dd = 0.0
            
            for metrics in recent_metrics:
                # Simulate portfolio value change
                portfolio_change = np.random.normal(-0.001, 0.02)  # Slight negative drift with 2% vol
                current_value = peak * (1 + portfolio_change)
                
                if current_value > peak:
                    peak = current_value
                else:
                    drawdown = (peak - current_value) / peak
                    max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    async def _check_risk_alerts(self, risk_metrics: Dict[str, Any]) -> None:
        """Check for risk limit breaches and generate alerts."""
        try:
            alerts = []
            
            # VaR limit check
            portfolio_var = risk_metrics.get('portfolio_var', 0.0)
            if portfolio_var > self.config['max_portfolio_var']:
                alerts.append({
                    'type': 'VAR_BREACH',
                    'message': f"Portfolio VaR ({portfolio_var:.3f}) exceeds limit ({self.config['max_portfolio_var']:.3f})",
                    'severity': 'HIGH',
                    'timestamp': datetime.now()
                })
            
            # Position concentration check
            for symbol, position in self.portfolio_positions.items():
                if position['weight'] > self.config['max_single_position_weight']:
                    alerts.append({
                        'type': 'POSITION_CONCENTRATION',
                        'message': f"{symbol} weight ({position['weight']:.3f}) exceeds limit ({self.config['max_single_position_weight']:.3f})",
                        'severity': 'MEDIUM',
                        'timestamp': datetime.now()
                    })
            
            # Max drawdown check
            max_drawdown = risk_metrics.get('max_drawdown', 0.0)
            if max_drawdown > self.config['max_drawdown_limit']:
                alerts.append({
                    'type': 'DRAWDOWN_BREACH',
                    'message': f"Max drawdown ({max_drawdown:.3f}) exceeds limit ({self.config['max_drawdown_limit']:.3f})",
                    'severity': 'HIGH',
                    'timestamp': datetime.now()
                })
            
            # Add alerts to history
            self.risk_alerts.extend(alerts)
            
            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.risk_alerts = [alert for alert in self.risk_alerts if alert['timestamp'] > cutoff_time]
            
            # Log high severity alerts
            for alert in alerts:
                if alert['severity'] == 'HIGH':
                    logger.warning(f"Risk Alert: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
    
    async def _calculate_position_risk_features(self, symbol: str) -> Dict[str, float]:
        """Calculate position-specific risk features."""
        features = {}
        
        try:
            position = self.portfolio_positions.get(symbol, {})
            
            # Position weight and size
            features['position_weight'] = position.get('weight', 0.0)
            features['position_notional'] = position.get('notional', 0.0)
            
            # Position risk metrics
            df = await self.get_market_data(symbol, 30)
            if df is not None and not df.empty and 'price' in df.columns:
                returns = df['price'].pct_change().dropna()
                if len(returns) > 10:
                    # Individual position volatility
                    features['position_volatility'] = returns.std()
                    
                    # Position VaR contribution
                    position_var = features['position_volatility'] * stats.norm.ppf(0.95) * features['position_weight']
                    features['position_var_contribution'] = position_var
                    
                    # Skewness and kurtosis
                    features['position_skewness'] = stats.skew(returns)
                    features['position_kurtosis'] = stats.kurtosis(returns)
                    
                    # Maximum individual loss in last 30 days
                    features['position_max_loss'] = abs(returns.min()) if len(returns) > 0 else 0.0
                else:
                    features.update({
                        'position_volatility': 0.0,
                        'position_var_contribution': 0.0,
                        'position_skewness': 0.0,
                        'position_kurtosis': 0.0,
                        'position_max_loss': 0.0
                    })
            else:
                features.update({
                    'position_volatility': 0.0,
                    'position_var_contribution': 0.0,
                    'position_skewness': 0.0,
                    'position_kurtosis': 0.0,
                    'position_max_loss': 0.0
                })
            
            # Weight relative to limits
            features['weight_utilization'] = features['position_weight'] / self.config['max_single_position_weight']
            
        except Exception as e:
            logger.error(f"Error calculating position risk features: {e}")
            features = {
                'position_weight': 0.0,
                'position_notional': 0.0,
                'position_volatility': 0.0,
                'position_var_contribution': 0.0,
                'position_skewness': 0.0,
                'position_kurtosis': 0.0,
                'position_max_loss': 0.0,
                'weight_utilization': 0.0
            }
        
        return features
    
    async def _calculate_portfolio_risk_features(self, symbol: str) -> Dict[str, float]:
        """Calculate portfolio-level risk features."""
        features = {}
        
        try:
            if self.risk_metrics_history:
                latest_metrics = self.risk_metrics_history[-1]
                
                features['portfolio_var'] = latest_metrics.get('portfolio_var', 0.0)
                features['portfolio_volatility'] = latest_metrics.get('portfolio_volatility', 0.0)
                features['max_drawdown'] = latest_metrics.get('max_drawdown', 0.0)
                features['total_exposure'] = latest_metrics.get('total_exposure', 0.0)
                features['num_positions'] = latest_metrics.get('num_positions', 0)
            else:
                features.update({
                    'portfolio_var': 0.0,
                    'portfolio_volatility': 0.0,
                    'max_drawdown': 0.0,
                    'total_exposure': 0.0,
                    'num_positions': 0
                })
            
            # Risk utilization ratios
            features['var_utilization'] = features['portfolio_var'] / self.config['max_portfolio_var']
            features['drawdown_utilization'] = features['max_drawdown'] / self.config['max_drawdown_limit']
            
            # Portfolio concentration
            active_weights = [pos['weight'] for pos in self.portfolio_positions.values() if pos['weight'] > 0.01]
            if active_weights:
                features['portfolio_concentration'] = sum(w**2 for w in active_weights)  # Herfindahl index
                features['avg_position_size'] = np.mean(active_weights)
                features['position_size_std'] = np.std(active_weights)
            else:
                features.update({
                    'portfolio_concentration': 0.0,
                    'avg_position_size': 0.0,
                    'position_size_std': 0.0
                })
            
            # Risk alerts count
            recent_alerts = [alert for alert in self.risk_alerts 
                           if alert['timestamp'] > datetime.now() - timedelta(hours=1)]
            features['recent_risk_alerts'] = len(recent_alerts)
            features['high_severity_alerts'] = len([a for a in recent_alerts if a['severity'] == 'HIGH'])
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk features: {e}")
            features = {
                'portfolio_var': 0.0,
                'portfolio_volatility': 0.0,
                'max_drawdown': 0.0,
                'total_exposure': 0.0,
                'num_positions': 0,
                'var_utilization': 0.0,
                'drawdown_utilization': 0.0,
                'portfolio_concentration': 0.0,
                'avg_position_size': 0.0,
                'position_size_std': 0.0,
                'recent_risk_alerts': 0,
                'high_severity_alerts': 0
            }
        
        return features
    
    async def _calculate_liquidity_risk_features(self, symbol: str) -> Dict[str, float]:
        """Calculate liquidity risk features."""
        features = {}
        
        try:
            # Get market data for liquidity analysis
            df = await self.get_market_data(symbol, 20)
            if df is not None and not df.empty:
                # Volume-based liquidity score
                if 'volume' in df.columns:
                    avg_volume = df['volume'].mean()
                    volume_std = df['volume'].std()
                    volume_consistency = 1 - (volume_std / avg_volume) if avg_volume > 0 else 0
                    
                    features['avg_daily_volume'] = avg_volume
                    features['volume_consistency'] = max(0, volume_consistency)
                    
                    # Liquidity score (simplified)
                    liquidity_score = min(avg_volume / 1000000, 1.0) * volume_consistency
                    features['liquidity_score'] = liquidity_score
                    self.liquidity_scores[symbol] = liquidity_score
                else:
                    features.update({
                        'avg_daily_volume': 0.0,
                        'volume_consistency': 0.0,
                        'liquidity_score': 0.5
                    })
                
                # Bid-ask spread proxy (using high-low range)
                if 'high' in df.columns and 'low' in df.columns and 'price' in df.columns:
                    spreads = (df['high'] - df['low']) / df['price']
                    features['avg_bid_ask_spread'] = spreads.mean()
                    features['bid_ask_volatility'] = spreads.std()
                else:
                    features.update({
                        'avg_bid_ask_spread': 0.02,  # Default 2%
                        'bid_ask_volatility': 0.005
                    })
            else:
                features.update({
                    'avg_daily_volume': 0.0,
                    'volume_consistency': 0.0,
                    'liquidity_score': 0.5,
                    'avg_bid_ask_spread': 0.02,
                    'bid_ask_volatility': 0.005
                })
            
            # Position liquidity risk
            position_weight = self.portfolio_positions.get(symbol, {}).get('weight', 0.0)
            features['liquidity_risk_score'] = position_weight / max(features['liquidity_score'], 0.1)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk features: {e}")
            features = {
                'avg_daily_volume': 0.0,
                'volume_consistency': 0.0,
                'liquidity_score': 0.5,
                'avg_bid_ask_spread': 0.02,
                'bid_ask_volatility': 0.005,
                'liquidity_risk_score': 0.0
            }
        
        return features
    
    async def _calculate_correlation_risk_features(self, symbol: str) -> Dict[str, float]:
        """Calculate correlation risk features."""
        features = {}
        
        try:
            if self.correlation_matrix.empty or symbol not in self.correlation_matrix.columns:
                features.update({
                    'avg_correlation': 0.0,
                    'max_correlation': 0.0,
                    'correlation_risk_score': 0.0,
                    'correlation_concentration': 0.0
                })
                return features
            
            # Calculate correlation metrics
            correlations = self.correlation_matrix[symbol].drop(symbol)  # Exclude self-correlation
            
            # Only consider correlations with positions
            weighted_correlations = []
            for other_symbol, corr in correlations.items():
                other_weight = self.portfolio_positions.get(other_symbol, {}).get('weight', 0.0)
                if other_weight > 0.01:  # Only consider significant positions
                    weighted_correlations.append(abs(corr))
            
            if weighted_correlations:
                features['avg_correlation'] = np.mean(weighted_correlations)
                features['max_correlation'] = np.max(weighted_correlations)
                
                # Correlation concentration (how many high correlations)
                high_corr_count = sum(1 for corr in weighted_correlations if corr > self.config['max_correlation_exposure'])
                features['correlation_concentration'] = high_corr_count / len(weighted_correlations)
                
                # Risk score based on correlation exposure
                position_weight = self.portfolio_positions.get(symbol, {}).get('weight', 0.0)
                features['correlation_risk_score'] = position_weight * features['max_correlation']
            else:
                features.update({
                    'avg_correlation': 0.0,
                    'max_correlation': 0.0,
                    'correlation_risk_score': 0.0,
                    'correlation_concentration': 0.0
                })
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk features: {e}")
            features = {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_risk_score': 0.0,
                'correlation_concentration': 0.0
            }
        
        return features
    
    async def _calculate_var_features(self, symbol: str) -> Dict[str, float]:
        """Calculate VaR-related features."""
        features = {}
        
        try:
            # Individual position VaR
            position = self.portfolio_positions.get(symbol, {})
            position_weight = position.get('weight', 0.0)
            
            if position_weight > 0.01:
                df = await self.get_market_data(symbol, 30)
                if df is not None and not df.empty and 'price' in df.columns:
                    returns = df['price'].pct_change().dropna()
                    if len(returns) > 10:
                        # Calculate position VaR
                        position_vol = returns.std()
                        position_var = position_vol * stats.norm.ppf(0.95) * position_weight
                        features['individual_var'] = position_var
                        
                        # VaR contribution to portfolio
                        portfolio_var = self.risk_metrics_history[-1].get('portfolio_var', 0.0) if self.risk_metrics_history else 0.0
                        features['var_contribution_pct'] = (position_var / portfolio_var * 100) if portfolio_var > 0 else 0.0
                    else:
                        features.update({
                            'individual_var': 0.0,
                            'var_contribution_pct': 0.0
                        })
                else:
                    features.update({
                        'individual_var': 0.0,
                        'var_contribution_pct': 0.0
                    })
            else:
                features.update({
                    'individual_var': 0.0,
                    'var_contribution_pct': 0.0
                })
            
            # VaR trend analysis
            if len(self.var_history) > 10:
                recent_var = np.mean(self.var_history[-5:])
                older_var = np.mean(self.var_history[-15:-10])
                features['var_trend'] = (recent_var - older_var) / older_var if older_var > 0 else 0.0
            else:
                features['var_trend'] = 0.0
            
            # VaR percentile
            if len(self.var_history) > 20:
                current_var = self.var_history[-1] if self.var_history else 0.0
                features['var_percentile'] = stats.percentileofscore(self.var_history, current_var) / 100.0
            else:
                features['var_percentile'] = 0.5
            
        except Exception as e:
            logger.error(f"Error calculating VaR features: {e}")
            features = {
                'individual_var': 0.0,
                'var_contribution_pct': 0.0,
                'var_trend': 0.0,
                'var_percentile': 0.5
            }
        
        return features
    
    async def _analyze_risk_exposure(self, symbol: str, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Analyze risk exposure for the symbol.
        
        Args:
            symbol: Trading symbol
            features: Calculated risk features
            
        Returns:
            Risk analysis results
        """
        try:
            analysis = {
                'risk_level': 'NORMAL',
                'recommended_action': 'HOLD',
                'risk_factors': [],
                'mitigation_suggestions': [],
                'position_sizing_recommendation': 'MAINTAIN',
                'urgency': 'LOW'
            }
            
            # Analyze position concentration
            position_weight = features.get('position_weight', 0.0)
            weight_utilization = features.get('weight_utilization', 0.0)
            
            if weight_utilization > 0.9:
                analysis['risk_level'] = 'HIGH'
                analysis['risk_factors'].append(f"Position weight ({position_weight:.2%}) near limit")
                analysis['recommended_action'] = 'REDUCE'
                analysis['urgency'] = 'HIGH'
            elif weight_utilization > 0.7:
                analysis['risk_level'] = 'ELEVATED'
                analysis['risk_factors'].append(f"Position weight ({position_weight:.2%}) elevated")
                analysis['mitigation_suggestions'].append("Consider reducing position size")
            
            # Analyze portfolio VaR
            var_utilization = features.get('var_utilization', 0.0)
            if var_utilization > 0.9:
                analysis['risk_level'] = 'HIGH'
                analysis['risk_factors'].append("Portfolio VaR near limit")
                analysis['recommended_action'] = 'REDUCE'
                analysis['urgency'] = 'HIGH'
            elif var_utilization > 0.75:
                analysis['risk_level'] = 'ELEVATED'
                analysis['risk_factors'].append("Portfolio VaR elevated")
            
            # Analyze correlation risk
            max_correlation = features.get('max_correlation', 0.0)
            if max_correlation > self.config['max_correlation_exposure']:
                analysis['risk_factors'].append(f"High correlation exposure ({max_correlation:.2f})")
                analysis['mitigation_suggestions'].append("Consider diversifying into uncorrelated assets")
            
            # Analyze liquidity risk
            liquidity_score = features.get('liquidity_score', 0.5)
            if liquidity_score < self.config['min_liquidity_score']:
                analysis['risk_factors'].append(f"Low liquidity score ({liquidity_score:.2f})")
                analysis['mitigation_suggestions'].append("Monitor position size relative to liquidity")
            
            # Check for recent risk alerts
            high_severity_alerts = features.get('high_severity_alerts', 0)
            if high_severity_alerts > 0:
                analysis['risk_level'] = 'HIGH'
                analysis['urgency'] = 'HIGH'
                analysis['risk_factors'].append(f"{high_severity_alerts} high severity alerts")
            
            # Position sizing recommendations
            if analysis['recommended_action'] == 'REDUCE':
                analysis['position_sizing_recommendation'] = 'DECREASE'
            elif analysis['risk_level'] == 'NORMAL' and features.get('var_utilization', 0.0) < 0.5:
                analysis['position_sizing_recommendation'] = 'CAN_INCREASE'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk exposure: {e}")
            return None
    
    async def _generate_risk_management_signal(
        self, 
        symbol: str, 
        analysis: Dict[str, Any], 
        features: Dict[str, float]
    ) -> Optional[TradingSignal]:
        """Generate risk management signal."""
        try:
            recommended_action = analysis['recommended_action']
            risk_level = analysis['risk_level']
            urgency = analysis['urgency']
            
            # Only generate signals for actionable risk management decisions
            if recommended_action == 'HOLD' and risk_level == 'NORMAL':
                return None
            
            # Determine signal type based on risk analysis
            if recommended_action == 'REDUCE':
                signal_type = SignalType.SELL
            elif analysis['position_sizing_recommendation'] == 'CAN_INCREASE':
                signal_type = SignalType.BUY
            else:
                return None  # No clear signal
            
            # Calculate confidence based on risk factors
            base_confidence = 0.6  # Risk management signals have moderate base confidence
            risk_factor_count = len(analysis['risk_factors'])
            
            if urgency == 'HIGH':
                confidence = min(0.9, base_confidence + risk_factor_count * 0.1)
            elif urgency == 'MEDIUM':
                confidence = min(0.8, base_confidence + risk_factor_count * 0.05)
            else:
                confidence = min(0.7, base_confidence + risk_factor_count * 0.02)
            
            # Position sizing for risk management
            current_weight = features.get('position_weight', 0.0)
            
            if signal_type == SignalType.SELL:
                # Reduce position - target weight based on risk level
                if risk_level == 'HIGH':
                    target_weight = current_weight * 0.5  # Reduce by 50%
                else:
                    target_weight = current_weight * 0.75  # Reduce by 25%
            else:
                # Can increase - but cap at risk limits
                max_safe_weight = self.config['max_single_position_weight'] * 0.8  # 80% of limit
                target_weight = min(current_weight * 1.25, max_safe_weight)
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None
            
            # Risk management targets (conservative)
            if signal_type == SignalType.SELL:
                target_price = current_price * 0.98  # Small price target for risk reduction
                stop_loss = current_price * 1.05  # Tight stop loss
            else:
                target_price = current_price * 1.02
                stop_loss = current_price * 0.95
            
            signal = TradingSignal(
                id=str(uuid.uuid4()),
                agent_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type=signal_type,
                confidence=confidence,
                strength=min(risk_factor_count / 5.0, 1.0),  # Strength based on risk factors
                reasoning={
                    'risk_analysis': analysis,
                    'risk_metrics': {
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'var_utilization': features.get('var_utilization', 0.0),
                        'liquidity_score': features.get('liquidity_score', 0.5)
                    },
                    'risk_management_action': recommended_action
                },
                features_used=features,
                prediction_horizon=24,  # 24 hour horizon for risk management
                target_price=target_price,
                stop_loss=stop_loss,
                risk_score=min(len(analysis['risk_factors']) / 3.0, 1.0),
                expected_return=0.01  # Small expected return for risk management
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating risk management signal: {e}")
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


# Convenience function for creating risk manager agent
def create_risk_manager_agent(symbols: List[str], **kwargs) -> RiskManagerAgent:
    """Create a risk manager agent with default configuration."""
    return RiskManagerAgent(symbols, kwargs)