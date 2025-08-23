"""
Arbitrage Agent for Bloomberg Terminal
Statistical arbitrage and pairs trading agent.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import zscore

from agents.base_agent import BaseAgent, TradingSignal, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class ArbitrageAgent(BaseAgent):
    """
    Statistical arbitrage agent specializing in:
    - Pairs trading based on cointegration
    - Mean reversion arbitrage
    - Cross-asset arbitrage opportunities
    - Index arbitrage
    - Calendar spread arbitrage
    - Volatility arbitrage
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        default_config = {
            'lookback_days': 60,
            'min_correlation': 0.7,
            'cointegration_pvalue': 0.05,
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'max_holding_period': 5,  # days
            'min_spread_volatility': 0.01,
            'max_spread_volatility': 0.1,
            'pair_selection_period': 30,  # days
            'rebalance_frequency': 'daily',
            'max_pairs': 10,
            'min_liquidity_score': 0.6,
            'risk_budget': 0.02,  # 2% per position
            'volatility_lookback': 20,
            'momentum_threshold': 0.15,
            'mean_reversion_strength': 0.8
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            name="ArbitrageAgent",
            symbols=symbols,
            config=default_config
        )
        
        # Agent-specific state
        self.active_pairs: Dict[str, Dict] = {}
        self.pair_statistics: Dict[str, Dict] = {}
        self.cointegration_cache: Dict[str, Dict] = {}
        self.spread_history: Dict[str, List] = {}
        
    async def initialize(self) -> None:
        """Initialize arbitrage-specific components."""
        logger.info(f"Initializing {self.name} for arbitrage trading")
        
        # Initialize pair combinations
        await self._initialize_pairs()
        
        # Calculate initial pair statistics
        await self._update_pair_statistics()
        
        logger.info(f"{self.name} initialized with {len(self.active_pairs)} potential pairs")
    
    async def cleanup(self) -> None:
        """Cleanup arbitrage-specific resources."""
        self.active_pairs.clear()
        self.pair_statistics.clear()
        self.cointegration_cache.clear()
        self.spread_history.clear()
        logger.info(f"{self.name} cleanup completed")
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate arbitrage-based trading signal.
        
        Args:
            symbol: Primary trading symbol
            
        Returns:
            TradingSignal or None if no arbitrage opportunity
        """
        try:
            # Find best arbitrage opportunities for this symbol
            opportunities = await self._find_arbitrage_opportunities(symbol)
            
            if not opportunities:
                return None
            
            # Select best opportunity
            best_opportunity = max(opportunities, key=lambda x: x['score'])
            
            # Generate signal based on opportunity
            signal = await self._generate_arbitrage_signal(symbol, best_opportunity)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating arbitrage signal for {symbol}: {e}")
            return None
    
    async def calculate_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate arbitrage-specific features.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of feature names to values
        """
        try:
            # Check cache first
            cache_key = f"arbitrage_features:{symbol}"
            cached_features = await self.get_cached_feature(cache_key, ttl=300)
            if cached_features:
                return cached_features
            
            features = {}
            
            # Calculate pair-based features
            pair_features = await self._calculate_pair_features(symbol)
            features.update(pair_features)
            
            # Calculate spread features
            spread_features = await self._calculate_spread_features(symbol)
            features.update(spread_features)
            
            # Calculate market microstructure features
            microstructure_features = await self._calculate_microstructure_features(symbol)
            features.update(microstructure_features)
            
            # Calculate volatility arbitrage features
            volatility_features = await self._calculate_volatility_features(symbol)
            features.update(volatility_features)
            
            # Cache the features
            await self.cache_feature(cache_key, features, ttl=300)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage features for {symbol}: {e}")
            return {}
    
    async def _initialize_pairs(self) -> None:
        """Initialize potential trading pairs."""
        try:
            # Generate all possible pairs from symbols
            for i, symbol1 in enumerate(self.symbols):
                for j, symbol2 in enumerate(self.symbols[i+1:], i+1):
                    pair_key = f"{symbol1}_{symbol2}"
                    
                    self.active_pairs[pair_key] = {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'status': 'evaluating',
                        'last_updated': datetime.now(),
                        'correlation': 0.0,
                        'cointegration_pvalue': 1.0,
                        'spread_zscore': 0.0,
                        'half_life': None,
                        'score': 0.0
                    }
                    
                    self.spread_history[pair_key] = []
            
        except Exception as e:
            logger.error(f"Error initializing pairs: {e}")
    
    async def _update_pair_statistics(self) -> None:
        """Update statistical properties of all pairs."""
        try:
            for pair_key, pair_info in self.active_pairs.items():
                symbol1 = pair_info['symbol1']
                symbol2 = pair_info['symbol2']
                
                # Get price data for both symbols
                df1 = await self.get_market_data(symbol1, self.config['lookback_days'])
                df2 = await self.get_market_data(symbol2, self.config['lookback_days'])
                
                if df1 is None or df2 is None or len(df1) < 30 or len(df2) < 30:
                    continue
                
                # Align data
                aligned_data = self._align_price_data(df1, df2)
                if aligned_data is None:
                    continue
                
                prices1, prices2 = aligned_data
                
                # Calculate correlation
                correlation = np.corrcoef(prices1, prices2)[0, 1]
                pair_info['correlation'] = correlation
                
                # Test for cointegration if correlation is high enough
                if abs(correlation) >= self.config['min_correlation']:
                    coint_result = await self._test_cointegration(prices1, prices2)
                    pair_info.update(coint_result)
                    
                    # Calculate current spread
                    current_spread = await self._calculate_current_spread(symbol1, symbol2)
                    if current_spread is not None:
                        pair_info['current_spread'] = current_spread
                        
                        # Calculate z-score
                        spread_history = self.spread_history.get(pair_key, [])
                        if len(spread_history) > 20:
                            spread_mean = np.mean(spread_history[-20:])
                            spread_std = np.std(spread_history[-20:])
                            if spread_std > 0:
                                pair_info['spread_zscore'] = (current_spread - spread_mean) / spread_std
                
                # Calculate pair score
                pair_info['score'] = self._calculate_pair_score(pair_info)
                pair_info['last_updated'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating pair statistics: {e}")
    
    def _align_price_data(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Align price data from two symbols."""
        try:
            if 'price' not in df1.columns or 'price' not in df2.columns:
                return None
            
            # Simple alignment by length (assumes same time intervals)
            min_length = min(len(df1), len(df2))
            prices1 = df1['price'].tail(min_length).values
            prices2 = df2['price'].tail(min_length).values
            
            return prices1, prices2
            
        except Exception as e:
            logger.error(f"Error aligning price data: {e}")
            return None
    
    async def _test_cointegration(self, prices1: np.ndarray, prices2: np.ndarray) -> Dict[str, Any]:
        """Test for cointegration between two price series."""
        result = {
            'cointegration_pvalue': 1.0,
            'cointegration_coefficient': 0.0,
            'half_life': None,
            'mean_reversion_strength': 0.0
        }
        
        try:
            # Simple cointegration test using linear regression residuals
            # In production, would use Johansen test or Engle-Granger test
            
            # Linear regression
            A = np.vstack([prices1, np.ones(len(prices1))]).T
            beta, alpha = np.linalg.lstsq(A, prices2, rcond=None)[0]
            
            # Calculate residuals (spread)
            residuals = prices2 - (beta * prices1 + alpha)
            
            # Test for stationarity of residuals (simplified)
            # In production, would use ADF test
            result['cointegration_coefficient'] = beta
            
            # Calculate half-life using AR(1) model
            if len(residuals) > 10:
                lag_residuals = residuals[:-1]
                residuals_diff = np.diff(residuals)
                
                if len(lag_residuals) > 0 and np.std(lag_residuals) > 0:
                    # AR(1) regression
                    ar_coeff = np.corrcoef(residuals_diff, lag_residuals)[0, 1]
                    
                    if ar_coeff < 0:  # Mean reverting
                        half_life = -np.log(2) / np.log(1 + ar_coeff)
                        result['half_life'] = half_life
                        result['mean_reversion_strength'] = abs(ar_coeff)
                        
                        # Simple p-value approximation
                        result['cointegration_pvalue'] = max(0.01, abs(ar_coeff) - 0.1)
            
        except Exception as e:
            logger.error(f"Error testing cointegration: {e}")
        
        return result
    
    async def _calculate_current_spread(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Calculate current spread between two symbols."""
        try:
            if self.market_data_service:
                price1 = await self.market_data_service.get_latest_price(symbol1)
                price2 = await self.market_data_service.get_latest_price(symbol2)
                
                if price1 and price2:
                    # Get hedge ratio from pair statistics
                    pair_key = f"{symbol1}_{symbol2}"
                    pair_info = self.active_pairs.get(pair_key, {})
                    hedge_ratio = pair_info.get('cointegration_coefficient', 1.0)
                    
                    spread = price2 - hedge_ratio * price1
                    
                    # Store in history
                    if pair_key in self.spread_history:
                        self.spread_history[pair_key].append(spread)
                        # Keep only recent history
                        self.spread_history[pair_key] = self.spread_history[pair_key][-100:]
                    
                    return spread
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating current spread: {e}")
            return None
    
    def _calculate_pair_score(self, pair_info: Dict[str, Any]) -> float:
        """Calculate overall score for a trading pair."""
        try:
            score = 0.0
            
            # Correlation component
            correlation = abs(pair_info.get('correlation', 0.0))
            if correlation >= self.config['min_correlation']:
                score += correlation * 0.3
            
            # Cointegration component
            coint_pvalue = pair_info.get('cointegration_pvalue', 1.0)
            if coint_pvalue <= self.config['cointegration_pvalue']:
                score += (1 - coint_pvalue) * 0.4
            
            # Mean reversion strength
            mr_strength = pair_info.get('mean_reversion_strength', 0.0)
            score += mr_strength * 0.2
            
            # Half-life component (prefer shorter half-lives)
            half_life = pair_info.get('half_life', None)
            if half_life and half_life > 0:
                # Optimal half-life between 1-10 days
                if 1 <= half_life <= 10:
                    score += (11 - half_life) / 10 * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating pair score: {e}")
            return 0.0
    
    async def _find_arbitrage_opportunities(self, symbol: str) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities for a given symbol."""
        opportunities = []
        
        try:
            # Update pair statistics
            await self._update_pair_statistics()
            
            # Check all pairs involving this symbol
            for pair_key, pair_info in self.active_pairs.items():
                if symbol not in [pair_info['symbol1'], pair_info['symbol2']]:
                    continue
                
                # Check if pair qualifies for trading
                if not self._pair_qualifies_for_trading(pair_info):
                    continue
                
                # Check for entry signal
                opportunity = await self._check_arbitrage_entry(symbol, pair_info)
                if opportunity:
                    opportunities.append(opportunity)
            
            # Sort by score
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            
            return opportunities[:5]  # Return top 5 opportunities
            
        except Exception as e:
            logger.error(f"Error finding arbitrage opportunities: {e}")
            return []
    
    def _pair_qualifies_for_trading(self, pair_info: Dict[str, Any]) -> bool:
        """Check if a pair qualifies for trading."""
        try:
            # Check correlation
            if abs(pair_info.get('correlation', 0.0)) < self.config['min_correlation']:
                return False
            
            # Check cointegration
            if pair_info.get('cointegration_pvalue', 1.0) > self.config['cointegration_pvalue']:
                return False
            
            # Check mean reversion strength
            if pair_info.get('mean_reversion_strength', 0.0) < self.config['mean_reversion_strength']:
                return False
            
            # Check half-life
            half_life = pair_info.get('half_life', None)
            if not half_life or half_life <= 0 or half_life > self.config['max_holding_period']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking pair qualification: {e}")
            return False
    
    async def _check_arbitrage_entry(self, symbol: str, pair_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for arbitrage entry opportunity."""
        try:
            spread_zscore = pair_info.get('spread_zscore', 0.0)
            
            # Check if z-score exceeds entry threshold
            if abs(spread_zscore) < self.config['entry_zscore']:
                return None
            
            # Determine direction
            if spread_zscore > self.config['entry_zscore']:
                # Spread is high - short the spread (sell symbol2, buy symbol1)
                direction = 'short_spread'
                if symbol == pair_info['symbol2']:
                    action = 'sell'
                else:
                    action = 'buy'
            elif spread_zscore < -self.config['entry_zscore']:
                # Spread is low - long the spread (buy symbol2, sell symbol1)
                direction = 'long_spread'
                if symbol == pair_info['symbol2']:
                    action = 'buy'
                else:
                    action = 'sell'
            else:
                return None
            
            opportunity = {
                'type': 'pairs_trade',
                'symbol': symbol,
                'pair_symbol': pair_info['symbol2'] if symbol == pair_info['symbol1'] else pair_info['symbol1'],
                'direction': direction,
                'action': action,
                'spread_zscore': spread_zscore,
                'hedge_ratio': pair_info.get('cointegration_coefficient', 1.0),
                'half_life': pair_info.get('half_life', 5),
                'score': abs(spread_zscore) * pair_info.get('score', 0.0),
                'pair_info': pair_info
            }
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error checking arbitrage entry: {e}")
            return None
    
    async def _calculate_pair_features(self, symbol: str) -> Dict[str, float]:
        """Calculate pair-related features."""
        features = {}
        
        try:
            # Find best pair for this symbol
            best_pair = None
            best_score = 0.0
            
            for pair_key, pair_info in self.active_pairs.items():
                if symbol in [pair_info['symbol1'], pair_info['symbol2']]:
                    if pair_info.get('score', 0.0) > best_score:
                        best_score = pair_info.get('score', 0.0)
                        best_pair = pair_info
            
            if best_pair:
                features.update({
                    'best_pair_correlation': best_pair.get('correlation', 0.0),
                    'best_pair_cointegration_pvalue': best_pair.get('cointegration_pvalue', 1.0),
                    'best_pair_spread_zscore': best_pair.get('spread_zscore', 0.0),
                    'best_pair_half_life': best_pair.get('half_life', 0.0) or 0.0,
                    'best_pair_score': best_score,
                    'best_pair_mean_reversion_strength': best_pair.get('mean_reversion_strength', 0.0)
                })
            else:
                features.update({
                    'best_pair_correlation': 0.0,
                    'best_pair_cointegration_pvalue': 1.0,
                    'best_pair_spread_zscore': 0.0,
                    'best_pair_half_life': 0.0,
                    'best_pair_score': 0.0,
                    'best_pair_mean_reversion_strength': 0.0
                })
            
            # Count qualifying pairs
            qualifying_pairs = sum(1 for pair_info in self.active_pairs.values()
                                 if symbol in [pair_info['symbol1'], pair_info['symbol2']] 
                                 and self._pair_qualifies_for_trading(pair_info))
            
            features['qualifying_pairs_count'] = qualifying_pairs
            
        except Exception as e:
            logger.error(f"Error calculating pair features: {e}")
            features = {
                'best_pair_correlation': 0.0,
                'best_pair_cointegration_pvalue': 1.0,
                'best_pair_spread_zscore': 0.0,
                'best_pair_half_life': 0.0,
                'best_pair_score': 0.0,
                'best_pair_mean_reversion_strength': 0.0,
                'qualifying_pairs_count': 0
            }
        
        return features
    
    async def _calculate_spread_features(self, symbol: str) -> Dict[str, float]:
        """Calculate spread-related features."""
        features = {
            'avg_spread_volatility': 0.0,
            'spread_momentum': 0.0,
            'spread_mean_reversion_signal': 0.0
        }
        
        try:
            # Calculate average spread volatility for pairs involving this symbol
            spread_vols = []
            spread_momentums = []
            
            for pair_key, pair_info in self.active_pairs.items():
                if symbol not in [pair_info['symbol1'], pair_info['symbol2']]:
                    continue
                
                spread_hist = self.spread_history.get(pair_key, [])
                if len(spread_hist) > 20:
                    spread_vol = np.std(spread_hist[-20:])
                    spread_vols.append(spread_vol)
                    
                    # Calculate momentum
                    recent_spreads = spread_hist[-10:]
                    older_spreads = spread_hist[-20:-10]
                    if len(recent_spreads) > 0 and len(older_spreads) > 0:
                        momentum = np.mean(recent_spreads) - np.mean(older_spreads)
                        spread_momentums.append(momentum)
            
            if spread_vols:
                features['avg_spread_volatility'] = np.mean(spread_vols)
            
            if spread_momentums:
                features['spread_momentum'] = np.mean(spread_momentums)
            
            # Mean reversion signal strength
            max_zscore = 0.0
            for pair_key, pair_info in self.active_pairs.items():
                if symbol in [pair_info['symbol1'], pair_info['symbol2']]:
                    zscore = abs(pair_info.get('spread_zscore', 0.0))
                    max_zscore = max(max_zscore, zscore)
            
            features['spread_mean_reversion_signal'] = min(max_zscore / 3.0, 1.0)  # Normalize
            
        except Exception as e:
            logger.error(f"Error calculating spread features: {e}")
        
        return features
    
    async def _calculate_microstructure_features(self, symbol: str) -> Dict[str, float]:
        """Calculate market microstructure features."""
        features = {
            'liquidity_score': 0.0,
            'bid_ask_spread': 0.0,
            'market_impact_cost': 0.0
        }
        
        try:
            # Get market data
            df = await self.get_market_data(symbol, 5)
            if df is None or df.empty:
                return features
            
            # Calculate liquidity score (simplified)
            if 'volume' in df.columns:
                avg_volume = df['volume'].mean()
                features['liquidity_score'] = min(avg_volume / 1000000, 1.0)  # Normalize to millions
            
            # Estimate bid-ask spread (simplified using high-low)
            if 'high' in df.columns and 'low' in df.columns:
                avg_spread = (df['high'] - df['low']).mean()
                avg_price = df['price'].mean() if 'price' in df.columns else 1.0
                features['bid_ask_spread'] = avg_spread / avg_price if avg_price > 0 else 0.0
            
            # Market impact cost (simplified)
            if 'volume' in df.columns:
                price_impact = df['price'].pct_change().abs().mean()
                volume_factor = 1 / (1 + features['liquidity_score'])
                features['market_impact_cost'] = price_impact * volume_factor
            
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
        
        return features
    
    async def _calculate_volatility_features(self, symbol: str) -> Dict[str, float]:
        """Calculate volatility arbitrage features."""
        features = {
            'realized_volatility': 0.0,
            'volatility_risk_premium': 0.0,
            'volatility_mean_reversion': 0.0
        }
        
        try:
            # Get market data
            df = await self.get_market_data(symbol, self.config['volatility_lookback'])
            if df is None or df.empty or 'price' not in df.columns:
                return features
            
            # Calculate realized volatility
            returns = df['price'].pct_change().dropna()
            if len(returns) > 10:
                realized_vol = returns.std() * np.sqrt(252)  # Annualized
                features['realized_volatility'] = realized_vol
                
                # Volatility mean reversion (simplified)
                vol_series = returns.rolling(10).std()
                if len(vol_series) > 20:
                    vol_mean = vol_series.mean()
                    current_vol = vol_series.iloc[-1]
                    features['volatility_mean_reversion'] = (vol_mean - current_vol) / vol_mean
            
            # Placeholder for volatility risk premium (would need options data)
            features['volatility_risk_premium'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility features: {e}")
        
        return features
    
    async def _generate_arbitrage_signal(
        self, 
        symbol: str, 
        opportunity: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on arbitrage opportunity."""
        try:
            spread_zscore = opportunity['spread_zscore']
            action = opportunity['action']
            half_life = opportunity['half_life']
            
            # Determine signal type
            if action == 'buy':
                if abs(spread_zscore) > 2.5:
                    signal_type = SignalType.STRONG_BUY
                else:
                    signal_type = SignalType.BUY
            else:  # sell
                if abs(spread_zscore) > 2.5:
                    signal_type = SignalType.STRONG_SELL
                else:
                    signal_type = SignalType.SELL
            
            # Calculate confidence based on statistical significance
            confidence = min(abs(spread_zscore) / 4.0, 0.85)  # Cap at 85%
            
            # Boost confidence for better pairs
            pair_score = opportunity.get('score', 0.0)
            confidence = min(confidence + pair_score * 0.1, 0.85)
            
            # Skip if confidence too low
            if confidence < 0.5:
                return None
            
            # Calculate targets based on expected mean reversion
            expected_reversion = min(abs(spread_zscore) * 0.3, 0.06)  # Up to 6% expected reversion
            stop_loss_pct = 0.03  # 3% stop loss
            
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None
            
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                target_price = current_price * (1 + expected_reversion)
                stop_loss = current_price * (1 - stop_loss_pct)
            else:
                target_price = current_price * (1 - expected_reversion)
                stop_loss = current_price * (1 + stop_loss_pct)
            
            # Create signal
            signal = TradingSignal(
                id=str(uuid.uuid4()),
                agent_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                signal_type=signal_type,
                confidence=confidence,
                strength=min(abs(spread_zscore) / 3.0, 1.0),
                reasoning={
                    'opportunity': opportunity,
                    'statistical_analysis': {
                        'spread_zscore': spread_zscore,
                        'half_life_days': half_life,
                        'pair_correlation': opportunity['pair_info'].get('correlation', 0.0),
                        'cointegration_pvalue': opportunity['pair_info'].get('cointegration_pvalue', 1.0)
                    }
                },
                features_used=await self.calculate_features(symbol),
                prediction_horizon=max(int(half_life * 24), 12),  # Convert half-life to hours
                target_price=target_price,
                stop_loss=stop_loss,
                risk_score=min(abs(spread_zscore) / 5.0, 1.0),
                expected_return=expected_reversion
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating arbitrage signal: {e}")
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


# Convenience function for creating arbitrage agent
def create_arbitrage_agent(symbols: List[str], **kwargs) -> ArbitrageAgent:
    """Create an arbitrage agent with default configuration."""
    return ArbitrageAgent(symbols, kwargs)