"""
Feature Store for Bloomberg Terminal
Advanced feature engineering and storage system for ML models.
"""

import asyncio
import logging
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import hashlib

from events.event_bus import EventBus, Event, EventType, get_event_bus

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Feature data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    TIMESERIES = "timeseries"
    EMBEDDING = "embedding"


class FeatureStatus(Enum):
    """Feature computation status."""
    PENDING = "pending"
    COMPUTING = "computing"
    READY = "ready"
    ERROR = "error"
    STALE = "stale"


@dataclass
class FeatureDefinition:
    """Feature definition and metadata."""
    name: str
    feature_type: FeatureType
    description: str
    computation_function: str
    dependencies: List[str] = field(default_factory=list)
    refresh_interval: int = 300  # seconds
    ttl: int = 3600  # time to live in seconds
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeatureValue:
    """Feature value with metadata."""
    feature_name: str
    symbol: str
    value: Any
    timestamp: datetime
    status: FeatureStatus = FeatureStatus.READY
    computation_time_ms: float = 0.0
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureStore:
    """
    Advanced feature store providing:
    - Real-time feature computation and caching
    - Feature versioning and lineage tracking
    - Batch and streaming feature serving
    - Feature quality monitoring
    - Feature discovery and catalog
    - ML model integration
    - Historical feature retrieval
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            # Storage configuration
            'cache_backend': 'redis',  # redis, memory, disk
            'batch_storage_backend': 'timescaledb',
            'feature_ttl_default': 3600,  # 1 hour
            'max_cache_size': 10000,  # features in memory
            
            # Computation configuration
            'parallel_workers': 4,
            'max_computation_time': 30,  # seconds
            'retry_attempts': 3,
            'computation_timeout': 60,
            
            # Quality monitoring
            'enable_monitoring': True,
            'quality_checks_enabled': True,
            'drift_detection_enabled': True,
            'anomaly_detection_threshold': 3.0,
            
            # Performance optimization
            'enable_batch_processing': True,
            'batch_size': 1000,
            'prefetch_enabled': True,
            'compression_enabled': True,
            
            # Feature serving
            'serve_stale_features': True,
            'max_staleness_seconds': 1800,  # 30 minutes
            'feature_warming_enabled': True,
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.event_bus: EventBus = get_event_bus()
        
        # Feature registry
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_cache: Dict[str, Dict[str, FeatureValue]] = {}  # feature_name -> symbol -> value
        self.computation_graph: Dict[str, List[str]] = {}  # dependency graph
        
        # Performance metrics
        self.metrics = {
            'features_computed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_errors': 0,
            'avg_computation_time_ms': 0.0,
            'last_update': datetime.now(timezone.utc)
        }
        
        # Quality monitoring
        self.feature_quality_stats: Dict[str, Dict] = {}
        self.drift_detectors: Dict[str, Any] = {}
        
        # Computation workers
        self.computation_semaphore = asyncio.Semaphore(self.config['parallel_workers'])
        self.computation_tasks: Dict[str, asyncio.Task] = {}
        
        self.is_running = False
        
    async def initialize(self) -> None:
        """Initialize the feature store."""
        try:
            logger.info("Initializing Feature Store")
            
            # Register built-in features
            await self._register_builtin_features()
            
            # Setup event subscriptions
            await self._setup_event_subscriptions()
            
            # Initialize storage backends
            await self._initialize_storage()
            
            # Start quality monitoring
            if self.config['enable_monitoring']:
                await self._initialize_monitoring()
            
            logger.info(f"Feature Store initialized with {len(self.feature_definitions)} features")
            
        except Exception as e:
            logger.error(f"Failed to initialize Feature Store: {e}")
            raise
    
    async def start(self) -> None:
        """Start the feature store."""
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._feature_refresh_loop())
        asyncio.create_task(self._quality_monitoring_loop())
        asyncio.create_task(self._cache_cleanup_loop())
        asyncio.create_task(self._metrics_update_loop())
        
        # Pre-warm critical features if enabled
        if self.config['feature_warming_enabled']:
            asyncio.create_task(self._warm_critical_features())
        
        logger.info("Feature Store started")
    
    async def stop(self) -> None:
        """Stop the feature store."""
        self.is_running = False
        
        # Cancel running computation tasks
        for task in self.computation_tasks.values():
            if not task.done():
                task.cancel()
        
        logger.info("Feature Store stopped")
    
    async def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """
        Register a new feature definition.
        
        Args:
            feature_def: Feature definition to register
            
        Returns:
            Success status
        """
        try:
            # Validate feature definition
            if not self._validate_feature_definition(feature_def):
                return False
            
            # Check for circular dependencies
            if self._has_circular_dependency(feature_def.name, feature_def.dependencies):
                logger.error(f"Circular dependency detected for feature {feature_def.name}")
                return False
            
            # Register feature
            self.feature_definitions[feature_def.name] = feature_def
            
            # Update dependency graph
            self.computation_graph[feature_def.name] = feature_def.dependencies
            
            # Initialize cache entry
            if feature_def.name not in self.feature_cache:
                self.feature_cache[feature_def.name] = {}
            
            # Initialize quality monitoring
            if self.config['enable_monitoring']:
                await self._initialize_feature_monitoring(feature_def.name)
            
            logger.info(f"Registered feature: {feature_def.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering feature {feature_def.name}: {e}")
            return False
    
    async def get_feature(
        self, 
        feature_name: str, 
        symbol: str,
        force_refresh: bool = False
    ) -> Optional[FeatureValue]:
        """
        Get a single feature value.
        
        Args:
            feature_name: Name of the feature
            symbol: Symbol to get feature for
            force_refresh: Force recomputation even if cached
            
        Returns:
            Feature value or None if not available
        """
        try:
            # Check if feature is registered
            if feature_name not in self.feature_definitions:
                logger.error(f"Feature {feature_name} not registered")
                return None
            
            # Check cache first (unless force refresh)
            if not force_refresh:
                cached_value = await self._get_cached_feature(feature_name, symbol)
                if cached_value and not self._is_stale(cached_value):
                    self.metrics['cache_hits'] += 1
                    return cached_value
            
            self.metrics['cache_misses'] += 1
            
            # Compute feature
            feature_value = await self._compute_feature(feature_name, symbol)
            
            if feature_value:
                # Cache the result
                await self._cache_feature(feature_value)
                
                # Update metrics
                self.metrics['features_computed'] += 1
                self._update_computation_time_metric(feature_value.computation_time_ms)
            
            return feature_value
            
        except Exception as e:
            logger.error(f"Error getting feature {feature_name} for {symbol}: {e}")
            self.metrics['computation_errors'] += 1
            return None
    
    async def get_features(
        self,
        feature_names: List[str],
        symbols: List[str],
        force_refresh: bool = False
    ) -> Dict[str, Dict[str, FeatureValue]]:
        """
        Get multiple features for multiple symbols efficiently.
        
        Args:
            feature_names: List of feature names
            symbols: List of symbols
            force_refresh: Force recomputation
            
        Returns:
            Dictionary of feature_name -> symbol -> FeatureValue
        """
        try:
            results = {}
            
            # Prepare computation tasks
            tasks = []
            for feature_name in feature_names:
                results[feature_name] = {}
                for symbol in symbols:
                    task = asyncio.create_task(
                        self.get_feature(feature_name, symbol, force_refresh)
                    )
                    tasks.append((feature_name, symbol, task))
            
            # Execute all tasks concurrently
            for feature_name, symbol, task in tasks:
                try:
                    feature_value = await task
                    if feature_value:
                        results[feature_name][symbol] = feature_value
                except Exception as e:
                    logger.error(f"Error computing {feature_name} for {symbol}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting multiple features: {e}")
            return {}
    
    async def get_feature_vector(
        self,
        symbol: str,
        feature_names: Optional[List[str]] = None,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Get feature vector for ML model input.
        
        Args:
            symbol: Symbol to get features for
            feature_names: Specific features to include (None for all)
            include_metadata: Include feature metadata
            
        Returns:
            Feature vector as dictionary
        """
        try:
            if feature_names is None:
                feature_names = list(self.feature_definitions.keys())
            
            # Get all requested features
            features = {}
            metadata = {}
            
            for feature_name in feature_names:
                feature_value = await self.get_feature(feature_name, symbol)
                if feature_value:
                    features[feature_name] = feature_value.value
                    if include_metadata:
                        metadata[feature_name] = {
                            'timestamp': feature_value.timestamp.isoformat(),
                            'status': feature_value.status.value,
                            'computation_time_ms': feature_value.computation_time_ms,
                            'version': feature_value.version
                        }
                else:
                    # Use default value for missing features
                    features[feature_name] = self._get_default_value(feature_name)
                    if include_metadata:
                        metadata[feature_name] = {'status': 'missing'}
            
            result = {'features': features}
            if include_metadata:
                result['metadata'] = metadata
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting feature vector for {symbol}: {e}")
            return {}
    
    async def get_historical_features(
        self,
        feature_names: List[str],
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Get historical feature values for backtesting and training.
        
        Args:
            feature_names: Features to retrieve
            symbols: Symbols to get data for
            start_time: Start time
            end_time: End time
            interval: Data interval (1m, 5m, 1h, 1d)
            
        Returns:
            DataFrame with historical features
        """
        try:
            # This would query the batch storage backend (TimescaleDB)
            # For now, return mock historical data
            
            time_range = pd.date_range(start=start_time, end=end_time, freq=interval)
            
            data = []
            for timestamp in time_range:
                for symbol in symbols:
                    row = {
                        'timestamp': timestamp,
                        'symbol': symbol
                    }
                    
                    # Generate mock feature values
                    for feature_name in feature_names:
                        if feature_name in self.feature_definitions:
                            feature_def = self.feature_definitions[feature_name]
                            if feature_def.feature_type == FeatureType.NUMERICAL:
                                row[feature_name] = np.random.normal(0, 1)
                            elif feature_def.feature_type == FeatureType.BOOLEAN:
                                row[feature_name] = np.random.choice([True, False])
                            elif feature_def.feature_type == FeatureType.CATEGORICAL:
                                row[feature_name] = np.random.choice(['A', 'B', 'C'])
                            else:
                                row[feature_name] = 0.0
                    
                    data.append(row)
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical features: {e}")
            return pd.DataFrame()
    
    async def get_feature_stats(self, feature_name: str) -> Dict[str, Any]:
        """Get statistics and quality metrics for a feature."""
        try:
            if feature_name not in self.feature_definitions:
                return {}
            
            feature_def = self.feature_definitions[feature_name]
            quality_stats = self.feature_quality_stats.get(feature_name, {})
            
            # Calculate cache statistics
            cache_data = self.feature_cache.get(feature_name, {})
            total_cached = len(cache_data)
            fresh_count = sum(1 for fv in cache_data.values() if not self._is_stale(fv))
            
            return {
                'feature_name': feature_name,
                'feature_type': feature_def.feature_type.value,
                'description': feature_def.description,
                'version': feature_def.version,
                'dependencies': feature_def.dependencies,
                'refresh_interval': feature_def.refresh_interval,
                'cached_symbols': total_cached,
                'fresh_values': fresh_count,
                'stale_values': total_cached - fresh_count,
                'quality_stats': quality_stats,
                'last_updated': feature_def.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting feature stats: {e}")
            return {}
    
    async def list_features(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all registered features with optional tag filtering."""
        try:
            features = []
            
            for feature_name, feature_def in self.feature_definitions.items():
                # Apply tag filter
                if tags and not any(tag in feature_def.tags for tag in tags):
                    continue
                
                feature_info = {
                    'name': feature_name,
                    'type': feature_def.feature_type.value,
                    'description': feature_def.description,
                    'version': feature_def.version,
                    'tags': feature_def.tags,
                    'dependencies': feature_def.dependencies,
                    'refresh_interval': feature_def.refresh_interval,
                    'created_at': feature_def.created_at.isoformat(),
                    'updated_at': feature_def.updated_at.isoformat()
                }
                
                features.append(feature_info)
            
            return sorted(features, key=lambda x: x['name'])
            
        except Exception as e:
            logger.error(f"Error listing features: {e}")
            return []
    
    async def invalidate_feature(self, feature_name: str, symbol: Optional[str] = None) -> bool:
        """Invalidate cached feature values."""
        try:
            if feature_name not in self.feature_cache:
                return True
            
            if symbol:
                # Invalidate specific symbol
                if symbol in self.feature_cache[feature_name]:
                    del self.feature_cache[feature_name][symbol]
            else:
                # Invalidate all symbols for this feature
                self.feature_cache[feature_name].clear()
            
            logger.info(f"Invalidated feature {feature_name}" + (f" for {symbol}" if symbol else ""))
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating feature: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get feature store performance metrics."""
        try:
            cache_size = sum(len(symbols) for symbols in self.feature_cache.values())
            hit_rate = self.metrics['cache_hits'] / max(
                self.metrics['cache_hits'] + self.metrics['cache_misses'], 1
            )
            
            return {
                **self.metrics,
                'registered_features': len(self.feature_definitions),
                'cached_features': cache_size,
                'cache_hit_rate': hit_rate,
                'running_computations': len(self.computation_tasks),
                'feature_types_distribution': self._get_feature_type_distribution(),
                'quality_issues': self._count_quality_issues()
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
    
    async def _register_builtin_features(self) -> None:
        """Register built-in technical and fundamental features."""
        try:
            builtin_features = [
                # Technical indicators
                FeatureDefinition(
                    name="price_sma_20",
                    feature_type=FeatureType.NUMERICAL,
                    description="20-period Simple Moving Average of price",
                    computation_function="compute_sma",
                    tags=["technical", "price", "trend"]
                ),
                FeatureDefinition(
                    name="price_ema_12",
                    feature_type=FeatureType.NUMERICAL,
                    description="12-period Exponential Moving Average",
                    computation_function="compute_ema",
                    tags=["technical", "price", "trend"]
                ),
                FeatureDefinition(
                    name="rsi_14",
                    feature_type=FeatureType.NUMERICAL,
                    description="14-period Relative Strength Index",
                    computation_function="compute_rsi",
                    tags=["technical", "momentum", "oscillator"]
                ),
                FeatureDefinition(
                    name="macd_signal",
                    feature_type=FeatureType.NUMERICAL,
                    description="MACD Signal Line",
                    computation_function="compute_macd",
                    dependencies=["price_ema_12", "price_ema_26"],
                    tags=["technical", "momentum", "trend"]
                ),
                FeatureDefinition(
                    name="bollinger_upper",
                    feature_type=FeatureType.NUMERICAL,
                    description="Bollinger Bands Upper Band",
                    computation_function="compute_bollinger_bands",
                    dependencies=["price_sma_20"],
                    tags=["technical", "volatility", "bands"]
                ),
                FeatureDefinition(
                    name="volume_sma_20",
                    feature_type=FeatureType.NUMERICAL,
                    description="20-period Volume Simple Moving Average",
                    computation_function="compute_volume_sma",
                    tags=["technical", "volume"]
                ),
                FeatureDefinition(
                    name="volatility_20d",
                    feature_type=FeatureType.NUMERICAL,
                    description="20-day realized volatility",
                    computation_function="compute_volatility",
                    tags=["risk", "volatility", "statistical"]
                ),
                
                # Market microstructure
                FeatureDefinition(
                    name="bid_ask_spread",
                    feature_type=FeatureType.NUMERICAL,
                    description="Bid-ask spread as percentage of mid price",
                    computation_function="compute_spread",
                    tags=["microstructure", "liquidity"]
                ),
                FeatureDefinition(
                    name="order_imbalance",
                    feature_type=FeatureType.NUMERICAL,
                    description="Order book imbalance ratio",
                    computation_function="compute_order_imbalance",
                    tags=["microstructure", "flow"]
                ),
                
                # Fundamental features
                FeatureDefinition(
                    name="market_cap_rank",
                    feature_type=FeatureType.NUMERICAL,
                    description="Market capitalization ranking",
                    computation_function="compute_market_cap_rank",
                    tags=["fundamental", "size"]
                ),
                FeatureDefinition(
                    name="sector_momentum",
                    feature_type=FeatureType.NUMERICAL,
                    description="Sector relative momentum",
                    computation_function="compute_sector_momentum",
                    tags=["fundamental", "sector", "momentum"]
                ),
                
                # Sentiment features
                FeatureDefinition(
                    name="news_sentiment_1d",
                    feature_type=FeatureType.NUMERICAL,
                    description="1-day aggregated news sentiment score",
                    computation_function="compute_news_sentiment",
                    tags=["sentiment", "news", "alternative"]
                ),
                FeatureDefinition(
                    name="social_sentiment_1d",
                    feature_type=FeatureType.NUMERICAL,
                    description="1-day social media sentiment",
                    computation_function="compute_social_sentiment",
                    tags=["sentiment", "social", "alternative"]
                ),
                
                # Risk features
                FeatureDefinition(
                    name="beta_60d",
                    feature_type=FeatureType.NUMERICAL,
                    description="60-day rolling beta vs market",
                    computation_function="compute_beta",
                    tags=["risk", "correlation", "systematic"]
                ),
                FeatureDefinition(
                    name="var_1d_95",
                    feature_type=FeatureType.NUMERICAL,
                    description="1-day 95% Value at Risk",
                    computation_function="compute_var",
                    dependencies=["volatility_20d"],
                    tags=["risk", "var", "quantitative"]
                ),
                
                # Categorical features
                FeatureDefinition(
                    name="trend_regime",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Current trend regime (uptrend/downtrend/sideways)",
                    computation_function="compute_trend_regime",
                    dependencies=["price_sma_20", "price_sma_50"],
                    tags=["regime", "trend", "classification"]
                ),
                FeatureDefinition(
                    name="volatility_regime",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Volatility regime (low/medium/high)",
                    computation_function="compute_vol_regime",
                    dependencies=["volatility_20d"],
                    tags=["regime", "volatility", "classification"]
                ),
                
                # Boolean features
                FeatureDefinition(
                    name="is_new_high",
                    feature_type=FeatureType.BOOLEAN,
                    description="Is current price a 20-day high",
                    computation_function="compute_new_high",
                    tags=["momentum", "extremes", "binary"]
                ),
                FeatureDefinition(
                    name="earnings_week",
                    feature_type=FeatureType.BOOLEAN,
                    description="Is earnings announcement within 1 week",
                    computation_function="compute_earnings_proximity",
                    tags=["fundamental", "events", "binary"]
                ),
            ]
            
            # Register all features
            for feature_def in builtin_features:
                await self.register_feature(feature_def)
            
        except Exception as e:
            logger.error(f"Error registering builtin features: {e}")
    
    async def _compute_feature(self, feature_name: str, symbol: str) -> Optional[FeatureValue]:
        """Compute a single feature value."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Get feature definition
            feature_def = self.feature_definitions[feature_name]
            
            # Compute dependencies first
            dependency_values = {}
            for dep_name in feature_def.dependencies:
                dep_value = await self.get_feature(dep_name, symbol)
                if dep_value:
                    dependency_values[dep_name] = dep_value.value
                else:
                    logger.warning(f"Failed to compute dependency {dep_name} for {feature_name}")
                    return None
            
            # Use semaphore to limit concurrent computations
            async with self.computation_semaphore:
                # Compute the feature
                computed_value = await self._execute_computation(
                    feature_def.computation_function,
                    symbol,
                    dependency_values
                )
                
                if computed_value is None:
                    return None
                
                # Create feature value
                end_time = asyncio.get_event_loop().time()
                computation_time_ms = (end_time - start_time) * 1000
                
                feature_value = FeatureValue(
                    feature_name=feature_name,
                    symbol=symbol,
                    value=computed_value,
                    timestamp=datetime.now(timezone.utc),
                    status=FeatureStatus.READY,
                    computation_time_ms=computation_time_ms,
                    version=feature_def.version
                )
                
                # Update quality monitoring
                if self.config['enable_monitoring']:
                    await self._update_feature_quality(feature_value)
                
                return feature_value
                
        except Exception as e:
            logger.error(f"Error computing feature {feature_name}: {e}")
            return FeatureValue(
                feature_name=feature_name,
                symbol=symbol,
                value=None,
                timestamp=datetime.now(timezone.utc),
                status=FeatureStatus.ERROR
            )
    
    async def _execute_computation(
        self,
        computation_function: str,
        symbol: str,
        dependencies: Dict[str, Any]
    ) -> Any:
        """Execute feature computation function."""
        try:
            # Mock computation functions - in production these would be real implementations
            if computation_function == "compute_sma":
                return np.random.uniform(100, 200)  # Mock price SMA
            elif computation_function == "compute_ema":
                return np.random.uniform(95, 205)   # Mock EMA
            elif computation_function == "compute_rsi":
                return np.random.uniform(20, 80)    # Mock RSI
            elif computation_function == "compute_macd":
                return np.random.uniform(-5, 5)     # Mock MACD
            elif computation_function == "compute_bollinger_bands":
                sma = dependencies.get("price_sma_20", 150)
                return sma * 1.02  # Mock upper band
            elif computation_function == "compute_volume_sma":
                return np.random.uniform(1000000, 5000000)  # Mock volume
            elif computation_function == "compute_volatility":
                return np.random.uniform(0.15, 0.45)  # Mock volatility
            elif computation_function == "compute_spread":
                return np.random.uniform(0.001, 0.01)  # Mock bid-ask spread
            elif computation_function == "compute_order_imbalance":
                return np.random.uniform(-1, 1)  # Mock order imbalance
            elif computation_function == "compute_market_cap_rank":
                return np.random.randint(1, 1000)  # Mock market cap rank
            elif computation_function == "compute_sector_momentum":
                return np.random.uniform(-0.1, 0.1)  # Mock sector momentum
            elif computation_function == "compute_news_sentiment":
                return np.random.uniform(-1, 1)  # Mock news sentiment
            elif computation_function == "compute_social_sentiment":
                return np.random.uniform(-1, 1)  # Mock social sentiment
            elif computation_function == "compute_beta":
                return np.random.uniform(0.5, 2.0)  # Mock beta
            elif computation_function == "compute_var":
                volatility = dependencies.get("volatility_20d", 0.2)
                return volatility * np.sqrt(1/252) * 1.96  # 95% VaR
            elif computation_function == "compute_trend_regime":
                return np.random.choice(["uptrend", "downtrend", "sideways"])
            elif computation_function == "compute_vol_regime":
                return np.random.choice(["low", "medium", "high"])
            elif computation_function == "compute_new_high":
                return np.random.choice([True, False])
            elif computation_function == "compute_earnings_proximity":
                return np.random.choice([True, False])
            else:
                logger.warning(f"Unknown computation function: {computation_function}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error executing computation {computation_function}: {e}")
            return None
    
    async def _get_cached_feature(self, feature_name: str, symbol: str) -> Optional[FeatureValue]:
        """Get feature value from cache."""
        try:
            if feature_name in self.feature_cache:
                if symbol in self.feature_cache[feature_name]:
                    return self.feature_cache[feature_name][symbol]
            return None
        except Exception:
            return None
    
    async def _cache_feature(self, feature_value: FeatureValue) -> None:
        """Cache feature value."""
        try:
            feature_name = feature_value.feature_name
            symbol = feature_value.symbol
            
            if feature_name not in self.feature_cache:
                self.feature_cache[feature_name] = {}
            
            self.feature_cache[feature_name][symbol] = feature_value
            
        except Exception as e:
            logger.error(f"Error caching feature: {e}")
    
    def _is_stale(self, feature_value: FeatureValue) -> bool:
        """Check if feature value is stale."""
        try:
            if feature_value.feature_name not in self.feature_definitions:
                return True
            
            feature_def = self.feature_definitions[feature_value.feature_name]
            age_seconds = (datetime.now(timezone.utc) - feature_value.timestamp).total_seconds()
            
            return age_seconds > feature_def.ttl
            
        except Exception:
            return True
    
    def _validate_feature_definition(self, feature_def: FeatureDefinition) -> bool:
        """Validate feature definition."""
        try:
            # Check required fields
            if not feature_def.name or not feature_def.computation_function:
                return False
            
            # Check feature name format
            if not feature_def.name.replace('_', '').replace('-', '').isalnum():
                return False
            
            # Check dependencies exist
            for dep in feature_def.dependencies:
                if dep not in self.feature_definitions and dep != feature_def.name:
                    logger.warning(f"Dependency {dep} not found for feature {feature_def.name}")
            
            return True
            
        except Exception:
            return False
    
    def _has_circular_dependency(self, feature_name: str, dependencies: List[str]) -> bool:
        """Check for circular dependencies."""
        def check_circular(current: str, visited: set, path: set) -> bool:
            if current in path:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            path.add(current)
            
            for dep in self.computation_graph.get(current, []):
                if check_circular(dep, visited, path):
                    return True
            
            path.remove(current)
            return False
        
        # Add new dependencies to graph temporarily
        temp_graph = self.computation_graph.copy()
        temp_graph[feature_name] = dependencies
        
        # Check for cycles
        visited = set()
        for node in temp_graph:
            if check_circular(node, visited, set()):
                return True
        
        return False
    
    def _get_default_value(self, feature_name: str) -> Any:
        """Get default value for missing feature."""
        if feature_name not in self.feature_definitions:
            return 0.0
        
        feature_type = self.feature_definitions[feature_name].feature_type
        
        if feature_type == FeatureType.NUMERICAL:
            return 0.0
        elif feature_type == FeatureType.BOOLEAN:
            return False
        elif feature_type == FeatureType.CATEGORICAL:
            return "unknown"
        else:
            return None
    
    def _update_computation_time_metric(self, computation_time_ms: float) -> None:
        """Update average computation time metric."""
        current_avg = self.metrics['avg_computation_time_ms']
        total_computed = self.metrics['features_computed']
        
        if total_computed > 0:
            self.metrics['avg_computation_time_ms'] = (
                (current_avg * (total_computed - 1) + computation_time_ms) / total_computed
            )
        else:
            self.metrics['avg_computation_time_ms'] = computation_time_ms
    
    def _get_feature_type_distribution(self) -> Dict[str, int]:
        """Get distribution of feature types."""
        distribution = {}
        for feature_def in self.feature_definitions.values():
            feature_type = feature_def.feature_type.value
            distribution[feature_type] = distribution.get(feature_type, 0) + 1
        return distribution
    
    def _count_quality_issues(self) -> int:
        """Count features with quality issues."""
        return sum(
            1 for stats in self.feature_quality_stats.values()
            if stats.get('has_issues', False)
        )
    
    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        try:
            # Subscribe to market data updates to trigger feature refresh
            await self.event_bus.subscribe(
                [EventType.MARKET_DATA_UPDATE],
                self._handle_market_data_event
            )
            
        except Exception as e:
            logger.error(f"Failed to setup event subscriptions: {e}")
    
    async def _handle_market_data_event(self, event: Event) -> None:
        """Handle market data events."""
        try:
            market_data = event.data
            symbol = market_data.get('symbol')
            
            if symbol:
                # Invalidate price-dependent features
                price_features = [name for name, def_ in self.feature_definitions.items() 
                                if 'price' in def_.tags]
                
                for feature_name in price_features:
                    await self.invalidate_feature(feature_name, symbol)
                    
        except Exception as e:
            logger.error(f"Error handling market data event: {e}")
    
    async def _initialize_storage(self) -> None:
        """Initialize storage backends."""
        # Would initialize Redis, TimescaleDB connections here
        pass
    
    async def _initialize_monitoring(self) -> None:
        """Initialize quality monitoring."""
        for feature_name in self.feature_definitions:
            await self._initialize_feature_monitoring(feature_name)
    
    async def _initialize_feature_monitoring(self, feature_name: str) -> None:
        """Initialize monitoring for a specific feature."""
        self.feature_quality_stats[feature_name] = {
            'computation_count': 0,
            'error_count': 0,
            'avg_computation_time': 0.0,
            'last_value': None,
            'has_issues': False
        }
    
    async def _update_feature_quality(self, feature_value: FeatureValue) -> None:
        """Update quality statistics for a feature."""
        try:
            feature_name = feature_value.feature_name
            stats = self.feature_quality_stats.get(feature_name, {})
            
            stats['computation_count'] = stats.get('computation_count', 0) + 1
            stats['last_value'] = feature_value.value
            stats['last_updated'] = datetime.now(timezone.utc)
            
            # Update average computation time
            current_avg = stats.get('avg_computation_time', 0.0)
            count = stats['computation_count']
            stats['avg_computation_time'] = (
                (current_avg * (count - 1) + feature_value.computation_time_ms) / count
            )
            
            # Check for anomalies
            if self.config['anomaly_detection_threshold'] > 0:
                if feature_value.computation_time_ms > self.config['anomaly_detection_threshold'] * stats['avg_computation_time']:
                    stats['has_issues'] = True
                    logger.warning(f"Slow computation detected for {feature_name}: {feature_value.computation_time_ms}ms")
            
            self.feature_quality_stats[feature_name] = stats
            
        except Exception as e:
            logger.error(f"Error updating feature quality: {e}")
    
    async def _feature_refresh_loop(self) -> None:
        """Background loop to refresh stale features."""
        while self.is_running:
            try:
                # Check for stale features and refresh them
                for feature_name, symbol_cache in self.feature_cache.items():
                    for symbol, feature_value in symbol_cache.items():
                        if self._is_stale(feature_value):
                            # Refresh in background
                            asyncio.create_task(self.get_feature(feature_name, symbol, force_refresh=True))
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in feature refresh loop: {e}")
                await asyncio.sleep(30)
    
    async def _quality_monitoring_loop(self) -> None:
        """Background quality monitoring loop."""
        while self.is_running:
            try:
                # Monitor feature quality and detect issues
                for feature_name, stats in self.feature_quality_stats.items():
                    # Check error rate
                    error_rate = stats.get('error_count', 0) / max(stats.get('computation_count', 1), 1)
                    if error_rate > 0.1:  # 10% error rate threshold
                        logger.warning(f"High error rate for feature {feature_name}: {error_rate:.2%}")
                        stats['has_issues'] = True
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _cache_cleanup_loop(self) -> None:
        """Background cache cleanup loop."""
        while self.is_running:
            try:
                # Remove very old cached values
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=6)
                
                for feature_name, symbol_cache in list(self.feature_cache.items()):
                    symbols_to_remove = []
                    for symbol, feature_value in symbol_cache.items():
                        if feature_value.timestamp < cutoff_time:
                            symbols_to_remove.append(symbol)
                    
                    for symbol in symbols_to_remove:
                        del symbol_cache[symbol]
                
                await asyncio.sleep(1800)  # Cleanup every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _metrics_update_loop(self) -> None:
        """Background metrics update loop."""
        while self.is_running:
            try:
                self.metrics['last_update'] = datetime.now(timezone.utc)
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(30)
    
    async def _warm_critical_features(self) -> None:
        """Pre-warm critical features for better performance."""
        try:
            # Identify critical features (most frequently used)
            critical_features = ['price_sma_20', 'rsi_14', 'volatility_20d', 'beta_60d']
            critical_symbols = ['AAPL', 'GOOGL', 'MSFT', 'SPY', 'QQQ']
            
            logger.info("Pre-warming critical features...")
            
            tasks = []
            for feature_name in critical_features:
                if feature_name in self.feature_definitions:
                    for symbol in critical_symbols:
                        task = asyncio.create_task(self.get_feature(feature_name, symbol))
                        tasks.append(task)
            
            # Execute warming tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"Pre-warmed {len(tasks)} feature values")
                
        except Exception as e:
            logger.error(f"Error pre-warming features: {e}")


# Convenience function
def create_feature_store(**kwargs) -> FeatureStore:
    """Create a feature store with configuration."""
    return FeatureStore(kwargs)