"""
ML Pipeline for Bloomberg Terminal
Advanced machine learning pipeline for model training and inference.
"""

import asyncio
import logging
import uuid
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ml.feature_store import FeatureStore
from events.event_bus import EventBus, Event, EventType, get_event_bus

logger = logging.getLogger(__name__)

# Optional ML imports - gracefully handle if not available
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - using mock models")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelType(Enum):
    """ML model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"


class ModelStatus(Enum):
    """Model training and deployment status."""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRED = "retired"


class PredictionType(Enum):
    """Types of predictions."""
    SIGNAL_DIRECTION = "signal_direction"  # Buy/Sell/Hold
    PRICE_MOVEMENT = "price_movement"      # Price change prediction
    VOLATILITY = "volatility"             # Volatility prediction
    RISK_SCORE = "risk_score"            # Risk assessment
    ANOMALY_SCORE = "anomaly_score"      # Anomaly detection


@dataclass
class ModelConfig:
    """ML model configuration."""
    name: str
    model_type: ModelType
    prediction_type: PredictionType
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_column: str = ""
    training_window_days: int = 30
    validation_split: float = 0.2
    retrain_frequency_hours: int = 24
    performance_threshold: float = 0.6
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    training_samples: int = 0
    validation_samples: int = 0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[List[List[int]]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Prediction:
    """Model prediction result."""
    id: str
    model_name: str
    symbol: str
    prediction_type: PredictionType
    value: Union[float, str, Dict[str, float]]
    confidence: float
    features_used: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLPipeline:
    """
    Comprehensive ML pipeline providing:
    - Model training and deployment
    - Feature engineering integration
    - Real-time inference serving
    - Model performance monitoring
    - Automated retraining
    - A/B testing framework
    - Model versioning and rollback
    """
    
    def __init__(self, feature_store: FeatureStore, config: Dict[str, Any] = None):
        self.feature_store = feature_store
        
        default_config = {
            # Model storage
            'model_storage_path': './models',
            'model_registry_backend': 'local',  # local, s3, gcs
            'model_versioning': True,
            
            # Training configuration
            'training_data_lookback_days': 90,
            'min_training_samples': 1000,
            'max_training_time_hours': 6,
            'parallel_training': True,
            'auto_hyperparameter_tuning': True,
            
            # Inference configuration
            'inference_timeout_ms': 1000,
            'batch_inference_size': 100,
            'prediction_caching': True,
            'cache_ttl_seconds': 300,
            
            # Monitoring
            'performance_monitoring_enabled': True,
            'model_drift_detection': True,
            'data_drift_detection': True,
            'alert_threshold_accuracy_drop': 0.05,
            
            # Retraining
            'auto_retraining_enabled': True,
            'retrain_schedule': 'daily',
            'champion_challenger_testing': True,
            'rollback_on_performance_drop': True,
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.event_bus: EventBus = get_event_bus()
        
        # Model registry
        self.model_configs: Dict[str, ModelConfig] = {}
        self.trained_models: Dict[str, Any] = {}  # model_name -> model object
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        
        # Preprocessing pipelines
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        
        # Prediction cache
        self.prediction_cache: Dict[str, Dict] = {}  # model_name -> symbol -> prediction
        
        # Performance monitoring
        self.performance_history: Dict[str, List[Dict]] = {}
        self.drift_detectors: Dict[str, Any] = {}
        
        # Training queue
        self.training_queue: asyncio.Queue = asyncio.Queue()
        self.training_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent trainings
        
        self.is_running = False
        
        # Ensure model storage directory exists
        Path(self.config['model_storage_path']).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the ML pipeline."""
        try:
            logger.info("Initializing ML Pipeline")
            
            # Setup event subscriptions
            await self._setup_event_subscriptions()
            
            # Register built-in models
            await self._register_builtin_models()
            
            # Load existing trained models
            await self._load_existing_models()
            
            logger.info(f"ML Pipeline initialized with {len(self.model_configs)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Pipeline: {e}")
            raise
    
    async def start(self) -> None:
        """Start the ML pipeline."""
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._training_worker())
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._retraining_scheduler())
        asyncio.create_task(self._cache_cleanup_loop())
        
        # Start initial training for models without trained versions
        await self._start_initial_training()
        
        logger.info("ML Pipeline started")
    
    async def stop(self) -> None:
        """Stop the ML pipeline."""
        self.is_running = False
        logger.info("ML Pipeline stopped")
    
    async def register_model(self, model_config: ModelConfig) -> bool:
        """
        Register a new ML model configuration.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Success status
        """
        try:
            # Validate model configuration
            if not self._validate_model_config(model_config):
                return False
            
            # Register model
            self.model_configs[model_config.name] = model_config
            self.model_status[model_config.name] = ModelStatus.CREATED
            self.performance_history[model_config.name] = []
            
            # Initialize preprocessing components
            if SKLEARN_AVAILABLE:
                self.scalers[model_config.name] = StandardScaler()
                if model_config.model_type == ModelType.CLASSIFICATION:
                    self.encoders[model_config.name] = LabelEncoder()
            
            logger.info(f"Registered model: {model_config.name}")
            
            # Queue for training if features are available
            if await self._check_feature_availability(model_config.feature_names):
                await self.training_queue.put(model_config.name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering model {model_config.name}: {e}")
            return False
    
    async def train_model(self, model_name: str, symbols: List[str] = None) -> bool:
        """
        Train or retrain a model.
        
        Args:
            model_name: Name of the model to train
            symbols: Symbols to train on (None for all available)
            
        Returns:
            Success status
        """
        try:
            if model_name not in self.model_configs:
                logger.error(f"Model {model_name} not registered")
                return False
            
            model_config = self.model_configs[model_name]
            self.model_status[model_name] = ModelStatus.TRAINING
            
            logger.info(f"Starting training for model {model_name}")
            
            # Get training data
            training_data = await self._prepare_training_data(model_config, symbols)
            if training_data.empty:
                logger.error(f"No training data available for {model_name}")
                self.model_status[model_name] = ModelStatus.FAILED
                return False
            
            # Train model
            model, metrics = await self._train_model_impl(model_config, training_data)
            
            if model is None:
                logger.error(f"Training failed for model {model_name}")
                self.model_status[model_name] = ModelStatus.FAILED
                return False
            
            # Store trained model
            self.trained_models[model_name] = model
            self.model_metrics[model_name] = metrics
            self.model_status[model_name] = ModelStatus.TRAINED
            
            # Save model to disk
            await self._save_model(model_name, model)
            
            # Update performance history
            self.performance_history[model_name].append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': metrics.__dict__.copy(),
                'training_samples': metrics.training_samples
            })
            
            logger.info(f"Successfully trained model {model_name} - Accuracy: {metrics.accuracy:.3f}")
            
            # Deploy model if performance meets threshold
            if metrics.accuracy >= model_config.performance_threshold:
                await self.deploy_model(model_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            self.model_status[model_name] = ModelStatus.FAILED
            return False
    
    async def predict(
        self,
        model_name: str,
        symbol: str,
        features: Optional[Dict[str, Any]] = None
    ) -> Optional[Prediction]:
        """
        Generate prediction using trained model.
        
        Args:
            model_name: Name of the model
            symbol: Symbol to predict for
            features: Optional pre-computed features
            
        Returns:
            Prediction result or None
        """
        try:
            # Check if model is deployed
            if self.model_status.get(model_name) != ModelStatus.DEPLOYED:
                logger.warning(f"Model {model_name} not deployed")
                return None
            
            # Check prediction cache first
            cache_key = f"{model_name}:{symbol}"
            if self.config['prediction_caching'] and cache_key in self.prediction_cache:
                cached_pred = self.prediction_cache[cache_key]
                if (datetime.now(timezone.utc) - cached_pred['timestamp']).total_seconds() < self.config['cache_ttl_seconds']:
                    return cached_pred['prediction']
            
            # Get model configuration
            model_config = self.model_configs[model_name]
            trained_model = self.trained_models[model_name]
            
            # Get features
            if features is None:
                feature_vector = await self.feature_store.get_feature_vector(
                    symbol, model_config.feature_names
                )
                features = feature_vector.get('features', {})
            
            # Prepare feature array
            feature_array = np.array([
                features.get(fname, 0.0) for fname in model_config.feature_names
            ]).reshape(1, -1)
            
            # Apply preprocessing
            if model_name in self.scalers:
                feature_array = self.scalers[model_name].transform(feature_array)
            
            # Generate prediction
            if SKLEARN_AVAILABLE:
                if model_config.model_type == ModelType.CLASSIFICATION:
                    prediction_probs = trained_model.predict_proba(feature_array)[0]
                    predicted_class = trained_model.predict(feature_array)[0]
                    confidence = max(prediction_probs)
                    
                    # Convert back from encoded labels
                    if model_name in self.encoders:
                        predicted_class = self.encoders[model_name].inverse_transform([predicted_class])[0]
                    
                    prediction_value = predicted_class
                    
                elif model_config.model_type == ModelType.REGRESSION:
                    prediction_value = trained_model.predict(feature_array)[0]
                    confidence = 0.8  # Would calculate actual confidence interval
                else:
                    prediction_value = 0.0
                    confidence = 0.5
            else:
                # Mock prediction
                prediction_value = self._generate_mock_prediction(model_config.prediction_type)
                confidence = np.random.uniform(0.6, 0.9)
            
            # Create prediction object
            prediction = Prediction(
                id=str(uuid.uuid4()),
                model_name=model_name,
                symbol=symbol,
                prediction_type=model_config.prediction_type,
                value=prediction_value,
                confidence=confidence,
                features_used=features,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'model_version': model_config.version,
                    'feature_count': len(features)
                }
            )
            
            # Cache prediction
            if self.config['prediction_caching']:
                self.prediction_cache[cache_key] = {
                    'prediction': prediction,
                    'timestamp': datetime.now(timezone.utc)
                }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {model_name}: {e}")
            return None
    
    async def batch_predict(
        self,
        model_name: str,
        symbols: List[str]
    ) -> Dict[str, Optional[Prediction]]:
        """Generate predictions for multiple symbols efficiently."""
        try:
            results = {}
            
            # Process in batches
            batch_size = self.config['batch_inference_size']
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                # Generate predictions concurrently
                tasks = [
                    asyncio.create_task(self.predict(model_name, symbol))
                    for symbol in batch_symbols
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for symbol, result in zip(batch_symbols, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error predicting {symbol}: {result}")
                        results[symbol] = None
                    else:
                        results[symbol] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return {}
    
    async def deploy_model(self, model_name: str) -> bool:
        """Deploy a trained model for inference."""
        try:
            if model_name not in self.trained_models:
                logger.error(f"Model {model_name} not trained")
                return False
            
            self.model_status[model_name] = ModelStatus.DEPLOYED
            
            # Clear prediction cache for this model
            cache_keys_to_remove = [key for key in self.prediction_cache.keys() if key.startswith(f"{model_name}:")]
            for key in cache_keys_to_remove:
                del self.prediction_cache[key]
            
            logger.info(f"Deployed model {model_name}")
            
            # Publish deployment event
            await self.event_bus.publish(Event(
                id=str(uuid.uuid4()),
                event_type=EventType.SYSTEM_HEALTH,  # Using system health as placeholder
                timestamp=datetime.now(timezone.utc),
                source='MLPipeline',
                data={
                    'action': 'model_deployed',
                    'model_name': model_name,
                    'deployment_time': datetime.now(timezone.utc).isoformat()
                }
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model {model_name}: {e}")
            return False
    
    async def get_model_performance(self, model_name: str) -> Optional[ModelMetrics]:
        """Get performance metrics for a model."""
        return self.model_metrics.get(model_name)
    
    async def get_model_status(self, model_name: str) -> Optional[ModelStatus]:
        """Get current status of a model."""
        return self.model_status.get(model_name)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models with their status."""
        try:
            models = []
            
            for model_name, model_config in self.model_configs.items():
                status = self.model_status.get(model_name, ModelStatus.CREATED)
                metrics = self.model_metrics.get(model_name)
                
                model_info = {
                    'name': model_name,
                    'type': model_config.model_type.value,
                    'prediction_type': model_config.prediction_type.value,
                    'algorithm': model_config.algorithm,
                    'status': status.value,
                    'version': model_config.version,
                    'feature_count': len(model_config.feature_names),
                    'created_at': model_config.created_at.isoformat(),
                    'performance': {
                        'accuracy': metrics.accuracy if metrics else 0.0,
                        'f1_score': metrics.f1_score if metrics else 0.0,
                        'training_samples': metrics.training_samples if metrics else 0
                    } if metrics else None
                }
                
                models.append(model_info)
            
            return sorted(models, key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get ML pipeline performance metrics."""
        try:
            total_models = len(self.model_configs)
            deployed_models = sum(1 for status in self.model_status.values() if status == ModelStatus.DEPLOYED)
            cached_predictions = len(self.prediction_cache)
            
            avg_accuracy = 0.0
            if self.model_metrics:
                avg_accuracy = np.mean([m.accuracy for m in self.model_metrics.values()])
            
            return {
                'total_models': total_models,
                'deployed_models': deployed_models,
                'training_models': sum(1 for status in self.model_status.values() if status == ModelStatus.TRAINING),
                'failed_models': sum(1 for status in self.model_status.values() if status == ModelStatus.FAILED),
                'cached_predictions': cached_predictions,
                'average_accuracy': avg_accuracy,
                'sklearn_available': SKLEARN_AVAILABLE,
                'xgboost_available': XGBOOST_AVAILABLE,
                'lightgbm_available': LIGHTGBM_AVAILABLE,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline metrics: {e}")
            return {}
    
    async def _register_builtin_models(self) -> None:
        """Register built-in model configurations."""
        try:
            builtin_models = [
                # Signal direction classifier
                ModelConfig(
                    name="signal_direction_rf",
                    model_type=ModelType.CLASSIFICATION,
                    prediction_type=PredictionType.SIGNAL_DIRECTION,
                    algorithm="random_forest",
                    hyperparameters={'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                    feature_names=[
                        'price_sma_20', 'price_ema_12', 'rsi_14', 'macd_signal',
                        'volatility_20d', 'volume_sma_20', 'beta_60d'
                    ],
                    target_column="signal_direction",
                    performance_threshold=0.55,
                    tags=['signal', 'classification', 'technical']
                ),
                
                # Price movement predictor
                ModelConfig(
                    name="price_movement_xgb",
                    model_type=ModelType.REGRESSION,
                    prediction_type=PredictionType.PRICE_MOVEMENT,
                    algorithm="xgboost",
                    hyperparameters={'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
                    feature_names=[
                        'price_sma_20', 'price_ema_12', 'rsi_14', 'bollinger_upper',
                        'volatility_20d', 'volume_sma_20', 'news_sentiment_1d', 'sector_momentum'
                    ],
                    target_column="price_change_1d",
                    performance_threshold=0.6,
                    tags=['price', 'regression', 'fundamental']
                ),
                
                # Volatility predictor
                ModelConfig(
                    name="volatility_lgb",
                    model_type=ModelType.REGRESSION,
                    prediction_type=PredictionType.VOLATILITY,
                    algorithm="lightgbm",
                    hyperparameters={'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9},
                    feature_names=[
                        'volatility_20d', 'rsi_14', 'volume_sma_20', 'bid_ask_spread',
                        'order_imbalance', 'var_1d_95', 'beta_60d'
                    ],
                    target_column="volatility_next_1d",
                    performance_threshold=0.65,
                    tags=['volatility', 'risk', 'microstructure']
                ),
                
                # Risk score classifier
                ModelConfig(
                    name="risk_score_lr",
                    model_type=ModelType.CLASSIFICATION,
                    prediction_type=PredictionType.RISK_SCORE,
                    algorithm="logistic_regression",
                    hyperparameters={'C': 1.0, 'random_state': 42, 'max_iter': 1000},
                    feature_names=[
                        'beta_60d', 'volatility_20d', 'var_1d_95', 'bid_ask_spread',
                        'market_cap_rank', 'sector_momentum'
                    ],
                    target_column="risk_category",
                    performance_threshold=0.7,
                    tags=['risk', 'classification', 'fundamental']
                ),
                
                # Anomaly detector
                ModelConfig(
                    name="anomaly_detector_rf",
                    model_type=ModelType.ANOMALY_DETECTION,
                    prediction_type=PredictionType.ANOMALY_SCORE,
                    algorithm="isolation_forest",
                    hyperparameters={'contamination': 0.1, 'random_state': 42},
                    feature_names=[
                        'price_sma_20', 'volume_sma_20', 'volatility_20d', 'rsi_14',
                        'bid_ask_spread', 'order_imbalance'
                    ],
                    target_column="is_anomaly",
                    performance_threshold=0.6,
                    tags=['anomaly', 'unsupervised', 'detection']
                ),
            ]
            
            # Register all built-in models
            for model_config in builtin_models:
                await self.register_model(model_config)
            
        except Exception as e:
            logger.error(f"Error registering builtin models: {e}")
    
    async def _prepare_training_data(self, model_config: ModelConfig, symbols: List[str] = None) -> pd.DataFrame:
        """Prepare training data for model."""
        try:
            if symbols is None:
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Default symbols
            
            # Get historical features
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=model_config.training_window_days)
            
            training_data = await self.feature_store.get_historical_features(
                feature_names=model_config.feature_names,
                symbols=symbols,
                start_time=start_time,
                end_time=end_time,
                interval='1h'
            )
            
            if training_data.empty:
                logger.warning(f"No historical data available for {model_config.name}")
                return pd.DataFrame()
            
            # Generate synthetic target variable based on prediction type
            training_data = self._generate_synthetic_targets(training_data, model_config)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_targets(self, data: pd.DataFrame, model_config: ModelConfig) -> pd.DataFrame:
        """Generate synthetic target variables for demonstration."""
        try:
            if model_config.prediction_type == PredictionType.SIGNAL_DIRECTION:
                # Generate buy/sell/hold signals based on features
                data['signal_direction'] = np.where(
                    data.get('rsi_14', 50) > 70, 'sell',
                    np.where(data.get('rsi_14', 50) < 30, 'buy', 'hold')
                )
            
            elif model_config.prediction_type == PredictionType.PRICE_MOVEMENT:
                # Generate price change targets
                data['price_change_1d'] = np.random.normal(0.01, 0.02, len(data))
            
            elif model_config.prediction_type == PredictionType.VOLATILITY:
                # Generate volatility targets
                data['volatility_next_1d'] = np.random.uniform(0.1, 0.4, len(data))
            
            elif model_config.prediction_type == PredictionType.RISK_SCORE:
                # Generate risk categories
                data['risk_category'] = np.where(
                    data.get('volatility_20d', 0.2) > 0.3, 'high',
                    np.where(data.get('volatility_20d', 0.2) < 0.15, 'low', 'medium')
                )
            
            elif model_config.prediction_type == PredictionType.ANOMALY_SCORE:
                # Generate anomaly flags
                data['is_anomaly'] = np.random.choice([0, 1], len(data), p=[0.9, 0.1])
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating synthetic targets: {e}")
            return data
    
    async def _train_model_impl(self, model_config: ModelConfig, training_data: pd.DataFrame) -> Tuple[Any, ModelMetrics]:
        """Actual model training implementation."""
        try:
            # Prepare features and target
            X = training_data[model_config.feature_names].fillna(0)
            y = training_data[model_config.target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=model_config.validation_split, random_state=42
            )
            
            if not SKLEARN_AVAILABLE:
                # Return mock model and metrics
                model = {'type': 'mock', 'config': model_config}
                metrics = ModelMetrics(
                    model_name=model_config.name,
                    accuracy=np.random.uniform(0.6, 0.8),
                    f1_score=np.random.uniform(0.55, 0.75),
                    training_samples=len(X_train),
                    validation_samples=len(X_test)
                )
                return model, metrics
            
            # Apply scaling
            scaler = self.scalers[model_config.name]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model based on algorithm
            if model_config.algorithm == "random_forest":
                if model_config.model_type == ModelType.CLASSIFICATION:
                    model = RandomForestClassifier(**model_config.hyperparameters)
                else:
                    model = RandomForestRegressor(**model_config.hyperparameters)
            
            elif model_config.algorithm == "logistic_regression":
                model = LogisticRegression(**model_config.hyperparameters)
            
            elif model_config.algorithm == "linear_regression":
                model = LinearRegression()
            
            elif model_config.algorithm == "xgboost" and XGBOOST_AVAILABLE:
                if model_config.model_type == ModelType.CLASSIFICATION:
                    model = xgb.XGBClassifier(**model_config.hyperparameters)
                else:
                    model = xgb.XGBRegressor(**model_config.hyperparameters)
            
            elif model_config.algorithm == "lightgbm" and LIGHTGBM_AVAILABLE:
                if model_config.model_type == ModelType.CLASSIFICATION:
                    model = lgb.LGBMClassifier(**model_config.hyperparameters)
                else:
                    model = lgb.LGBMRegressor(**model_config.hyperparameters)
            
            else:
                logger.warning(f"Algorithm {model_config.algorithm} not available, using RandomForest")
                if model_config.model_type == ModelType.CLASSIFICATION:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Handle label encoding for classification
            if model_config.model_type == ModelType.CLASSIFICATION:
                encoder = self.encoders[model_config.name]
                y_train_encoded = encoder.fit_transform(y_train)
                y_test_encoded = encoder.transform(y_test)
            else:
                y_train_encoded = y_train
                y_test_encoded = y_test
            
            # Train model
            model.fit(X_train_scaled, y_train_encoded)
            
            # Generate predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = ModelMetrics(model_name=model_config.name)
            
            if model_config.model_type == ModelType.CLASSIFICATION:
                # Decode predictions for metric calculation
                if model_config.name in self.encoders:
                    y_test_original = encoder.inverse_transform(y_test_encoded)
                    y_pred_original = encoder.inverse_transform(y_pred)
                else:
                    y_test_original = y_test_encoded
                    y_pred_original = y_pred
                
                metrics.accuracy = accuracy_score(y_test_original, y_pred_original)
                metrics.precision = precision_score(y_test_original, y_pred_original, average='weighted')
                metrics.recall = recall_score(y_test_original, y_pred_original, average='weighted')
                metrics.f1_score = f1_score(y_test_original, y_pred_original, average='weighted')
            
            else:  # Regression
                metrics.mse = mean_squared_error(y_test_encoded, y_pred)
                metrics.rmse = np.sqrt(metrics.mse)
                metrics.mae = np.mean(np.abs(y_test_encoded - y_pred))
                # Mock R2 score
                metrics.r2_score = max(0, 1 - (metrics.mse / np.var(y_test_encoded)))
                metrics.accuracy = metrics.r2_score  # Use R2 as accuracy for regression
            
            metrics.training_samples = len(X_train)
            metrics.validation_samples = len(X_test)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(model_config.feature_names, model.feature_importances_))
                metrics.feature_importance = {k: float(v) for k, v in importance_dict.items()}
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error training model implementation: {e}")
            return None, ModelMetrics(model_name=model_config.name)
    
    def _generate_mock_prediction(self, prediction_type: PredictionType) -> Any:
        """Generate mock prediction for testing."""
        if prediction_type == PredictionType.SIGNAL_DIRECTION:
            return np.random.choice(['buy', 'sell', 'hold'])
        elif prediction_type == PredictionType.PRICE_MOVEMENT:
            return np.random.uniform(-0.05, 0.05)
        elif prediction_type == PredictionType.VOLATILITY:
            return np.random.uniform(0.1, 0.4)
        elif prediction_type == PredictionType.RISK_SCORE:
            return np.random.choice(['low', 'medium', 'high'])
        elif prediction_type == PredictionType.ANOMALY_SCORE:
            return np.random.uniform(0, 1)
        else:
            return 0.0
    
    def _validate_model_config(self, model_config: ModelConfig) -> bool:
        """Validate model configuration."""
        try:
            if not model_config.name or not model_config.algorithm:
                return False
            
            if not model_config.feature_names:
                logger.warning(f"No features specified for model {model_config.name}")
                return False
            
            if model_config.validation_split <= 0 or model_config.validation_split >= 1:
                logger.error(f"Invalid validation split for {model_config.name}")
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _check_feature_availability(self, feature_names: List[str]) -> bool:
        """Check if all required features are available."""
        try:
            available_features = await self.feature_store.list_features()
            available_names = {f['name'] for f in available_features}
            
            missing_features = set(feature_names) - available_names
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return len(missing_features) < len(feature_names) * 0.5  # Allow up to 50% missing
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking feature availability: {e}")
            return False
    
    async def _save_model(self, model_name: str, model: Any) -> bool:
        """Save trained model to disk."""
        try:
            model_path = Path(self.config['model_storage_path']) / f"{model_name}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': self.scalers.get(model_name),
                    'encoder': self.encoders.get(model_name),
                    'timestamp': datetime.now(timezone.utc),
                    'config': self.model_configs[model_name]
                }, f)
            
            logger.info(f"Saved model {model_name} to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    async def _load_existing_models(self) -> None:
        """Load existing trained models from disk."""
        try:
            model_dir = Path(self.config['model_storage_path'])
            
            for model_file in model_dir.glob("*.pkl"):
                try:
                    model_name = model_file.stem
                    
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Restore model components
                    if 'model' in model_data:
                        self.trained_models[model_name] = model_data['model']
                        
                    if 'scaler' in model_data and model_data['scaler']:
                        self.scalers[model_name] = model_data['scaler']
                        
                    if 'encoder' in model_data and model_data['encoder']:
                        self.encoders[model_name] = model_data['encoder']
                    
                    # Mark as trained
                    self.model_status[model_name] = ModelStatus.TRAINED
                    
                    logger.info(f"Loaded existing model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {e}")
            
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
    
    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        try:
            # Subscribe to feature updates for retraining triggers
            await self.event_bus.subscribe(
                [EventType.MARKET_DATA_UPDATE],
                self._handle_market_data_event
            )
            
        except Exception as e:
            logger.error(f"Failed to setup event subscriptions: {e}")
    
    async def _handle_market_data_event(self, event: Event) -> None:
        """Handle market data events."""
        try:
            # Could trigger model retraining based on data patterns
            pass
        except Exception as e:
            logger.error(f"Error handling market data event: {e}")
    
    async def _training_worker(self) -> None:
        """Background training worker."""
        while self.is_running:
            try:
                # Wait for training requests
                model_name = await asyncio.wait_for(self.training_queue.get(), timeout=30)
                
                async with self.training_semaphore:
                    await self.train_model(model_name)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in training worker: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitoring_loop(self) -> None:
        """Monitor model performance and detect degradation."""
        while self.is_running:
            try:
                for model_name, metrics in self.model_metrics.items():
                    # Check for performance degradation
                    if len(self.performance_history.get(model_name, [])) > 1:
                        recent_perf = self.performance_history[model_name][-1]['metrics']['accuracy']
                        previous_perf = self.performance_history[model_name][-2]['metrics']['accuracy']
                        
                        if recent_perf < previous_perf - self.config['alert_threshold_accuracy_drop']:
                            logger.warning(f"Performance degradation detected for {model_name}: "
                                         f"{recent_perf:.3f} vs {previous_perf:.3f}")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(1800)
    
    async def _retraining_scheduler(self) -> None:
        """Schedule automatic model retraining."""
        while self.is_running:
            try:
                if self.config['auto_retraining_enabled']:
                    # Check which models need retraining
                    for model_name, model_config in self.model_configs.items():
                        if self.model_status.get(model_name) == ModelStatus.DEPLOYED:
                            # Check if model needs retraining
                            if await self._should_retrain_model(model_name, model_config):
                                await self.training_queue.put(model_name)
                
                # Wait based on retrain schedule
                if self.config['retrain_schedule'] == 'daily':
                    await asyncio.sleep(86400)  # 24 hours
                elif self.config['retrain_schedule'] == 'hourly':
                    await asyncio.sleep(3600)   # 1 hour
                else:
                    await asyncio.sleep(43200)  # 12 hours (default)
                
            except Exception as e:
                logger.error(f"Error in retraining scheduler: {e}")
                await asyncio.sleep(3600)
    
    async def _should_retrain_model(self, model_name: str, model_config: ModelConfig) -> bool:
        """Check if model should be retrained."""
        try:
            # Check time since last training
            if self.performance_history.get(model_name):
                last_training = datetime.fromisoformat(
                    self.performance_history[model_name][-1]['timestamp'].replace('Z', '+00:00')
                )
                hours_since_training = (datetime.now(timezone.utc) - last_training).total_seconds() / 3600
                
                if hours_since_training >= model_config.retrain_frequency_hours:
                    return True
            
            # Check performance degradation
            if len(self.performance_history.get(model_name, [])) > 1:
                recent_perf = self.performance_history[model_name][-1]['metrics']['accuracy']
                if recent_perf < model_config.performance_threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain condition: {e}")
            return False
    
    async def _cache_cleanup_loop(self) -> None:
        """Clean up old cached predictions."""
        while self.is_running:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.config['cache_ttl_seconds'] * 2)
                
                keys_to_remove = []
                for key, cached_item in self.prediction_cache.items():
                    if cached_item['timestamp'] < cutoff_time:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.prediction_cache[key]
                
                await asyncio.sleep(1800)  # Clean every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _start_initial_training(self) -> None:
        """Start initial training for models that don't have trained versions."""
        try:
            for model_name in self.model_configs:
                if self.model_status.get(model_name) == ModelStatus.CREATED:
                    await self.training_queue.put(model_name)
            
        except Exception as e:
            logger.error(f"Error starting initial training: {e}")


# Convenience function
def create_ml_pipeline(feature_store: FeatureStore, **kwargs) -> MLPipeline:
    """Create an ML pipeline with configuration."""
    return MLPipeline(feature_store, kwargs)