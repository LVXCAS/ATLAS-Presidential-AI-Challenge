"""Machine learning model configuration."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class ModelConfig(BaseModel):
    """Base model configuration."""
    
    name: str
    model_type: str  # "sklearn", "pytorch", "xgboost", "reinforcement"
    version: str = "1.0.0"
    description: Optional[str] = None
    
    # Training settings
    train_test_split: float = 0.8
    validation_split: float = 0.2
    random_state: int = 42
    
    # Model-specific parameters
    hyperparameters: Dict[str, Any] = {}
    
    # Feature engineering
    features: List[str] = []
    target: str = "returns"
    
    # Data preprocessing
    normalize_features: bool = True
    handle_missing: str = "drop"  # "drop", "fill", "interpolate"
    
    class Config:
        extra = "allow"


class SklearnModelConfig(ModelConfig):
    """Scikit-learn model configuration."""
    
    model_type: str = "sklearn"
    
    # Common sklearn parameters
    hyperparameters: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }


class XGBoostModelConfig(ModelConfig):
    """XGBoost model configuration."""
    
    model_type: str = "xgboost"
    
    hyperparameters: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }


class PyTorchModelConfig(ModelConfig):
    """PyTorch model configuration."""
    
    model_type: str = "pytorch"
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Model architecture
    hidden_layers: List[int] = [64, 32, 16]
    dropout_rate: float = 0.2
    activation: str = "relu"
    
    # Training settings
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    early_stopping: bool = True
    patience: int = 10
    
    # Optimization
    optimizer: str = "adam"
    scheduler: Optional[str] = "reduce_lr_on_plateau"
    
    hyperparameters: Dict[str, Any] = {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.0001
    }


class ReinforcementLearningConfig(ModelConfig):
    """Reinforcement learning configuration."""
    
    model_type: str = "reinforcement"
    
    # RL-specific parameters
    algorithm: str = "PPO"  # PPO, A2C, DQN, etc.
    policy: str = "MlpPolicy"
    
    # Training parameters
    total_timesteps: int = 100000
    learning_rate: Union[float, str] = 0.0003
    
    # Environment settings
    observation_space_size: int = 20
    action_space_size: int = 3  # buy, sell, hold
    
    # Reward function
    reward_function: str = "sharpe_ratio"
    transaction_cost: float = 0.001
    
    hyperparameters: Dict[str, Any] = {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5
    }


class MLPipelineConfig(BaseModel):
    """Machine learning pipeline configuration."""
    
    # Data pipeline
    data_lookback_days: int = 252  # 1 year
    feature_engineering: Dict[str, Any] = {
        "technical_indicators": True,
        "fundamental_ratios": True,
        "sentiment_scores": False,
        "macro_economic": True
    }
    
    # Model ensemble
    ensemble_models: List[str] = [
        "random_forest",
        "xgboost", 
        "neural_network"
    ]
    ensemble_method: str = "voting"  # voting, stacking
    
    # Cross-validation
    cv_folds: int = 5
    cv_method: str = "time_series"  # time_series, k_fold
    
    # Model evaluation
    metrics: List[str] = [
        "accuracy", "precision", "recall", "f1",
        "sharpe_ratio", "sortino_ratio", "max_drawdown"
    ]
    
    # Model persistence
    save_models: bool = True
    model_registry_path: str = "models/registry"
    
    # Monitoring
    model_drift_threshold: float = 0.1
    retrain_frequency: str = "monthly"  # daily, weekly, monthly
    
    # Feature selection
    feature_selection: bool = True
    max_features: Optional[int] = 50
    feature_importance_threshold: float = 0.001


# Predefined model configurations
MODEL_CONFIGS = {
    "price_prediction_rf": SklearnModelConfig(
        name="price_prediction_rf",
        description="Random Forest for price prediction",
        hyperparameters={
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        },
        features=["sma_20", "rsi", "macd", "volume", "volatility"]
    ),
    
    "momentum_xgb": XGBoostModelConfig(
        name="momentum_xgb", 
        description="XGBoost for momentum strategy",
        hyperparameters={
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05
        },
        features=["returns_1d", "returns_5d", "rsi", "momentum_12_1"]
    ),
    
    "lstm_price": PyTorchModelConfig(
        name="lstm_price",
        description="LSTM for price forecasting",
        hidden_layers=[128, 64],
        epochs=200,
        features=["price", "volume", "returns"]
    ),
    
    "trading_agent_ppo": ReinforcementLearningConfig(
        name="trading_agent_ppo",
        description="PPO agent for trading",
        algorithm="PPO",
        total_timesteps=1000000,
        observation_space_size=30
    )
}