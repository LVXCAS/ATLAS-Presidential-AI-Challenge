"""Unit tests for configuration management."""

import pytest
import tempfile
import os
from pathlib import Path

from quant_research.config import BaseConfig, get_config
from quant_research.config.environments import DevelopmentConfig, ProductionConfig


class TestBaseConfig:
    """Test base configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = BaseConfig()
        
        assert config.ENVIRONMENT == "development"
        assert config.DEBUG is True
        assert config.LOG_LEVEL == "INFO"
        assert config.MAX_WORKERS == 4
        assert config.BATCH_SIZE == 1000
    
    def test_environment_override(self):
        """Test environment variable override."""
        # Set environment variable
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["MAX_WORKERS"] = "8"
        
        try:
            config = BaseConfig()
            assert config.LOG_LEVEL == "DEBUG"
            assert config.MAX_WORKERS == 8
        finally:
            # Clean up
            del os.environ["LOG_LEVEL"] 
            del os.environ["MAX_WORKERS"]
    
    def test_directory_creation(self):
        """Test automatic directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BaseConfig()
            config.DATA_DIR = Path(temp_dir) / "test_data"
            config.MODELS_DIR = Path(temp_dir) / "test_models"
            
            # Trigger directory creation
            config.__post_init__()
            
            assert config.DATA_DIR.exists()
            assert config.MODELS_DIR.exists()


class TestEnvironmentConfigs:
    """Test environment-specific configurations."""
    
    def test_development_config(self):
        """Test development configuration."""
        config = DevelopmentConfig()
        
        assert config.ENVIRONMENT == "development"
        assert config.DEBUG is True
        assert config.LOG_LEVEL == "DEBUG"
        assert config.MAX_POSITION_SIZE == 0.05
    
    def test_production_config(self):
        """Test production configuration.""" 
        config = ProductionConfig()
        
        assert config.ENVIRONMENT == "production"
        assert config.DEBUG is False
        assert config.LOG_LEVEL == "INFO"
        assert config.MAX_POSITION_SIZE == 0.02
        assert config.EMAIL_ALERTS is True
    
    def test_get_config_default(self):
        """Test get_config with default environment."""
        # Ensure no environment variable is set
        env_var = os.environ.get("ENVIRONMENT")
        if env_var:
            del os.environ["ENVIRONMENT"]
        
        try:
            config = get_config()
            assert isinstance(config, DevelopmentConfig)
        finally:
            # Restore environment variable if it existed
            if env_var:
                os.environ["ENVIRONMENT"] = env_var
    
    def test_get_config_production(self):
        """Test get_config with production environment."""
        os.environ["ENVIRONMENT"] = "production"
        
        try:
            config = get_config()
            assert isinstance(config, ProductionConfig)
        finally:
            del os.environ["ENVIRONMENT"]


class TestDataConfig:
    """Test data configuration."""
    
    def test_default_symbols(self):
        """Test default symbol lists."""
        from quant_research.config.data import MarketDataConfig
        
        config = MarketDataConfig()
        
        assert "SPY" in config.default_symbols
        assert "AAPL" in config.default_symbols
        assert "BTC/USD" in config.crypto_symbols
    
    def test_data_source_config(self):
        """Test data source configuration."""
        from quant_research.config.data import MarketDataConfig
        
        config = MarketDataConfig()
        
        # Check Alpaca configuration
        alpaca_config = config.sources["alpaca"]
        assert alpaca_config.name == "alpaca"
        assert alpaca_config.rate_limit == 200
        assert alpaca_config.enabled is True


class TestModelConfigs:
    """Test model configurations."""
    
    def test_sklearn_config(self):
        """Test scikit-learn model configuration."""
        from quant_research.config.models import SklearnModelConfig
        
        config = SklearnModelConfig(name="test_rf")
        
        assert config.model_type == "sklearn"
        assert config.train_test_split == 0.8
        assert "n_estimators" in config.hyperparameters
    
    def test_pytorch_config(self):
        """Test PyTorch model configuration."""
        from quant_research.config.models import PyTorchModelConfig
        
        config = PyTorchModelConfig(name="test_nn")
        
        assert config.model_type == "pytorch"
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.device in ["cpu", "cuda"]
    
    def test_predefined_configs(self):
        """Test predefined model configurations."""
        from quant_research.config.models import MODEL_CONFIGS
        
        assert "price_prediction_rf" in MODEL_CONFIGS
        assert "momentum_xgb" in MODEL_CONFIGS
        assert "lstm_price" in MODEL_CONFIGS
        
        # Test specific config
        rf_config = MODEL_CONFIGS["price_prediction_rf"]
        assert rf_config.name == "price_prediction_rf"
        assert rf_config.model_type == "sklearn"


class TestBacktestConfig:
    """Test backtesting configuration."""
    
    def test_default_backtest_config(self):
        """Test default backtest configuration."""
        from quant_research.config.backtesting import BacktestConfig
        
        config = BacktestConfig()
        
        assert config.initial_capital == 100000.0
        assert config.commission == 0.001
        assert config.benchmark == "SPY"
        assert "sharpe_ratio" in config.metrics_to_calculate
    
    def test_strategy_config(self):
        """Test strategy-specific configuration."""
        from quant_research.config.backtesting import StrategyBacktestConfig
        
        config = StrategyBacktestConfig(
            strategy_name="test_momentum",
            strategy_type="momentum"
        )
        
        assert config.strategy_name == "test_momentum"
        assert config.position_sizing_method == "fixed_percent"
        assert config.rebalance_frequency == "monthly"
    
    def test_predefined_backtest_configs(self):
        """Test predefined backtest configurations."""
        from quant_research.config.backtesting import BACKTEST_CONFIGS
        
        assert "momentum_strategy" in BACKTEST_CONFIGS
        assert "mean_reversion" in BACKTEST_CONFIGS
        
        # Test specific config
        momentum_config = BACKTEST_CONFIGS["momentum_strategy"]
        assert momentum_config.strategy_type == "momentum"
        assert "lookback_period" in momentum_config.parameters


if __name__ == "__main__":
    pytest.main([__file__])