"""
Secure configuration loader that integrates with the security system.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from config.security import SecurityManager, SecurityConfig, get_security_manager
from config.settings import Settings
import logging

logger = logging.getLogger(__name__)


class SecureSettings(BaseModel):
    """Secure settings that handle encrypted configuration."""
    
    environment: str = Field(default="development")
    security_config: SecurityConfig = Field(default_factory=SecurityConfig)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._security_manager = None
        self._initialize_secure_config()
    
    def _get_security_manager(self):
        """Get or create security manager instance."""
        if self._security_manager is None:
            self._security_manager = SecurityManager(self.security_config)
        return self._security_manager
    
    def _initialize_secure_config(self):
        """Initialize secure configuration."""
        try:
            # Load API keys from environment into secure storage
            self._get_security_manager().initialize_api_keys_from_env()
            logger.info("Secure configuration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize secure configuration: {e}")
            raise
    
    def get_broker_config(self, broker: str) -> Dict[str, Any]:
        """Get broker configuration securely."""
        credentials = self._get_security_manager().get_broker_credentials(broker)
        if not credentials:
            logger.warning(f"No credentials found for broker: {broker}")
            return {}
        
        return {
            "api_key": credentials.get("api_key"),
            "secret_key": credentials.get("secret_key"),
            "service": credentials.get("service")
        }
    
    def get_ai_config(self, service: str) -> Dict[str, Any]:
        """Get AI service configuration securely."""
        credentials = self._get_security_manager().get_ai_credentials(service)
        if not credentials:
            logger.warning(f"No credentials found for AI service: {service}")
            return {}
        
        return {
            "api_key": credentials.get("api_key"),
            "service": credentials.get("service")
        }
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user."""
        return self._get_security_manager().authenticate_user(username, password)
    
    def authorize_action(self, session_token: str, action: str) -> bool:
        """Authorize user action."""
        return self._get_security_manager().authorize_action(session_token, action)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self._get_security_manager().encrypt_sensitive_data(data)
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self._get_security_manager().decrypt_sensitive_data(encrypted_data)


class EnvironmentManager:
    """Manages environment-specific configurations."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.secure_settings = SecureSettings(environment=self.environment)
        self.base_settings = Settings()
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for current environment."""
        db_config = {
            "host": self.base_settings.database.host,
            "port": self.base_settings.database.port,
            "database": self.base_settings.database.database,
            "username": self.base_settings.database.username,
        }
        
        # Get encrypted password
        if self.base_settings.database.password:
            db_config["password"] = self.base_settings.database.password
        
        return db_config
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration for current environment."""
        redis_config = {
            "host": self.base_settings.redis.host,
            "port": self.base_settings.redis.port,
            "db": self.base_settings.redis.db,
        }
        
        if self.base_settings.redis.password:
            redis_config["password"] = self.base_settings.redis.password
        
        return redis_config
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration for current environment."""
        return {
            "paper_trading": self.base_settings.trading.paper_trading,
            "live_trading": self.base_settings.trading.live_trading,
            "initial_capital": self.base_settings.trading.initial_capital,
            "target_symbols": self.base_settings.trading.target_symbols,
        }
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return {
            "max_daily_loss_pct": self.base_settings.risk.max_daily_loss_pct,
            "max_position_size_pct": self.base_settings.risk.max_position_size_pct,
            "max_portfolio_leverage": self.base_settings.risk.max_portfolio_leverage,
            "var_confidence_level": self.base_settings.risk.var_confidence_level,
            "emergency_stop_loss_pct": self.base_settings.risk.emergency_stop_loss_pct,
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all configuration components."""
        validation_results = {}
        
        # Validate database configuration
        try:
            db_config = self.get_database_config()
            validation_results["database"] = all([
                db_config.get("host"),
                db_config.get("port"),
                db_config.get("database"),
                db_config.get("username")
            ])
        except Exception as e:
            logger.error(f"Database configuration validation failed: {e}")
            validation_results["database"] = False
        
        # Validate broker configurations
        brokers = ["alpaca", "polygon"]
        for broker in brokers:
            try:
                broker_config = self.secure_settings.get_broker_config(broker)
                validation_results[f"broker_{broker}"] = bool(broker_config.get("api_key"))
            except Exception as e:
                logger.error(f"Broker {broker} configuration validation failed: {e}")
                validation_results[f"broker_{broker}"] = False
        
        # Validate AI service configurations
        ai_services = ["openai", "google", "deepseek"]
        for service in ai_services:
            try:
                ai_config = self.secure_settings.get_ai_config(service)
                validation_results[f"ai_{service}"] = bool(ai_config.get("api_key"))
            except Exception as e:
                logger.error(f"AI service {service} configuration validation failed: {e}")
                validation_results[f"ai_{service}"] = False
        
        return validation_results


# Global environment manager instance (lazy initialization)
_env_manager = None

def get_env_manager() -> EnvironmentManager:
    """Get or create the global environment manager instance."""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager

# For backward compatibility
env_manager = None

def get_api_keys() -> Dict[str, str]:
    """
    Get API keys from secure storage for market data providers.
    
    Returns:
        Dict[str, str]: Dictionary of API keys
    """
    try:
        env_mgr = get_env_manager()
        
        # Get broker configurations
        alpaca_config = env_mgr.secure_settings.get_broker_config("alpaca")
        polygon_config = env_mgr.secure_settings.get_broker_config("polygon")
        
        # Get AI configurations
        openai_config = env_mgr.secure_settings.get_ai_config("openai")
        google_config = env_mgr.secure_settings.get_ai_config("google")
        deepseek_config = env_mgr.secure_settings.get_ai_config("deepseek")
        
        # Combine all API keys
        api_keys = {
            'ALPACA_API_KEY': alpaca_config.get('api_key', ''),
            'ALPACA_SECRET_KEY': alpaca_config.get('secret_key', ''),
            'ALPACA_BASE_URL': env_mgr.base_settings.brokers.alpaca_base_url,
            'POLYGON_API_KEY': polygon_config.get('api_key', ''),
            'OPENAI_API_KEY': openai_config.get('api_key', ''),
            'GOOGLE_API_KEY': google_config.get('api_key', ''),
            'GEMINI_API_KEY': google_config.get('api_key', ''),  # Alias for Google
            'DEEPSEEK_API_KEY': deepseek_config.get('api_key', ''),
        }
        
        # Fallback to environment variables if secure storage is empty
        for key in api_keys:
            if not api_keys[key]:
                api_keys[key] = os.getenv(key, '')
        
        return api_keys
        
    except Exception as e:
        logger.error(f"Failed to load API keys: {e}")
        # Fallback to environment variables
        return {
            'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY', ''),
            'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY', ''),
            'ALPACA_BASE_URL': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY', ''),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY', ''),
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', ''),
            'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY', ''),
        }