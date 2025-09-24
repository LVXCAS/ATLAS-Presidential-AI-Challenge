"""
Configuration Management System

Enterprise-grade configuration management with environment-specific settings,
dynamic updates, validation, encryption, and centralized configuration for
distributed trading systems.
"""

import os
import json
import yaml
import toml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
import logging
from datetime import datetime, timezone
import threading
import time
from abc import ABC, abstractmethod
import hashlib
import redis
import sqlite3
from cryptography.fernet import Fernet
import secrets
from pydantic import BaseModel, ValidationError, validator
from pydantic.env_settings import BaseSettings
import warnings
warnings.filterwarnings('ignore')

# Configuration watching
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    username: str = "trader"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    ssl_mode: str = "prefer"
    timeout: int = 30

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl: bool = False
    pool_size: int = 10
    timeout: int = 5

@dataclass
class BrokerConfig:
    """Broker configuration"""
    name: str
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    sandbox: bool = True
    rate_limit: int = 100
    timeout: int = 30
    retry_attempts: int = 3
    enabled: bool = True

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = ""
    jwt_expiration_hours: int = 24
    password_min_length: int = 12
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    encryption_key: str = ""
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class TradingConfig:
    """Trading system configuration"""
    initial_capital: float = 1000000.0
    max_leverage: float = 3.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02    # 2% daily loss limit
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    risk_free_rate: float = 0.02
    enable_paper_trading: bool = True

@dataclass
class AlertConfig:
    """Alerting configuration"""
    enable_email: bool = False
    enable_slack: bool = False
    enable_sms: bool = False
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    slack_webhook_url: str = ""
    alert_frequency_seconds: int = 300

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    file_path: str = "logs/trading_system.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_remote: bool = False
    remote_endpoint: str = ""

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_grafana: bool = True
    grafana_port: int = 3000
    metrics_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    enable_profiling: bool = False

@dataclass
class SystemConfig:
    """Complete system configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "1.0.0"
    timezone: str = "UTC"

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Broker configurations
    brokers: Dict[str, BrokerConfig] = field(default_factory=dict)

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

class ConfigurationValidator:
    """Validates configuration settings"""

    @staticmethod
    def validate_database_config(config: DatabaseConfig) -> List[str]:
        """Validate database configuration"""
        errors = []

        if not config.host:
            errors.append("Database host is required")

        if config.port <= 0 or config.port > 65535:
            errors.append("Database port must be between 1 and 65535")

        if not config.database:
            errors.append("Database name is required")

        if not config.username:
            errors.append("Database username is required")

        if config.pool_size <= 0:
            errors.append("Database pool size must be positive")

        return errors

    @staticmethod
    def validate_trading_config(config: TradingConfig) -> List[str]:
        """Validate trading configuration"""
        errors = []

        if config.initial_capital <= 0:
            errors.append("Initial capital must be positive")

        if config.max_leverage <= 0:
            errors.append("Max leverage must be positive")

        if config.max_position_size <= 0 or config.max_position_size > 1:
            errors.append("Max position size must be between 0 and 1")

        if config.max_daily_loss <= 0 or config.max_daily_loss > 1:
            errors.append("Max daily loss must be between 0 and 1")

        if config.commission_rate < 0:
            errors.append("Commission rate cannot be negative")

        return errors

    @staticmethod
    def validate_security_config(config: SecurityConfig) -> List[str]:
        """Validate security configuration"""
        errors = []

        if not config.jwt_secret_key:
            errors.append("JWT secret key is required")

        if config.jwt_expiration_hours <= 0:
            errors.append("JWT expiration must be positive")

        if config.password_min_length < 8:
            errors.append("Minimum password length should be at least 8")

        if config.max_login_attempts <= 0:
            errors.append("Max login attempts must be positive")

        return errors

    @staticmethod
    def validate_system_config(config: SystemConfig) -> List[str]:
        """Validate complete system configuration"""
        errors = []

        # Validate sub-configurations
        errors.extend(ConfigurationValidator.validate_database_config(config.database))
        errors.extend(ConfigurationValidator.validate_trading_config(config.trading))
        errors.extend(ConfigurationValidator.validate_security_config(config.security))

        # Validate broker configurations
        for broker_name, broker_config in config.brokers.items():
            if not broker_config.name:
                errors.append(f"Broker {broker_name} must have a name")

        return errors

class ConfigurationLoader:
    """Loads configuration from various sources"""

    @staticmethod
    def load_from_file(file_path: str, format_type: ConfigFormat = None) -> Dict[str, Any]:
        """Load configuration from file"""

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        # Determine format from extension if not specified
        if format_type is None:
            ext = path.suffix.lower()
            if ext == '.json':
                format_type = ConfigFormat.JSON
            elif ext in ['.yml', '.yaml']:
                format_type = ConfigFormat.YAML
            elif ext == '.toml':
                format_type = ConfigFormat.TOML
            else:
                raise ValueError(f"Cannot determine format for file: {file_path}")

        # Load based on format
        with open(file_path, 'r', encoding='utf-8') as f:
            if format_type == ConfigFormat.JSON:
                return json.load(f)
            elif format_type == ConfigFormat.YAML:
                return yaml.safe_load(f)
            elif format_type == ConfigFormat.TOML:
                return toml.load(f)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

    @staticmethod
    def load_from_env(prefix: str = "TRADING_") -> Dict[str, Any]:
        """Load configuration from environment variables"""

        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()

                # Try to parse as JSON first, then as string
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config[config_key] = value

        return config

    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from dictionary"""
        return config_dict.copy()

class ConfigurationEncryption:
    """Handles encryption/decryption of sensitive configuration values"""

    def __init__(self, encryption_key: str = None):
        if encryption_key:
            self.key = encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
        else:
            self.key = Fernet.generate_key()

        self.fernet = Fernet(self.key)

    def encrypt_value(self, value: str) -> str:
        """Encrypt configuration value"""
        return self.fernet.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt configuration value"""
        return self.fernet.decrypt(encrypted_value.encode()).decode()

    def encrypt_config_section(self, config: Dict[str, Any],
                             sensitive_keys: List[str]) -> Dict[str, Any]:
        """Encrypt sensitive keys in configuration section"""

        encrypted_config = config.copy()

        for key in sensitive_keys:
            if key in encrypted_config and encrypted_config[key]:
                encrypted_config[key] = self.encrypt_value(str(encrypted_config[key]))

        return encrypted_config

    def decrypt_config_section(self, config: Dict[str, Any],
                             sensitive_keys: List[str]) -> Dict[str, Any]:
        """Decrypt sensitive keys in configuration section"""

        decrypted_config = config.copy()

        for key in sensitive_keys:
            if key in decrypted_config and decrypted_config[key]:
                try:
                    decrypted_config[key] = self.decrypt_value(decrypted_config[key])
                except Exception as e:
                    logging.warning(f"Failed to decrypt key {key}: {e}")

        return decrypted_config

class ConfigurationWatcher(FileSystemEventHandler):
    """Watches configuration files for changes"""

    def __init__(self, config_manager: 'ConfigurationManager'):
        self.config_manager = config_manager
        self.last_modified = {}

    def on_modified(self, event):
        """Handle file modification events"""

        if event.is_directory:
            return

        file_path = event.src_path

        # Avoid duplicate events
        current_time = time.time()
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < 1:  # 1 second debounce
                return

        self.last_modified[file_path] = current_time

        # Check if this is a configuration file we're watching
        if file_path in self.config_manager.watched_files:
            logging.info(f"Configuration file changed: {file_path}")

            try:
                # Reload configuration
                self.config_manager.reload_from_file(file_path)

                # Notify listeners
                self.config_manager._notify_change_listeners()

            except Exception as e:
                logging.error(f"Error reloading configuration from {file_path}: {e}")

class ConfigurationManager:
    """Central configuration management system"""

    def __init__(self,
                 encryption_key: str = None,
                 redis_client: redis.Redis = None,
                 enable_watching: bool = True):

        self.config = SystemConfig()
        self.encryption = ConfigurationEncryption(encryption_key)
        self.redis_client = redis_client
        self.enable_watching = enable_watching

        # Configuration sources and watchers
        self.config_sources: List[Dict[str, Any]] = []
        self.watched_files: Set[str] = set()
        self.file_observer = None
        self.watcher = None

        # Change listeners
        self.change_listeners: List[Callable] = []

        # Thread safety
        self.config_lock = threading.RLock()

        # Sensitive configuration keys that should be encrypted
        self.sensitive_keys = {
            'database': ['password'],
            'redis': ['password'],
            'security': ['jwt_secret_key', 'encryption_key'],
            'brokers': ['api_key', 'api_secret', 'api_passphrase'],
            'alerts': ['email_password', 'slack_webhook_url']
        }

    def load_from_file(self, file_path: str,
                      environment: Environment = None,
                      format_type: ConfigFormat = None,
                      watch: bool = True) -> 'ConfigurationManager':
        """Load configuration from file"""

        try:
            # Load raw configuration
            raw_config = ConfigurationLoader.load_from_file(file_path, format_type)

            # Apply environment-specific overrides
            if environment:
                env_key = f"environments.{environment.value}"
                if env_key in raw_config:
                    env_config = raw_config[env_key]
                    raw_config.update(env_config)

            # Merge with current configuration
            self._merge_configuration(raw_config)

            # Decrypt sensitive values
            self._decrypt_sensitive_values()

            # Add to watched files
            if watch and self.enable_watching:
                self.watched_files.add(str(Path(file_path).absolute()))
                self._start_file_watching()

            # Store source information
            self.config_sources.append({
                'type': 'file',
                'path': file_path,
                'format': format_type,
                'loaded_at': datetime.now(timezone.utc),
                'checksum': self._calculate_file_checksum(file_path)
            })

            logging.info(f"Configuration loaded from file: {file_path}")

        except Exception as e:
            logging.error(f"Failed to load configuration from {file_path}: {e}")
            raise

        return self

    def load_from_environment(self, prefix: str = "TRADING_") -> 'ConfigurationManager':
        """Load configuration from environment variables"""

        try:
            env_config = ConfigurationLoader.load_from_env(prefix)
            self._merge_configuration(env_config)

            self.config_sources.append({
                'type': 'environment',
                'prefix': prefix,
                'loaded_at': datetime.now(timezone.utc)
            })

            logging.info(f"Configuration loaded from environment variables with prefix: {prefix}")

        except Exception as e:
            logging.error(f"Failed to load configuration from environment: {e}")
            raise

        return self

    def load_from_redis(self, key: str = "trading_system:config") -> 'ConfigurationManager':
        """Load configuration from Redis"""

        if not self.redis_client:
            raise ValueError("Redis client not configured")

        try:
            config_data = self.redis_client.get(key)
            if config_data:
                redis_config = json.loads(config_data)
                self._merge_configuration(redis_config)

                self.config_sources.append({
                    'type': 'redis',
                    'key': key,
                    'loaded_at': datetime.now(timezone.utc)
                })

                logging.info(f"Configuration loaded from Redis key: {key}")

        except Exception as e:
            logging.error(f"Failed to load configuration from Redis: {e}")
            raise

        return self

    def save_to_file(self, file_path: str,
                    format_type: ConfigFormat = ConfigFormat.YAML,
                    encrypt_sensitive: bool = True) -> 'ConfigurationManager':
        """Save configuration to file"""

        try:
            # Get configuration as dictionary
            config_dict = self._config_to_dict()

            # Encrypt sensitive values if requested
            if encrypt_sensitive:
                config_dict = self._encrypt_sensitive_values(config_dict)

            # Save based on format
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type == ConfigFormat.JSON:
                    json.dump(config_dict, f, indent=2, default=str)
                elif format_type == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                elif format_type == ConfigFormat.TOML:
                    toml.dump(config_dict, f)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")

            logging.info(f"Configuration saved to file: {file_path}")

        except Exception as e:
            logging.error(f"Failed to save configuration to {file_path}: {e}")
            raise

        return self

    def save_to_redis(self, key: str = "trading_system:config",
                     ttl: int = None) -> 'ConfigurationManager':
        """Save configuration to Redis"""

        if not self.redis_client:
            raise ValueError("Redis client not configured")

        try:
            config_dict = self._config_to_dict()
            config_json = json.dumps(config_dict, default=str)

            if ttl:
                self.redis_client.setex(key, ttl, config_json)
            else:
                self.redis_client.set(key, config_json)

            logging.info(f"Configuration saved to Redis key: {key}")

        except Exception as e:
            logging.error(f"Failed to save configuration to Redis: {e}")
            raise

        return self

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path"""

        with self.config_lock:
            try:
                keys = key_path.split('.')
                value = self.config

                for key in keys:
                    if hasattr(value, key):
                        value = getattr(value, key)
                    elif isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return default

                return value

            except Exception as e:
                logging.error(f"Error getting configuration value for {key_path}: {e}")
                return default

    def set(self, key_path: str, value: Any) -> 'ConfigurationManager':
        """Set configuration value by dot-separated path"""

        with self.config_lock:
            try:
                keys = key_path.split('.')
                config_obj = self.config

                # Navigate to the parent object
                for key in keys[:-1]:
                    if hasattr(config_obj, key):
                        config_obj = getattr(config_obj, key)
                    elif isinstance(config_obj, dict):
                        if key not in config_obj:
                            config_obj[key] = {}
                        config_obj = config_obj[key]
                    else:
                        raise ValueError(f"Cannot navigate to path: {key_path}")

                # Set the final value
                final_key = keys[-1]
                if hasattr(config_obj, final_key):
                    setattr(config_obj, final_key, value)
                elif isinstance(config_obj, dict):
                    config_obj[final_key] = value
                else:
                    raise ValueError(f"Cannot set value at path: {key_path}")

                # Notify listeners
                self._notify_change_listeners()

                logging.info(f"Configuration value set: {key_path} = {value}")

            except Exception as e:
                logging.error(f"Error setting configuration value for {key_path}: {e}")
                raise

        return self

    def validate(self) -> List[str]:
        """Validate current configuration"""

        with self.config_lock:
            return ConfigurationValidator.validate_system_config(self.config)

    def add_change_listener(self, listener: Callable) -> 'ConfigurationManager':
        """Add configuration change listener"""

        self.change_listeners.append(listener)
        return self

    def remove_change_listener(self, listener: Callable) -> 'ConfigurationManager':
        """Remove configuration change listener"""

        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
        return self

    def reload_from_file(self, file_path: str):
        """Reload configuration from specific file"""

        try:
            # Find the source information
            source_info = None
            for source in self.config_sources:
                if source.get('type') == 'file' and source.get('path') == file_path:
                    source_info = source
                    break

            if not source_info:
                raise ValueError(f"File {file_path} is not a known configuration source")

            # Check if file has actually changed
            current_checksum = self._calculate_file_checksum(file_path)
            if current_checksum == source_info.get('checksum'):
                return  # No changes

            # Reload configuration
            raw_config = ConfigurationLoader.load_from_file(
                file_path,
                source_info.get('format')
            )

            self._merge_configuration(raw_config)
            self._decrypt_sensitive_values()

            # Update source information
            source_info['loaded_at'] = datetime.now(timezone.utc)
            source_info['checksum'] = current_checksum

            logging.info(f"Configuration reloaded from file: {file_path}")

        except Exception as e:
            logging.error(f"Failed to reload configuration from {file_path}: {e}")
            raise

    def _merge_configuration(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing configuration"""

        with self.config_lock:
            # Convert dictionary to SystemConfig object
            self._update_config_from_dict(self.config, new_config)

    def _update_config_from_dict(self, config_obj: Any, config_dict: Dict[str, Any]):
        """Update configuration object from dictionary"""

        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                current_value = getattr(config_obj, key)

                if isinstance(current_value, dict) and isinstance(value, dict):
                    # Merge dictionaries
                    current_value.update(value)
                elif hasattr(current_value, '__dict__') and isinstance(value, dict):
                    # Recursively update dataclass objects
                    self._update_config_from_dict(current_value, value)
                else:
                    # Direct assignment
                    setattr(config_obj, key, value)
            elif hasattr(config_obj, 'custom') and isinstance(config_obj.custom, dict):
                # Add to custom configuration
                config_obj.custom[key] = value

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""

        with self.config_lock:
            return asdict(self.config)

    def _encrypt_sensitive_values(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive values in configuration dictionary"""

        encrypted_config = config_dict.copy()

        for section, sensitive_keys in self.sensitive_keys.items():
            if section in encrypted_config:
                if section == 'brokers':
                    # Handle broker configurations specially
                    for broker_name, broker_config in encrypted_config[section].items():
                        if isinstance(broker_config, dict):
                            encrypted_config[section][broker_name] = \
                                self.encryption.encrypt_config_section(broker_config, sensitive_keys)
                else:
                    # Handle regular sections
                    if isinstance(encrypted_config[section], dict):
                        encrypted_config[section] = \
                            self.encryption.encrypt_config_section(encrypted_config[section], sensitive_keys)

        return encrypted_config

    def _decrypt_sensitive_values(self):
        """Decrypt sensitive values in current configuration"""

        with self.config_lock:
            # Decrypt database passwords
            if hasattr(self.config.database, 'password') and self.config.database.password:
                try:
                    self.config.database.password = self.encryption.decrypt_value(self.config.database.password)
                except:
                    pass  # Value might not be encrypted

            # Decrypt Redis passwords
            if hasattr(self.config.redis, 'password') and self.config.redis.password:
                try:
                    self.config.redis.password = self.encryption.decrypt_value(self.config.redis.password)
                except:
                    pass

            # Decrypt broker credentials
            for broker_name, broker_config in self.config.brokers.items():
                for key in ['api_key', 'api_secret', 'api_passphrase']:
                    if hasattr(broker_config, key):
                        value = getattr(broker_config, key)
                        if value:
                            try:
                                setattr(broker_config, key, self.encryption.decrypt_value(value))
                            except:
                                pass

    def _start_file_watching(self):
        """Start watching configuration files for changes"""

        if not self.enable_watching or self.file_observer:
            return

        try:
            self.watcher = ConfigurationWatcher(self)
            self.file_observer = Observer()

            # Watch directories containing configuration files
            watched_dirs = set()
            for file_path in self.watched_files:
                dir_path = str(Path(file_path).parent)
                if dir_path not in watched_dirs:
                    self.file_observer.schedule(self.watcher, dir_path, recursive=False)
                    watched_dirs.add(dir_path)

            self.file_observer.start()
            logging.info("Configuration file watching started")

        except Exception as e:
            logging.error(f"Failed to start configuration file watching: {e}")

    def _stop_file_watching(self):
        """Stop watching configuration files"""

        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.file_observer = None
            logging.info("Configuration file watching stopped")

    def _notify_change_listeners(self):
        """Notify all change listeners"""

        for listener in self.change_listeners:
            try:
                listener(self.config)
            except Exception as e:
                logging.error(f"Error notifying configuration change listener: {e}")

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate file checksum for change detection"""

        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self._stop_file_watching()

# Example usage and setup
def create_default_configuration() -> SystemConfig:
    """Create default configuration with sensible defaults"""

    config = SystemConfig()

    # Setup default brokers
    config.brokers['alpaca'] = BrokerConfig(
        name='alpaca',
        sandbox=True,
        rate_limit=200,
        enabled=True
    )

    config.brokers['interactive_brokers'] = BrokerConfig(
        name='interactive_brokers',
        sandbox=True,
        rate_limit=50,
        enabled=False
    )

    # Setup security defaults
    config.security.jwt_secret_key = secrets.token_urlsafe(32)
    config.security.encryption_key = Fernet.generate_key().decode()

    return config

def setup_configuration_management(config_dir: str = "config") -> ConfigurationManager:
    """Setup configuration management system"""

    # Create configuration directory
    Path(config_dir).mkdir(exist_ok=True)

    # Initialize configuration manager
    config_manager = ConfigurationManager(enable_watching=True)

    # Load configuration from multiple sources in order of priority
    try:
        # 1. Load base configuration
        base_config_path = Path(config_dir) / "base.yaml"
        if base_config_path.exists():
            config_manager.load_from_file(str(base_config_path))
        else:
            # Create default configuration
            default_config = create_default_configuration()
            config_manager.config = default_config
            config_manager.save_to_file(str(base_config_path))

        # 2. Load environment-specific configuration
        env = Environment(os.getenv('TRADING_ENV', 'development'))
        env_config_path = Path(config_dir) / f"{env.value}.yaml"
        if env_config_path.exists():
            config_manager.load_from_file(str(env_config_path))

        # 3. Load from environment variables (highest priority)
        config_manager.load_from_environment()

        # Validate configuration
        errors = config_manager.validate()
        if errors:
            logging.warning(f"Configuration validation errors: {errors}")

        logging.info(f"Configuration management setup complete for environment: {env.value}")

    except Exception as e:
        logging.error(f"Failed to setup configuration management: {e}")
        raise

    return config_manager

if __name__ == "__main__":
    # Example usage
    config_manager = setup_configuration_management()

    # Add change listener
    def on_config_change(config: SystemConfig):
        print(f"Configuration changed - Environment: {config.environment.value}")

    config_manager.add_change_listener(on_config_change)

    # Access configuration values
    print(f"Database host: {config_manager.get('database.host')}")
    print(f"Trading capital: {config_manager.get('trading.initial_capital')}")

    # Update configuration
    config_manager.set('trading.max_leverage', 2.5)

    # Validate configuration
    validation_errors = config_manager.validate()
    if validation_errors:
        print(f"Validation errors: {validation_errors}")
    else:
        print("Configuration is valid")

    # Keep running to test file watching
    try:
        print("Configuration system running... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down configuration system")
        config_manager._stop_file_watching()