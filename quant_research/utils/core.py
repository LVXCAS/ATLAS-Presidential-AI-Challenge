"""Core utilities for the quantitative research platform."""

import logging
import structlog
from typing import Dict, Any, Optional
import os
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        structured: Use structured logging
    """
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *(
                [logging.FileHandler(log_file)]
                if log_file else []
            )
        ]
    )
    
    if structured:
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(colors=True) if not log_file
                else structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


def ensure_directory_exists(path: str) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles zero denominator.
    
    Args:
        numerator: Numerator
        denominator: Denominator  
        default: Default value if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__)
    # Go up from utils/core.py to project root
    return current_file.parent.parent.parent


def load_environment_config(env_file: str = ".env") -> Dict[str, str]:
    """Load environment configuration from file.
    
    Args:
        env_file: Environment file name
        
    Returns:
        Dictionary of environment variables
    """
    config = {}
    env_path = get_project_root() / env_file
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    config[key.strip()] = value.strip()
    
    return config


class SingletonMeta(type):
    """Singleton metaclass."""
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def memoize(func):
    """Simple memoization decorator.
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    wrapper.cache = cache
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to retry function on exception.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        import time
                        time.sleep(delay * (attempt + 1))
                    continue
            
            # All retries failed
            raise last_exception
        
        return wrapper
    return decorator


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format value as percentage.
    
    Args:
        value: Value to format (e.g., 0.05 for 5%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def validate_symbol(symbol: str) -> str:
    """Validate and clean a trading symbol.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        Cleaned symbol
        
    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    # Clean symbol
    cleaned = symbol.strip().upper()
    
    if not cleaned:
        raise ValueError("Symbol cannot be empty after cleaning")
    
    # Basic validation (alphanumeric and common separators)
    if not all(c.isalnum() or c in '-._^' for c in cleaned):
        raise ValueError(f"Invalid characters in symbol: {cleaned}")
    
    return cleaned


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result