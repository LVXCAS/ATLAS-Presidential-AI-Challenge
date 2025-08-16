"""
Logging configuration for the LangGraph Trading System.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from rich.logging import RichHandler


def setup_logging(
    level: int = logging.INFO, log_file: Optional[str] = None, json_logs: bool = False
) -> None:
    """
    Setup structured logging for the trading system.

    Args:
        level: Logging level
        log_file: Optional log file path
        json_logs: Whether to use JSON formatting
    """
    # Configure structlog
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
            structlog.processors.JSONRenderer()
            if json_logs
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Setup standard library logging
    handlers = []

    # Console handler with Rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True, markup=True, show_time=True, show_path=True
    )
    console_handler.setLevel(level)
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        if json_logs:
            formatter = logging.Formatter("%(message)s")
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers, format="%(message)s")

    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("kafka").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)
