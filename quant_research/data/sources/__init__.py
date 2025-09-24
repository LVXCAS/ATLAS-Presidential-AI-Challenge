"""Data source implementations."""

from .base import BaseDataSource, DataSourceError
from .yahoo import YahooDataSource
from .manager import DataSourceManager

# Optional imports
try:
    from .alpaca import AlpacaDataSource
except ImportError:
    AlpacaDataSource = None

__all__ = [
    'BaseDataSource',
    'DataSourceError',
    'YahooDataSource',
    'DataSourceManager'
]

if AlpacaDataSource:
    __all__.append('AlpacaDataSource')