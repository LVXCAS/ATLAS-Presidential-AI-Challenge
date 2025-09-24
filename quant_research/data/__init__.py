"""Data management infrastructure for quantitative research."""

from .sources import (
    YahooDataSource,
    DataSourceManager,
    BaseDataSource,
    DataSourceError
)

# Optional imports
try:
    from .sources import AlpacaDataSource
except ImportError:
    AlpacaDataSource = None

__all__ = [
    'YahooDataSource',
    'DataSourceManager',
    'BaseDataSource',
    'DataSourceError'
]

if AlpacaDataSource:
    __all__.append('AlpacaDataSource')