"""Base data source interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
from datetime import datetime, date
from dataclasses import dataclass
import asyncio
import aiohttp
from structlog import get_logger

logger = get_logger(__name__)


class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass


@dataclass
class DataRequest:
    """Data request specification."""
    
    symbols: List[str]
    start_date: Union[str, date, datetime]
    end_date: Union[str, date, datetime]
    frequency: str = "1D"  # 1min, 5min, 1H, 1D
    fields: Optional[List[str]] = None
    limit: Optional[int] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if isinstance(self.start_date, str):
            self.start_date = pd.to_datetime(self.start_date).date()
        if isinstance(self.end_date, str):
            self.end_date = pd.to_datetime(self.end_date).date()


@dataclass 
class DataResponse:
    """Data response container."""
    
    data: Union[pd.DataFrame, Any]  # Any to handle polars if available
    metadata: Dict[str, Any]
    source: str
    request: DataRequest
    timestamp: datetime
    
    @property
    def is_empty(self) -> bool:
        """Check if response contains data."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.empty
        elif POLARS_AVAILABLE and pl and hasattr(self.data, 'is_empty'):
            return self.data.is_empty()
        return len(self.data) == 0 if hasattr(self.data, '__len__') else True
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if POLARS_AVAILABLE and pl and hasattr(self.data, 'to_pandas'):
            return self.data.to_pandas()
        return self.data
    
    def to_polars(self):
        """Convert to polars DataFrame.""" 
        if not POLARS_AVAILABLE or not pl:
            raise ImportError("Polars not available. Install with: pip install polars")
        
        if isinstance(self.data, pd.DataFrame):
            return pl.from_pandas(self.data)
        return self.data


class BaseDataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        rate_limit: Optional[int] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """Initialize data source.
        
        Args:
            name: Data source name
            api_key: API key if required
            rate_limit: Requests per minute limit
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            **kwargs: Additional configuration
        """
        self.name = name
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.config = kwargs
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter: Optional[asyncio.Semaphore] = None
        
        if rate_limit:
            self._rate_limiter = asyncio.Semaphore(rate_limit)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Initialize connection."""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        logger.info(f"Connected to {self.name} data source")
    
    async def disconnect(self):
        """Clean up connection."""
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info(f"Disconnected from {self.name} data source")
    
    @abstractmethod
    async def get_bars(self, request: DataRequest) -> DataResponse:
        """Get price bars (OHLCV data).
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse containing price bars
        """
        pass
    
    @abstractmethod
    async def get_trades(self, request: DataRequest) -> DataResponse:
        """Get trade data.
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse containing trade data
        """
        pass
    
    @abstractmethod
    async def get_quotes(self, request: DataRequest) -> DataResponse:
        """Get quote data (bid/ask).
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse containing quote data
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols.
        
        Returns:
            List of available symbols
        """
        pass
    
    async def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and retries.
        
        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response data as dictionary
            
        Raises:
            DataSourceError: If request fails
        """
        if not self._session:
            await self.connect()
        
        # Apply rate limiting
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.get(
                    url, params=params, headers=headers
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    logger.debug(
                        f"Request successful: {url}",
                        status=response.status,
                        attempt=attempt + 1
                    )
                    
                    return data
                    
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Request failed after {self.max_retries + 1} attempts",
                        url=url,
                        error=str(e)
                    )
                    raise DataSourceError(f"Request failed: {e}")
                
                logger.warning(
                    f"Request attempt {attempt + 1} failed, retrying",
                    url=url,
                    error=str(e)
                )
                
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # Release rate limiter semaphore
        if self._rate_limiter:
            self._rate_limiter.release()
    
    def _validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate and clean symbol list.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            List of valid symbols
        """
        valid_symbols = []
        
        for symbol in symbols:
            # Clean symbol (remove whitespace, convert to uppercase)
            clean_symbol = symbol.strip().upper()
            
            if clean_symbol:
                valid_symbols.append(clean_symbol)
        
        return valid_symbols
    
    def _standardize_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str
    ) -> pd.DataFrame:
        """Standardize DataFrame format.
        
        Args:
            df: Input DataFrame
            symbol: Symbol name
            data_type: Type of data (bars, trades, quotes)
            
        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return df
        
        # Add symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        # Ensure timestamp is datetime index
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Sort by timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        
        return df