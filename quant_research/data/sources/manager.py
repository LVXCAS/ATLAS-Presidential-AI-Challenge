"""Data source manager for coordinating multiple data sources."""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import asyncio
from datetime import datetime, date
from contextlib import asynccontextmanager

from .base import BaseDataSource, DataRequest, DataResponse, DataSourceError
from .alpaca import AlpacaDataSource
from .yahoo import YahooDataSource
from structlog import get_logger

logger = get_logger(__name__)


class DataSourceManager:
    """Manages multiple data sources and provides unified interface."""
    
    def __init__(self):
        """Initialize data source manager."""
        self.sources: Dict[str, BaseDataSource] = {}
        self.primary_source: Optional[str] = None
        self.fallback_order: List[str] = []
    
    def add_source(
        self,
        name: str,
        source: BaseDataSource,
        is_primary: bool = False
    ):
        """Add a data source.
        
        Args:
            name: Source name
            source: Data source instance
            is_primary: Whether this is the primary source
        """
        self.sources[name] = source
        
        if is_primary or self.primary_source is None:
            self.primary_source = name
        
        # Add to fallback order if not already present
        if name not in self.fallback_order:
            self.fallback_order.append(name)
        
        logger.info(f"Added data source: {name}", is_primary=is_primary)
    
    def remove_source(self, name: str):
        """Remove a data source.
        
        Args:
            name: Source name to remove
        """
        if name in self.sources:
            del self.sources[name]
        
        if name in self.fallback_order:
            self.fallback_order.remove(name)
        
        if self.primary_source == name:
            self.primary_source = self.fallback_order[0] if self.fallback_order else None
        
        logger.info(f"Removed data source: {name}")
    
    def set_fallback_order(self, order: List[str]):
        """Set the fallback order for data sources.
        
        Args:
            order: List of source names in fallback order
        """
        # Validate all sources exist
        for name in order:
            if name not in self.sources:
                raise ValueError(f"Unknown data source: {name}")
        
        self.fallback_order = order
        logger.info(f"Set fallback order: {order}")
    
    @asynccontextmanager
    async def connect_all(self):
        """Context manager to connect to all data sources."""
        # Connect to all sources
        for name, source in self.sources.items():
            try:
                await source.connect()
                logger.info(f"Connected to {name}")
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
        
        try:
            yield self
        finally:
            # Disconnect from all sources
            for name, source in self.sources.items():
                try:
                    await source.disconnect()
                    logger.info(f"Disconnected from {name}")
                except Exception as e:
                    logger.error(f"Error disconnecting from {name}: {e}")
    
    async def get_bars(
        self,
        request: DataRequest,
        sources: Optional[List[str]] = None,
        use_fallback: bool = True
    ) -> DataResponse:
        """Get price bars with fallback support.
        
        Args:
            request: Data request specification
            sources: Specific sources to try (None for all)
            use_fallback: Whether to use fallback sources
            
        Returns:
            DataResponse containing price bars
        """
        return await self._get_data_with_fallback(
            "get_bars", request, sources, use_fallback
        )
    
    async def get_trades(
        self,
        request: DataRequest,
        sources: Optional[List[str]] = None,
        use_fallback: bool = True
    ) -> DataResponse:
        """Get trade data with fallback support.
        
        Args:
            request: Data request specification
            sources: Specific sources to try (None for all)
            use_fallback: Whether to use fallback sources
            
        Returns:
            DataResponse containing trade data
        """
        return await self._get_data_with_fallback(
            "get_trades", request, sources, use_fallback
        )
    
    async def get_quotes(
        self,
        request: DataRequest,
        sources: Optional[List[str]] = None,
        use_fallback: bool = True
    ) -> DataResponse:
        """Get quote data with fallback support.
        
        Args:
            request: Data request specification
            sources: Specific sources to try (None for all)
            use_fallback: Whether to use fallback sources
            
        Returns:
            DataResponse containing quote data
        """
        return await self._get_data_with_fallback(
            "get_quotes", request, sources, use_fallback
        )
    
    async def _get_data_with_fallback(
        self,
        method_name: str,
        request: DataRequest,
        sources: Optional[List[str]] = None,
        use_fallback: bool = True
    ) -> DataResponse:
        """Get data with automatic fallback to other sources.
        
        Args:
            method_name: Method name to call on sources
            request: Data request specification
            sources: Specific sources to try
            use_fallback: Whether to use fallback sources
            
        Returns:
            DataResponse from successful source
            
        Raises:
            DataSourceError: If all sources fail
        """
        # Determine sources to try
        if sources:
            sources_to_try = sources
        elif use_fallback:
            sources_to_try = self.fallback_order
        else:
            sources_to_try = [self.primary_source] if self.primary_source else []
        
        last_error = None
        
        for source_name in sources_to_try:
            if source_name not in self.sources:
                logger.warning(f"Source {source_name} not available, skipping")
                continue
            
            source = self.sources[source_name]
            
            try:
                # Get the method from the source
                method = getattr(source, method_name)
                
                # Call the method
                response = await method(request)
                
                if not response.is_empty:
                    logger.info(
                        f"Successfully retrieved data from {source_name}",
                        method=method_name,
                        records=len(response.to_pandas()) if hasattr(response.data, '__len__') else "unknown"
                    )
                    return response
                else:
                    logger.warning(f"Empty response from {source_name}")
                    
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed to get data from {source_name}: {e}",
                    method=method_name
                )
                continue
        
        # All sources failed
        error_msg = f"All data sources failed for {method_name}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        
        logger.error(error_msg)
        raise DataSourceError(error_msg)
    
    async def get_parallel_data(
        self,
        requests: List[DataRequest],
        method_name: str = "get_bars"
    ) -> List[DataResponse]:
        """Get data from multiple requests in parallel.
        
        Args:
            requests: List of data requests
            method_name: Method to call (get_bars, get_trades, get_quotes)
            
        Returns:
            List of DataResponse objects
        """
        # Create tasks for parallel execution
        tasks = []
        
        for request in requests:
            method = getattr(self, method_name)
            task = asyncio.create_task(method(request))
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Request {i} failed: {response}")
                # Create empty response for failed requests
                valid_responses.append(DataResponse(
                    data=pd.DataFrame(),
                    metadata={"error": str(response)},
                    source="failed",
                    request=requests[i],
                    timestamp=datetime.now()
                ))
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources.
        
        Returns:
            List of source names
        """
        return list(self.sources.keys())
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all data sources.
        
        Returns:
            Dictionary mapping source names to health status
        """
        health_status = {}
        
        for name, source in self.sources.items():
            try:
                # Try to get a simple data request
                test_request = DataRequest(
                    symbols=["SPY"],
                    start_date=datetime.now().date(),
                    end_date=datetime.now().date(),
                    limit=1
                )
                
                start_time = datetime.now()
                response = await source.get_bars(test_request)
                end_time = datetime.now()
                
                response_time = (end_time - start_time).total_seconds()
                
                health_status[name] = {
                    "status": "healthy",
                    "response_time_seconds": response_time,
                    "last_check": datetime.now().isoformat(),
                    "data_available": not response.is_empty
                }
                
            except Exception as e:
                health_status[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        return health_status
    
    def get_source_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """Get capabilities of each data source.
        
        Returns:
            Dictionary mapping sources to their capabilities
        """
        capabilities = {}
        
        for name, source in self.sources.items():
            capabilities[name] = {
                "bars": hasattr(source, 'get_bars'),
                "trades": hasattr(source, 'get_trades'), 
                "quotes": hasattr(source, 'get_quotes'),
                "fundamentals": hasattr(source, 'get_fundamentals'),
                "realtime": hasattr(source, 'subscribe_realtime'),
                "symbols": hasattr(source, 'get_available_symbols')
            }
        
        return capabilities