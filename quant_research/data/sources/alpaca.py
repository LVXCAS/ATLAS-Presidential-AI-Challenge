"""Alpaca data source implementation."""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import asyncio

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest, StockTradesRequest, StockQuotesRequest
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    ALPACA_AVAILABLE = True
except ImportError:
    # Use alpaca-py instead
    try:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data import (
            StockBarsRequest, StockTradesRequest, StockQuotesRequest
        )
        from alpaca.data import TimeFrame, TimeFrameUnit
        from alpaca.trading import TradingClient
        ALPACA_AVAILABLE = True
    except ImportError:
        ALPACA_AVAILABLE = False

from .base import BaseDataSource, DataRequest, DataResponse, DataSourceError
from structlog import get_logger

logger = get_logger(__name__)


class AlpacaDataSource(BaseDataSource):
    """Alpaca data source for US equities."""
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        **kwargs
    ):
        """Initialize Alpaca data source.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading environment
            **kwargs: Additional configuration
        """
        super().__init__(
            name="alpaca",
            api_key=api_key,
            rate_limit=200,  # 200 requests per minute
            **kwargs
        )
        
        self.secret_key = secret_key
        self.paper = paper
        
        # Initialize clients
        self._data_client = StockHistoricalDataClient(api_key, secret_key)
        self._trading_client = TradingClient(
            api_key, secret_key, paper=paper
        )
    
    def _get_timeframe(self, frequency: str) -> TimeFrame:
        """Convert frequency string to Alpaca TimeFrame.
        
        Args:
            frequency: Frequency string (1min, 5min, 1H, 1D)
            
        Returns:
            Alpaca TimeFrame object
        """
        frequency_map = {
            "1min": TimeFrame(1, TimeFrameUnit.Minute),
            "5min": TimeFrame(5, TimeFrameUnit.Minute),
            "15min": TimeFrame(15, TimeFrameUnit.Minute),
            "30min": TimeFrame(30, TimeFrameUnit.Minute),
            "1H": TimeFrame(1, TimeFrameUnit.Hour),
            "1D": TimeFrame(1, TimeFrameUnit.Day),
            "1W": TimeFrame(1, TimeFrameUnit.Week),
            "1M": TimeFrame(1, TimeFrameUnit.Month),
        }
        
        if frequency not in frequency_map:
            raise DataSourceError(f"Unsupported frequency: {frequency}")
        
        return frequency_map[frequency]
    
    async def get_bars(self, request: DataRequest) -> DataResponse:
        """Get price bars from Alpaca.
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse containing OHLCV data
        """
        try:
            symbols = self._validate_symbols(request.symbols)
            timeframe = self._get_timeframe(request.frequency)
            
            # Create Alpaca request
            bars_request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=request.start_date,
                end=request.end_date,
                limit=request.limit
            )
            
            # Get data from Alpaca
            bars_response = self._data_client.get_stock_bars(bars_request)
            
            # Convert to DataFrame
            df_list = []
            
            for symbol in symbols:
                if symbol in bars_response.data:
                    symbol_bars = bars_response.data[symbol]
                    
                    # Convert to DataFrame
                    data = []
                    for bar in symbol_bars:
                        data.append({
                            'timestamp': bar.timestamp,
                            'open': bar.open,
                            'high': bar.high, 
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'vwap': bar.vwap,
                            'trade_count': bar.trade_count,
                            'symbol': symbol
                        })
                    
                    if data:
                        symbol_df = pd.DataFrame(data)
                        symbol_df = self._standardize_dataframe(
                            symbol_df, symbol, "bars"
                        )
                        df_list.append(symbol_df)
            
            # Combine all symbols
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=False)
            else:
                combined_df = pd.DataFrame()
            
            logger.info(
                f"Retrieved {len(combined_df)} bars from Alpaca",
                symbols=symbols,
                frequency=request.frequency
            )
            
            return DataResponse(
                data=combined_df,
                metadata={
                    "source": "alpaca",
                    "data_type": "bars",
                    "symbols_count": len(symbols),
                    "records_count": len(combined_df),
                    "frequency": request.frequency
                },
                source=self.name,
                request=request,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting bars from Alpaca: {e}")
            raise DataSourceError(f"Alpaca bars request failed: {e}")
    
    async def get_trades(self, request: DataRequest) -> DataResponse:
        """Get trade data from Alpaca.
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse containing trade data
        """
        try:
            symbols = self._validate_symbols(request.symbols)
            
            # Create Alpaca request
            trades_request = StockTradesRequest(
                symbol_or_symbols=symbols,
                start=request.start_date,
                end=request.end_date,
                limit=request.limit
            )
            
            # Get data from Alpaca
            trades_response = self._data_client.get_stock_trades(trades_request)
            
            # Convert to DataFrame
            df_list = []
            
            for symbol in symbols:
                if symbol in trades_response.data:
                    symbol_trades = trades_response.data[symbol]
                    
                    # Convert to DataFrame
                    data = []
                    for trade in symbol_trades:
                        data.append({
                            'timestamp': trade.timestamp,
                            'price': trade.price,
                            'size': trade.size,
                            'conditions': trade.conditions,
                            'exchange': trade.exchange,
                            'symbol': symbol
                        })
                    
                    if data:
                        symbol_df = pd.DataFrame(data)
                        symbol_df = self._standardize_dataframe(
                            symbol_df, symbol, "trades"
                        )
                        df_list.append(symbol_df)
            
            # Combine all symbols
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=False)
            else:
                combined_df = pd.DataFrame()
            
            logger.info(
                f"Retrieved {len(combined_df)} trades from Alpaca",
                symbols=symbols
            )
            
            return DataResponse(
                data=combined_df,
                metadata={
                    "source": "alpaca",
                    "data_type": "trades",
                    "symbols_count": len(symbols),
                    "records_count": len(combined_df)
                },
                source=self.name,
                request=request,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting trades from Alpaca: {e}")
            raise DataSourceError(f"Alpaca trades request failed: {e}")
    
    async def get_quotes(self, request: DataRequest) -> DataResponse:
        """Get quote data from Alpaca.
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse containing quote data
        """
        try:
            symbols = self._validate_symbols(request.symbols)
            
            # Create Alpaca request
            quotes_request = StockQuotesRequest(
                symbol_or_symbols=symbols,
                start=request.start_date,
                end=request.end_date,
                limit=request.limit
            )
            
            # Get data from Alpaca
            quotes_response = self._data_client.get_stock_quotes(quotes_request)
            
            # Convert to DataFrame
            df_list = []
            
            for symbol in symbols:
                if symbol in quotes_response.data:
                    symbol_quotes = quotes_response.data[symbol]
                    
                    # Convert to DataFrame
                    data = []
                    for quote in symbol_quotes:
                        data.append({
                            'timestamp': quote.timestamp,
                            'bid_price': quote.bid_price,
                            'bid_size': quote.bid_size,
                            'ask_price': quote.ask_price,
                            'ask_size': quote.ask_size,
                            'bid_exchange': quote.bid_exchange,
                            'ask_exchange': quote.ask_exchange,
                            'symbol': symbol
                        })
                    
                    if data:
                        symbol_df = pd.DataFrame(data)
                        symbol_df = self._standardize_dataframe(
                            symbol_df, symbol, "quotes"
                        )
                        df_list.append(symbol_df)
            
            # Combine all symbols
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=False)
            else:
                combined_df = pd.DataFrame()
            
            logger.info(
                f"Retrieved {len(combined_df)} quotes from Alpaca",
                symbols=symbols
            )
            
            return DataResponse(
                data=combined_df,
                metadata={
                    "source": "alpaca",
                    "data_type": "quotes",
                    "symbols_count": len(symbols),
                    "records_count": len(combined_df)
                },
                source=self.name,
                request=request,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting quotes from Alpaca: {e}")
            raise DataSourceError(f"Alpaca quotes request failed: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of tradeable symbols from Alpaca.
        
        Returns:
            List of available symbols
        """
        try:
            # Get active assets
            assets = self._trading_client.get_all_assets()
            
            # Filter for active, tradeable US equities
            symbols = [
                asset.symbol for asset in assets
                if (
                    asset.status == "active" and
                    asset.tradable and
                    asset.class_ == "us_equity"
                )
            ]
            
            logger.info(f"Retrieved {len(symbols)} symbols from Alpaca")
            
            return sorted(symbols)
            
        except Exception as e:
            logger.error(f"Error getting symbols from Alpaca: {e}")
            raise DataSourceError(f"Failed to get symbols: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information.
        
        Returns:
            Dictionary containing account information
        """
        try:
            account = self._trading_client.get_account()
            
            return {
                "account_id": account.id,
                "status": account.status,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "multiplier": account.multiplier,
                "day_trade_count": account.day_trade_count,
                "pattern_day_trader": account.pattern_day_trader
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise DataSourceError(f"Failed to get account info: {e}")