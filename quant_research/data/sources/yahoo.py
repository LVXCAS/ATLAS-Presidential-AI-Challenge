"""Yahoo Finance data source implementation."""

import pandas as pd
import yfinance as yf
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import asyncio

from .base import BaseDataSource, DataRequest, DataResponse, DataSourceError
from structlog import get_logger

logger = get_logger(__name__)


class YahooDataSource(BaseDataSource):
    """Yahoo Finance data source."""
    
    def __init__(self, **kwargs):
        """Initialize Yahoo Finance data source.
        
        Args:
            **kwargs: Additional configuration
        """
        super().__init__(
            name="yahoo",
            rate_limit=2000,  # Very generous limit
            **kwargs
        )
    
    def _get_yf_interval(self, frequency: str) -> str:
        """Convert frequency string to Yahoo Finance interval.
        
        Args:
            frequency: Frequency string (1min, 5min, 1H, 1D)
            
        Returns:
            Yahoo Finance interval string
        """
        interval_map = {
            "1min": "1m",
            "2min": "2m", 
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "60min": "60m",
            "90min": "90m",
            "1H": "1h",
            "1D": "1d",
            "5D": "5d",
            "1W": "1wk",
            "1M": "1mo",
            "3M": "3mo"
        }
        
        if frequency not in interval_map:
            raise DataSourceError(f"Unsupported frequency: {frequency}")
        
        return interval_map[frequency]
    
    async def get_bars(self, request: DataRequest) -> DataResponse:
        """Get price bars from Yahoo Finance.
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse containing OHLCV data
        """
        try:
            symbols = self._validate_symbols(request.symbols)
            interval = self._get_yf_interval(request.frequency)
            
            # Download data
            df_list = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Get historical data
                    hist = ticker.history(
                        start=request.start_date,
                        end=request.end_date,
                        interval=interval,
                        auto_adjust=True,
                        prepost=False
                    )
                    
                    if not hist.empty:
                        # Reset index to get timestamp as column
                        hist = hist.reset_index()
                        
                        # Standardize column names
                        hist.columns = [col.lower() for col in hist.columns]
                        hist = hist.rename(columns={
                            'date': 'timestamp',
                            'datetime': 'timestamp'
                        })
                        
                        # Add symbol column
                        hist['symbol'] = symbol
                        
                        # Standardize DataFrame
                        hist = self._standardize_dataframe(hist, symbol, "bars")
                        df_list.append(hist)
                        
                        logger.debug(f"Retrieved {len(hist)} bars for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue
            
            # Combine all symbols
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=False)
            else:
                combined_df = pd.DataFrame()
            
            logger.info(
                f"Retrieved {len(combined_df)} bars from Yahoo Finance",
                symbols=symbols,
                frequency=request.frequency
            )
            
            return DataResponse(
                data=combined_df,
                metadata={
                    "source": "yahoo",
                    "data_type": "bars",
                    "symbols_count": len(symbols),
                    "records_count": len(combined_df),
                    "frequency": request.frequency,
                    "interval": interval
                },
                source=self.name,
                request=request,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting bars from Yahoo Finance: {e}")
            raise DataSourceError(f"Yahoo Finance bars request failed: {e}")
    
    async def get_trades(self, request: DataRequest) -> DataResponse:
        """Get trade data (not available from Yahoo Finance).
        
        Args:
            request: Data request specification
            
        Returns:
            Empty DataResponse
        """
        logger.warning("Trade data not available from Yahoo Finance")
        
        return DataResponse(
            data=pd.DataFrame(),
            metadata={
                "source": "yahoo",
                "data_type": "trades",
                "available": False,
                "message": "Trade data not available from Yahoo Finance"
            },
            source=self.name,
            request=request,
            timestamp=datetime.now()
        )
    
    async def get_quotes(self, request: DataRequest) -> DataResponse:
        """Get current quote data from Yahoo Finance.
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse containing current quote data
        """
        try:
            symbols = self._validate_symbols(request.symbols)
            
            # Get current quotes
            data = yf.download(
                tickers=symbols,
                period="1d",
                interval="1m",
                group_by="ticker",
                auto_adjust=True,
                prepost=False
            )
            
            df_list = []
            
            if len(symbols) == 1:
                # Single symbol case
                if not data.empty:
                    latest = data.tail(1).reset_index()
                    latest['symbol'] = symbols[0]
                    
                    # Create quote-like data from OHLC
                    latest['bid_price'] = latest['Low'] 
                    latest['ask_price'] = latest['High']
                    latest['bid_size'] = 0  # Not available
                    latest['ask_size'] = 0  # Not available
                    
                    df_list.append(latest)
            else:
                # Multiple symbols case
                for symbol in symbols:
                    if symbol in data.columns.levels[0]:
                        symbol_data = data[symbol]
                        
                        if not symbol_data.empty:
                            latest = symbol_data.tail(1).reset_index()
                            latest['symbol'] = symbol
                            
                            # Create quote-like data from OHLC
                            latest['bid_price'] = latest['Low']
                            latest['ask_price'] = latest['High'] 
                            latest['bid_size'] = 0  # Not available
                            latest['ask_size'] = 0  # Not available
                            
                            df_list.append(latest)
            
            # Combine all symbols
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
            else:
                combined_df = pd.DataFrame()
            
            logger.info(
                f"Retrieved {len(combined_df)} quotes from Yahoo Finance",
                symbols=symbols
            )
            
            return DataResponse(
                data=combined_df,
                metadata={
                    "source": "yahoo",
                    "data_type": "quotes",
                    "symbols_count": len(symbols),
                    "records_count": len(combined_df),
                    "note": "Quotes derived from latest OHLC data"
                },
                source=self.name,
                request=request,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting quotes from Yahoo Finance: {e}")
            raise DataSourceError(f"Yahoo Finance quotes request failed: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of popular symbols (Yahoo Finance doesn't provide full list).
        
        Returns:
            List of popular symbols
        """
        # Common symbols for testing/demo purposes
        popular_symbols = [
            # Major indices
            "^GSPC", "^DJI", "^IXIC", "^RUT",
            
            # ETFs
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLB",
            "GLD", "SLV", "TLT", "HYG", "LQD",
            
            # Tech
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NFLX",
            "NVDA", "CRM", "ADBE", "ORCL", "INTC", "AMD", "PYPL",
            
            # Finance
            "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC",
            "BRK-A", "BRK-B", "V", "MA",
            
            # Healthcare
            "JNJ", "PFE", "UNH", "ABBV", "TMO", "DHR", "BMY", "AMGN", "GILD",
            
            # Consumer
            "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "DIS",
            
            # Energy
            "XOM", "CVX", "COP", "EOG", "SLB", "OXY",
            
            # Industrials
            "BA", "CAT", "GE", "MMM", "UPS", "RTX",
            
            # Crypto
            "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD"
        ]
        
        logger.info(f"Returning {len(popular_symbols)} popular symbols")
        
        return popular_symbols
    
    async def get_fundamentals(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get fundamental data for symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary mapping symbols to fundamental data
        """
        try:
            symbols = self._validate_symbols(symbols)
            fundamentals = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Get various fundamental data
                    info = ticker.info
                    financials = ticker.financials
                    balance_sheet = ticker.balance_sheet
                    cash_flow = ticker.cash_flow
                    
                    fundamentals[symbol] = {
                        "info": info,
                        "financials": financials.to_dict() if not financials.empty else {},
                        "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
                        "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {}
                    }
                    
                    logger.debug(f"Retrieved fundamentals for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
                    fundamentals[symbol] = {}
            
            logger.info(f"Retrieved fundamentals for {len(fundamentals)} symbols")
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error getting fundamentals: {e}")
            raise DataSourceError(f"Yahoo Finance fundamentals request failed: {e}")