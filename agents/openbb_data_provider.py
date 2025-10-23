#!/usr/bin/env python3
"""
OpenBB Data Provider for PC-HIVE-TRADING
Professional financial data integration using OpenBB Platform 4.5
Provides enhanced market data, options chains, fundamentals, and economic indicators
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Try to import OpenBB - gracefully fall back if not available
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
    print("[OK] OpenBB Platform loaded successfully")
except ImportError as e:
    OPENBB_AVAILABLE = False
    print(f"[WARN] OpenBB not available: {e}")
except Exception as e:
    OPENBB_AVAILABLE = False
    print(f"[WARN] OpenBB initialization issue: {e}")

# Import yfinance as fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Setup logging
try:
    from config.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class OptionsChainData:
    """Structured options chain data"""
    symbol: str
    expiration: datetime
    calls: pd.DataFrame
    puts: pd.DataFrame
    underlying_price: float
    timestamp: datetime


class DataProvider(str, Enum):
    """Available data providers in OpenBB"""
    YFINANCE = "yfinance"
    INTRINIO = "intrinio"
    POLYGON = "polygon"
    TIINGO = "tiingo"
    FMP = "fmp"
    CBOE = "cboe"
    NASDAQ = "nasdaq"
    FRED = "fred"
    BENZINGA = "benzinga"


class OpenBBDataProvider:
    """
    Professional financial data provider using OpenBB Platform 4.5
    Provides enhanced data quality with automatic fallback to yfinance
    """

    def __init__(self):
        self.available = OPENBB_AVAILABLE
        self.fallback_available = YFINANCE_AVAILABLE
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 60  # Cache data for 60 seconds
        self.last_cache_time: Dict[str, datetime] = {}

        if self.available:
            logger.info("[OK] OpenBB Data Provider initialized with full capabilities")
        elif self.fallback_available:
            logger.info("[WARN] OpenBB Data Provider using yfinance fallback")
        else:
            logger.warning("[ERROR] No data provider available!")

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache or key not in self.last_cache_time:
            return False

        age = (datetime.now() - self.last_cache_time[key]).total_seconds()
        return age < self.cache_ttl

    def _cache_data(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[key] = data
        self.last_cache_time[key] = datetime.now()

    async def get_equity_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        provider: DataProvider = DataProvider.YFINANCE
    ) -> pd.DataFrame:
        """
        Get equity historical data with enhanced quality

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'SPY')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1m', '5m', '1h', '1d', '1wk', '1mo')
            provider: Data provider to use

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"equity_{symbol}_{period}_{interval}"

        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached equity data for {symbol}")
            return self.cache[cache_key]

        try:
            if self.available:
                # Use OpenBB for professional-grade data
                logger.debug(f"Fetching {symbol} data from OpenBB ({provider.value})")

                # Convert period to start/end dates
                end_date = datetime.now()
                period_days = {
                    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
                    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
                }
                start_date = end_date - timedelta(days=period_days.get(period, 365))

                # Fetch data using OpenBB
                result = obb.equity.price.historical(
                    symbol=symbol,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    provider=provider.value
                )

                # Convert to DataFrame
                df = result.to_df() if hasattr(result, 'to_df') else pd.DataFrame(result)

                if not df.empty:
                    self._cache_data(cache_key, df)
                    logger.info(f"[OK] Fetched {len(df)} bars for {symbol} from OpenBB")
                    return df

            # Fallback to yfinance
            if self.fallback_available:
                logger.debug(f"Falling back to yfinance for {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                if not df.empty:
                    self._cache_data(cache_key, df)
                    logger.info(f"[OK] Fetched {len(df)} bars for {symbol} from yfinance")
                    return df

            logger.warning(f"[ERROR] No data available for {symbol}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching equity data for {symbol}: {e}")

            # Emergency fallback to yfinance
            if self.fallback_available:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period, interval=interval)
                    return df
                except:
                    pass

            return pd.DataFrame()

    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[datetime] = None,
        provider: DataProvider = DataProvider.YFINANCE
    ) -> Optional[OptionsChainData]:
        """
        Get comprehensive options chain data

        Args:
            symbol: Underlying stock ticker
            expiration: Specific expiration date (optional)
            provider: Data provider to use

        Returns:
            OptionsChainData object with calls and puts
        """
        cache_key = f"options_{symbol}_{expiration or 'all'}"

        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached options chain for {symbol}")
            return self.cache[cache_key]

        try:
            if self.available:
                try:
                    # Try OpenBB first for better data quality
                    logger.debug(f"Fetching options chain for {symbol} from OpenBB")

                    result = obb.derivatives.options.chains(
                        symbol=symbol,
                        provider=provider.value
                    )

                    # Convert to DataFrame
                    df = result.to_df() if hasattr(result, 'to_df') else pd.DataFrame(result)

                    if not df.empty:
                        # Get underlying price
                        equity_data = await self.get_equity_data(symbol, period="1d")
                        underlying_price = float(equity_data['Close'].iloc[-1]) if not equity_data.empty else 0.0

                        # Separate calls and puts
                        calls_df = df[df['option_type'] == 'call'] if 'option_type' in df.columns else pd.DataFrame()
                        puts_df = df[df['option_type'] == 'put'] if 'option_type' in df.columns else pd.DataFrame()

                        # Get expiration
                        exp_date = expiration or datetime.now() + timedelta(days=30)

                        options_data = OptionsChainData(
                            symbol=symbol,
                            expiration=exp_date,
                            calls=calls_df,
                            puts=puts_df,
                            underlying_price=underlying_price,
                            timestamp=datetime.now()
                        )

                        self._cache_data(cache_key, options_data)
                        logger.info(f"[OK] Fetched options chain for {symbol} from OpenBB: "
                                  f"{len(calls_df)} calls, {len(puts_df)} puts")
                        return options_data
                except Exception as e:
                    logger.debug(f"OpenBB options fetch failed, trying yfinance: {e}")

            # Fallback to yfinance
            if self.fallback_available:
                logger.debug(f"Fetching options chain for {symbol} from yfinance")
                ticker = yf.Ticker(symbol)

                # Get available expirations
                expirations = ticker.options
                if not expirations:
                    logger.warning(f"No options available for {symbol}")
                    return None

                # Use specified expiration or nearest
                target_exp = expiration.strftime('%Y-%m-%d') if expiration else expirations[0]
                if target_exp not in expirations:
                    target_exp = expirations[0]

                # Get option chain
                chain = ticker.option_chain(target_exp)

                # Get underlying price
                hist = ticker.history(period="1d")
                underlying_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0.0

                options_data = OptionsChainData(
                    symbol=symbol,
                    expiration=datetime.strptime(target_exp, '%Y-%m-%d'),
                    calls=chain.calls,
                    puts=chain.puts,
                    underlying_price=underlying_price,
                    timestamp=datetime.now()
                )

                self._cache_data(cache_key, options_data)
                logger.info(f"[OK] Fetched options chain for {symbol} from yfinance: "
                          f"{len(chain.calls)} calls, {len(chain.puts)} puts")
                return options_data

            logger.warning(f"[ERROR] No options data available for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return None

    async def get_company_fundamentals(
        self,
        symbol: str,
        provider: DataProvider = DataProvider.FMP
    ) -> Dict[str, Any]:
        """
        Get company fundamental data

        Args:
            symbol: Stock ticker
            provider: Data provider (fmp, intrinio, polygon, yfinance)

        Returns:
            Dict with fundamental metrics
        """
        cache_key = f"fundamentals_{symbol}"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            if self.available:
                logger.debug(f"Fetching fundamentals for {symbol} from OpenBB")

                # Get key metrics
                metrics = obb.equity.fundamental.metrics(
                    symbol=symbol,
                    provider=provider.value
                )

                fundamentals = metrics.to_dict() if hasattr(metrics, 'to_dict') else {}

                self._cache_data(cache_key, fundamentals)
                logger.info(f"[OK] Fetched fundamentals for {symbol}")
                return fundamentals

            # Fallback to yfinance
            if self.fallback_available:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                fundamentals = {
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'earnings_growth': info.get('earningsGrowth', 0),
                }

                self._cache_data(cache_key, fundamentals)
                logger.info(f"[OK] Fetched fundamentals for {symbol} from yfinance")
                return fundamentals

            return {}

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}

    async def get_market_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
        provider: DataProvider = DataProvider.BENZINGA
    ) -> List[Dict[str, Any]]:
        """
        Get financial news with sentiment analysis

        Args:
            symbol: Stock ticker (None for general market news)
            limit: Number of news items to fetch
            provider: News provider (benzinga, polygon, tiingo)

        Returns:
            List of news articles with metadata
        """
        try:
            if self.available:
                logger.debug(f"Fetching news for {symbol or 'market'} from OpenBB")

                if symbol:
                    news_result = obb.news.company(
                        symbol=symbol,
                        limit=limit,
                        provider=provider.value
                    )
                else:
                    news_result = obb.news.world(
                        limit=limit,
                        provider=provider.value
                    )

                news = news_result.to_dict('records') if hasattr(news_result, 'to_dict') else []

                logger.info(f"[OK] Fetched {len(news)} news items")
                return news

            logger.info("News data requires OpenBB (not available)")
            return []

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    async def get_economic_indicator(
        self,
        indicator: str = "GDP",
        provider: DataProvider = DataProvider.FRED
    ) -> pd.DataFrame:
        """
        Get economic indicators (GDP, inflation, unemployment, etc.)

        Args:
            indicator: Economic indicator code
            provider: Economic data provider (fred, bls, oecd)

        Returns:
            DataFrame with indicator time series
        """
        try:
            if self.available:
                logger.debug(f"Fetching {indicator} from {provider.value}")

                # Examples of available indicators
                if indicator.upper() == "GDP":
                    result = obb.economy.gdp(provider=provider.value)
                elif indicator.upper() == "CPI":
                    result = obb.economy.cpi(provider=provider.value)
                elif indicator.upper() == "UNEMPLOYMENT":
                    result = obb.economy.unemployment(provider=provider.value)
                else:
                    # Generic FRED series fetch
                    result = obb.economy.fred_series(series_id=indicator)

                df = result.to_df() if hasattr(result, 'to_df') else pd.DataFrame()

                logger.info(f"[OK] Fetched economic indicator: {indicator}")
                return df

            logger.info("Economic data requires OpenBB (not available)")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching economic indicator {indicator}: {e}")
            return pd.DataFrame()

    async def get_market_indices(self) -> Dict[str, float]:
        """
        Get major market indices (SPY, QQQ, DIA, IWM, VIX)

        Returns:
            Dict with index symbols and current prices
        """
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            '^VIX': 'Volatility Index'
        }

        results = {}

        for symbol, name in indices.items():
            try:
                df = await self.get_equity_data(symbol, period="1d")
                if not df.empty:
                    results[symbol] = float(df['Close'].iloc[-1])
                    logger.debug(f"{name} ({symbol}): ${results[symbol]:.2f}")
            except Exception as e:
                logger.warning(f"Error fetching {name}: {e}")
                results[symbol] = 0.0

        return results

    async def calculate_technical_indicators(
        self,
        symbol: str,
        period: str = "3mo"
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive technical indicators

        Args:
            symbol: Stock ticker
            period: Historical period for calculation

        Returns:
            Dict with technical indicators
        """
        try:
            # Get historical data
            df = await self.get_equity_data(symbol, period=period)

            if df.empty:
                return {}

            # Calculate indicators
            indicators = {}

            # RSI (14-day)
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = float(100 - (100 / (1 + rs.iloc[-1])))

            # Moving Averages
            if len(df) >= 50:
                indicators['sma_20'] = float(df['Close'].rolling(window=20).mean().iloc[-1])
                indicators['sma_50'] = float(df['Close'].rolling(window=50).mean().iloc[-1])
                indicators['ema_20'] = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])

            # MACD
            if len(df) >= 26:
                ema12 = df['Close'].ewm(span=12, adjust=False).mean()
                ema26 = df['Close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                indicators['macd'] = float(macd.iloc[-1])
                indicators['macd_signal'] = float(signal.iloc[-1])
                indicators['macd_histogram'] = float(macd.iloc[-1] - signal.iloc[-1])

            # Bollinger Bands
            if len(df) >= 20:
                sma = df['Close'].rolling(window=20).mean()
                std = df['Close'].rolling(window=20).std()
                indicators['bb_upper'] = float(sma.iloc[-1] + (2 * std.iloc[-1]))
                indicators['bb_middle'] = float(sma.iloc[-1])
                indicators['bb_lower'] = float(sma.iloc[-1] - (2 * std.iloc[-1]))

            # Volatility
            indicators['volatility'] = float(df['Close'].pct_change().std() * np.sqrt(252))

            # Momentum
            if len(df) >= 10:
                indicators['momentum_10d'] = float((df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100)

            logger.info(f"[OK] Calculated technical indicators for {symbol}")
            return indicators

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return {}

    async def get_options_greeks_estimate(
        self,
        symbol: str,
        strike: float,
        expiration: datetime,
        option_type: str = "call"
    ) -> Dict[str, float]:
        """
        Estimate options Greeks using simplified models

        Args:
            symbol: Underlying ticker
            strike: Strike price
            expiration: Expiration date
            option_type: 'call' or 'put'

        Returns:
            Dict with estimated Greeks
        """
        try:
            # Get current price and volatility
            df = await self.get_equity_data(symbol, period="1mo")
            if df.empty:
                return {}

            current_price = float(df['Close'].iloc[-1])
            volatility = float(df['Close'].pct_change().std() * np.sqrt(252))

            # Days to expiration
            dte = (expiration - datetime.now()).days
            time_to_expiry = dte / 365.0

            # Simplified Greeks estimation
            moneyness = current_price / strike

            if option_type.lower() == 'call':
                # Rough delta estimate for calls
                if moneyness > 1.1:  # Deep ITM
                    delta = 0.85
                elif moneyness > 1.0:  # ITM
                    delta = 0.65
                elif moneyness > 0.95:  # Near ATM
                    delta = 0.50
                elif moneyness > 0.90:  # OTM
                    delta = 0.35
                else:  # Deep OTM
                    delta = 0.15
            else:  # put
                if moneyness < 0.9:  # Deep ITM
                    delta = -0.85
                elif moneyness < 1.0:  # ITM
                    delta = -0.65
                elif moneyness < 1.05:  # Near ATM
                    delta = -0.50
                elif moneyness < 1.10:  # OTM
                    delta = -0.35
                else:  # Deep OTM
                    delta = -0.15

            # Rough gamma (highest near ATM)
            gamma = 0.1 * (1 - abs(moneyness - 1.0) * 2) if abs(moneyness - 1.0) < 0.5 else 0.01

            # Rough theta (increases as expiration approaches)
            theta = -0.05 * (1 + (1 / max(time_to_expiry, 0.01)))

            # Rough vega
            vega = 0.15 * np.sqrt(time_to_expiry)

            greeks = {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'implied_volatility': volatility
            }

            logger.debug(f"Estimated Greeks for {symbol} {strike} {option_type}: {greeks}")
            return greeks

        except Exception as e:
            logger.error(f"Error estimating Greeks: {e}")
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get provider status and capabilities"""
        return {
            'openbb_available': self.available,
            'yfinance_fallback': self.fallback_available,
            'cache_size': len(self.cache),
            'cache_ttl': self.cache_ttl,
            'capabilities': {
                'equity_data': True,
                'options_chains': True,
                'fundamentals': self.available or self.fallback_available,
                'news': self.available,
                'economic_data': self.available,
                'technical_indicators': True,
                'greeks_estimation': True
            }
        }


# Global singleton instance
openbb_provider = OpenBBDataProvider()


# Test function
async def test_openbb_integration():
    """Test OpenBB integration with sample data"""
    print("\n" + "="*60)
    print("TESTING OPENBB DATA PROVIDER")
    print("="*60)

    provider = OpenBBDataProvider()

    # Test 1: Equity data
    print("\n[TEST 1] Fetching SPY equity data...")
    spy_data = await provider.get_equity_data("SPY", period="5d")
    if not spy_data.empty:
        print(f"[OK] Success: Retrieved {len(spy_data)} bars")
        print(f"   Latest close: ${spy_data['Close'].iloc[-1]:.2f}")
    else:
        print("[FAIL] Failed to fetch equity data")

    # Test 2: Options chain
    print("\n[TEST 2] Fetching AAPL options chain...")
    options = await provider.get_options_chain("AAPL")
    if options:
        print(f"[OK] Success: {len(options.calls)} calls, {len(options.puts)} puts")
        print(f"   Underlying price: ${options.underlying_price:.2f}")
    else:
        print("[FAIL] Failed to fetch options chain")

    # Test 3: Technical indicators
    print("\n[TEST 3] Calculating technical indicators for MSFT...")
    indicators = await provider.calculate_technical_indicators("MSFT")
    if indicators:
        print(f"[OK] Success: Calculated {len(indicators)} indicators")
        if 'rsi' in indicators:
            print(f"   RSI: {indicators['rsi']:.2f}")
    else:
        print("[FAIL] Failed to calculate indicators")

    # Test 4: Market indices
    print("\n[TEST 4] Fetching market indices...")
    indices = await provider.get_market_indices()
    if indices:
        print(f"[OK] Success: Retrieved {len(indices)} indices")
        for symbol, price in indices.items():
            print(f"   {symbol}: ${price:.2f}")
    else:
        print("[FAIL] Failed to fetch indices")

    # Status
    print("\n" + "="*60)
    status = provider.get_status()
    print("PROVIDER STATUS:")
    print(f"OpenBB Available: {status['openbb_available']}")
    print(f"YFinance Fallback: {status['yfinance_fallback']}")
    print(f"Cache Size: {status['cache_size']} items")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_openbb_integration())
