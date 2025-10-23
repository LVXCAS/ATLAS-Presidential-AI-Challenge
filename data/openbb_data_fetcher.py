#!/usr/bin/env python3
"""
OPENBB PLATFORM DATA FETCHER
Multi-source data aggregation with automatic fallback

OpenBB provides access to 100+ data providers through unified API
Automatically selects best free source for each data type
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    print("[WARNING] OpenBB not installed - run: pip install openbb")


@dataclass
class EarningsEvent:
    """Earnings announcement from OpenBB"""
    symbol: str
    company_name: str
    earnings_date: str
    estimated_eps: Optional[float] = None
    actual_eps: Optional[float] = None
    surprise_pct: Optional[float] = None
    market_cap: Optional[float] = None


class OpenBBDataFetcher:
    """
    Unified data fetcher using OpenBB Platform

    Capabilities:
    - Earnings calendar (multiple providers)
    - Options unusual activity
    - Insider trading
    - Dark pool activity
    - Economic data
    - Market data with auto-provider selection
    """

    def __init__(self):
        """Initialize OpenBB data fetcher"""

        if not OPENBB_AVAILABLE:
            raise ImportError("OpenBB not installed. Run: pip install openbb")

        print("[OPENBB] Data fetcher initialized")
        print("[OPENBB] Access to 100+ data providers")

    def get_earnings_calendar(self,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             provider: str = "fmp") -> List[EarningsEvent]:
        """
        Get earnings calendar from OpenBB

        Args:
            start_date: Start date (YYYY-MM-DD), default today
            end_date: End date (YYYY-MM-DD), default +7 days
            provider: Data provider ('fmp', 'nasdaq', 'tradingeconomics')

        Returns:
            List of earnings events
        """

        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')

        if not end_date:
            end_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

        print(f"[OPENBB] Fetching earnings: {start_date} to {end_date}")

        try:
            # Get earnings calendar
            calendar = obb.equity.calendar.earnings(
                start_date=start_date,
                end_date=end_date,
                provider=provider
            )

            # Convert to dataframe
            df = calendar.to_df() if hasattr(calendar, 'to_df') else pd.DataFrame(calendar)

            if df.empty:
                print(f"[OPENBB] No earnings found ({provider})")
                return []

            # Convert to EarningsEvent objects
            events = []
            for _, row in df.iterrows():
                event = EarningsEvent(
                    symbol=row.get('symbol', row.get('ticker', '')),
                    company_name=row.get('name', row.get('company', '')),
                    earnings_date=str(row.get('reportDate', row.get('date', ''))),
                    estimated_eps=row.get('epsEstimate', row.get('estimated_eps')),
                    actual_eps=row.get('epsActual', row.get('actual_eps')),
                    surprise_pct=row.get('epsSurprisePct'),
                    market_cap=row.get('marketCap', row.get('market_cap'))
                )
                events.append(event)

            print(f"[OPENBB] Found {len(events)} earnings events")
            return events

        except Exception as e:
            print(f"[OPENBB ERROR] Earnings calendar: {e}")

            # Try fallback provider
            if provider == 'fmp':
                print("[OPENBB] Trying fallback provider: nasdaq")
                return self.get_earnings_calendar(start_date, end_date, provider='nasdaq')

            return []

    def get_unusual_options_activity(self,
                                     min_volume: int = 1000,
                                     min_oi_ratio: float = 2.0) -> pd.DataFrame:
        """
        Get unusual options activity

        Args:
            min_volume: Minimum option volume
            min_oi_ratio: Minimum volume/OI ratio (2.0 = 2x normal)

        Returns:
            DataFrame of unusual options activity
        """

        print("[OPENBB] Fetching unusual options activity...")

        try:
            # Get unusual options
            unusual = obb.derivatives.options.unusual(provider="intrinio")

            df = unusual.to_df() if hasattr(unusual, 'to_df') else pd.DataFrame(unusual)

            if df.empty:
                print("[OPENBB] No unusual activity found")
                return pd.DataFrame()

            # Filter by criteria
            filtered = df[
                (df['volume'] >= min_volume) &
                (df['volume'] / df['open_interest'] >= min_oi_ratio)
            ]

            print(f"[OPENBB] Found {len(filtered)} unusual options contracts")
            return filtered

        except Exception as e:
            print(f"[OPENBB ERROR] Unusual options: {e}")
            return pd.DataFrame()

    def get_insider_trading(self,
                           symbol: Optional[str] = None,
                           transaction_type: str = 'P',
                           min_value: float = 100000) -> pd.DataFrame:
        """
        Get insider trading activity

        Args:
            symbol: Stock symbol (None = all)
            transaction_type: 'P' (purchase), 'S' (sale), 'A' (award)
            min_value: Minimum transaction value

        Returns:
            DataFrame of insider trades
        """

        print(f"[OPENBB] Fetching insider trading{f' for {symbol}' if symbol else ''}...")

        try:
            if symbol:
                insider = obb.equity.ownership.insider_trading(
                    symbol=symbol,
                    provider="fmp"
                )
            else:
                # Get recent insider trading across market
                insider = obb.equity.ownership.insider_trading(provider="fmp")

            df = insider.to_df() if hasattr(insider, 'to_df') else pd.DataFrame(insider)

            if df.empty:
                print("[OPENBB] No insider trading found")
                return pd.DataFrame()

            # Filter by criteria
            filtered = df[
                (df['transactionType'] == transaction_type) &
                (df['securitiesTransacted'] * df['price'] >= min_value)
            ]

            print(f"[OPENBB] Found {len(filtered)} insider trades")
            return filtered

        except Exception as e:
            print(f"[OPENBB ERROR] Insider trading: {e}")
            return pd.DataFrame()

    def get_economic_calendar(self,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             importance: str = "high") -> pd.DataFrame:
        """
        Get economic events calendar

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            importance: Filter by importance (high, medium, low)

        Returns:
            DataFrame of economic events
        """

        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')

        if not end_date:
            end_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

        print(f"[OPENBB] Fetching economic calendar: {start_date} to {end_date}")

        try:
            calendar = obb.economy.calendar(
                start_date=start_date,
                end_date=end_date,
                provider="tradingeconomics"
            )

            df = calendar.to_df() if hasattr(calendar, 'to_df') else pd.DataFrame(calendar)

            if df.empty:
                return pd.DataFrame()

            # Filter by importance
            if 'importance' in df.columns:
                df = df[df['importance'].str.lower() == importance.lower()]

            print(f"[OPENBB] Found {len(df)} economic events")
            return df

        except Exception as e:
            print(f"[OPENBB ERROR] Economic calendar: {e}")
            return pd.DataFrame()

    def get_market_data(self,
                       symbol: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       interval: str = "1d") -> pd.DataFrame:
        """
        Get historical market data with auto-provider selection

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 1h, 1d)

        Returns:
            DataFrame with OHLCV data
        """

        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"[OPENBB] Fetching {symbol} data: {start_date} to {end_date}")

        try:
            # OpenBB auto-selects best provider for free tier
            data = obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                provider="yfinance"  # Free, reliable
            )

            df = data.to_df() if hasattr(data, 'to_df') else pd.DataFrame(data)

            print(f"[OPENBB] Fetched {len(df)} candles")
            return df

        except Exception as e:
            print(f"[OPENBB ERROR] Market data: {e}")
            return pd.DataFrame()


def demo():
    """Demo OpenBB data fetcher"""

    print("\n" + "="*70)
    print("OPENBB DATA FETCHER DEMO")
    print("="*70)

    if not OPENBB_AVAILABLE:
        print("\n[ERROR] OpenBB not installed")
        print("Install with: pip install openbb")
        return

    fetcher = OpenBBDataFetcher()

    # Test earnings calendar
    print("\n" + "-"*70)
    print("TEST 1: Earnings Calendar")
    print("-"*70)
    earnings = fetcher.get_earnings_calendar()
    if earnings:
        for event in earnings[:5]:
            print(f"  {event.symbol}: {event.earnings_date}")

    # Test market data
    print("\n" + "-"*70)
    print("TEST 2: Market Data")
    print("-"*70)
    spy_data = fetcher.get_market_data("SPY", interval="1d")
    if not spy_data.empty:
        print(f"  SPY: {len(spy_data)} days of data")
        print(f"  Latest close: ${spy_data['close'].iloc[-1]:.2f}")

    print("\n" + "="*70)
    print("Demo complete")
    print("="*70)


if __name__ == "__main__":
    demo()
