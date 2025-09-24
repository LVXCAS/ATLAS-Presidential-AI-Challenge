import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import concurrent.futures
import threading
from collections import defaultdict
import yfinance as yf
import time
import json
from pathlib import Path

try:
    import openbb_terminal.sdk as openbb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    logging.warning("OpenBB SDK not available")

from event_bus import TradingEventBus, Event, Priority


class OpportunityType(Enum):
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLUME_SPIKE = "volume_spike"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    EARNINGS_PLAY = "earnings_play"
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    BOLLINGER_SQUEEZE = "bollinger_squeeze"


class DataSource(Enum):
    YFINANCE = "yfinance"
    POLYGON = "polygon"
    OPENBB = "openbb"
    ALPHA_VANTAGE = "alpha_vantage"


@dataclass
class ScanFilter:
    min_price: float = 1.0
    max_price: float = 1000.0
    min_volume: int = 100000
    min_market_cap: float = 100_000_000
    max_market_cap: float = 500_000_000_000
    sectors: List[str] = field(default_factory=list)
    exchanges: List[str] = field(default_factory=lambda: ['NASDAQ', 'NYSE'])
    exclude_etfs: bool = True
    exclude_penny_stocks: bool = True
    min_avg_volume_20d: int = 500000
    
    def matches(self, stock_data: Dict[str, Any]) -> bool:
        price = stock_data.get('price', 0)
        volume = stock_data.get('volume', 0)
        market_cap = stock_data.get('market_cap', 0)
        
        if not (self.min_price <= price <= self.max_price):
            return False
        if volume < self.min_volume:
            return False
        if not (self.min_market_cap <= market_cap <= self.max_market_cap):
            return False
        if self.exclude_penny_stocks and price < 5.0:
            return False
        if self.sectors and stock_data.get('sector') not in self.sectors:
            return False
        if self.exchanges and stock_data.get('exchange') not in self.exchanges:
            return False
        
        return True


@dataclass 
class TradingOpportunity:
    symbol: str
    opportunity_type: OpportunityType
    confidence: float
    target_price: float
    stop_loss: float
    timeframe: str
    volume: int
    price: float
    change_percent: float
    discovered_at: datetime
    data_source: DataSource
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    fundamental_data: Dict[str, Any] = field(default_factory=dict)
    risk_reward_ratio: float = 0.0
    
    def __post_init__(self):
        if self.target_price > 0 and self.stop_loss > 0:
            potential_gain = abs(self.target_price - self.price)
            potential_loss = abs(self.price - self.stop_loss)
            if potential_loss > 0:
                self.risk_reward_ratio = potential_gain / potential_loss
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'opportunity_type': self.opportunity_type.value,
            'confidence': self.confidence,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'timeframe': self.timeframe,
            'volume': self.volume,
            'price': self.price,
            'change_percent': self.change_percent,
            'discovered_at': self.discovered_at.isoformat(),
            'data_source': self.data_source.value,
            'technical_indicators': self.technical_indicators,
            'fundamental_data': self.fundamental_data,
            'risk_reward_ratio': self.risk_reward_ratio
        }


class MarketDataSource:
    def __init__(self, source_type: DataSource):
        self.source_type = source_type
        self.logger = logging.getLogger(f"{__name__}.{source_type.value}")
        self.rate_limit = 0.1  # seconds between requests
        self.last_request = 0
        
    async def get_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
    
    async def get_bulk_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError
    
    def _rate_limit_wait(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()


class YFinanceSource(MarketDataSource):
    def __init__(self):
        super().__init__(DataSource.YFINANCE)
        self.rate_limit = 0.1
    
    async def get_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            self._rate_limit_wait()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="30d", interval="1d")
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            
            # Technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else sma_20
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = hist['Close'].rolling(window=bb_period).mean()
            bb_upper = bb_middle + (hist['Close'].rolling(window=bb_period).std() * bb_std)
            bb_lower = bb_middle - (hist['Close'].rolling(window=bb_period).std() * bb_std)
            
            # Volume analysis
            avg_volume_20d = hist['Volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = volume / avg_volume_20d if avg_volume_20d > 0 else 1
            
            # Price change
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_percent = ((current_price - prev_close) / prev_close) * 100
            
            return {
                'symbol': symbol,
                'price': float(current_price),
                'volume': int(volume),
                'change_percent': float(change_percent),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'exchange': info.get('exchange', ''),
                'avg_volume_20d': float(avg_volume_20d),
                'volume_ratio': float(volume_ratio),
                'technical_indicators': {
                    'sma_20': float(sma_20) if pd.notna(sma_20) else 0,
                    'sma_50': float(sma_50) if pd.notna(sma_50) else 0,
                    'rsi': float(current_rsi) if pd.notna(current_rsi) else 50,
                    'bb_upper': float(bb_upper.iloc[-1]) if not bb_upper.empty else 0,
                    'bb_lower': float(bb_lower.iloc[-1]) if not bb_lower.empty else 0,
                    'bb_middle': float(bb_middle.iloc[-1]) if not bb_middle.empty else 0
                },
                'fundamental_data': {
                    'pe_ratio': info.get('trailingPE', 0),
                    'pb_ratio': info.get('priceToBook', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'profit_margin': info.get('profitMargins', 0)
                }
            }
        except Exception as e:
            self.logger.warning(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def get_bulk_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        
        # Process in chunks to avoid overwhelming the API
        chunk_size = 10
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self._sync_get_stock_data, symbol): symbol 
                          for symbol in chunk}
                
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        data = future.result()
                        if data:
                            results[symbol] = data
                    except Exception as e:
                        self.logger.warning(f"Error processing {symbol}: {e}")
            
            # Rate limiting between chunks
            await asyncio.sleep(0.5)
        
        return results
    
    def _sync_get_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.get_stock_data(symbol))
        finally:
            loop.close()


class PolygonSource(MarketDataSource):
    def __init__(self, api_key: str):
        super().__init__(DataSource.POLYGON)
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit = 0.2  # Free tier limit
    
    async def get_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            async with aiohttp.ClientSession() as session:
                # Get current price and volume
                ticker_url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
                params = {'apikey': self.api_key}
                
                async with session.get(ticker_url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    ticker_data = data.get('results', {})
                    
                    if not ticker_data:
                        return None
                    
                    day = ticker_data.get('day', {})
                    prev_day = ticker_data.get('prevDay', {})
                    
                    current_price = day.get('c', 0)
                    volume = day.get('v', 0)
                    change_percent = ((current_price - prev_day.get('c', current_price)) / 
                                    prev_day.get('c', current_price)) * 100 if prev_day.get('c') else 0
                    
                    return {
                        'symbol': symbol,
                        'price': float(current_price),
                        'volume': int(volume),
                        'change_percent': float(change_percent),
                        'market_cap': 0,  # Polygon doesn't provide this in snapshot
                        'sector': '',
                        'industry': '',
                        'exchange': 'US',
                        'avg_volume_20d': volume,  # Simplified
                        'volume_ratio': 1.0,
                        'technical_indicators': {},
                        'fundamental_data': {}
                    }
        except Exception as e:
            self.logger.warning(f"Error fetching Polygon data for {symbol}: {e}")
            return None
    
    async def get_bulk_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        
        for symbol in symbols:
            await asyncio.sleep(self.rate_limit)
            data = await self.get_stock_data(symbol)
            if data:
                results[symbol] = data
        
        return results


class OpportunityDetector:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.OpportunityDetector")
    
    def detect_opportunities(self, stock_data: Dict[str, Any], 
                           data_source: DataSource) -> List[TradingOpportunity]:
        opportunities = []
        symbol = stock_data['symbol']
        price = stock_data['price']
        volume = stock_data['volume']
        change_percent = stock_data['change_percent']
        technical = stock_data.get('technical_indicators', {})
        
        # Breakout detection
        if self._is_breakout(stock_data):
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                opportunity_type=OpportunityType.BREAKOUT,
                confidence=self._calculate_breakout_confidence(stock_data),
                target_price=price * 1.1,
                stop_loss=price * 0.95,
                timeframe="1-5 days",
                volume=volume,
                price=price,
                change_percent=change_percent,
                discovered_at=datetime.now(),
                data_source=data_source,
                technical_indicators=technical
            ))
        
        # Momentum detection
        if self._is_momentum_play(stock_data):
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                opportunity_type=OpportunityType.MOMENTUM,
                confidence=self._calculate_momentum_confidence(stock_data),
                target_price=price * 1.08,
                stop_loss=price * 0.97,
                timeframe="1-3 days",
                volume=volume,
                price=price,
                change_percent=change_percent,
                discovered_at=datetime.now(),
                data_source=data_source,
                technical_indicators=technical
            ))
        
        # Volume spike detection
        volume_ratio = stock_data.get('volume_ratio', 1)
        if volume_ratio > 2.0:
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                opportunity_type=OpportunityType.VOLUME_SPIKE,
                confidence=min(0.9, volume_ratio / 5.0),
                target_price=price * 1.05,
                stop_loss=price * 0.98,
                timeframe="intraday",
                volume=volume,
                price=price,
                change_percent=change_percent,
                discovered_at=datetime.now(),
                data_source=data_source,
                technical_indicators=technical
            ))
        
        # RSI-based opportunities
        rsi = technical.get('rsi', 50)
        if rsi < 30:  # Oversold
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                opportunity_type=OpportunityType.RSI_OVERSOLD,
                confidence=max(0.3, (30 - rsi) / 30),
                target_price=price * 1.06,
                stop_loss=price * 0.96,
                timeframe="3-7 days",
                volume=volume,
                price=price,
                change_percent=change_percent,
                discovered_at=datetime.now(),
                data_source=data_source,
                technical_indicators=technical
            ))
        elif rsi > 70:  # Overbought
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                opportunity_type=OpportunityType.RSI_OVERBOUGHT,
                confidence=max(0.3, (rsi - 70) / 30),
                target_price=price * 0.94,
                stop_loss=price * 1.04,
                timeframe="1-3 days",
                volume=volume,
                price=price,
                change_percent=change_percent,
                discovered_at=datetime.now(),
                data_source=data_source,
                technical_indicators=technical
            ))
        
        # Gap detection
        if abs(change_percent) > 5:
            opp_type = OpportunityType.GAP_UP if change_percent > 0 else OpportunityType.GAP_DOWN
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                opportunity_type=opp_type,
                confidence=min(0.8, abs(change_percent) / 10),
                target_price=price * (1.03 if change_percent > 0 else 0.97),
                stop_loss=price * (0.98 if change_percent > 0 else 1.02),
                timeframe="intraday",
                volume=volume,
                price=price,
                change_percent=change_percent,
                discovered_at=datetime.now(),
                data_source=data_source,
                technical_indicators=technical
            ))
        
        return opportunities
    
    def _is_breakout(self, stock_data: Dict[str, Any]) -> bool:
        technical = stock_data.get('technical_indicators', {})
        price = stock_data['price']
        sma_20 = technical.get('sma_20', 0)
        sma_50 = technical.get('sma_50', 0)
        volume_ratio = stock_data.get('volume_ratio', 1)
        
        # Price above both SMAs with volume confirmation
        return (price > sma_20 > sma_50 and volume_ratio > 1.5 and 
                stock_data['change_percent'] > 2)
    
    def _is_momentum_play(self, stock_data: Dict[str, Any]) -> bool:
        change_percent = stock_data['change_percent']
        volume_ratio = stock_data.get('volume_ratio', 1)
        rsi = stock_data.get('technical_indicators', {}).get('rsi', 50)
        
        return (change_percent > 3 and volume_ratio > 1.3 and 
                30 < rsi < 70)  # Not oversold/overbought
    
    def _calculate_breakout_confidence(self, stock_data: Dict[str, Any]) -> float:
        volume_ratio = stock_data.get('volume_ratio', 1)
        change_percent = stock_data['change_percent']
        
        volume_score = min(1.0, volume_ratio / 3.0)
        price_score = min(1.0, change_percent / 5.0)
        
        return (volume_score + price_score) / 2
    
    def _calculate_momentum_confidence(self, stock_data: Dict[str, Any]) -> float:
        change_percent = stock_data['change_percent']
        volume_ratio = stock_data.get('volume_ratio', 1)
        
        return min(0.9, (change_percent + volume_ratio) / 10)


class MarketScanner:
    def __init__(self, 
                 event_bus: Optional[TradingEventBus] = None,
                 polygon_api_key: Optional[str] = None,
                 max_workers: int = 20):
        
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        self.max_workers = max_workers
        
        # Initialize data sources
        self.data_sources: List[MarketDataSource] = [
            YFinanceSource()
        ]
        
        if polygon_api_key:
            self.data_sources.append(PolygonSource(polygon_api_key))
        
        self.opportunity_detector = OpportunityDetector()
        self.scan_results: Dict[str, List[TradingOpportunity]] = {}
        
        # Stock universe
        self.stock_universe: List[str] = []
        self._load_stock_universe()
    
    def _load_stock_universe(self):
        """Load a comprehensive list of stocks to scan"""
        try:
            # S&P 500 stocks (simplified list - in production, load from file/API)
            sp500_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 
                'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
                'PFE', 'AVGO', 'KO', 'LLY', 'TMO', 'COST', 'PEP', 'WMT', 'MRK',
                'DHR', 'VZ', 'ABT', 'ADBE', 'NFLX', 'CRM', 'ACN', 'NKE', 'TXN',
                'LIN', 'RTX', 'QCOM', 'PM', 'HON', 'SBUX', 'T', 'LOW', 'UPS',
                'AMGN', 'IBM', 'SPGI', 'CAT', 'GS', 'NEE', 'AMD', 'INTU', 'AXP',
                'BLK', 'DE', 'BKNG', 'MDT', 'ISRG', 'TJX', 'GE', 'AMAT', 'SYK',
                'MMM', 'ADP', 'MO', 'GILD', 'LRCX', 'CI', 'CB', 'CME', 'USB',
                'PLD', 'TGT', 'CSX', 'MDLZ', 'REGN', 'SO', 'MU', 'ZTS', 'EOG',
                'DUK', 'SHW', 'BSX', 'ITW', 'CL', 'APD', 'BMY', 'HUM', 'AON',
                'KLAC', 'WM', 'GD', 'INTC', 'COP', 'MMC', 'EMR', 'FDX', 'NSC',
                'ADI', 'ECL', 'PNC', 'ICE', 'FCX', 'EQIX', 'MAR', 'GM', 'SNPS',
                'DFS', 'CMG', 'EL', 'NOW', 'NXPI', 'MCK', 'PSA', 'ROP', 'SLB',
                'TFC', 'AEP', 'EW', 'AFL', 'APH', 'CDNS', 'MET', 'PXD', 'FIS',
                'CCI', 'HCA', 'COF', 'F', 'MSI', 'CARR', 'ORLY', 'SRE', 'WFC',
                'KMB', 'BIIB', 'IQV', 'NOC', 'CTAS', 'BDX', 'OXY', 'AZO', 'D',
                'INFO', 'JCI', 'MPC', 'ALL', 'CMI', 'GPN', 'PEG', 'HPQ', 'FAST',
                'TRV', 'EXC', 'KMI', 'PAYX', 'ED', 'FISV', 'PRU', 'ROST', 'GLW',
                'WBA', 'VRSK', 'EA', 'WELL', 'CTSH', 'VRTX', 'ALGN', 'YUM',
                'DXCM', 'OTIS', 'DAL', 'HLT', 'MCHP', 'KHC', 'XEL', 'DLR',
                'HSY', 'TEL', 'DOW', 'DD', 'EBAY', 'AMT', 'DG', 'ANSS', 'EXR',
                'CHD', 'FTNT', 'AWK', 'MLM', 'RMD', 'ILMN', 'PPG', 'CERN', 'O',
                'FRC', 'ETN', 'HPE', 'STZ', 'ADM', 'A', 'WEC', 'MNST', 'LVS',
                'PSX', 'SPG', 'ES', 'ROK', 'LYB', 'SBAC', 'KEYS', 'LHX', 'IT',
                'GWW', 'TSCO', 'AEE', 'AVB', 'WY', 'PAYC', 'ANET', 'ARE', 'PCG',
                'IEX', 'EIX', 'MTB', 'RSG', 'NTRS', 'VRSN', 'CAH', 'CLX', 'DTE',
                'VMC', 'STT', 'HBAN', 'FTV', 'IDXX', 'ETR', 'PKI', 'TROW', 'CNP',
                'WMB', 'RF', 'K', 'FE', 'ULTA', 'MPWR', 'UAL', 'NTAP', 'HOLX',
                'DRE', 'SWKS', 'PPL', 'ZBRA', 'DOV', 'WAT', 'PEAK', 'COO', 'PFG',
                'AVY', 'DGX', 'LDOS', 'AKAM', 'CFG', 'EXPD', 'CPRT', 'JBHT',
                'TDY', 'NVR', 'LH', 'EQR', 'LUV', 'ABMD', 'CHRW', 'STE', 'CTLT',
                'TYL', 'URI', 'FLT', 'TECH', 'SIVB', 'BXP', 'MAS', 'CAG', 'VTRS',
                'XRAY', 'UDR', 'BF-B', 'GRMN', 'J', 'TXT', 'CE', 'SYF', 'JKHY',
                'MAA', 'PKG', 'POOL', 'TPG', 'HIG', 'FFIV', 'LNT', 'WST', 'IP',
                'UHS', 'ATO', 'CINF', 'HSIC', 'AOS', 'CMS', 'TAP', 'PNR', 'LW',
                'NUE', 'KIM', 'NDAQ', 'REG', 'ALLE', 'IRM', 'DISH', 'SJM', 'APA',
                'HRL', 'PBCT', 'MKC', 'INCY', 'VFC', 'CPB', 'OMC', 'IPG', 'HAS'
            ]
            
            # Add Russell 2000 representative stocks
            russell_stocks = [
                'PENN', 'RKT', 'SKLZ', 'CRSR', 'PLTR', 'WISH', 'CLOV', 'SOFI',
                'HOOD', 'LCID', 'RIVN', 'ABNB', 'COIN', 'RBLX', 'SNOW', 'ZM',
                'PTON', 'TDOC', 'ROKU', 'SQ', 'SHOP', 'TWLO', 'OKTA', 'DDOG',
                'CRWD', 'ZS', 'NET', 'FSLY', 'MDB', 'TEAM', 'WDAY', 'SPLK',
                'VEEV', 'CZR', 'DKNG', 'UBER', 'LYFT', 'DASH', 'PINS', 'SNAP',
                'SPOT', 'SQ', 'PYPL', 'ZG', 'ZILLOW', 'DOCU', 'BYND', 'MRNA',
                'BNTX', 'PFE', 'JNJ', 'GILD', 'REGN', 'VRTX', 'BIIB', 'AMGN'
            ]
            
            self.stock_universe = list(set(sp500_stocks + russell_stocks))
            self.logger.info(f"Loaded {len(self.stock_universe)} stocks for scanning")
            
        except Exception as e:
            self.logger.error(f"Error loading stock universe: {e}")
            self.stock_universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Fallback
    
    async def scan_market(self, 
                         scan_filter: Optional[ScanFilter] = None,
                         max_symbols: int = 2000) -> Dict[str, List[TradingOpportunity]]:
        
        self.logger.info(f"Starting market scan of {min(max_symbols, len(self.stock_universe))} symbols")
        
        if scan_filter is None:
            scan_filter = ScanFilter()
        
        symbols_to_scan = self.stock_universe[:max_symbols]
        all_opportunities = {}
        
        # Process symbols in parallel chunks
        chunk_size = min(50, max_symbols // self.max_workers)
        chunks = [symbols_to_scan[i:i+chunk_size] 
                 for i in range(0, len(symbols_to_scan), chunk_size)]
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_chunk(chunk: List[str], source: MarketDataSource):
            async with semaphore:
                try:
                    stock_data = await source.get_bulk_data(chunk)
                    chunk_opportunities = {}
                    
                    for symbol, data in stock_data.items():
                        if scan_filter.matches(data):
                            opportunities = self.opportunity_detector.detect_opportunities(
                                data, source.source_type
                            )
                            if opportunities:
                                chunk_opportunities[symbol] = opportunities
                                
                                # Publish high-confidence opportunities immediately
                                for opp in opportunities:
                                    if opp.confidence > 0.7 and self.event_bus:
                                        await self.event_bus.publish(
                                            "trading_opportunity_discovered",
                                            opp.to_dict(),
                                            priority=Priority.HIGH
                                        )
                    
                    return chunk_opportunities
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk: {e}")
                    return {}
        
        # Process all chunks across all data sources
        tasks = []
        for source in self.data_sources:
            for chunk in chunks:
                task = asyncio.create_task(process_chunk(chunk, source))
                tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        for result in results:
            if isinstance(result, dict):
                for symbol, opportunities in result.items():
                    if symbol not in all_opportunities:
                        all_opportunities[symbol] = []
                    all_opportunities[symbol].extend(opportunities)
        
        # Deduplicate and rank opportunities
        final_opportunities = self._process_and_rank_opportunities(all_opportunities)
        
        self.scan_results = final_opportunities
        
        # Publish scan completion
        if self.event_bus:
            await self.event_bus.publish(
                "market_scan_completed",
                {
                    'total_symbols_scanned': len([s for chunk_result in results 
                                                if isinstance(chunk_result, dict) 
                                                for s in chunk_result]),
                    'opportunities_found': sum(len(opps) for opps in final_opportunities.values()),
                    'unique_symbols': len(final_opportunities),
                    'scan_duration': time.time(),
                    'top_opportunities': [opp.to_dict() for opps in list(final_opportunities.values())[:10] for opp in opps[:3]]
                },
                priority=Priority.NORMAL
            )
        
        self.logger.info(f"Market scan completed. Found {sum(len(opps) for opps in final_opportunities.values())} opportunities across {len(final_opportunities)} symbols")
        
        return final_opportunities
    
    def _process_and_rank_opportunities(self, 
                                     opportunities: Dict[str, List[TradingOpportunity]]) -> Dict[str, List[TradingOpportunity]]:
        processed = {}
        
        for symbol, opps in opportunities.items():
            # Remove duplicates based on opportunity type and confidence
            unique_opps = {}
            for opp in opps:
                key = (opp.opportunity_type, opp.timeframe)
                if key not in unique_opps or opp.confidence > unique_opps[key].confidence:
                    unique_opps[key] = opp
            
            # Sort by confidence and risk/reward ratio
            sorted_opps = sorted(
                unique_opps.values(),
                key=lambda x: (x.confidence, x.risk_reward_ratio),
                reverse=True
            )
            
            # Keep top 5 opportunities per symbol
            processed[symbol] = sorted_opps[:5]
        
        return processed
    
    async def get_top_opportunities(self, 
                                   limit: int = 20,
                                   min_confidence: float = 0.6) -> List[TradingOpportunity]:
        all_opportunities = []
        
        for symbol, opportunities in self.scan_results.items():
            for opp in opportunities:
                if opp.confidence >= min_confidence:
                    all_opportunities.append(opp)
        
        # Sort by confidence and risk/reward
        all_opportunities.sort(
            key=lambda x: (x.confidence, x.risk_reward_ratio),
            reverse=True
        )
        
        return all_opportunities[:limit]
    
    async def continuous_scan(self, 
                            interval_minutes: int = 15,
                            scan_filter: Optional[ScanFilter] = None):
        """Run continuous market scanning"""
        self.logger.info(f"Starting continuous scan with {interval_minutes} minute intervals")
        
        while True:
            try:
                start_time = time.time()
                await self.scan_market(scan_filter)
                
                scan_duration = time.time() - start_time
                self.logger.info(f"Scan completed in {scan_duration:.2f} seconds")
                
                # Wait for next scan
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in continuous scan: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def save_scan_results(self, filename: str = "scan_results.json"):
        """Save scan results to JSON file"""
        try:
            results_dict = {}
            for symbol, opportunities in self.scan_results.items():
                results_dict[symbol] = [opp.to_dict() for opp in opportunities]
            
            with open(filename, 'w') as f:
                json.dump({
                    'scan_timestamp': datetime.now().isoformat(),
                    'total_symbols': len(self.scan_results),
                    'total_opportunities': sum(len(opps) for opps in self.scan_results.values()),
                    'results': results_dict
                }, f, indent=2)
            
            self.logger.info(f"Scan results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving scan results: {e}")


# Example usage and testing
async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create event bus
    event_bus = TradingEventBus()
    await event_bus.start()
    
    # Create market scanner
    scanner = MarketScanner(event_bus=event_bus)
    
    # Define scan filters
    scan_filter = ScanFilter(
        min_price=5.0,
        max_price=500.0,
        min_volume=500000,
        min_market_cap=500_000_000
    )
    
    try:
        # Run scan
        results = await scanner.scan_market(scan_filter, max_symbols=100)
        
        # Get top opportunities
        top_opportunities = await scanner.get_top_opportunities(limit=10)
        
        print(f"Found {len(top_opportunities)} high-confidence opportunities:")
        for opp in top_opportunities:
            print(f"  {opp.symbol}: {opp.opportunity_type.value} - {opp.confidence:.2f} confidence")
        
        # Save results
        scanner.save_scan_results()
        
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())