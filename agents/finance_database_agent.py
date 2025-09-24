"""
FinanceDatabase Integration Agent
Integrates JerBouma's FinanceDatabase for comprehensive market data and ML training enhancement
Provides advanced screening, fundamental data, and market universe expansion
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# FinanceDatabase integration
try:
    import financedatabase as fdb
    FINANCE_DB_AVAILABLE = True
    print("FinanceDatabase successfully imported")
except ImportError as e:
    print(f"FinanceDatabase not available: {e}")
    FINANCE_DB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketUniverse:
    """Market universe data structure"""
    equities: Dict[str, Dict]
    etfs: Dict[str, Dict]
    indices: Dict[str, Dict]
    sectors: Set[str]
    industries: Set[str]
    countries: Set[str]
    market_caps: Dict[str, str]

@dataclass
class ScreeningResult:
    """Advanced screening result"""
    symbol: str
    name: str
    sector: str
    industry: str
    country: str
    market_cap: str
    score: float
    factors: Dict[str, float]
    fundamental_data: Dict[str, Any]
    ml_features: Dict[str, float]

class FinanceDatabaseAgent:
    """
    Advanced FinanceDatabase integration for ML enhancement:
    - Comprehensive market universe (159k+ equities, ETFs, funds)
    - Advanced multi-factor screening
    - Fundamental data integration for ML features
    - Sector/industry analysis for relative strength
    - Market cap and geographic diversification
    - Enhanced training data for ML models
    """

    def __init__(self):
        self.name = "FinanceDatabase Agent"
        self.market_universe = None
        self.cache = {}
        self.screening_criteria = {}

        if FINANCE_DB_AVAILABLE:
            self._initialize_databases()

        logger.info("FinanceDatabase Agent initialized")

    def _initialize_databases(self):
        """Initialize FinanceDatabase connections (stocks, ETFs, funds only)"""
        try:
            self.equities_db = fdb.Equities()
            self.etfs_db = fdb.ETFs()
            self.funds_db = fdb.Funds()
            self.indices_db = fdb.Indices()

            # Only initialize what's available and needed for stock trading
            try:
                self.currencies_db = fdb.Currencies()
            except AttributeError:
                self.currencies_db = None
                logger.info("Currencies database not available")

            try:
                self.commodities_db = fdb.Commodities()
            except AttributeError:
                self.commodities_db = None
                logger.info("Commodities database not available")

            # Skip cryptocurrencies - not needed for stock trading
            self.cryptocurrencies_db = None

            logger.info("FinanceDatabase connections initialized (stocks, ETFs, funds, indices)")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    async def build_comprehensive_market_universe(self) -> MarketUniverse:
        """Build comprehensive market universe for ML training"""
        try:
            if not FINANCE_DB_AVAILABLE:
                return self._build_fallback_universe()

            # Get all equities data
            equities_data = self.equities_db.select()
            etfs_data = self.etfs_db.select()
            indices_data = self.indices_db.select()

            # Extract unique categories for analysis
            sectors = set()
            industries = set()
            countries = set()
            market_caps = {}

            # Process equities for categories
            for symbol, data in equities_data.items():
                if isinstance(data, dict):
                    sector = data.get('sector', '')
                    industry = data.get('industry', '')
                    country = data.get('country', '')
                    market_cap = data.get('market_cap', '')

                    if sector:
                        sectors.add(sector)
                    if industry:
                        industries.add(industry)
                    if country:
                        countries.add(country)
                    if market_cap:
                        market_caps[symbol] = market_cap

            universe = MarketUniverse(
                equities=equities_data,
                etfs=etfs_data,
                indices=indices_data,
                sectors=sectors,
                industries=industries,
                countries=countries,
                market_caps=market_caps
            )

            self.market_universe = universe
            logger.info(f"Market universe built: {len(equities_data)} equities, {len(etfs_data)} ETFs")

            return universe

        except Exception as e:
            logger.error(f"Market universe building error: {e}")
            return self._build_fallback_universe()

    def _build_fallback_universe(self) -> MarketUniverse:
        """Fallback universe if FinanceDatabase unavailable"""
        # Use common tickers as fallback
        fallback_equities = {
            'AAPL': {'sector': 'Technology', 'industry': 'Consumer Electronics', 'country': 'US'},
            'MSFT': {'sector': 'Technology', 'industry': 'Software', 'country': 'US'},
            'GOOGL': {'sector': 'Technology', 'industry': 'Internet Content', 'country': 'US'},
            'AMZN': {'sector': 'Consumer Discretionary', 'industry': 'E-commerce', 'country': 'US'},
            'TSLA': {'sector': 'Consumer Discretionary', 'industry': 'Auto Manufacturing', 'country': 'US'},
            'META': {'sector': 'Technology', 'industry': 'Social Media', 'country': 'US'},
            'NVDA': {'sector': 'Technology', 'industry': 'Semiconductors', 'country': 'US'},
            'JPM': {'sector': 'Financial Services', 'industry': 'Banking', 'country': 'US'},
            'JNJ': {'sector': 'Healthcare', 'industry': 'Pharmaceuticals', 'country': 'US'},
            'V': {'sector': 'Financial Services', 'industry': 'Payment Processing', 'country': 'US'}
        }

        return MarketUniverse(
            equities=fallback_equities,
            etfs={},
            indices={},
            sectors={'Technology', 'Consumer Discretionary', 'Financial Services', 'Healthcare'},
            industries={'Software', 'E-commerce', 'Banking', 'Pharmaceuticals'},
            countries={'US'},
            market_caps={}
        )

    async def advanced_multi_factor_screening(self,
                                            min_market_cap: float = 1e9,
                                            sectors: List[str] = None,
                                            countries: List[str] = None,
                                            max_results: int = 50) -> List[ScreeningResult]:
        """Advanced multi-factor screening for ML training data"""
        try:
            if not self.market_universe:
                await self.build_comprehensive_market_universe()

            results = []
            processed_count = 0

            # Filter equities based on criteria
            for symbol, data in self.market_universe.equities.items():
                if processed_count >= max_results * 2:  # Process more to filter down
                    break

                try:
                    # Basic filters
                    if not isinstance(data, dict):
                        continue

                    sector = data.get('sector', '')
                    country = data.get('country', '')

                    # Apply filters
                    if sectors and sector not in sectors:
                        continue
                    if countries and country not in countries:
                        continue

                    # Get additional data for scoring
                    score, factors, fundamental_data, ml_features = await self._calculate_screening_score(symbol, data)

                    if score > 0.5:  # Minimum score threshold
                        result = ScreeningResult(
                            symbol=symbol,
                            name=data.get('short_name', symbol),
                            sector=sector,
                            industry=data.get('industry', ''),
                            country=country,
                            market_cap=data.get('market_cap', ''),
                            score=score,
                            factors=factors,
                            fundamental_data=fundamental_data,
                            ml_features=ml_features
                        )
                        results.append(result)

                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue

            # Sort by score and return top results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:max_results]

        except Exception as e:
            logger.error(f"Multi-factor screening error: {e}")
            return []

    async def _calculate_screening_score(self, symbol: str, base_data: Dict) -> Tuple[float, Dict, Dict, Dict]:
        """Calculate comprehensive screening score with ML features"""
        try:
            # Initialize scores
            factors = {
                'technical_score': 0.0,
                'fundamental_score': 0.0,
                'liquidity_score': 0.0,
                'momentum_score': 0.0,
                'quality_score': 0.0
            }

            fundamental_data = {}
            ml_features = {}

            # Get market data for technical analysis
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                hist = stock.history(period='6mo')

                if hist.empty:
                    return 0.0, factors, fundamental_data, ml_features

                # Technical factors
                current_price = hist['Close'].iloc[-1]
                sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]

                # Technical score
                if current_price > sma_20 > sma_50:
                    factors['technical_score'] = 0.8
                elif current_price > sma_20:
                    factors['technical_score'] = 0.6
                else:
                    factors['technical_score'] = 0.3

                # Momentum score
                if len(hist) >= 20:
                    momentum_20d = (current_price / hist['Close'].iloc[-20] - 1) * 100
                    if momentum_20d > 5:
                        factors['momentum_score'] = 0.8
                    elif momentum_20d > 0:
                        factors['momentum_score'] = 0.6
                    else:
                        factors['momentum_score'] = 0.3

                # Liquidity score (volume)
                avg_volume = hist['Volume'].mean()
                if avg_volume > 1000000:
                    factors['liquidity_score'] = 0.8
                elif avg_volume > 100000:
                    factors['liquidity_score'] = 0.6
                else:
                    factors['liquidity_score'] = 0.3

                # Fundamental data from yfinance
                fundamental_data = {
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'revenue_growth': info.get('revenueGrowth', 0)
                }

                # Fundamental score
                pe_ratio = fundamental_data.get('pe_ratio', 0)
                roe = fundamental_data.get('roe', 0)
                profit_margin = fundamental_data.get('profit_margin', 0)

                fund_score = 0.5  # Base score
                if 0 < pe_ratio < 25:  # Reasonable P/E
                    fund_score += 0.1
                if roe > 0.15:  # Good ROE
                    fund_score += 0.2
                if profit_margin > 0.1:  # Good profit margin
                    fund_score += 0.2

                factors['fundamental_score'] = min(fund_score, 1.0)

                # Quality score (combine fundamental factors)
                quality_factors = [
                    1 if fundamental_data.get('debt_to_equity', 100) < 50 else 0,
                    1 if fundamental_data.get('roe', 0) > 0.1 else 0,
                    1 if fundamental_data.get('profit_margin', 0) > 0.05 else 0
                ]
                factors['quality_score'] = sum(quality_factors) / len(quality_factors)

                # ML Features for training
                ml_features = {
                    'price_vs_sma20': (current_price / sma_20 - 1) * 100 if sma_20 > 0 else 0,
                    'price_vs_sma50': (current_price / sma_50 - 1) * 100 if sma_50 > 0 else 0,
                    'volume_ratio': avg_volume / hist['Volume'].std() if hist['Volume'].std() > 0 else 1,
                    'volatility_20d': hist['Close'].pct_change().rolling(20).std().iloc[-1] * 100,
                    'rsi_14': self._calculate_rsi(hist['Close'], 14),
                    'log_market_cap': np.log(fundamental_data.get('market_cap', 1)),
                    'pe_ratio_normalized': min(pe_ratio / 25, 2) if pe_ratio > 0 else 0,
                    'roe_normalized': min(roe * 5, 1) if roe > 0 else 0,
                    'momentum_20d': momentum_20d if 'momentum_20d' in locals() else 0
                }

            except Exception as e:
                logger.warning(f"Data retrieval error for {symbol}: {e}")
                return 0.0, factors, fundamental_data, ml_features

            # Calculate composite score
            weights = {
                'technical_score': 0.25,
                'fundamental_score': 0.25,
                'liquidity_score': 0.20,
                'momentum_score': 0.20,
                'quality_score': 0.10
            }

            composite_score = sum(factors[factor] * weights[factor] for factor in weights)

            return composite_score, factors, fundamental_data, ml_features

        except Exception as e:
            logger.error(f"Scoring error for {symbol}: {e}")
            return 0.0, {}, {}, {}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

    async def get_sector_analysis(self, sector: str) -> Dict[str, Any]:
        """Get comprehensive sector analysis for ML training"""
        try:
            if not self.market_universe:
                await self.build_comprehensive_market_universe()

            # Filter stocks by sector
            sector_stocks = {
                symbol: data for symbol, data in self.market_universe.equities.items()
                if isinstance(data, dict) and data.get('sector', '') == sector
            }

            if not sector_stocks:
                return {}

            # Analyze sector performance
            sector_analysis = {
                'sector': sector,
                'total_stocks': len(sector_stocks),
                'sample_stocks': list(sector_stocks.keys())[:20],
                'industries': set(data.get('industry', '') for data in sector_stocks.values() if data.get('industry')),
                'countries': set(data.get('country', '') for data in sector_stocks.values() if data.get('country'))
            }

            return sector_analysis

        except Exception as e:
            logger.error(f"Sector analysis error for {sector}: {e}")
            return {}

    async def enhance_ml_training_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Enhance ML training data with FinanceDatabase information"""
        try:
            enhanced_data = {}

            for symbol in symbols:
                # Get base data from FinanceDatabase
                equity_data = self.market_universe.equities.get(symbol, {}) if self.market_universe else {}

                # Calculate ML features
                _, factors, fundamental_data, ml_features = await self._calculate_screening_score(symbol, equity_data)

                enhanced_data[symbol] = {
                    'base_info': equity_data,
                    'screening_factors': factors,
                    'fundamental_data': fundamental_data,
                    'ml_features': ml_features,
                    'sector': equity_data.get('sector', 'Unknown'),
                    'industry': equity_data.get('industry', 'Unknown'),
                    'country': equity_data.get('country', 'Unknown')
                }

            return enhanced_data

        except Exception as e:
            logger.error(f"ML training data enhancement error: {e}")
            return {}

    async def get_market_universe_stats(self) -> Dict[str, Any]:
        """Get comprehensive market universe statistics"""
        try:
            if not self.market_universe:
                await self.build_comprehensive_market_universe()

            stats = {
                'total_equities': len(self.market_universe.equities),
                'total_etfs': len(self.market_universe.etfs),
                'total_indices': len(self.market_universe.indices),
                'unique_sectors': len(self.market_universe.sectors),
                'unique_industries': len(self.market_universe.industries),
                'unique_countries': len(self.market_universe.countries),
                'sectors_list': list(self.market_universe.sectors)[:10],  # Top 10
                'industries_list': list(self.market_universe.industries)[:10],  # Top 10
                'countries_list': list(self.market_universe.countries)[:10]  # Top 10
            }

            return stats

        except Exception as e:
            logger.error(f"Market universe stats error: {e}")
            return {}

# Create singleton instance
finance_database_agent = FinanceDatabaseAgent()