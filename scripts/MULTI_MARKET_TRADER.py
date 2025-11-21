"""
UNIFIED MULTI-MARKET TRADING PROGRAM
Consolidates FOREX, FUTURES, and CRYPTO trading into ONE system
Runs as a single process instead of 3 separate bots

Architecture:
- ForexMarketHandler: Handles EUR_USD, USD_JPY, GBP_USD, GBP_JPY (E8 Markets)
- FuturesMarketHandler: Handles ES, NQ, CL, GC contracts (Apex Trader Funding)
- CryptoMarketHandler: Handles BTC/USD, ETH/USD (Crypto Fund Trader)
- UnifiedTradingEngine: Orchestrates all 3 markets in one main loop

Shared Libraries: technical_analysis.py, kelly_criterion.py, multi_timeframe.py
"""
import os
import sys
import time
import threading
from datetime import datetime
import numpy as np

# Import shared libraries
from SHARED.technical_analysis import ta
from SHARED.kelly_criterion import kelly
from SHARED.multi_timeframe import mtf
from SHARED.ai_confirmation import ai_agent  # AI confirmation layer (DeepSeek + MiniMax)
from SHARED.trade_logger import trade_logger  # Comprehensive logging for A/B testing

# OANDA API for Forex
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
# FOREX MARKET HANDLER - E8 MARKETS
# ============================================================================

class ForexMarketHandler:
    """
    Handles forex trading for E8 Markets prop firm
    Pairs: EUR_USD, USD_JPY, GBP_USD, GBP_JPY
    """

    def __init__(self):
        self.name = "FOREX (E8 Markets)"

        # OANDA API
        self.oanda_token = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

        if not self.oanda_token:
            raise ValueError("Missing OANDA_API_KEY in .env")

        self.client = API(access_token=self.oanda_token, environment='practice')

        # Trading pairs
        self.pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'GBP_JPY']

        # Risk parameters
        self.min_score = 2.5
        self.risk_per_trade = 0.01  # 1%
        self.max_positions = 3
        self.leverage = 5

        # Time-based filters
        self.AVOID_HOURS = {
            'EUR_USD': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'GBP_USD': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'USD_JPY': [0, 1, 2, 3, 4, 16, 17, 18],
            'GBP_JPY': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18],
        }

        self.scan_interval = 3600  # 1 hour

    def get_data(self, pair, granularity='H1', count=200):
        """Fetch historical forex data from OANDA"""
        try:
            params = {"count": count, "granularity": granularity}
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = self.client.request(r)

            candles = response['candles']
            closes = [float(c['mid']['c']) for c in candles]
            highs = [float(c['mid']['h']) for c in candles]
            lows = [float(c['mid']['l']) for c in candles]

            return {
                'closes': np.array(closes),
                'highs': np.array(highs),
                'lows': np.array(lows),
                'current_price': closes[-1]
            }
        except Exception as e:
            print(f"    [ERROR] Getting data for {pair}: {e}")
            return None

    def calculate_score(self, pair, data):
        """Calculate trading score using shared TA library"""
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

        # Shared technical analysis
        rsi = ta.calculate_rsi(closes)
        macd = ta.calculate_macd(closes)
        ema_fast = ta.calculate_ema(closes, period=10)
        ema_slow = ta.calculate_ema(closes, period=21)
        adx = ta.calculate_adx(highs, lows, closes)
        atr = ta.calculate_atr(highs, lows, closes)

        # 4H trend confirmation (shared MTF)
        data_4h = self.get_data(pair, granularity='H4', count=100)
        if data_4h:
            trend_4h = mtf.get_higher_timeframe_trend(data_4h['closes'], current_price)
        else:
            trend_4h = 'neutral'

        # Score LONG signals
        long_score = 0
        long_signals = []

        if rsi < 30:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")
        elif rsi < 40:
            long_score += 1
            long_signals.append("RSI_LOW")

        if macd['macd'] > macd['signal'] and macd['histogram'] > 0:
            long_score += 2
            long_signals.append("MACD_BULL_CROSS")

        if current_price > ema_fast and ema_fast > ema_slow:
            long_score += 2
            long_signals.append("EMA_BULLISH")

        if adx > 25:
            long_score += 1
            long_signals.append("STRONG_TREND")

        if trend_4h == 'bullish':
            long_score += 2
            long_signals.append("4H_BULLISH_TREND")
        elif trend_4h == 'bearish':
            long_score -= 1.5

        # Score SHORT signals
        short_score = 0
        short_signals = []

        if rsi > 70:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")
        elif rsi > 60:
            short_score += 1
            short_signals.append("RSI_HIGH")

        if macd['macd'] < macd['signal'] and macd['histogram'] < 0:
            short_score += 2
            short_signals.append("MACD_BEAR_CROSS")

        if current_price < ema_fast and ema_fast < ema_slow:
            short_score += 2
            short_signals.append("EMA_BEARISH")

        if adx > 25:
            short_score += 1
            short_signals.append("STRONG_TREND")

        if trend_4h == 'bearish':
            short_score += 2
            short_signals.append("4H_BEARISH_TREND")
        elif trend_4h == 'bullish':
            short_score -= 1.5

        # Determine best direction
        if long_score > short_score and long_score >= self.min_score:
            return {
                'market': 'FOREX',
                'symbol': pair,
                'score': long_score,
                'direction': 'long',
                'signals': long_signals,
                'rsi': rsi,
                'adx': adx,
                'current_price': current_price
            }
        elif short_score >= self.min_score:
            return {
                'market': 'FOREX',
                'symbol': pair,
                'score': short_score,
                'direction': 'short',
                'signals': short_signals,
                'rsi': rsi,
                'adx': adx,
                'current_price': current_price
            }
        else:
            return {'score': 0, 'direction': None}

    def scan(self):
        """Scan all forex pairs for opportunities"""
        opportunities = []
        current_hour = datetime.now().hour

        for pair in self.pairs:
            # Time filter
            if current_hour in self.AVOID_HOURS.get(pair, []):
                print(f"    [SKIP] {pair} - Avoiding hour {current_hour}")
                continue

            data = self.get_data(pair)
            if data is None:
                continue

            result = self.calculate_score(pair, data)

            if result['score'] >= self.min_score:
                opportunities.append(result)
                print(f"    [FOUND] {pair} {result['direction'].upper()}: {result['score']:.1f}/10")
                print(f"            Signals: {', '.join(result['signals'])}")
            else:
                print(f"    [SKIP] {pair}: Score {result['score']:.1f}/10")

        return opportunities

    def get_balance(self):
        """Get current account balance"""
        try:
            r = accounts.AccountSummary(accountID=self.oanda_account_id)
            response = self.client.request(r)
            return float(response['account']['balance'])
        except Exception as e:
            print(f"    [ERROR] Getting balance: {e}")
            return 0


# ============================================================================
# FUTURES MARKET HANDLER - APEX TRADER FUNDING
# ============================================================================

class FuturesMarketHandler:
    """
    Handles futures trading for Apex Trader Funding
    Contracts: ES (S&P 500), NQ (Nasdaq), CL (Oil), GC (Gold)
    """

    def __init__(self):
        self.name = "FUTURES (Apex Trader Funding)"

        # Futures contracts
        self.contracts = {
            'ES': {
                'name': 'E-mini S&P 500',
                'point_value': 50,
                'tick_size': 0.25,
                'margin': 12000,
            },
            'NQ': {
                'name': 'E-mini Nasdaq',
                'point_value': 20,
                'tick_size': 0.25,
                'margin': 15000,
            },
            'CL': {
                'name': 'Crude Oil',
                'point_value': 1000,
                'tick_size': 0.01,
                'margin': 6000,
            },
            'GC': {
                'name': 'Gold',
                'point_value': 100,
                'tick_size': 0.10,
                'margin': 8000,
            }
        }

        # Risk parameters
        self.account_size = 150000
        self.min_score = 3.0
        self.risk_per_trade = 0.01
        self.max_positions = 2

        # Time filters (US market hours)
        self.BEST_HOURS = {
            'ES': [9, 10, 11, 12, 13, 14, 15],
            'NQ': [9, 10, 11, 12, 13, 14, 15],
            'CL': [8, 9, 10, 11, 12, 13],
            'GC': [8, 9, 10, 11, 12, 13],
        }

        self.scan_interval = 1800  # 30 minutes

    def get_data(self, contract, granularity='1H', count=200):
        """
        Fetch historical futures data
        NOTE: Placeholder - replace with actual Apex/Rithmic feed in production
        """
        # Simulated data structure for now
        base_price = {
            'ES': 4500,
            'NQ': 15000,
            'CL': 75,
            'GC': 2000
        }.get(contract, 4500)

        return {
            'closes': np.random.randn(count) * 10 + base_price,
            'highs': np.random.randn(count) * 10 + base_price + 10,
            'lows': np.random.randn(count) * 10 + base_price - 10,
            'current_price': base_price + np.random.randn() * 5,
            'volume': np.random.randint(1000, 10000, count)
        }

    def calculate_score(self, contract, data):
        """Calculate trading score using shared TA library"""
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

        # Shared technical analysis
        rsi = ta.calculate_rsi(closes)
        macd = ta.calculate_macd(closes)
        ema_fast = ta.calculate_ema(closes, period=10)
        ema_slow = ta.calculate_ema(closes, period=21)
        adx = ta.calculate_adx(highs, lows, closes)
        atr = ta.calculate_atr(highs, lows, closes)

        # 4H trend confirmation (shared MTF)
        data_4h = self.get_data(contract, granularity='4H', count=100)
        trend_4h = mtf.get_higher_timeframe_trend(data_4h['closes'], current_price)

        # Score LONG signals
        long_score = 0
        long_signals = []

        if rsi < 30:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")
        elif rsi < 40:
            long_score += 1
            long_signals.append("RSI_LOW")

        if macd['macd'] > macd['signal']:
            long_score += 2
            long_signals.append("MACD_BULL_CROSS")

        if current_price > ema_fast and ema_fast > ema_slow:
            long_score += 2
            long_signals.append("EMA_BULLISH")

        if adx > 25:
            long_score += 1
            long_signals.append("STRONG_TREND")

        if trend_4h == 'bullish':
            long_score += 2
            long_signals.append("4H_BULLISH_TREND")

        # Score SHORT signals
        short_score = 0
        short_signals = []

        if rsi > 70:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")

        if macd['macd'] < macd['signal']:
            short_score += 2
            short_signals.append("MACD_BEAR_CROSS")

        if current_price < ema_fast and ema_fast < ema_slow:
            short_score += 2
            short_signals.append("EMA_BEARISH")

        if adx > 25:
            short_score += 1
            short_signals.append("STRONG_TREND")

        if trend_4h == 'bearish':
            short_score += 2
            short_signals.append("4H_BEARISH_TREND")

        # Determine best direction
        if long_score > short_score and long_score >= self.min_score:
            # Calculate position size using shared Kelly
            position_data = kelly.calculate_position_size(
                technical_score=long_score,
                fundamental_score=0,
                account_balance=self.account_size,
                risk_per_trade=self.risk_per_trade
            )

            num_contracts = kelly.calculate_futures_contracts(
                position_size_dollars=position_data['final_size'],
                point_value=self.contracts[contract]['point_value'],
                current_price=current_price
            )

            return {
                'market': 'FUTURES',
                'symbol': contract,
                'score': long_score,
                'direction': 'long',
                'signals': long_signals,
                'rsi': rsi,
                'adx': adx,
                'current_price': current_price,
                'num_contracts': num_contracts,
                'confidence': position_data['confidence']
            }
        elif short_score >= self.min_score:
            position_data = kelly.calculate_position_size(
                technical_score=short_score,
                fundamental_score=0,
                account_balance=self.account_size,
                risk_per_trade=self.risk_per_trade
            )

            num_contracts = kelly.calculate_futures_contracts(
                position_size_dollars=position_data['final_size'],
                point_value=self.contracts[contract]['point_value'],
                current_price=current_price
            )

            return {
                'market': 'FUTURES',
                'symbol': contract,
                'score': short_score,
                'direction': 'short',
                'signals': short_signals,
                'rsi': rsi,
                'adx': adx,
                'current_price': current_price,
                'num_contracts': num_contracts,
                'confidence': position_data['confidence']
            }
        else:
            return {'score': 0, 'direction': None}

    def scan(self):
        """Scan all futures contracts for opportunities"""
        opportunities = []
        current_hour = datetime.now().hour

        for contract_name in self.contracts.keys():
            # Time filter
            if current_hour not in self.BEST_HOURS.get(contract_name, []):
                print(f"    [SKIP] {contract_name} - Outside best hours (current: {current_hour})")
                continue

            data = self.get_data(contract_name)
            result = self.calculate_score(contract_name, data)

            if result['score'] >= self.min_score:
                opportunities.append(result)
                print(f"    [FOUND] {contract_name} {result['direction'].upper()}: {result['score']:.1f}/10")
                print(f"            Signals: {', '.join(result['signals'])}")
                print(f"            Contracts: {result['num_contracts']}")
            else:
                print(f"    [SKIP] {contract_name}: Score {result['score']:.1f}/10")

        return opportunities


# ============================================================================
# CRYPTO MARKET HANDLER - CRYPTO FUND TRADER
# ============================================================================

class CryptoMarketHandler:
    """
    Handles crypto trading for Crypto Fund Trader
    Pairs: BTC/USD, ETH/USD
    Trading: 24/7/365
    """

    def __init__(self):
        self.name = "CRYPTO (Crypto Fund Trader)"

        # Crypto pairs
        self.pairs = {
            'BTCUSD': {
                'name': 'Bitcoin',
                'min_size': 0.001,
                'precision': 8,
            },
            'ETHUSD': {
                'name': 'Ethereum',
                'min_size': 0.01,
                'precision': 8,
            }
        }

        # Risk parameters
        self.account_size = 200000
        self.min_score = 2.5
        self.risk_per_trade = 0.008  # 0.8% (crypto more volatile)
        self.max_positions = 2
        self.volatility_multiplier = 2.0

        # Time filters (crypto trades 24/7 but best during US hours)
        self.BEST_HOURS = list(range(9, 17))  # 9 AM - 5 PM EST
        self.AVOID_HOURS = list(range(0, 5))  # 12 AM - 5 AM EST

        self.scan_interval = 1800  # 30 minutes

    def get_data(self, pair, granularity='1H', count=200):
        """
        Fetch historical crypto data
        NOTE: Placeholder - replace with actual crypto exchange API in production
        Consider using ccxt library for multi-exchange support
        """
        base_price = 65000 if 'BTC' in pair else 3500
        return {
            'closes': np.random.randn(count) * 1000 + base_price,
            'highs': np.random.randn(count) * 1000 + base_price + 500,
            'lows': np.random.randn(count) * 1000 + base_price - 500,
            'current_price': base_price + np.random.randn() * 500,
            'volume': np.random.randint(100, 1000, count)
        }

    def calculate_score(self, pair, data):
        """Calculate trading score using shared TA library"""
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

        # Shared technical analysis
        rsi = ta.calculate_rsi(closes)
        macd = ta.calculate_macd(closes)
        ema_fast = ta.calculate_ema(closes, period=10)
        ema_slow = ta.calculate_ema(closes, period=21)
        adx = ta.calculate_adx(highs, lows, closes)
        atr = ta.calculate_atr(highs, lows, closes)
        bb = ta.calculate_bollinger_bands(closes)

        # 4H trend confirmation (shared MTF)
        data_4h = self.get_data(pair, granularity='4H', count=100)
        trend_4h = mtf.get_higher_timeframe_trend(data_4h['closes'], current_price)

        # Score LONG signals
        long_score = 0
        long_signals = []

        if rsi < 30:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")
        elif rsi < 40:
            long_score += 1
            long_signals.append("RSI_LOW")

        if macd['macd'] > macd['signal']:
            long_score += 2
            long_signals.append("MACD_BULL_CROSS")

        if current_price > ema_fast and ema_fast > ema_slow:
            long_score += 2
            long_signals.append("EMA_BULLISH")

        if current_price < bb['lower']:
            long_score += 1
            long_signals.append("BB_BOUNCE")

        if adx > 25:
            long_score += 1
            long_signals.append("STRONG_TREND")

        if trend_4h == 'bullish':
            long_score += 2
            long_signals.append("4H_BULLISH_TREND")

        # Score SHORT signals
        short_score = 0
        short_signals = []

        if rsi > 70:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")
        elif rsi > 60:
            short_score += 1
            short_signals.append("RSI_HIGH")

        if macd['macd'] < macd['signal']:
            short_score += 2
            short_signals.append("MACD_BEAR_CROSS")

        if current_price < ema_fast and ema_fast < ema_slow:
            short_score += 2
            short_signals.append("EMA_BEARISH")

        if current_price > bb['upper']:
            short_score += 1
            short_signals.append("BB_REJECTION")

        if adx > 25:
            short_score += 1
            short_signals.append("STRONG_TREND")

        if trend_4h == 'bearish':
            short_score += 2
            short_signals.append("4H_BEARISH_TREND")

        # Determine best direction
        if long_score > short_score and long_score >= self.min_score:
            # Calculate position size using shared Kelly
            position_data = kelly.calculate_position_size(
                technical_score=long_score,
                fundamental_score=0,
                account_balance=self.account_size,
                risk_per_trade=self.risk_per_trade
            )

            crypto_units = kelly.calculate_crypto_units(
                position_size_dollars=position_data['final_size'],
                current_price=current_price
            )

            return {
                'market': 'CRYPTO',
                'symbol': pair,
                'score': long_score,
                'direction': 'long',
                'signals': long_signals,
                'rsi': rsi,
                'adx': adx,
                'current_price': current_price,
                'crypto_units': crypto_units,
                'dollar_value': crypto_units * current_price,
                'confidence': position_data['confidence']
            }
        elif short_score >= self.min_score:
            position_data = kelly.calculate_position_size(
                technical_score=short_score,
                fundamental_score=0,
                account_balance=self.account_size,
                risk_per_trade=self.risk_per_trade
            )

            crypto_units = kelly.calculate_crypto_units(
                position_size_dollars=position_data['final_size'],
                current_price=current_price
            )

            return {
                'market': 'CRYPTO',
                'symbol': pair,
                'score': short_score,
                'direction': 'short',
                'signals': short_signals,
                'rsi': rsi,
                'adx': adx,
                'current_price': current_price,
                'crypto_units': crypto_units,
                'dollar_value': crypto_units * current_price,
                'confidence': position_data['confidence']
            }
        else:
            return {'score': 0, 'direction': None}

    def scan(self):
        """Scan all crypto pairs for opportunities (24/7)"""
        opportunities = []
        current_hour = datetime.now().hour

        if current_hour in self.AVOID_HOURS:
            print(f"    [INFO] Hour {current_hour} - Low liquidity period (still scanning)")

        for pair_name in self.pairs.keys():
            data = self.get_data(pair_name)
            result = self.calculate_score(pair_name, data)

            if result['score'] >= self.min_score:
                opportunities.append(result)
                print(f"    [FOUND] {pair_name} {result['direction'].upper()}: {result['score']:.1f}/10")
                print(f"            Signals: {', '.join(result['signals'])}")
                print(f"            Units: {result['crypto_units']:.4f} (${result['dollar_value']:,.2f})")
            else:
                print(f"    [SKIP] {pair_name}: Score {result['score']:.1f}/10")

        return opportunities


# ============================================================================
# UNIFIED TRADING ENGINE - ORCHESTRATES ALL 3 MARKETS
# ============================================================================

class UnifiedTradingEngine:
    """
    Main orchestrator for multi-market trading
    Runs ONE loop that scans forex -> futures -> crypto in sequence
    Provides unified dashboard and P/L tracking
    """

    def __init__(self):
        print("=" * 80)
        print(" " * 20 + "UNIFIED MULTI-MARKET TRADING SYSTEM")
        print("=" * 80)
        print()

        # Initialize market handlers
        try:
            self.forex_handler = ForexMarketHandler()
            print(f"[INITIALIZED] {self.forex_handler.name}")
            print(f"              Pairs: {', '.join(self.forex_handler.pairs)}")
            print(f"              Min Score: {self.forex_handler.min_score}/10")
            print(f"              Scan Interval: {self.forex_handler.scan_interval/60:.0f} min")
            print()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Forex handler: {e}")
            self.forex_handler = None

        try:
            self.futures_handler = FuturesMarketHandler()
            print(f"[INITIALIZED] {self.futures_handler.name}")
            print(f"              Contracts: {', '.join(self.futures_handler.contracts.keys())}")
            print(f"              Min Score: {self.futures_handler.min_score}/10")
            print(f"              Account: ${self.futures_handler.account_size:,}")
            print()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Futures handler: {e}")
            self.futures_handler = None

        try:
            self.crypto_handler = CryptoMarketHandler()
            print(f"[INITIALIZED] {self.crypto_handler.name}")
            print(f"              Pairs: {', '.join(self.crypto_handler.pairs.keys())}")
            print(f"              Min Score: {self.crypto_handler.min_score}/10")
            print(f"              Account: ${self.crypto_handler.account_size:,}")
            print(f"              Trading: 24/7/365")
            print()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Crypto handler: {e}")
            self.crypto_handler = None

        print("=" * 80)
        print("SHARED LIBRARIES:")
        print("  - Technical Analysis (TA-Lib): RSI, MACD, EMA, ADX, ATR, Bollinger Bands")
        print("  - Kelly Criterion: Position sizing based on signal confidence")
        print("  - Multi-Timeframe: 4H trend confirmation for 1H entries")
        print(f"  - AI Confirmation: {'ENABLED (DeepSeek + MiniMax)' if ai_agent.enabled else 'DISABLED (TA-only mode)'}")
        print("=" * 80)
        print()

        # Unified scanning interval (use shortest interval from handlers)
        self.scan_interval = 1800  # 30 minutes (futures/crypto pace)

        # Tracking
        self.iteration = 0
        self.total_opportunities = {
            'forex': 0,
            'futures': 0,
            'crypto': 0
        }

        # AI confirmation tracking
        self.ai_stats = {
            'total_analyzed': 0,
            'approved': 0,
            'rejected': 0,
            'reduced': 0,
            'consensus_reached': 0
        }

        # Enable/disable AI confirmation (can toggle for A/B testing)
        self.use_ai_confirmation = True  # Set to False for pure TA-only mode

    def scan_all_markets(self):
        """
        Scan all markets in sequence: FOREX -> FUTURES -> CRYPTO
        Returns: dict of opportunities by market
        """
        all_opportunities = {
            'forex': [],
            'futures': [],
            'crypto': []
        }

        # Scan FOREX
        if self.forex_handler:
            print(f"\n[SCANNING FOREX] {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 80)
            try:
                forex_opps = self.forex_handler.scan()
                all_opportunities['forex'] = forex_opps
                self.total_opportunities['forex'] += len(forex_opps)
            except Exception as e:
                print(f"    [ERROR] Forex scan failed: {e}")

        # Scan FUTURES
        if self.futures_handler:
            print(f"\n[SCANNING FUTURES] {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 80)
            try:
                futures_opps = self.futures_handler.scan()
                all_opportunities['futures'] = futures_opps
                self.total_opportunities['futures'] += len(futures_opps)
            except Exception as e:
                print(f"    [ERROR] Futures scan failed: {e}")

        # Scan CRYPTO
        if self.crypto_handler:
            print(f"\n[SCANNING CRYPTO] {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 80)
            try:
                crypto_opps = self.crypto_handler.scan()
                all_opportunities['crypto'] = crypto_opps
                self.total_opportunities['crypto'] += len(crypto_opps)
            except Exception as e:
                print(f"    [ERROR] Crypto scan failed: {e}")

        return all_opportunities

    def apply_ai_confirmation(self, opportunities):
        """
        Apply AI confirmation layer to high-confidence trade candidates
        Uses DeepSeek V3.1 + MiniMax multi-model voting

        Args:
            opportunities: dict of opportunities by market

        Returns:
            dict of AI-validated opportunities (only approved/reduced trades)
        """
        if not self.use_ai_confirmation or not ai_agent.enabled:
            print("\n[AI CONFIRMATION] Disabled - using TA-only mode")
            return opportunities

        validated_opportunities = {
            'forex': [],
            'futures': [],
            'crypto': []
        }

        print("\n" + "=" * 80)
        print(" " * 25 + "AI CONFIRMATION LAYER")
        print("=" * 80)

        for market_type, opps in opportunities.items():
            if not opps:
                continue

            print(f"\n[AI ANALYZING {market_type.upper()}] {len(opps)} candidates")
            print("-" * 80)

            for opp in opps:
                # Log all TA signals (for A/B testing comparison)
                trade_logger.log_ta_signal(market_type, opp)

                # Only use AI for high-confidence trades (score >= 6.0)
                if opp['score'] < 6.0:
                    print(f"  {opp['symbol']} ({opp['score']:.1f}/10) - LOW SCORE, AUTO-APPROVE")
                    validated_opportunities[market_type].append(opp)
                    continue

                # Prepare trade data for AI analysis
                trade_data = {
                    'symbol': opp['symbol'],
                    'direction': opp['direction'],
                    'score': opp['score'],
                    'rsi': opp.get('rsi', 50),
                    'macd': opp.get('macd', {}),
                    'adx': opp.get('adx', 20),
                    'current_price': opp.get('current_price', 0),
                    'trend_4h': opp.get('trend_4h', 'neutral'),
                    'bollinger': opp.get('bollinger', {})
                }

                # AI analysis
                print(f"  {opp['symbol']} ({opp['score']:.1f}/10) - Requesting AI confirmation...")
                self.ai_stats['total_analyzed'] += 1

                ai_decision = ai_agent.analyze_trade(trade_data, market_type=market_type)

                # Log AI decision
                trade_logger.log_ai_decision(market_type, opp['symbol'], opp['score'], ai_decision)

                # Track AI statistics
                if ai_decision['action'] == 'APPROVE':
                    self.ai_stats['approved'] += 1
                elif ai_decision['action'] == 'REJECT':
                    self.ai_stats['rejected'] += 1
                elif ai_decision['action'] == 'REDUCE_SIZE':
                    self.ai_stats['reduced'] += 1

                if ai_decision.get('consensus', False):
                    self.ai_stats['consensus_reached'] += 1

                # Display AI decision
                action_emoji = {
                    'APPROVE': '[APPROVED]',
                    'REJECT': '[REJECTED]',
                    'REDUCE_SIZE': '[REDUCED]'
                }
                print(f"    {action_emoji.get(ai_decision['action'], '[UNKNOWN]')} "
                      f"{ai_decision['confidence']:.0f}% confidence")
                print(f"    Reason: {ai_decision['reason']}")

                # Apply AI decision
                if ai_decision['action'] == 'REJECT':
                    print(f"    Skipping trade due to AI rejection")
                    continue

                elif ai_decision['action'] == 'REDUCE_SIZE':
                    # Adjust position size based on AI recommendation
                    if 'units' in opp:  # Forex
                        opp['units'] = int(opp['units'] * 0.5)
                        print(f"    Reducing position to {opp['units']} units")
                    elif 'num_contracts' in opp:  # Futures
                        opp['num_contracts'] = max(1, int(opp['num_contracts'] * 0.5))
                        print(f"    Reducing position to {opp['num_contracts']} contracts")
                    elif 'crypto_units' in opp:  # Crypto
                        opp['crypto_units'] *= 0.5
                        opp['dollar_value'] *= 0.5
                        print(f"    Reducing position to {opp['crypto_units']:.4f} units")

                # Add AI metadata to opportunity
                opp['ai_decision'] = ai_decision
                opp['ai_approved'] = True

                validated_opportunities[market_type].append(opp)

        # Print AI summary
        print("\n" + "-" * 80)
        print("AI CONFIRMATION SUMMARY:")
        print(f"  Total Analyzed: {self.ai_stats['total_analyzed']}")
        print(f"  Approved: {self.ai_stats['approved']} | "
              f"Reduced: {self.ai_stats['reduced']} | "
              f"Rejected: {self.ai_stats['rejected']}")
        if self.ai_stats['total_analyzed'] > 0:
            consensus_rate = (self.ai_stats['consensus_reached'] / self.ai_stats['total_analyzed']) * 100
            print(f"  Model Consensus Rate: {consensus_rate:.1f}%")
        print("=" * 80)

        return validated_opportunities

    def print_unified_dashboard(self, opportunities):
        """
        Print unified status dashboard showing all markets
        """
        print("\n" + "=" * 80)
        print(" " * 25 + "UNIFIED MARKET DASHBOARD")
        print("=" * 80)

        total_opps = sum(len(opps) for opps in opportunities.values())

        if total_opps == 0:
            print("\n[NO OPPORTUNITIES FOUND ACROSS ALL MARKETS]")
        else:
            print(f"\n[FOUND {total_opps} TOTAL OPPORTUNITIES]")
            print()

            # Display by market
            for market_name, opps in opportunities.items():
                if opps:
                    print(f"  {market_name.upper()} ({len(opps)} opportunities):")
                    for opp in opps:
                        symbol = opp['symbol']
                        direction = opp['direction'].upper()
                        score = opp['score']

                        # Market-specific details
                        if market_name == 'forex':
                            print(f"    - {symbol} {direction}: {score:.1f}/10")
                        elif market_name == 'futures':
                            contracts = opp.get('num_contracts', 'N/A')
                            print(f"    - {symbol} {direction}: {score:.1f}/10 ({contracts} contracts)")
                        elif market_name == 'crypto':
                            units = opp.get('crypto_units', 0)
                            value = opp.get('dollar_value', 0)
                            print(f"    - {symbol} {direction}: {score:.1f}/10 ({units:.4f} units, ${value:,.2f})")
                    print()

        # Account balances
        print("-" * 80)
        print("ACCOUNT STATUS:")
        if self.forex_handler:
            try:
                balance = self.forex_handler.get_balance()
                print(f"  FOREX (E8):   ${balance:,.2f}")
            except:
                print(f"  FOREX (E8):   [Unable to fetch]")
        if self.futures_handler:
            print(f"  FUTURES (Apex): ${self.futures_handler.account_size:,} (simulated)")
        if self.crypto_handler:
            print(f"  CRYPTO (CFT):  ${self.crypto_handler.account_size:,} (simulated)")

        print("-" * 80)
        print(f"Session Totals: FOREX={self.total_opportunities['forex']} | "
              f"FUTURES={self.total_opportunities['futures']} | "
              f"CRYPTO={self.total_opportunities['crypto']}")
        print("=" * 80)

    def run(self):
        """
        Main unified trading loop
        Scans all markets every 30 minutes
        """
        print("\n[STARTING UNIFIED TRADING SYSTEM]")
        print(f"Scanning ALL markets every {self.scan_interval/60:.0f} minutes")
        print("Markets: FOREX + FUTURES + CRYPTO")
        print("Press Ctrl+C to stop")
        print()

        try:
            while True:
                self.iteration += 1

                print("\n" + "=" * 80)
                print(f"ITERATION #{self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)

                # Scan all markets
                opportunities = self.scan_all_markets()

                # Apply AI confirmation layer (filters/validates high-confidence trades)
                validated_opportunities = self.apply_ai_confirmation(opportunities)

                # Display unified dashboard (now showing AI-validated opportunities)
                self.print_unified_dashboard(validated_opportunities)

                # TODO: Execute validated trades across all markets here
                # For now, just scanning, AI confirming, and reporting

                print(f"\n[Next scan in {self.scan_interval/60:.0f} minutes...]")
                print(f"[Sleeping until {(datetime.now().timestamp() + self.scan_interval):.0f}]")

                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            print("\n\n[SHUTTING DOWN UNIFIED TRADING SYSTEM]")
            self.shutdown()

    def shutdown(self):
        """
        Graceful shutdown: Export logs and print session summary
        """
        print("\n" + "=" * 80)
        print(" " * 25 + "SHUTDOWN SEQUENCE")
        print("=" * 80)

        # Print session performance summary
        trade_logger.print_session_summary()

        # Export all logs to JSON
        trade_logger.export_session_logs()

        print("\n[SHUTDOWN COMPLETE]")
        print("=" * 80)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        engine = UnifiedTradingEngine()
        engine.run()
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] System stopped by user (Ctrl+C)")
        print("=" * 80)
        print("Final session stats:")
        print(f"  FOREX opportunities found: {engine.total_opportunities['forex']}")
        print(f"  FUTURES opportunities found: {engine.total_opportunities['futures']}")
        print(f"  CRYPTO opportunities found: {engine.total_opportunities['crypto']}")
        print("=" * 80)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
