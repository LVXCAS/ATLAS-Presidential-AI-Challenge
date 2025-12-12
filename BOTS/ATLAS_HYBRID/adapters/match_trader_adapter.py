"""
MatchTrader REST API Adapter for ATLAS

Connects ATLAS to E8 MatchTrader demo accounts via REST API.

Documentation: https://app.theneo.io/match-trade/platform-api
"""

import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchTraderAdapter:
    """
    MatchTrader REST API client for E8 demo accounts.

    Features:
    - Login/authentication with token refresh
    - Real-time market data
    - Position management (open, close, modify)
    - Account balance tracking
    - Rate limiting (500 req/min)
    """

    def __init__(self, email: str, password: str, broker_id: int = 1):
        """
        Initialize MatchTrader adapter.

        Args:
            email: E8 account email (kkdo@hotmail.com)
            password: E8 account password
            broker_id: Broker ID (default: 1 for E8)
        """
        self.base_url = "https://mtr.e8markets.com/api"
        self.email = email
        self.password = password
        self.broker_id = broker_id

        self.token = None
        self.token_expires_at = None
        self.account_id = None

        # Rate limiting (500 requests/minute)
        self.requests_count = 0
        self.requests_reset_time = time.time() + 60

        # Symbol mapping (ATLAS -> MatchTrader)
        self.symbol_map = {
            "EUR_USD": "EURUSD",
            "GBP_USD": "GBPUSD",
            "USD_JPY": "USDJPY",
            "AUD_USD": "AUDUSD",
            "USD_CHF": "USDCHF",
            "NZD_USD": "NZDUSD",
            "USD_CAD": "USDCAD",
        }

        # Reverse mapping
        self.reverse_symbol_map = {v: k for k, v in self.symbol_map.items()}

    def _check_rate_limit(self):
        """Enforce 500 requests/minute rate limit"""
        current_time = time.time()

        # Reset counter every minute
        if current_time > self.requests_reset_time:
            self.requests_count = 0
            self.requests_reset_time = current_time + 60

        # Check if we're at limit
        if self.requests_count >= 500:
            wait_time = self.requests_reset_time - current_time
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.requests_count = 0
                self.requests_reset_time = time.time() + 60

        self.requests_count += 1

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """
        Make HTTP request with rate limiting and error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional requests kwargs

        Returns:
            Response JSON or None on error
        """
        self._check_rate_limit()

        url = f"{self.base_url}/{endpoint}"

        # Add auth token if available
        headers = kwargs.get('headers', {})
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        kwargs['headers'] = headers

        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}, Response: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def login(self) -> bool:
        """
        Authenticate with MatchTrader API.

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Logging in to MatchTrader as {self.email}")

        payload = {
            "email": self.email,
            "password": self.password,
            "brokerId": self.broker_id
        }

        response = self._make_request('POST', 'login', json=payload)

        if response and 'token' in response:
            self.token = response['token']
            # Token expires in 15 minutes
            self.token_expires_at = datetime.now() + timedelta(minutes=15)
            self.account_id = response.get('accountId')

            logger.info(f"✓ Login successful, account ID: {self.account_id}")
            return True
        else:
            logger.error("Login failed")
            return False

    def refresh_token(self) -> bool:
        """
        Refresh authentication token (before 15min expiry).

        Returns:
            True if successful, False otherwise
        """
        if not self.token:
            return self.login()

        response = self._make_request('POST', 'refresh-token')

        if response and 'token' in response:
            self.token = response['token']
            self.token_expires_at = datetime.now() + timedelta(minutes=15)
            logger.info("✓ Token refreshed")
            return True
        else:
            logger.warning("Token refresh failed, re-logging in")
            return self.login()

    def _ensure_authenticated(self):
        """Ensure we have a valid token, refresh if needed"""
        if not self.token:
            self.login()
            return

        # Refresh token if it expires in < 2 minutes
        if datetime.now() + timedelta(minutes=2) > self.token_expires_at:
            self.refresh_token()

    def get_account_balance(self) -> Optional[Dict]:
        """
        Get current account balance and equity.

        Returns:
            {
                'balance': 200000.00,
                'equity': 199500.00,
                'margin_used': 500.00,
                'margin_available': 199000.00,
                'currency': 'USD'
            }
        """
        self._ensure_authenticated()

        response = self._make_request('GET', 'balance')

        if response:
            return {
                'balance': response.get('balance', 0.0),
                'equity': response.get('equity', 0.0),
                'margin_used': response.get('marginUsed', 0.0),
                'margin_available': response.get('marginAvailable', 0.0),
                'currency': response.get('currency', 'USD')
            }
        return None

    def get_symbols(self) -> Optional[List[Dict]]:
        """
        Get available trading symbols.

        Returns:
            List of symbols with details (name, digits, spread, etc.)
        """
        self._ensure_authenticated()

        response = self._make_request('GET', 'symbols')
        return response.get('symbols', []) if response else None

    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Get current market data (bid, ask, spread) for a symbol.

        Args:
            symbol: Symbol name (ATLAS format, e.g. "EUR_USD")

        Returns:
            {
                'symbol': 'EUR_USD',
                'bid': 1.08450,
                'ask': 1.08455,
                'spread': 0.00005,
                'time': datetime object
            }
        """
        self._ensure_authenticated()

        # Convert ATLAS symbol to MatchTrader format
        mt_symbol = self.symbol_map.get(symbol, symbol.replace('_', ''))

        response = self._make_request('GET', f'market-watch/{mt_symbol}')

        if response:
            return {
                'symbol': symbol,
                'bid': response.get('bid', 0.0),
                'ask': response.get('ask', 0.0),
                'spread': response.get('spread', 0.0),
                'time': datetime.fromtimestamp(response.get('timestamp', time.time()))
            }
        return None

    def get_candles(self, symbol: str, timeframe: str = 'H1', count: int = 100) -> Optional[List[Dict]]:
        """
        Get historical candle data.

        Args:
            symbol: Symbol name (ATLAS format)
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            count: Number of candles to retrieve

        Returns:
            List of candles [{open, high, low, close, volume, time}, ...]
        """
        self._ensure_authenticated()

        mt_symbol = self.symbol_map.get(symbol, symbol.replace('_', ''))

        params = {
            'symbol': mt_symbol,
            'timeframe': timeframe,
            'count': count
        }

        response = self._make_request('GET', 'candles', params=params)

        if response and 'candles' in response:
            candles = []
            for candle in response['candles']:
                candles.append({
                    'time': datetime.fromtimestamp(candle.get('timestamp', 0)),
                    'open': candle.get('open', 0.0),
                    'high': candle.get('high', 0.0),
                    'low': candle.get('low', 0.0),
                    'close': candle.get('close', 0.0),
                    'volume': candle.get('volume', 0)
                })
            return candles
        return None

    def get_open_positions(self) -> Optional[List[Dict]]:
        """
        Get all open positions.

        Returns:
            List of positions with details (id, symbol, type, lots, PnL, etc.)
        """
        self._ensure_authenticated()

        response = self._make_request('GET', 'opened-positions')

        if response and 'positions' in response:
            positions = []
            for pos in response['positions']:
                # Convert MatchTrader symbol back to ATLAS format
                mt_symbol = pos.get('symbol', '')
                atlas_symbol = self.reverse_symbol_map.get(mt_symbol, mt_symbol)

                positions.append({
                    'id': pos.get('id'),
                    'symbol': atlas_symbol,
                    'type': 'long' if pos.get('type') == 'BUY' else 'short',
                    'lots': pos.get('volume', 0.0),
                    'open_price': pos.get('openPrice', 0.0),
                    'current_price': pos.get('currentPrice', 0.0),
                    'pnl': pos.get('profit', 0.0),
                    'swap': pos.get('swap', 0.0),
                    'commission': pos.get('commission', 0.0),
                    'stop_loss': pos.get('stopLoss'),
                    'take_profit': pos.get('takeProfit'),
                    'open_time': datetime.fromtimestamp(pos.get('openTime', 0))
                })
            return positions
        return None

    def open_position(self, symbol: str, direction: str, lots: float,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> Optional[Dict]:
        """
        Open a new position.

        Args:
            symbol: Symbol name (ATLAS format)
            direction: 'long' or 'short'
            lots: Position size in lots
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Position details if successful, None otherwise
        """
        self._ensure_authenticated()

        mt_symbol = self.symbol_map.get(symbol, symbol.replace('_', ''))

        payload = {
            'symbol': mt_symbol,
            'type': 'BUY' if direction.lower() == 'long' else 'SELL',
            'volume': lots
        }

        if stop_loss:
            payload['stopLoss'] = stop_loss
        if take_profit:
            payload['takeProfit'] = take_profit

        response = self._make_request('POST', 'open-position', json=payload)

        if response and response.get('success'):
            logger.info(f"✓ Opened {direction} position: {symbol} {lots} lots")
            return {
                'id': response.get('positionId'),
                'symbol': symbol,
                'type': direction,
                'lots': lots,
                'open_price': response.get('openPrice'),
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        else:
            logger.error(f"Failed to open position: {response}")
            return None

    def close_position(self, position_id: int, lots: Optional[float] = None) -> bool:
        """
        Close a position (fully or partially).

        Args:
            position_id: Position ID to close
            lots: Lots to close (None = close all)

        Returns:
            True if successful, False otherwise
        """
        self._ensure_authenticated()

        if lots:
            # Partial close
            payload = {
                'positionId': position_id,
                'volume': lots
            }
            response = self._make_request('POST', 'partial-close', json=payload)
        else:
            # Full close
            payload = {
                'positionId': position_id
            }
            response = self._make_request('POST', 'close-positions', json=payload)

        if response and response.get('success'):
            logger.info(f"✓ Closed position {position_id}")
            return True
        else:
            logger.error(f"Failed to close position: {response}")
            return False

    def modify_position(self, position_id: int,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> bool:
        """
        Modify position stop loss or take profit.

        Args:
            position_id: Position ID to modify
            stop_loss: New stop loss price (None = no change)
            take_profit: New take profit price (None = no change)

        Returns:
            True if successful, False otherwise
        """
        self._ensure_authenticated()

        payload = {
            'positionId': position_id
        }

        if stop_loss is not None:
            payload['stopLoss'] = stop_loss
        if take_profit is not None:
            payload['takeProfit'] = take_profit

        response = self._make_request('POST', 'edit-position', json=payload)

        if response and response.get('success'):
            logger.info(f"✓ Modified position {position_id}")
            return True
        else:
            logger.error(f"Failed to modify position: {response}")
            return False

    def close_all_positions(self) -> bool:
        """
        Close all open positions (emergency stop).

        Returns:
            True if successful, False otherwise
        """
        positions = self.get_open_positions()

        if not positions:
            logger.info("No open positions to close")
            return True

        logger.warning(f"Closing {len(positions)} positions")

        success_count = 0
        for pos in positions:
            if self.close_position(pos['id']):
                success_count += 1

        logger.info(f"Closed {success_count}/{len(positions)} positions")
        return success_count == len(positions)

    def calculate_pnl(self) -> Dict:
        """
        Calculate total P/L from all open positions.

        Returns:
            {
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_pnl': 0.0,
                'open_positions_count': 0
            }
        """
        positions = self.get_open_positions()

        if not positions:
            return {
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_pnl': 0.0,
                'open_positions_count': 0
            }

        unrealized_pnl = sum(pos['pnl'] for pos in positions)

        return {
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': 0.0,  # MatchTrader API doesn't provide this directly
            'total_pnl': unrealized_pnl,
            'open_positions_count': len(positions)
        }


def test_connection(email: str, password: str):
    """
    Test MatchTrader connection with provided credentials.

    Args:
        email: E8 account email
        password: E8 account password
    """
    print("\n" + "="*70)
    print("MATCHTRADER CONNECTION TEST")
    print("="*70)

    # Initialize adapter
    client = MatchTraderAdapter(email=email, password=password)

    # Test login
    print("\n[1/4] Testing login...")
    if not client.login():
        print("[FAILED] Could not authenticate")
        return False
    print("✓ Login successful")

    # Test balance
    print("\n[2/4] Testing balance retrieval...")
    balance = client.get_account_balance()
    if balance:
        print(f"✓ Balance: ${balance['balance']:,.2f}")
        print(f"  Equity: ${balance['equity']:,.2f}")
    else:
        print("[FAILED] Could not retrieve balance")
        return False

    # Test market data
    print("\n[3/4] Testing market data...")
    market_data = client.get_market_data('EUR_USD')
    if market_data:
        print(f"✓ EUR/USD Bid: {market_data['bid']:.5f}")
        print(f"  EUR/USD Ask: {market_data['ask']:.5f}")
    else:
        print("[FAILED] Could not retrieve market data")
        return False

    # Test candle data
    print("\n[4/4] Testing historical data...")
    candles = client.get_candles('EUR_USD', 'H1', count=10)
    if candles:
        print(f"✓ Retrieved {len(candles)} H1 candles")
        print(f"  Latest close: {candles[-1]['close']:.5f}")
    else:
        print("[FAILED] Could not retrieve candles")
        return False

    print("\n" + "="*70)
    print("[SUCCESS] MatchTrader connection working!")
    print("="*70)
    print("\n✓ Ready to deploy ATLAS on E8 demo account")
    print(f"✓ Account balance: ${balance['balance']:,.2f}")
    print(f"✓ Connection stable, {client.requests_count} API calls made")
    print("="*70)

    return True


if __name__ == "__main__":
    # Test with provided credentials
    EMAIL = "kkdo@hotmail.com"
    PASSWORD = "56H2K*kd"

    test_connection(EMAIL, PASSWORD)
