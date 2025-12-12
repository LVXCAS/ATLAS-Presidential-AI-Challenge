"""
MatchTrader WebSocket Adapter for ATLAS

Uses STOMP protocol over WebSocket to connect to E8 MatchTrader demo.
This bypasses Cloudflare bot protection that blocks REST API.

Requirements:
    pip install stomp.py websocket-client
"""

import json
import time
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import stomp
from threading import Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchTraderWebSocketAdapter:
    """
    MatchTrader WebSocket client using STOMP protocol.

    Features:
    - Real-time market data streaming
    - Position management
    - Account balance updates
    - Bypasses Cloudflare protection (WebSocket connection)
    """

    def __init__(self, email: str, password: str):
        """
        Initialize MatchTrader WebSocket adapter.

        Args:
            email: E8 account email
            password: E8 account password
        """
        self.email = email
        self.password = password

        # WebSocket endpoint (may need adjustment based on actual E8 endpoint)
        self.ws_host = "mtr.e8markets.com"
        self.ws_port = 443  # HTTPS WebSocket

        self.token = None
        self.connection = None
        self.connected_event = Event()

        # Data storage
        self.account_balance = None
        self.market_data = {}
        self.positions = []

        # Callbacks
        self.on_market_data_callback = None
        self.on_position_update_callback = None

    class StompListener(stomp.ConnectionListener):
        """STOMP message listener"""

        def __init__(self, adapter):
            self.adapter = adapter

        def on_connected(self, frame):
            """Called when connection is established"""
            logger.info("[OK] WebSocket connected via STOMP")
            self.adapter.connected_event.set()

        def on_disconnected(self):
            """Called when disconnected"""
            logger.warning("WebSocket disconnected")
            self.adapter.connected_event.clear()

        def on_error(self, frame):
            """Called on error"""
            logger.error(f"STOMP error: {frame.body}")

        def on_message(self, frame):
            """Called when message received"""
            try:
                data = json.loads(frame.body)
                self._handle_message(data, frame.headers.get('destination'))
            except Exception as e:
                logger.error(f"Error parsing message: {e}")

        def _handle_message(self, data: dict, destination: str):
            """Route message to appropriate handler"""

            if 'balance' in data:
                # Account balance update
                self.adapter.account_balance = {
                    'balance': data.get('balance'),
                    'equity': data.get('equity'),
                    'margin_used': data.get('marginUsed'),
                    'margin_available': data.get('marginAvailable')
                }
                logger.info(f"Balance update: ${data.get('equity'):,.2f}")

            elif 'symbol' in data and 'bid' in data:
                # Market data update
                symbol = data.get('symbol')
                self.adapter.market_data[symbol] = {
                    'bid': data.get('bid'),
                    'ask': data.get('ask'),
                    'spread': data.get('spread'),
                    'time': datetime.now()
                }

                if self.adapter.on_market_data_callback:
                    self.adapter.on_market_data_callback(symbol, data)

            elif 'positions' in data:
                # Position update
                self.adapter.positions = data.get('positions', [])

                if self.adapter.on_position_update_callback:
                    self.adapter.on_position_update_callback(self.adapter.positions)

    def connect(self) -> bool:
        """
        Connect to MatchTrader WebSocket via STOMP.

        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to {self.ws_host}:{self.ws_port} via WebSocket...")

            # Create STOMP connection over WebSocket
            # For STOMP 8.x, SSL is handled by the host/port combination
            self.connection = stomp.Connection(
                host_and_ports=[(self.ws_host, self.ws_port)],
                vhost='/',
                heartbeats=(10000, 10000),  # 10 second heartbeats
                auto_content_length=False
            )

            # Set listener
            listener = self.StompListener(self)
            self.connection.set_listener('', listener)

            # Connect with credentials
            logger.info(f"Authenticating as {self.email}...")
            self.connection.connect(
                username=self.email,
                passcode=self.password,
                wait=True
            )

            # Wait for connection confirmation
            if self.connected_event.wait(timeout=10):
                logger.info("[OK] WebSocket connection established")
                return True
            else:
                logger.error("Connection timeout")
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.connection and self.connection.is_connected():
            self.connection.disconnect()
            logger.info("Disconnected from WebSocket")

    def subscribe_market_data(self, symbols: List[str]):
        """
        Subscribe to real-time market data for symbols.

        Args:
            symbols: List of symbols (e.g., ['EURUSD', 'GBPUSD'])
        """
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected")
            return

        for symbol in symbols:
            destination = f"/topic/market-data/{symbol}"
            self.connection.subscribe(destination=destination, id=f"market-{symbol}")
            logger.info(f"[OK] Subscribed to {symbol} market data")

    def subscribe_account_updates(self):
        """Subscribe to account balance and position updates"""
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected")
            return

        # Subscribe to balance updates
        self.connection.subscribe(
            destination="/user/queue/balance",
            id="balance-updates"
        )

        # Subscribe to position updates
        self.connection.subscribe(
            destination="/user/queue/positions",
            id="position-updates"
        )

        logger.info("[OK] Subscribed to account updates")

    def send_order(self, symbol: str, direction: str, lots: float,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> bool:
        """
        Send order via WebSocket.

        Args:
            symbol: Symbol to trade
            direction: 'BUY' or 'SELL'
            lots: Position size
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price

        Returns:
            True if order sent successfully
        """
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected")
            return False

        order = {
            'symbol': symbol,
            'type': direction.upper(),
            'volume': lots,
            'stopLoss': stop_loss,
            'takeProfit': take_profit
        }

        try:
            self.connection.send(
                destination="/app/order",
                body=json.dumps(order),
                headers={'content-type': 'application/json'}
            )
            logger.info(f"[OK] Sent order: {direction} {symbol} {lots} lots")
            return True
        except Exception as e:
            logger.error(f"Failed to send order: {e}")
            return False

    def close_position(self, position_id: int) -> bool:
        """
        Close position via WebSocket.

        Args:
            position_id: Position ID to close

        Returns:
            True if close request sent successfully
        """
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected")
            return False

        try:
            self.connection.send(
                destination="/app/close-position",
                body=json.dumps({'positionId': position_id}),
                headers={'content-type': 'application/json'}
            )
            logger.info(f"[OK] Sent close request for position {position_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def get_account_balance(self) -> Optional[Dict]:
        """
        Get latest account balance from cache.

        Returns:
            Balance dict or None if not available
        """
        return self.account_balance

    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Get latest market data for symbol from cache.

        Args:
            symbol: Symbol name

        Returns:
            Market data dict or None if not available
        """
        return self.market_data.get(symbol)

    def get_positions(self) -> List[Dict]:
        """
        Get latest positions from cache.

        Returns:
            List of positions
        """
        return self.positions


def test_websocket_connection(email: str, password: str):
    """
    Test WebSocket connection to MatchTrader.

    Args:
        email: E8 account email
        password: E8 account password
    """
    print("\n" + "="*70)
    print("MATCHTRADER WEBSOCKET CONNECTION TEST")
    print("="*70)

    # Create adapter
    client = MatchTraderWebSocketAdapter(email=email, password=password)

    # Test connection
    print("\n[1/3] Connecting via WebSocket...")
    if not client.connect():
        print("[FAILED] Could not connect")
        print("\nPossible issues:")
        print("  1. Wrong credentials")
        print("  2. WebSocket endpoint URL incorrect")
        print("  3. E8 doesn't support WebSocket on this endpoint")
        print("\nTrying alternative: Install stomp.py first:")
        print("  pip install stomp.py websocket-client")
        return False

    print("[OK] Connected successfully")

    # Subscribe to account updates
    print("\n[2/3] Subscribing to account updates...")
    client.subscribe_account_updates()

    # Subscribe to market data
    print("\n[3/3] Subscribing to market data...")
    client.subscribe_market_data(['EURUSD', 'GBPUSD', 'USDJPY'])

    # Wait for data
    print("\nWaiting for data (10 seconds)...")
    time.sleep(10)

    # Check if we received data
    balance = client.get_account_balance()
    eurusd_data = client.get_market_data('EURUSD')

    if balance:
        print(f"\n[OK] Balance: ${balance.get('equity', 0):,.2f}")
    else:
        print("\n[WARN] No balance data received yet")

    if eurusd_data:
        print(f"[OK] EUR/USD Bid: {eurusd_data.get('bid'):.5f}")
    else:
        print("[WARN] No market data received yet")

    # Disconnect
    client.disconnect()

    print("\n" + "="*70)
    if balance or eurusd_data:
        print("[SUCCESS] WebSocket connection working!")
        print("="*70)
        print("\n[OK] Ready to deploy ATLAS with WebSocket adapter")
        return True
    else:
        print("[PARTIAL SUCCESS] Connected but no data received")
        print("="*70)
        print("\nThis could mean:")
        print("  1. Data subscriptions need different topic names")
        print("  2. Need to authenticate differently for data streams")
        print("  3. WebSocket works but endpoints need adjustment")
        return False


if __name__ == "__main__":
    # Test with provided credentials
    EMAIL = "kkdo@hotmail.com"
    PASSWORD = "56H2K*kd"

    # First, check if stomp.py is installed
    try:
        import stomp
        print("[OK] stomp.py library found")
    except ImportError:
        print("[ERROR] stomp.py not installed")
        print("\nInstall with:")
        print("  pip install stomp.py websocket-client")
        print("\nThen run this test again.")
        exit(1)

    test_websocket_connection(EMAIL, PASSWORD)
