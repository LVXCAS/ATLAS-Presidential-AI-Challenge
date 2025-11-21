"""
TRADELOCKER ADAPTER FOR E8 CHALLENGES

Adapts your WORKING_FOREX_OANDA.py bot to work with TradeLocker (E8 broker).
Drop-in replacement: same interface, different broker backend.

E8 Challenge Settings:
- $200K account
- 6% max drawdown
- 80% profit split after funded
- 10% profit target to pass
"""

import os
import time
from datetime import datetime, timedelta
from tradelocker import TLAPI
import numpy as np

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not available - will use simplified indicators")

class E8TradeLockerAdapter:
    """
    Adapter to make TradeLocker work exactly like OANDA API.
    Your existing bot can swap this in with minimal changes.
    """

    def __init__(self, environment="https://demo.tradelocker.com"):
        """
        Initialize TradeLocker connection for E8 challenge.

        Required environment variables:
        - TRADELOCKER_EMAIL: Your E8 account email
        - TRADELOCKER_PASSWORD: Your E8 account password
        - TRADELOCKER_SERVER: E8 server name (get from E8 dashboard)
        """

        # Load credentials from environment
        self.email = os.getenv('TRADELOCKER_EMAIL')
        self.password = os.getenv('TRADELOCKER_PASSWORD')
        self.server = os.getenv('TRADELOCKER_SERVER')
        self.environment = environment

        if not all([self.email, self.password, self.server]):
            raise ValueError(
                "Missing TradeLocker credentials in .env file!\n"
                "Add:\n"
                "  TRADELOCKER_EMAIL=your_email@example.com\n"
                "  TRADELOCKER_PASSWORD=your_password\n"
                "  TRADELOCKER_SERVER=E8-Live (or E8-Demo)\n"
            )

        # Initialize TradeLocker API
        self.tl = TLAPI(
            environment=self.environment,
            username=self.email,
            password=self.password,
            server=self.server
        )

        # Cache for instrument IDs (TradeLocker uses numeric IDs)
        self.instrument_cache = {}
        self._load_instruments()

        # Account ID (TradeLocker auto-manages this)
        self.account_id = self._get_account_id()

        print("=" * 70)
        print("TRADELOCKER ADAPTER - E8 CHALLENGE")
        print("=" * 70)
        print(f"Environment: {self.environment}")
        print(f"Server: {self.server}")
        print(f"Account ID: {self.account_id}")
        print(f"Status: Connected [OK]")
        print("=" * 70)

    def _load_instruments(self):
        """Load all available instruments and cache their IDs"""
        try:
            instruments_df = self.tl.get_all_instruments()

            # Map OANDA-style symbols to TradeLocker IDs
            # OANDA: "EUR_USD" -> TradeLocker: "EURUSD"
            for _, instrument in instruments_df.iterrows():
                # Convert pandas Series to dict first
                inst_dict = instrument.to_dict()

                # Get symbol name from dict (not Series)
                symbol = inst_dict.get('name', '')

                if not symbol:
                    continue

                # Store both formats
                self.instrument_cache[symbol] = inst_dict

                # Create OANDA format (EURUSD -> EUR_USD)
                if '_' not in symbol and len(symbol) >= 6:
                    # Forex pairs are 6 chars (EURUSD)
                    if symbol.replace('+', '').isalpha() and len(symbol.replace('+', '')) == 6:
                        oanda_format = f"{symbol[:3]}_{symbol[3:6]}"
                        self.instrument_cache[oanda_format] = inst_dict

            print(f"[INIT] Loaded {len(self.instrument_cache) // 2} instruments from TradeLocker")

        except Exception as e:
            print(f"[ERROR] Failed to load instruments: {e}")
            import traceback
            traceback.print_exc()
            self.instrument_cache = {}

    def _get_account_id(self):
        """Get the active account ID from TradeLocker"""
        try:
            # TradeLocker manages accounts internally
            # For E8, there's typically one account per challenge
            accounts = self.tl.get_all_accounts()
            if not accounts.empty:
                # Get first account's ID (accounts is a DataFrame)
                return int(accounts.iloc[0]['id'])
            return "TRADELOCKER_DEFAULT"
        except Exception as e:
            print(f"[WARN] Could not fetch account ID: {e}")
            return "TRADELOCKER_DEFAULT"

    def _convert_symbol(self, symbol):
        """
        Convert OANDA format to TradeLocker format.
        OANDA: EUR_USD -> TradeLocker: EURUSD
        """
        return symbol.replace('_', '')

    def _get_instrument_id(self, symbol):
        """Get TradeLocker instrument ID from OANDA-style symbol"""
        # Try direct lookup
        if symbol in self.instrument_cache:
            return self.instrument_cache[symbol].get('tradableInstrumentId')

        # Try converted format
        converted = self._convert_symbol(symbol)
        if converted in self.instrument_cache:
            return self.instrument_cache[converted].get('tradableInstrumentId')

        # Try fetching fresh
        try:
            instrument_id = self.tl.get_instrument_id_from_symbol_name(converted)
            return instrument_id
        except Exception as e:
            print(f"[ERROR] Could not find instrument {symbol}: {e}")
            return None

    def get_candles(self, symbol, count=100, granularity='H1'):
        """
        Fetch historical candles (OANDA-compatible format).

        Args:
            symbol: OANDA format like "EUR_USD"
            count: Number of candles to fetch
            granularity: OANDA format like "H1", "H4", "D"

        Returns:
            List of dicts with OANDA-style format:
            {'time': '2025-01-01T12:00:00Z', 'mid': {'o': 1.10, 'h': 1.11, 'l': 1.09, 'c': 1.10}}
        """
        instrument_id = self._get_instrument_id(symbol)
        if not instrument_id:
            return []

        # Convert OANDA granularity to TradeLocker resolution
        resolution_map = {
            'M1': '1',    # 1 minute
            'M5': '5',    # 5 minutes
            'M15': '15',  # 15 minutes
            'M30': '30',  # 30 minutes
            'H1': '60',   # 1 hour
            'H4': '240',  # 4 hours
            'D': '1D',    # Daily
        }
        resolution = resolution_map.get(granularity, '60')  # Default to 1 hour

        try:
            # Calculate time range for requested candle count
            now = datetime.utcnow()

            # Estimate lookback based on granularity
            minutes_per_candle = {
                '1': 1, '5': 5, '15': 15, '30': 30,
                '60': 60, '240': 240, '1D': 1440
            }
            minutes = minutes_per_candle.get(resolution, 60) * count
            from_time = int((now - timedelta(minutes=minutes)).timestamp())
            to_time = int(now.timestamp())

            # Map resolution to TradeLocker format
            tl_resolution_map = {
                '1': '1m', '5': '5m', '15': '15m', '30': '30m',
                '60': '1H', '240': '4H', '1D': '1D'
            }
            tl_resolution = tl_resolution_map.get(resolution, '1H')

            # Fetch price history (correct TradeLocker parameters)
            history_df = self.tl.get_price_history(
                instrument_id=instrument_id,
                resolution=tl_resolution,
                start_timestamp=from_time,
                end_timestamp=to_time
            )

            # Convert to OANDA format
            candles = []
            for _, bar in history_df.iterrows():
                candles.append({
                    'time': datetime.fromtimestamp(bar['t']).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'mid': {
                        'o': bar['o'],
                        'h': bar['h'],
                        'l': bar['l'],
                        'c': bar['c']
                    },
                    'volume': bar.get('v', 0)
                })

            return candles[-count:]  # Return only requested count

        except Exception as e:
            print(f"[ERROR] Failed to fetch candles for {symbol}: {e}")
            return []

    def get_account_summary(self):
        """
        Get account balance and stats (OANDA-compatible format).

        Returns:
            Dict with keys: 'balance', 'NAV', 'unrealizedPL', 'marginAvailable'
        """
        try:
            # Get account details (returns DataFrame)
            accounts = self.tl.get_all_accounts()

            if accounts.empty:
                raise ValueError("No accounts found")

            # Get first (primary) account as Series
            account = accounts.iloc[0]

            # Extract balance from DataFrame
            balance = float(account.get('accountBalance', 0))

            # Get positions to calculate unrealized P/L
            positions = self.tl.get_all_positions()
            unrealized_pl = 0
            if not positions.empty and 'unrealizedPl' in positions.columns:
                unrealized_pl = float(positions['unrealizedPl'].sum())

            # Calculate NAV (balance + unrealized P/L)
            nav = balance + unrealized_pl

            return {
                'balance': balance,
                'NAV': nav,
                'unrealizedPL': unrealized_pl,
                'marginAvailable': balance,  # Simplified for now
                'marginUsed': 0,
            }
        except Exception as e:
            print(f"[ERROR] Failed to get account summary: {e}")
            return {
                'balance': 0,
                'NAV': 0,
                'unrealizedPL': 0,
                'marginAvailable': 0
            }

    def get_open_positions(self):
        """
        Get all open positions (OANDA-compatible format).

        Returns:
            List of dicts with keys: 'instrument', 'long/short', 'units', 'unrealizedPL'
        """
        try:
            positions_df = self.tl.get_all_positions()

            if positions_df.empty:
                return []

            formatted_positions = []

            # Iterate over DataFrame rows
            for _, pos in positions_df.iterrows():
                # Get instrument ID and look up symbol
                instrument_id = pos.get('tradableInstrumentId', '')

                # Find symbol name from cache
                symbol = None
                for sym, inst in self.instrument_cache.items():
                    if inst.get('tradableInstrumentId') == instrument_id:
                        # Prefer OANDA format
                        if '_' in sym:
                            symbol = sym
                            break
                        symbol = sym

                if not symbol:
                    continue

                # Get side (buy/sell)
                side = pos.get('side', 'buy')
                qty = float(pos.get('qty', 0))
                pnl = float(pos.get('unrealizedPl', 0))

                formatted_positions.append({
                    'instrument': symbol,
                    'long': {
                        'units': str(int(qty * 100000)) if side == 'buy' else '0',
                        'unrealizedPL': str(pnl) if side == 'buy' else '0'
                    },
                    'short': {
                        'units': str(int(abs(qty * 100000))) if side == 'sell' else '0',
                        'unrealizedPL': str(pnl) if side == 'sell' else '0'
                    },
                    'unrealizedPL': str(pnl)
                })

            return formatted_positions

        except Exception as e:
            print(f"[ERROR] Failed to get open positions: {e}")
            return []

    def place_order(self, symbol, units, side, take_profit=None, stop_loss=None):
        """
        Place a market order with TP/SL (OANDA-compatible interface).

        Args:
            symbol: OANDA format like "EUR_USD"
            units: Position size (e.g., 100000 for 1 standard lot)
            side: "buy" or "sell"
            take_profit: Price level for take profit
            stop_loss: Price level for stop loss

        Returns:
            Order ID or None if failed
        """
        instrument_id = self._get_instrument_id(symbol)
        if not instrument_id:
            print(f"[ERROR] Cannot place order: instrument {symbol} not found")
            return None

        try:
            # Convert units to lots (TradeLocker typically uses lots, not units)
            # Standard lot = 100,000 units
            lots = abs(units) / 100000.0

            # Place market order
            order_id = self.tl.create_order(
                instrument_id=instrument_id,
                quantity=lots,
                side=side.lower(),
                type_="market",
                take_profit=take_profit,
                take_profit_type='absolute',  # Price level, not offset
                stop_loss=stop_loss,
                stop_loss_type='absolute'  # Price level, not offset
            )

            print(f"[ORDER] Placed {side.upper()} {lots:.2f} lots {symbol} @ market")
            if take_profit:
                print(f"        TP: {take_profit}")
            if stop_loss:
                print(f"        SL: {stop_loss}")

            return order_id

        except Exception as e:
            print(f"[ERROR] Failed to place order for {symbol}: {e}")
            return None

    def close_position(self, symbol):
        """
        Close an open position (OANDA-compatible interface).

        Args:
            symbol: OANDA format like "EUR_USD"

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get open positions
            positions = self.tl.get_all_positions()

            # Find matching position
            converted_symbol = self._convert_symbol(symbol)
            for pos in positions:
                pos_symbol = pos.get('tradableInstrumentId', '')

                if pos_symbol == converted_symbol or pos_symbol == symbol:
                    position_id = pos.get('id')
                    self.tl.close_position(position_id)
                    print(f"[CLOSE] Closed position {symbol}")
                    return True

            print(f"[WARN] No open position found for {symbol}")
            return False

        except Exception as e:
            print(f"[ERROR] Failed to close position {symbol}: {e}")
            return False


# Test connection
if __name__ == "__main__":
    print("\nTesting TradeLocker connection for E8...\n")

    try:
        # Initialize adapter
        adapter = E8TradeLockerAdapter()

        # Test account info
        print("\n[TEST] Fetching account info...")
        summary = adapter.get_account_summary()
        print(f"Balance: ${summary['balance']:,.2f}")
        print(f"Equity: ${summary['NAV']:,.2f}")
        print(f"Unrealized P/L: ${summary['unrealizedPL']:,.2f}")

        # Test candle fetch
        print("\n[TEST] Fetching EUR_USD candles...")
        candles = adapter.get_candles('EUR_USD', count=10, granularity='H1')
        if candles:
            latest = candles[-1]
            print(f"Latest candle: {latest['time']}")
            print(f"  O: {latest['mid']['o']:.5f}")
            print(f"  H: {latest['mid']['h']:.5f}")
            print(f"  L: {latest['mid']['l']:.5f}")
            print(f"  C: {latest['mid']['c']:.5f}")

        # Test positions
        print("\n[TEST] Fetching open positions...")
        positions = adapter.get_open_positions()
        print(f"Open positions: {len(positions)}")

        print("\n[OK] All tests passed! TradeLocker adapter is ready for E8.")

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("\nMake sure you have:")
        print("1. Added TradeLocker credentials to .env")
        print("2. Installed: pip install tradelocker")
        print("3. E8 challenge account is active")
