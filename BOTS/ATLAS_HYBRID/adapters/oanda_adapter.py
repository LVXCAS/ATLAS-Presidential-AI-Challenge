"""
OANDA REST API Adapter for ATLAS

Clean, simple REST API access to OANDA demo/live accounts.
Perfect for 60-day paper training validation.

Documentation: https://developer.oanda.com/rest-live-v20/
"""

import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
# Path: adapters/ -> ATLAS_HYBRID/ -> BOTS/ -> PC-HIVE-TRADING/
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / '.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OandaAdapter:
    """
    OANDA v20 REST API client for ATLAS.

    Features:
    - Real-time market data
    - Position management
    - Order execution
    - Account tracking
    - No Cloudflare, no MQL5, just works
    """

    def __init__(self, api_key: str = None, account_id: str = None, practice: bool = True):
        """
        Initialize OANDA adapter.

        Args:
            api_key: OANDA API key (reads from env if None)
            account_id: OANDA account ID (reads from env if None)
            practice: Use practice (demo) environment (default: True)
        """
        self.api_key = api_key or os.getenv('OANDA_API_KEY')
        self.account_id = account_id or os.getenv('OANDA_ACCOUNT_ID')

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA credentials not found. Set OANDA_API_KEY and OANDA_ACCOUNT_ID")

        # API endpoints
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # Symbol mapping (ATLAS -> OANDA)
        self.symbol_map = {
            "EUR_USD": "EUR_USD",
            "GBP_USD": "GBP_USD",
            "USD_JPY": "USD_JPY",
            "AUD_USD": "AUD_USD",
            "USD_CHF": "USD_CHF",
            "NZD_USD": "NZD_USD",
            "USD_CAD": "USD_CAD",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request to OANDA API"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"OANDA API error: {e}, Response: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def get_account_balance(self) -> Optional[Dict]:
        """
        Get current account balance and equity.

        Returns:
            {
                'balance': 100000.00,
                'equity': 99500.00,
                'unrealized_pnl': -500.00,
                'margin_used': 1000.00,
                'currency': 'USD'
            }
        """
        response = self._request('GET', f'/v3/accounts/{self.account_id}')

        if response and 'account' in response:
            account = response['account']
            return {
                'balance': float(account.get('balance', 0)),
                'equity': float(account.get('NAV', 0)),  # Net Asset Value
                'unrealized_pnl': float(account.get('unrealizedPL', 0)),
                'margin_used': float(account.get('marginUsed', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'currency': account.get('currency', 'USD')
            }
        return None

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
        oanda_symbol = self.symbol_map.get(symbol, symbol)

        response = self._request('GET', f'/v3/accounts/{self.account_id}/pricing',
                                params={'instruments': oanda_symbol})

        if response and 'prices' in response and len(response['prices']) > 0:
            price = response['prices'][0]
            bids = price.get('bids', [{}])
            asks = price.get('asks', [{}])

            bid = float(bids[0].get('price', 0)) if bids else 0
            ask = float(asks[0].get('price', 0)) if asks else 0

            return {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'spread': ask - bid,
                'time': datetime.fromisoformat(price.get('time', '').replace('Z', '+00:00'))
            }
        return None

    def get_candles(self, symbol: str, timeframe: str = 'H1', count: int = 100) -> Optional[List[Dict]]:
        """
        Get historical candle data.

        Args:
            symbol: Symbol name (ATLAS format)
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D)
            count: Number of candles to retrieve

        Returns:
            List of candles [{open, high, low, close, volume, time}, ...]
        """
        oanda_symbol = self.symbol_map.get(symbol, symbol)

        response = self._request('GET', f'/v3/instruments/{oanda_symbol}/candles',
                                params={
                                    'granularity': timeframe,
                                    'count': count
                                })

        if response and 'candles' in response:
            candles = []
            for candle in response['candles']:
                if candle.get('complete'):  # Only include complete candles
                    mid = candle.get('mid', {})
                    candles.append({
                        'time': datetime.fromisoformat(candle.get('time', '').replace('Z', '+00:00')),
                        'open': float(mid.get('o', 0)),
                        'high': float(mid.get('h', 0)),
                        'low': float(mid.get('l', 0)),
                        'close': float(mid.get('c', 0)),
                        'volume': int(candle.get('volume', 0))
                    })
            return candles
        return None

    def get_open_positions(self) -> Optional[List[Dict]]:
        """
        Get all open positions with trade IDs.

        Returns:
            List of positions with details including trade_id for trailing stops
        """
        # Get positions summary
        pos_response = self._request('GET', f'/v3/accounts/{self.account_id}/openPositions')

        if not pos_response or 'positions' not in pos_response:
            return []

        # Get individual trades for trade IDs
        trades_response = self._request('GET', f'/v3/accounts/{self.account_id}/trades')
        trades_by_instrument = {}

        if trades_response and 'trades' in trades_response:
            for trade in trades_response['trades']:
                instrument = trade['instrument']
                if instrument not in trades_by_instrument:
                    trades_by_instrument[instrument] = []
                trades_by_instrument[instrument].append(trade['id'])

        positions = []
        for pos in pos_response['positions']:
            long_units = float(pos.get('long', {}).get('units', 0))
            short_units = float(pos.get('short', {}).get('units', 0))

            if long_units != 0 or short_units != 0:
                instrument = pos.get('instrument')

                # Get first trade ID for this instrument (for trailing stops)
                trade_id = None
                if instrument in trades_by_instrument and trades_by_instrument[instrument]:
                    trade_id = trades_by_instrument[instrument][0]

                positions.append({
                    'instrument': instrument,  # Add 'instrument' key for live_trader
                    'symbol': instrument,      # Keep 'symbol' for backward compatibility
                    'type': 'long' if long_units > 0 else 'short',
                    'units': abs(long_units if long_units != 0 else short_units),
                    'unrealized_pnl': float(pos.get('unrealizedPL', 0)),
                    'margin_used': float(pos.get('marginUsed', 0)),
                    'avg_price': float(pos.get('long' if long_units > 0 else 'short', {}).get('averagePrice', 0)),
                    'trade_id': trade_id  # For trailing stop updates
                })
        return positions

    def open_position(self, symbol: str, direction: str, units: int,
                     stop_loss_pips: Optional[float] = None,
                     take_profit_pips: Optional[float] = None) -> Optional[Dict]:
        """
        Open a new position.

        Args:
            symbol: Symbol name (ATLAS format)
            direction: 'long' or 'short'
            units: Position size in units (10000 = 0.1 lot for OANDA)
            stop_loss_pips: Stop loss in pips (optional)
            take_profit_pips: Take profit in pips (optional)

        Returns:
            Order details if successful
        """
        oanda_symbol = self.symbol_map.get(symbol, symbol)

        # DEBUG: Log incoming parameters
        print(f"  [DEBUG OANDA] Received direction: {direction}")
        print(f"  [DEBUG OANDA] Received units: {units}")

        # OANDA uses positive units for long, negative for short
        order_units = units if direction.lower() == 'long' else -units

        # DEBUG: Log conversion
        print(f"  [DEBUG OANDA] Converted to order_units: {order_units}")

        order_data = {
            'order': {
                'type': 'MARKET',
                'instrument': oanda_symbol,
                'units': str(order_units),
                'timeInForce': 'FOK',  # Fill or Kill
                'positionFill': 'DEFAULT'
            }
        }

        # Add stop loss if specified
        if stop_loss_pips:
            # Get current price to calculate SL price
            market_data = self.get_market_data(symbol)
            if market_data:
                current_price = market_data['ask'] if direction.lower() == 'long' else market_data['bid']
                pip_value = 0.0001 if 'JPY' not in symbol else 0.01

                if direction.lower() == 'long':
                    sl_price = current_price - (stop_loss_pips * pip_value)
                else:
                    sl_price = current_price + (stop_loss_pips * pip_value)

                order_data['order']['stopLossOnFill'] = {
                    'price': f"{sl_price:.5f}"
                }

        # Add take profit if specified
        if take_profit_pips:
            market_data = self.get_market_data(symbol)
            if market_data:
                current_price = market_data['ask'] if direction.lower() == 'long' else market_data['bid']
                pip_value = 0.0001 if 'JPY' not in symbol else 0.01

                if direction.lower() == 'long':
                    tp_price = current_price + (take_profit_pips * pip_value)
                else:
                    tp_price = current_price - (take_profit_pips * pip_value)

                order_data['order']['takeProfitOnFill'] = {
                    'price': f"{tp_price:.5f}"
                }

        response = self._request('POST', f'/v3/accounts/{self.account_id}/orders',
                                json=order_data)

        if response and 'orderFillTransaction' in response:
            fill = response['orderFillTransaction']
            logger.info(f"[OK] Opened {direction} position: {symbol} {units} units")
            return {
                'id': fill.get('id'),
                'symbol': symbol,
                'type': direction,
                'units': units,
                'price': float(fill.get('price', 0))
            }
        else:
            logger.error(f"Failed to open position: {response}")
            return None

    def close_position(self, symbol: str, direction: str = 'long') -> bool:
        """
        Close position for a symbol.

        Args:
            symbol: Symbol to close
            direction: 'long' or 'short'

        Returns:
            True if successful
        """
        oanda_symbol = self.symbol_map.get(symbol, symbol)
        long_short = 'long' if direction.lower() == 'long' else 'short'

        response = self._request('PUT',
                                f'/v3/accounts/{self.account_id}/positions/{oanda_symbol}/close',
                                json={f'{long_short}Units': 'ALL'})

        if response:
            logger.info(f"[OK] Closed {direction} position: {symbol}")
            return True
        else:
            logger.error(f"Failed to close position: {symbol}")
            return False

    def update_trailing_stop(self, trade_id: str, trailing_distance_pips: float, symbol: str) -> bool:
        """
        Update trailing stop for an open trade.

        Args:
            trade_id: OANDA trade ID
            trailing_distance_pips: Distance in pips to trail behind current price
            symbol: Trading pair (for pip calculation)

        Returns:
            True if successful
        """
        try:
            # Get current market price
            market_data = self.get_market_data(symbol)
            if not market_data:
                logger.error(f"Cannot get market data for {symbol}")
                return False

            # Determine pip value
            pip_value = 0.0001 if 'JPY' not in symbol else 0.01

            # Get trade details to determine direction
            response = self._request('GET', f'/v3/accounts/{self.account_id}/trades/{trade_id}')
            if not response or 'trade' not in response:
                logger.error(f"Cannot fetch trade {trade_id}")
                return False

            trade = response['trade']
            current_units = float(trade['currentUnits'])
            is_long = current_units > 0

            # Calculate trailing stop price
            if is_long:
                # For long positions, trail below current bid
                current_price = market_data['bid']
                new_sl_price = current_price - (trailing_distance_pips * pip_value)
            else:
                # For short positions, trail above current ask
                current_price = market_data['ask']
                new_sl_price = current_price + (trailing_distance_pips * pip_value)

            # Get existing stop loss if any
            existing_sl_id = trade.get('stopLossOrderID')
            existing_sl_price = None

            if existing_sl_id:
                # Check if we should update (only move SL in favorable direction)
                sl_response = self._request('GET', f'/v3/accounts/{self.account_id}/orders/{existing_sl_id}')
                if sl_response and 'order' in sl_response:
                    existing_sl_price = float(sl_response['order']['price'])

                    # Only update if new SL is better (closer to price, locking in more profit)
                    if is_long:
                        if new_sl_price <= existing_sl_price:
                            # Don't move SL down for longs
                            return True
                    else:
                        if new_sl_price >= existing_sl_price:
                            # Don't move SL up for shorts
                            return True

            # Update the stop loss
            sl_data = {
                'stopLoss': {
                    'price': f"{new_sl_price:.5f}",
                    'timeInForce': 'GTC',
                    'triggerMode': 'TOP_OF_BOOK'
                }
            }

            response = self._request('PUT',
                                    f'/v3/accounts/{self.account_id}/trades/{trade_id}/orders',
                                    json=sl_data)

            if response and 'stopLossOrderTransaction' in response:
                logger.info(f"[TRAILING STOP] Updated {symbol} SL to {new_sl_price:.5f} ({trailing_distance_pips} pips trailing)")
                return True
            else:
                logger.error(f"Failed to update trailing stop: {response}")
                return False

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all open positions"""
        positions = self.get_open_positions()

        if not positions:
            logger.info("No open positions to close")
            return True

        success_count = 0
        for pos in positions:
            if self.close_position(pos['symbol'], pos['type']):
                success_count += 1

        logger.info(f"Closed {success_count}/{len(positions)} positions")
        return success_count == len(positions)


def test_oanda_connection():
    """Test OANDA connection"""
    print("\n" + "="*70)
    print("OANDA CONNECTION TEST")
    print("="*70)

    try:
        client = OandaAdapter(practice=True)
        print("[OK] OANDA adapter initialized")
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("\nMake sure .env file has:")
        print("  OANDA_API_KEY=your_key_here")
        print("  OANDA_ACCOUNT_ID=your_account_here")
        return False

    # Test account balance
    print("\n[1/3] Testing account access...")
    balance = client.get_account_balance()
    if balance:
        print(f"[OK] Balance: ${balance['balance']:,.2f}")
        print(f"  Equity: ${balance['equity']:,.2f}")
        print(f"  Unrealized P/L: ${balance['unrealized_pnl']:,.2f}")
    else:
        print("[FAILED] Could not retrieve balance")
        return False

    # Test market data
    print("\n[2/3] Testing market data...")
    market_data = client.get_market_data('EUR_USD')
    if market_data:
        print(f"[OK] EUR/USD Bid: {market_data['bid']:.5f}")
        print(f"  EUR/USD Ask: {market_data['ask']:.5f}")
        print(f"  Spread: {market_data['spread']:.5f}")
    else:
        print("[FAILED] Could not retrieve market data")
        return False

    # Test historical data
    print("\n[3/3] Testing historical data...")
    candles = client.get_candles('EUR_USD', 'H1', count=10)
    if candles:
        print(f"[OK] Retrieved {len(candles)} H1 candles")
        print(f"  Latest close: {candles[-1]['close']:.5f}")
        print(f"  Latest time: {candles[-1]['time']}")
    else:
        print("[FAILED] Could not retrieve candles")
        return False

    print("\n" + "="*70)
    print("[SUCCESS] OANDA connection working!")
    print("="*70)
    print("\n[OK] Ready to deploy ATLAS on OANDA demo")
    print(f"[OK] Account balance: ${balance['balance']:,.2f}")
    print("[OK] Market data streaming")
    print("="*70)

    return True


if __name__ == "__main__":
    test_oanda_connection()
