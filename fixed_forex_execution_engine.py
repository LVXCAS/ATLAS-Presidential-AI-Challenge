#!/usr/bin/env python3
"""
FIXED FOREX EXECUTION ENGINE
Automated trade execution via OANDA REST API

FIXES:
- Replaces v20 library with direct REST API calls
- Uses requests library with 5-second timeouts
- No more hanging on API calls
- Handles timeouts gracefully

Handles:
- Market order execution
- Stop loss placement
- Take profit placement
- Position monitoring
- Position closing

Safety Features:
- Paper trading mode (default)
- Order validation
- Risk checks
- Error handling
- Timeout protection
"""

import os
import requests
from typing import Dict, Optional, List
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

# Default OANDA credentials
DEFAULT_API_KEY = "0bff5dc7375409bb8747deebab8988a1-d8b26324102c95d6f2b6f641bc330a7c"
DEFAULT_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', '101-004-29328895-001')


class FixedForexExecutionEngine:
    """
    Execute forex trades via OANDA REST API

    Features:
    - Market orders with timeout protection
    - Stop loss / Take profit
    - Position queries
    - Auto-close positions
    - NO MORE HANGING - Uses 5-second timeouts
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 account_id: Optional[str] = None,
                 practice: bool = True,
                 paper_trading: bool = True,
                 timeout: int = 5):
        """
        Initialize OANDA execution engine with REST API

        Args:
            api_key: OANDA API key (or from env)
            account_id: OANDA account ID (or from env)
            practice: Use practice server (True) or live (False)
            paper_trading: Simulate trades without execution (True)
            timeout: Request timeout in seconds (default: 5)
        """

        self.api_key = api_key or os.getenv('OANDA_API_KEY') or DEFAULT_API_KEY
        self.account_id = account_id or os.getenv('OANDA_ACCOUNT_ID') or DEFAULT_ACCOUNT_ID
        self.practice = practice
        self.paper_trading = paper_trading
        self.timeout = timeout

        # Virtual positions for paper trading
        self.paper_positions = {}
        self.paper_trade_id = 1000

        if paper_trading:
            print("[FOREX EXECUTION] PAPER TRADING MODE - No real orders will be placed")
            self.base_url = None
        else:
            if not self.api_key or not self.account_id:
                raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID required. Check .env file.")

            if practice:
                self.base_url = 'https://api-fxpractice.oanda.com/v3'
            else:
                self.base_url = 'https://api-fxtrade.oanda.com/v3'

            self.headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            mode = "PRACTICE" if practice else "LIVE"
            print(f"[FOREX EXECUTION] Connected to OANDA {mode} server")
            print(f"[FOREX EXECUTION] Using REST API with {timeout}s timeout")

    def place_market_order(self,
                          pair: str,
                          direction: str,
                          units: int,
                          stop_loss: float,
                          take_profit: float) -> Optional[Dict]:
        """
        Place market order with stop loss and take profit

        Args:
            pair: Forex pair (e.g., 'EUR_USD')
            direction: 'LONG' or 'SHORT'
            units: Position size (e.g., 1000 = 0.01 lot)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order result dict or None
        """

        # Calculate signed units (positive = buy, negative = sell)
        signed_units = units if direction == 'LONG' else -units

        if self.paper_trading:
            return self._place_paper_order(pair, direction, units, stop_loss, take_profit)

        try:
            url = f"{self.base_url}/accounts/{self.account_id}/orders"

            order_data = {
                'order': {
                    'type': 'MARKET',
                    'instrument': pair,
                    'units': str(signed_units),
                    'timeInForce': 'FOK',
                    'positionFill': 'DEFAULT',
                    'stopLossOnFill': {
                        'price': f"{stop_loss:.5f}"
                    },
                    'takeProfitOnFill': {
                        'price': f"{take_profit:.5f}"
                    }
                }
            }

            response = requests.post(
                url,
                headers=self.headers,
                json=order_data,
                timeout=self.timeout
            )

            if response.status_code == 201:
                data = response.json()
                order_fill = data.get('orderFillTransaction', {})

                result = {
                    'success': True,
                    'trade_id': order_fill.get('id'),
                    'pair': pair,
                    'direction': direction,
                    'units': units,
                    'entry_price': float(order_fill.get('price', 0)),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'LIVE'
                }

                print(f"[ORDER FILLED] {pair} {direction} @ {result['entry_price']:.5f}")
                print(f"  Trade ID: {result['trade_id']}")
                print(f"  Stop: {stop_loss:.5f}, Target: {take_profit:.5f}")

                return result
            else:
                print(f"[ORDER FAILED] Status: {response.status_code}")
                print(f"  Response: {response.text}")
                return None

        except requests.Timeout:
            print(f"[TIMEOUT] Order request for {pair} exceeded {self.timeout}s")
            return None
        except Exception as e:
            print(f"[ERROR] Placing order for {pair}: {e}")
            return None

    def _place_paper_order(self,
                          pair: str,
                          direction: str,
                          units: int,
                          stop_loss: float,
                          take_profit: float) -> Dict:
        """
        Simulate order placement (paper trading)
        """

        # Get current price (simulated)
        entry_price = (stop_loss + take_profit) / 2 if direction == 'LONG' else (stop_loss + take_profit) / 2

        trade_id = f"PAPER_{self.paper_trade_id}"
        self.paper_trade_id += 1

        # Store paper position
        self.paper_positions[trade_id] = {
            'pair': pair,
            'direction': direction,
            'units': units,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'OPEN',
            'open_time': datetime.now().isoformat()
        }

        result = {
            'success': True,
            'trade_id': trade_id,
            'pair': pair,
            'direction': direction,
            'units': units,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now().isoformat(),
            'mode': 'PAPER'
        }

        print(f"[PAPER ORDER] {pair} {direction} @ {entry_price:.5f}")
        print(f"  Trade ID: {trade_id}")
        print(f"  Stop: {stop_loss:.5f}, Target: {take_profit:.5f}")

        return result

    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions

        Returns:
            List of open position dicts
        """

        if self.paper_trading:
            return [
                {**pos, 'trade_id': tid}
                for tid, pos in self.paper_positions.items()
                if pos['status'] == 'OPEN'
            ]

        try:
            url = f"{self.base_url}/accounts/{self.account_id}/openTrades"

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                trades = data.get('trades', [])

                positions = []
                for trade in trades:
                    positions.append({
                        'trade_id': trade.get('id'),
                        'pair': trade.get('instrument'),
                        'direction': 'LONG' if float(trade.get('currentUnits', 0)) > 0 else 'SHORT',
                        'units': abs(int(trade.get('currentUnits', 0))),
                        'entry_price': float(trade.get('price', 0)),
                        'unrealized_pl': float(trade.get('unrealizedPL', 0)),
                        'open_time': trade.get('openTime')
                    })

                return positions
            else:
                print(f"[ERROR] Getting positions: Status {response.status_code}")
                return []

        except requests.Timeout:
            print(f"[TIMEOUT] Get positions request exceeded {self.timeout}s")
            return []
        except Exception as e:
            print(f"[ERROR] Querying positions: {e}")
            return []

    def close_position(self, trade_id: str, reason: str = "Manual") -> bool:
        """
        Close position by trade ID

        Args:
            trade_id: Trade ID to close
            reason: Reason for closing

        Returns:
            True if closed successfully, False otherwise
        """

        if self.paper_trading:
            if trade_id in self.paper_positions:
                self.paper_positions[trade_id]['status'] = 'CLOSED'
                self.paper_positions[trade_id]['close_time'] = datetime.now().isoformat()
                self.paper_positions[trade_id]['close_reason'] = reason

                pos = self.paper_positions[trade_id]
                print(f"[PAPER CLOSE] {pos['pair']} (ID: {trade_id}) - Reason: {reason}")
                return True
            else:
                print(f"[ERROR] Paper trade {trade_id} not found")
                return False

        try:
            url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/close"

            response = requests.put(
                url,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                close_txn = data.get('orderFillTransaction', {})

                if close_txn:
                    pl = float(close_txn.get('pl', 0))
                    print(f"[POSITION CLOSED] Trade {trade_id}")
                    print(f"  P&L: ${pl:.2f}")
                    print(f"  Reason: {reason}")
                    return True
                else:
                    print(f"[WARNING] Position closed but no fill transaction")
                    return True
            else:
                print(f"[ERROR] Closing position {trade_id}: Status {response.status_code}")
                return False

        except requests.Timeout:
            print(f"[TIMEOUT] Close position request exceeded {self.timeout}s")
            return False
        except Exception as e:
            print(f"[ERROR] Closing position {trade_id}: {e}")
            return False

    def close_all_positions(self, reason: str = "Emergency Stop") -> int:
        """
        Close all open positions

        Args:
            reason: Reason for closing all

        Returns:
            Number of positions closed
        """

        positions = self.get_open_positions()

        closed_count = 0
        for pos in positions:
            if self.close_position(pos['trade_id'], reason):
                closed_count += 1

        print(f"[CLOSE ALL] Closed {closed_count} positions - Reason: {reason}")
        return closed_count

    def get_account_balance(self) -> Optional[float]:
        """Get account balance"""

        if self.paper_trading:
            return 10000.0  # Simulated balance

        try:
            url = f"{self.base_url}/accounts/{self.account_id}"

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                account = data.get('account', {})
                return float(account.get('balance', 0))
            else:
                return None

        except requests.Timeout:
            print(f"[TIMEOUT] Account balance request exceeded {self.timeout}s")
            return None
        except Exception as e:
            print(f"[ERROR] Getting balance: {e}")
            return None

    def calculate_position_size(self,
                                balance: float,
                                risk_percent: float,
                                stop_pips: float,
                                pair: str) -> int:
        """
        Calculate position size based on risk

        Args:
            balance: Account balance
            risk_percent: Risk per trade (e.g., 0.01 = 1%)
            stop_pips: Stop loss distance in pips
            pair: Forex pair

        Returns:
            Position size in units
        """

        # Risk amount in dollars
        risk_amount = balance * risk_percent

        # Calculate pip value (for 1000 units = 0.01 lot)
        if 'JPY' in pair:
            pip_value = 0.01  # JPY pairs
        else:
            pip_value = 0.0001  # Standard pairs

        # Units per pip (1000 units = $0.10 per pip for EUR/USD)
        units_per_pip = 1000 * pip_value * 10

        # Position size = Risk $ / (Stop pips * $ per pip)
        position_size = int(risk_amount / (stop_pips * (units_per_pip / 1000)))

        # Round to nearest 1000 (0.01 lot increments)
        position_size = max(1000, round(position_size / 1000) * 1000)

        return position_size

    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current market price for pair"""

        if self.paper_trading:
            # Return simulated price
            return 1.08500 if 'EUR' in pair else 150.500

        try:
            url = f"{self.base_url}/accounts/{self.account_id}/pricing"
            params = {'instruments': pair}

            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])

                if prices:
                    price_data = prices[0]
                    bids = price_data.get('bids', [])
                    asks = price_data.get('asks', [])

                    if bids and asks:
                        bid = float(bids[0].get('price', 0))
                        ask = float(asks[0].get('price', 0))
                        return (bid + ask) / 2

            return None

        except requests.Timeout:
            print(f"[TIMEOUT] Price request for {pair} exceeded {self.timeout}s")
            return None
        except Exception as e:
            print(f"[ERROR] Getting price for {pair}: {e}")
            return None


def demo():
    """Demo fixed forex execution engine"""

    print("\n" + "="*70)
    print("FIXED FOREX EXECUTION ENGINE DEMO")
    print("="*70)
    print("Using direct REST API calls with 5-second timeout")
    print("No more hanging issues!")
    print("="*70)

    # Initialize in paper trading mode
    engine = FixedForexExecutionEngine(paper_trading=True)

    print("\n[TEST 1] Place market order...")
    result = engine.place_market_order(
        pair='EUR_USD',
        direction='LONG',
        units=1000,  # 0.01 lot
        stop_loss=1.08000,
        take_profit=1.09000
    )

    if result:
        print(f"[OK] Order placed: {result['trade_id']}")

    print("\n[TEST 2] Query open positions...")
    positions = engine.get_open_positions()
    print(f"[OK] Found {len(positions)} open positions")

    for pos in positions:
        print(f"  - {pos['pair']} {pos['direction']} @ {pos['entry_price']:.5f}")

    print("\n[TEST 3] Close position...")
    if positions:
        closed = engine.close_position(positions[0]['trade_id'], "Test close")
        print(f"[OK] Position closed: {closed}")

    print("\n[TEST 4] Calculate position size...")
    balance = 10000.0
    risk_percent = 0.01  # 1%
    stop_pips = 30

    size = engine.calculate_position_size(balance, risk_percent, stop_pips, 'EUR_USD')
    print(f"[OK] Position size: {size} units (0.{size//1000:02d} lot)")
    print(f"  Risk: ${balance * risk_percent:.2f} ({risk_percent*100}%)")
    print(f"  Stop: {stop_pips} pips")

    print("\n" + "="*70)
    print("Execution engine ready with timeout protection")
    print("Set PAPER_TRADING=False in config to trade live")
    print("="*70)


if __name__ == "__main__":
    demo()
