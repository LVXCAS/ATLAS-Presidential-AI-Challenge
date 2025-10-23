#!/usr/bin/env python3
"""
COMPREHENSIVE POSITION MONITOR
Tracks all active trades across OPTIONS, FOREX, and FUTURES
Real-time P&L calculations with color-coded output

Usage:
    python monitor_positions.py           # Single snapshot
    python monitor_positions.py --watch   # Auto-refresh every 30 seconds
    python monitor_positions.py --json    # JSON output
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from decimal import Decimal

# Try to import required libraries
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from dotenv import load_dotenv
    load_dotenv()
    ALPACA_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Alpaca not available: {e}")
    ALPACA_AVAILABLE = False

try:
    import oandapyV20
    from oandapyV20 import API
    from oandapyV20.endpoints.positions import OpenPositions
    from oandapyV20.endpoints.pricing import PricingInfo
    OANDA_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] OANDA not available: {e}")
    OANDA_AVAILABLE = False

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    @staticmethod
    def disable():
        """Disable colors for non-TTY output"""
        Colors.GREEN = ''
        Colors.RED = ''
        Colors.YELLOW = ''
        Colors.BLUE = ''
        Colors.MAGENTA = ''
        Colors.CYAN = ''
        Colors.WHITE = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''
        Colors.RESET = ''


class PositionMonitor:
    """Monitor positions across all brokers and asset types"""

    def __init__(self):
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.alpaca_base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        self.oanda_api_key = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.oanda_base_url = os.getenv('OANDA_BASE_URL', 'https://api-fxpractice.oanda.com')

        # Initialize clients
        self.alpaca_trading_client = None
        self.alpaca_data_client = None
        self.oanda_client = None

        if ALPACA_AVAILABLE and self.alpaca_api_key and self.alpaca_secret_key:
            try:
                self.alpaca_trading_client = TradingClient(
                    self.alpaca_api_key,
                    self.alpaca_secret_key,
                    paper=True if 'paper' in self.alpaca_base_url else False
                )
                self.alpaca_data_client = StockHistoricalDataClient(
                    self.alpaca_api_key,
                    self.alpaca_secret_key
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize Alpaca: {e}")

        if OANDA_AVAILABLE and self.oanda_api_key and self.oanda_account_id:
            try:
                self.oanda_client = API(
                    access_token=self.oanda_api_key,
                    environment='practice' if 'practice' in self.oanda_base_url else 'live'
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize OANDA: {e}")

    def get_alpaca_options_positions(self) -> List[Dict]:
        """Get all options positions from Alpaca"""
        if not self.alpaca_trading_client:
            return []

        try:
            positions = self.alpaca_trading_client.get_all_positions()
            options_positions = []

            for pos in positions:
                # Check if it's an options contract (OCC format: SYMBOL + YYMMDD + C/P + STRIKE)
                if len(pos.symbol) > 10:
                    try:
                        # Parse options symbol
                        underlying = self._parse_options_symbol(pos.symbol)

                        position_data = {
                            'symbol': pos.symbol,
                            'underlying': underlying,
                            'asset_class': pos.asset_class,
                            'qty': float(pos.qty),
                            'side': pos.side,
                            'entry_price': float(pos.avg_entry_price),
                            'current_price': float(pos.current_price),
                            'market_value': float(pos.market_value),
                            'cost_basis': float(pos.cost_basis),
                            'unrealized_pl': float(pos.unrealized_pl),
                            'unrealized_plpc': float(pos.unrealized_plpc),
                            'unrealized_intraday_pl': float(pos.unrealized_intraday_pl),
                            'unrealized_intraday_plpc': float(pos.unrealized_intraday_plpc),
                            'change_today': float(pos.change_today)
                        }
                        options_positions.append(position_data)
                    except Exception as e:
                        print(f"[WARNING] Error parsing options position {pos.symbol}: {e}")

            return options_positions

        except Exception as e:
            print(f"[ERROR] Failed to get Alpaca options positions: {e}")
            return []

    def get_alpaca_futures_positions(self) -> List[Dict]:
        """Get all futures positions from Alpaca"""
        if not self.alpaca_trading_client:
            return []

        try:
            positions = self.alpaca_trading_client.get_all_positions()
            futures_positions = []

            for pos in positions:
                # Futures typically have specific formats (e.g., ESZ2025)
                if pos.asset_class == 'us_option':
                    continue  # Skip options

                # Check for futures format
                if any(month in pos.symbol[-6:] for month in ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']):
                    position_data = {
                        'symbol': pos.symbol,
                        'asset_class': pos.asset_class,
                        'qty': float(pos.qty),
                        'side': pos.side,
                        'entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'market_value': float(pos.market_value),
                        'cost_basis': float(pos.cost_basis),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc),
                        'change_today': float(pos.change_today)
                    }
                    futures_positions.append(position_data)

            return futures_positions

        except Exception as e:
            print(f"[ERROR] Failed to get Alpaca futures positions: {e}")
            return []

    def get_oanda_forex_positions(self) -> List[Dict]:
        """Get all forex positions from OANDA"""
        if not self.oanda_client or not self.oanda_account_id:
            return []

        try:
            # Get open positions
            r = OpenPositions(accountID=self.oanda_account_id)
            response = self.oanda_client.request(r)

            forex_positions = []

            if 'positions' in response:
                for pos in response['positions']:
                    # Get current pricing
                    try:
                        pricing_params = {"instruments": pos['instrument']}
                        pricing_request = PricingInfo(
                            accountID=self.oanda_account_id,
                            params=pricing_params
                        )
                        pricing_response = self.oanda_client.request(pricing_request)

                        current_price = None
                        if 'prices' in pricing_response and len(pricing_response['prices']) > 0:
                            price_data = pricing_response['prices'][0]
                            current_price = float(price_data['closeoutBid'])
                    except:
                        current_price = None

                    # Calculate P&L
                    long_units = float(pos['long']['units']) if 'long' in pos else 0
                    short_units = float(pos['short']['units']) if 'short' in pos else 0

                    long_pl = float(pos['long']['unrealizedPL']) if 'long' in pos else 0
                    short_pl = float(pos['short']['unrealizedPL']) if 'short' in pos else 0

                    total_units = long_units + short_units
                    total_pl = long_pl + short_pl

                    if total_units != 0:
                        position_data = {
                            'instrument': pos['instrument'],
                            'units': total_units,
                            'long_units': long_units,
                            'short_units': short_units,
                            'unrealized_pl': total_pl,
                            'long_pl': long_pl,
                            'short_pl': short_pl,
                            'current_price': current_price,
                            'long_avg_price': float(pos['long']['averagePrice']) if 'long' in pos and long_units != 0 else None,
                            'short_avg_price': float(pos['short']['averagePrice']) if 'short' in pos and short_units != 0 else None
                        }
                        forex_positions.append(position_data)

            return forex_positions

        except Exception as e:
            print(f"[ERROR] Failed to get OANDA forex positions: {e}")
            return []

    def _parse_options_symbol(self, symbol: str) -> str:
        """Parse OCC options symbol to get underlying"""
        # OCC format: SYMBOL + YYMMDD + C/P + STRIKE (padded to 8 digits)
        # Example: AAPL251121C00150000 -> AAPL

        # Find where the date starts (6 consecutive digits)
        for i in range(len(symbol) - 5):
            if symbol[i:i+6].isdigit():
                return symbol[:i]
        return symbol

    def _format_options_contract(self, symbol: str) -> Tuple[str, str, str, str]:
        """Parse and format options contract details"""
        # Parse OCC symbol
        for i in range(len(symbol) - 5):
            if symbol[i:i+6].isdigit():
                underlying = symbol[:i]
                date_str = symbol[i:i+6]
                option_type = symbol[i+6]
                strike_str = symbol[i+7:]

                # Format date
                exp_date = f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]}"

                # Format strike
                strike = float(strike_str) / 1000

                return underlying, exp_date, option_type, f"${strike:.2f}"

        return symbol, "Unknown", "?", "$0.00"

    def _group_options_spreads(self, positions: List[Dict]) -> List[List[Dict]]:
        """Group options positions into spreads"""
        # Group by underlying and expiration
        from collections import defaultdict

        groups = defaultdict(list)

        for pos in positions:
            underlying, exp_date, _, _ = self._format_options_contract(pos['symbol'])
            key = f"{underlying}_{exp_date}"
            groups[key].append(pos)

        # Convert to list of groups
        return [group for group in groups.values() if len(group) > 0]

    def display_positions_terminal(self):
        """Display positions in terminal with color coding"""
        print("\n" + "=" * 80)
        print(f"{Colors.BOLD}{Colors.CYAN}POSITION MONITOR - All Active Trades{Colors.RESET}")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%I:%M:%S %p %Z')}")
        print(f"Account: {Colors.YELLOW}Alpaca Paper{Colors.RESET} | {Colors.YELLOW}OANDA Practice{Colors.RESET}")
        print()

        total_pl = 0.0

        # OPTIONS POSITIONS
        print(f"{Colors.BOLD}{Colors.MAGENTA}OPTIONS POSITIONS (Alpaca Paper):{Colors.RESET}")
        print("-" * 80)

        options_positions = self.get_alpaca_options_positions()

        if options_positions:
            # Group into spreads
            spread_groups = self._group_options_spreads(options_positions)

            for idx, group in enumerate(spread_groups, 1):
                group_pl = sum(pos['unrealized_pl'] for pos in group)
                total_pl += group_pl

                # Determine if it's a spread
                if len(group) > 1:
                    underlying, exp_date, _, _ = self._format_options_contract(group[0]['symbol'])

                    # Identify spread type
                    if all(pos['side'] == 'short' for pos in group):
                        spread_type = "Credit Spread"
                    elif all(pos['side'] == 'long' for pos in group):
                        spread_type = "Debit Spread"
                    else:
                        spread_type = "Bull Put Spread" if any(pos['side'] == 'short' for pos in group) else "Spread"

                    print(f"{idx}. {Colors.BOLD}{underlying} {spread_type}{Colors.RESET}")

                    for pos in sorted(group, key=lambda x: x['side'], reverse=True):
                        _, exp_date, opt_type, strike = self._format_options_contract(pos['symbol'])
                        side_label = "Sell" if pos['side'] == 'short' else "Buy"

                        print(f"   {side_label}: {pos['symbol'][:6]}{exp_date[2:]}{opt_type}{strike} "
                              f"x{abs(pos['qty']):.0f} @ ${pos['entry_price']:.2f}")

                    # Calculate net credit/debit
                    net_credit = sum(
                        pos['cost_basis'] if pos['side'] == 'short' else -pos['cost_basis']
                        for pos in group
                    )

                    print(f"   Net {'Credit' if net_credit > 0 else 'Debit'}: "
                          f"{Colors.GREEN if net_credit > 0 else Colors.RED}${abs(net_credit):.2f}{Colors.RESET}")

                    # Get current underlying price
                    underlying_symbol, _, _, _ = self._format_options_contract(group[0]['symbol'])
                    try:
                        if self.alpaca_data_client:
                            request = StockLatestQuoteRequest(symbol_or_symbols=underlying_symbol)
                            quote = self.alpaca_data_client.get_stock_latest_quote(request)
                            current_price = float(quote[underlying_symbol].ask_price)
                            print(f"   Current: {underlying_symbol} @ ${current_price:.2f}")
                    except:
                        pass

                    # Status
                    status_color = Colors.GREEN if group_pl > 0 else Colors.RED
                    status_symbol = "+" if group_pl > 0 else "-"
                    status_text = "WINNING" if group_pl > 0 else "LOSING"

                    print(f"   Status: {status_color}[{status_symbol}] {status_text}{Colors.RESET}")

                    # Unrealized P&L
                    pl_pct = (group_pl / abs(net_credit) * 100) if net_credit != 0 else 0
                    pl_color = Colors.GREEN if group_pl >= 0 else Colors.RED
                    pl_sign = "+" if group_pl >= 0 else ""

                    print(f"   Unrealized P&L: {pl_color}{pl_sign}${group_pl:.2f} ({pl_sign}{pl_pct:.1f}%){Colors.RESET}")

                else:
                    # Single leg position
                    pos = group[0]
                    underlying, exp_date, opt_type, strike = self._format_options_contract(pos['symbol'])

                    side_label = "Short" if pos['side'] == 'short' else "Long"
                    opt_type_full = "Call" if opt_type == 'C' else "Put"

                    print(f"{idx}. {Colors.BOLD}{underlying} {side_label} {opt_type_full}{Colors.RESET}")
                    print(f"   {pos['symbol']} x{abs(pos['qty']):.0f} @ ${pos['entry_price']:.2f}")
                    print(f"   Strike: {strike} | Exp: {exp_date}")

                    # Current price and P&L
                    pl_color = Colors.GREEN if pos['unrealized_pl'] >= 0 else Colors.RED
                    pl_sign = "+" if pos['unrealized_pl'] >= 0 else ""

                    print(f"   Current Price: ${pos['current_price']:.2f}")
                    print(f"   Market Value: ${pos['market_value']:.2f}")
                    print(f"   Unrealized P&L: {pl_color}{pl_sign}${pos['unrealized_pl']:.2f} "
                          f"({pl_sign}{pos['unrealized_plpc']:.1f}%){Colors.RESET}")

                print()

            # Options total
            options_total_pl = sum(pos['unrealized_pl'] for pos in options_positions)
            pl_color = Colors.GREEN if options_total_pl >= 0 else Colors.RED
            pl_sign = "+" if options_total_pl >= 0 else ""
            print(f"{Colors.BOLD}OPTIONS TOTAL: {pl_color}{pl_sign}${options_total_pl:.2f}{Colors.RESET}")
            print()
        else:
            print(f"{Colors.YELLOW}No open positions{Colors.RESET}")
            print()

        # FOREX POSITIONS
        print(f"{Colors.BOLD}{Colors.MAGENTA}FOREX POSITIONS (OANDA):{Colors.RESET}")
        print("-" * 80)

        forex_positions = self.get_oanda_forex_positions()

        if forex_positions:
            for idx, pos in enumerate(forex_positions, 1):
                print(f"{idx}. {Colors.BOLD}{pos['instrument']}{Colors.RESET}")

                if pos['long_units'] != 0:
                    print(f"   Long: {pos['long_units']:,.0f} units @ {pos['long_avg_price']:.5f}")
                    pl_color = Colors.GREEN if pos['long_pl'] >= 0 else Colors.RED
                    pl_sign = "+" if pos['long_pl'] >= 0 else ""
                    print(f"   Long P&L: {pl_color}{pl_sign}${pos['long_pl']:.2f}{Colors.RESET}")

                if pos['short_units'] != 0:
                    print(f"   Short: {pos['short_units']:,.0f} units @ {pos['short_avg_price']:.5f}")
                    pl_color = Colors.GREEN if pos['short_pl'] >= 0 else Colors.RED
                    pl_sign = "+" if pos['short_pl'] >= 0 else ""
                    print(f"   Short P&L: {pl_color}{pl_sign}${pos['short_pl']:.2f}{Colors.RESET}")

                if pos['current_price']:
                    print(f"   Current Price: {pos['current_price']:.5f}")

                # Total P&L for this instrument
                total_pos_pl = pos['unrealized_pl']
                total_pl += total_pos_pl

                pl_color = Colors.GREEN if total_pos_pl >= 0 else Colors.RED
                pl_sign = "+" if total_pos_pl >= 0 else ""
                print(f"   {Colors.BOLD}Total P&L: {pl_color}{pl_sign}${total_pos_pl:.2f}{Colors.RESET}")
                print()

            # Forex total
            forex_total_pl = sum(pos['unrealized_pl'] for pos in forex_positions)
            pl_color = Colors.GREEN if forex_total_pl >= 0 else Colors.RED
            pl_sign = "+" if forex_total_pl >= 0 else ""
            print(f"{Colors.BOLD}FOREX TOTAL: {pl_color}{pl_sign}${forex_total_pl:.2f}{Colors.RESET}")
            print()
        else:
            print(f"{Colors.YELLOW}No open positions{Colors.RESET}")
            print()

        # FUTURES POSITIONS
        print(f"{Colors.BOLD}{Colors.MAGENTA}FUTURES POSITIONS (Alpaca):{Colors.RESET}")
        print("-" * 80)

        futures_positions = self.get_alpaca_futures_positions()

        if futures_positions:
            for idx, pos in enumerate(futures_positions, 1):
                print(f"{idx}. {Colors.BOLD}{pos['symbol']}{Colors.RESET}")
                print(f"   Quantity: {pos['qty']:.0f} contracts")
                print(f"   Entry Price: ${pos['entry_price']:.2f}")
                print(f"   Current Price: ${pos['current_price']:.2f}")
                print(f"   Market Value: ${pos['market_value']:.2f}")

                pl_color = Colors.GREEN if pos['unrealized_pl'] >= 0 else Colors.RED
                pl_sign = "+" if pos['unrealized_pl'] >= 0 else ""

                print(f"   Unrealized P&L: {pl_color}{pl_sign}${pos['unrealized_pl']:.2f} "
                      f"({pl_sign}{pos['unrealized_plpc']:.1f}%){Colors.RESET}")
                print()

                total_pl += pos['unrealized_pl']

            # Futures total
            futures_total_pl = sum(pos['unrealized_pl'] for pos in futures_positions)
            pl_color = Colors.GREEN if futures_total_pl >= 0 else Colors.RED
            pl_sign = "+" if futures_total_pl >= 0 else ""
            print(f"{Colors.BOLD}FUTURES TOTAL: {pl_color}{pl_sign}${futures_total_pl:.2f}{Colors.RESET}")
            print()
        else:
            print(f"{Colors.YELLOW}No open positions{Colors.RESET}")
            print()

        # OVERALL SUMMARY
        print("=" * 80)
        pl_color = Colors.GREEN if total_pl >= 0 else Colors.RED
        pl_sign = "+" if total_pl >= 0 else ""
        print(f"{Colors.BOLD}OVERALL P&L: {pl_color}{pl_sign}${total_pl:.2f}{Colors.RESET}")
        print("=" * 80)
        print()

    def get_positions_json(self) -> Dict:
        """Get all positions as JSON"""
        return {
            'timestamp': datetime.now().isoformat(),
            'options': self.get_alpaca_options_positions(),
            'forex': self.get_oanda_forex_positions(),
            'futures': self.get_alpaca_futures_positions(),
            'summary': {
                'options_pl': sum(pos['unrealized_pl'] for pos in self.get_alpaca_options_positions()),
                'forex_pl': sum(pos['unrealized_pl'] for pos in self.get_oanda_forex_positions()),
                'futures_pl': sum(pos['unrealized_pl'] for pos in self.get_alpaca_futures_positions()),
                'total_pl': (
                    sum(pos['unrealized_pl'] for pos in self.get_alpaca_options_positions()) +
                    sum(pos['unrealized_pl'] for pos in self.get_oanda_forex_positions()) +
                    sum(pos['unrealized_pl'] for pos in self.get_alpaca_futures_positions())
                )
            }
        }


def main():
    """Main entry point"""
    # Set UTF-8 encoding for Windows console
    if sys.platform.startswith('win'):
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        except:
            pass  # If fails, continue with default encoding

    parser = argparse.ArgumentParser(description='Monitor all trading positions')
    parser.add_argument('--watch', action='store_true', help='Auto-refresh every 30 seconds')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval in seconds (default: 30)')

    args = parser.parse_args()

    # Disable colors for JSON output or non-TTY
    if args.json or not sys.stdout.isatty():
        Colors.disable()

    monitor = PositionMonitor()

    if args.json:
        # JSON output
        data = monitor.get_positions_json()
        print(json.dumps(data, indent=2))
    elif args.watch:
        # Watch mode - refresh periodically
        try:
            while True:
                # Clear screen (works on both Windows and Unix)
                os.system('cls' if os.name == 'nt' else 'clear')
                monitor.display_positions_terminal()
                print(f"Refreshing in {args.interval} seconds... (Ctrl+C to stop)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    else:
        # Single snapshot
        monitor.display_positions_terminal()


if __name__ == "__main__":
    main()
