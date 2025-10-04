#!/usr/bin/env python3
"""
Real Alpaca Options Executor
Submits actual options orders to Alpaca paper trading account
"""

import os
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

load_dotenv('.env.paper')

class AlpacaOptionsExecutor:
    """Execute real options orders on Alpaca paper account"""

    def __init__(self):
        self.api = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )

    def find_option_contract(self, symbol: str, strike: float, expiry_days: int, option_type: str):
        """Find the appropriate option contract"""
        try:
            # Calculate target expiry date (closest Friday after expiry_days)
            target_date = datetime.now() + timedelta(days=expiry_days)
            # Move to next Friday
            days_to_friday = (4 - target_date.weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            expiry_date = target_date + timedelta(days=days_to_friday)

            # Search for contracts
            request = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status="active",
                expiration_date_gte=expiry_date.strftime('%Y-%m-%d'),
                expiration_date_lte=(expiry_date + timedelta(days=7)).strftime('%Y-%m-%d'),
                type=option_type.lower(),  # 'call' or 'put' (lowercase required)
                strike_price_gte=str(strike - 2.5),  # Must be string
                strike_price_lte=str(strike + 2.5)   # Must be string
            )

            contracts_response = self.api.get_option_contracts(request)

            # The response is (contracts_dict, headers) tuple
            if isinstance(contracts_response, tuple):
                contracts_dict = contracts_response[0]
            else:
                contracts_dict = contracts_response

            # Extract contracts list
            if hasattr(contracts_dict, 'option_contracts'):
                contracts = contracts_dict.option_contracts
            elif isinstance(contracts_dict, dict) and 'option_contracts' in contracts_dict:
                contracts = contracts_dict['option_contracts']
            else:
                contracts = []

            if not contracts:
                print(f"  WARNING: No {option_type} contracts found for {symbol} ${strike}")
                return None

            # Find closest strike
            best_contract = min(contracts, key=lambda c: abs(float(c.strike_price) - strike))

            return best_contract.symbol

        except Exception as e:
            print(f"  ERROR finding contract: {e}")
            return None

    def execute_straddle(self, symbol: str, current_price: float, contracts: int = 1, expiry_days: int = 14):
        """Execute a long straddle (earnings play)"""
        print(f"\n>>> SUBMITTING REAL OPTIONS ORDERS <<<")
        print(f"Symbol: {symbol}")
        print(f"Strategy: Long Straddle")
        print(f"Strike: ${current_price:.0f} (ATM)")
        print(f"Contracts: {contracts}")
        print(f"Expiry: ~{expiry_days} days")

        results = {
            'symbol': symbol,
            'strategy': 'long_straddle',
            'strike': round(current_price),
            'contracts': contracts,
            'orders': []
        }

        strike = round(current_price)

        # Find and execute CALL
        call_contract = self.find_option_contract(symbol, strike, expiry_days, 'CALL')
        if call_contract:
            try:
                order_request = MarketOrderRequest(
                    symbol=call_contract,
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                order = self.api.submit_order(order_data=order_request)
                print(f"  [OK] CALL order submitted: {order.id}")
                results['orders'].append({
                    'type': 'CALL',
                    'contract': call_contract,
                    'order_id': order.id,
                    'status': order.status
                })
            except Exception as e:
                print(f"  [FAIL] CALL order failed: {e}")
                results['orders'].append({
                    'type': 'CALL',
                    'error': str(e)
                })

        # Find and execute PUT
        put_contract = self.find_option_contract(symbol, strike, expiry_days, 'PUT')
        if put_contract:
            try:
                order_request = MarketOrderRequest(
                    symbol=put_contract,
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                order = self.api.submit_order(order_data=order_request)
                print(f"  [OK] PUT order submitted: {order.id}")
                results['orders'].append({
                    'type': 'PUT',
                    'contract': put_contract,
                    'order_id': order.id,
                    'status': order.status
                })
            except Exception as e:
                print(f"  [FAIL] PUT order failed: {e}")
                results['orders'].append({
                    'type': 'PUT',
                    'error': str(e)
                })

        print(f"\n>>> EXECUTION COMPLETE <<<")
        print(f"Orders submitted: {len([o for o in results['orders'] if 'order_id' in o])}/2")

        return results

    def execute_intel_dual(self, symbol: str, current_price: float, contracts: int = 2, expiry_days: int = 21):
        """Execute Intel dual strategy (cash-secured put + long call)"""
        print(f"\n>>> SUBMITTING REAL OPTIONS ORDERS <<<")
        print(f"Symbol: {symbol}")
        print(f"Strategy: Intel Dual (CSP + Long Call)")
        print(f"Contracts: {contracts}")
        print(f"Expiry: ~{expiry_days} days")

        results = {
            'symbol': symbol,
            'strategy': 'intel_dual',
            'contracts': contracts,
            'orders': []
        }

        # Cash-secured put (4% OTM)
        put_strike = round(current_price * 0.96)
        put_contract = self.find_option_contract(symbol, put_strike, expiry_days, 'PUT')
        if put_contract:
            try:
                # SELL put to collect premium
                order_request = MarketOrderRequest(
                    symbol=put_contract,
                    qty=contracts,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                order = self.api.submit_order(order_data=order_request)
                print(f"  [OK] SELL PUT order submitted: {order.id} @ ${put_strike}")
                results['orders'].append({
                    'type': 'CASH_SECURED_PUT',
                    'strike': put_strike,
                    'contract': put_contract,
                    'order_id': order.id,
                    'status': order.status
                })
            except Exception as e:
                print(f"  [FAIL] SELL PUT failed: {e}")
                results['orders'].append({
                    'type': 'CASH_SECURED_PUT',
                    'error': str(e)
                })

        # Long call (4% OTM)
        call_strike = round(current_price * 1.04)
        call_contract = self.find_option_contract(symbol, call_strike, expiry_days, 'CALL')
        if call_contract:
            try:
                # BUY call for upside
                order_request = MarketOrderRequest(
                    symbol=call_contract,
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                order = self.api.submit_order(order_data=order_request)
                print(f"  [OK] BUY CALL order submitted: {order.id} @ ${call_strike}")
                results['orders'].append({
                    'type': 'LONG_CALL',
                    'strike': call_strike,
                    'contract': call_contract,
                    'order_id': order.id,
                    'status': order.status
                })
            except Exception as e:
                print(f"  [FAIL] BUY CALL failed: {e}")
                results['orders'].append({
                    'type': 'LONG_CALL',
                    'error': str(e)
                })

        print(f"\n>>> EXECUTION COMPLETE <<<")
        print(f"Orders submitted: {len([o for o in results['orders'] if 'order_id' in o])}/2")

        return results

    def get_positions(self):
        """Get current option positions"""
        try:
            positions = self.api.get_all_positions()
            option_positions = [p for p in positions if len(p.symbol) > 10]  # Options have long symbols
            return option_positions
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def get_account_status(self):
        """Get account info"""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'options_enabled': hasattr(account, 'options_trading_level'),
                'options_level': getattr(account, 'options_trading_level', None)
            }
        except Exception as e:
            print(f"Error getting account: {e}")
            return None


if __name__ == "__main__":
    # Test the executor
    executor = AlpacaOptionsExecutor()

    print("ALPACA OPTIONS EXECUTOR - TEST")
    print("=" * 60)

    # Check account
    account = executor.get_account_status()
    if account:
        print(f"\nAccount Status:")
        print(f"  Buying Power: ${account['buying_power']:,.2f}")
        print(f"  Cash: ${account['cash']:,.2f}")
        print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"  Options Level: {account['options_level']}")

    # Check positions
    positions = executor.get_positions()
    print(f"\nCurrent Option Positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.qty} @ ${pos.avg_entry_price}")

    print("\n" + "=" * 60)
    print("Executor ready for live trading!")
