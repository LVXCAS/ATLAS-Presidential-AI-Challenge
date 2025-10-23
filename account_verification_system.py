#!/usr/bin/env python3
"""
ACCOUNT VERIFICATION SYSTEM
Verifies account capabilities before scanning to prevent wrong-account trading
"""

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi


class AccountVerificationSystem:
    """Verify account is ready for options trading before scanning"""

    def __init__(self, api_key=None, secret_key=None, base_url=None):
        """
        Initialize account verification system

        Args:
            api_key: Optional Alpaca API key (if not provided, loads from environment)
            secret_key: Optional Alpaca secret key
            base_url: Optional Alpaca base URL
        """
        # Only load environment if credentials not provided
        # This prevents overriding credentials set by parent systems
        if not api_key:
            # Use override=True to ensure we load the CORRECT credentials
            load_dotenv(override=True)
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL')

        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )

    def verify_account_ready(self, strategy_type='BULL_PUT_SPREAD'):
        """
        Verify account has necessary capabilities for trading strategy

        Args:
            strategy_type: 'BULL_PUT_SPREAD', 'DUAL_OPTIONS', 'STOCK'

        Returns:
            dict: {
                'ready': bool,
                'account_id': str,
                'equity': float,
                'options_buying_power': float,
                'issues': list,
                'warnings': list
            }
        """

        print("\n" + "="*70)
        print("ACCOUNT VERIFICATION SYSTEM")
        print("="*70)

        issues = []
        warnings = []

        # Get account details
        try:
            account = self.api.get_account()
        except Exception as e:
            return {
                'ready': False,
                'account_id': None,
                'equity': 0,
                'options_buying_power': 0,
                'issues': [f"Cannot connect to account: {e}"],
                'warnings': []
            }

        account_id = account.account_number
        equity = float(account.equity)
        cash = float(account.cash)
        buying_power = float(account.buying_power)
        options_buying_power = float(account.options_buying_power)
        options_level = int(account.options_trading_level)

        print(f"\n[ACCOUNT DETAILS]")
        print(f"  Account ID: {account_id}")
        print(f"  Equity: ${equity:,.2f}")
        print(f"  Cash: ${cash:,.2f}")
        print(f"  Buying Power: ${buying_power:,.2f}")
        print(f"  Options Buying Power: ${options_buying_power:,.2f}")
        print(f"  Options Level: {options_level}")

        # Check 1: Account equity
        if equity < 1000:
            issues.append(f"Account equity too low: ${equity:.2f} (minimum $1,000)")
        elif equity < 10000:
            warnings.append(f"Low account equity: ${equity:.2f} (recommended $10,000+)")

        # Check 2: Negative cash (margin call risk)
        # NOTE: Negative cash is NORMAL for options spreads (collateral held)
        # Only flag as critical if buying power is ALSO low (true margin call)
        if cash < 0 and buying_power < 1000:
            issues.append(f"MARGIN CALL RISK: Negative cash ${cash:.2f} + low buying power ${buying_power:.2f}")
        elif cash < 0:
            warnings.append(f"Negative cash ${cash:.2f} (normal for options spreads, buying power ${buying_power:,.2f} available)")
        elif cash < 1000:
            warnings.append(f"Low cash: ${cash:.2f} (may limit trading)")

        # Check 3: Options buying power (strategy-specific)
        if strategy_type in ['BULL_PUT_SPREAD', 'IRON_CONDOR', 'BUTTERFLY']:
            if options_buying_power == 0:
                issues.append(f"ZERO options buying power (cannot trade spreads!)")
            elif options_buying_power < 5000:
                warnings.append(f"Low options buying power: ${options_buying_power:.2f}")

        elif strategy_type == 'DUAL_OPTIONS':
            # Dual options needs HUGE buying power for cash-secured puts
            if options_buying_power < 10000:
                issues.append(f"Insufficient options buying power for Dual Options: ${options_buying_power:.2f} (need $10,000+)")

        # Check 4: Options approval level
        if options_level < 2:
            issues.append(f"Options level {options_level} too low (need level 2+ for spreads)")

        # Check 5: Position count
        positions = self.api.list_positions()
        position_count = len(positions)

        if position_count > 20:
            warnings.append(f"High position count: {position_count} positions open")

        # Check 6: Verify this is the INTENDED account
        # Paper trading accounts typically start with $100k, which is VALID
        # Only flag as wrong account if it's LIVE (not paper) and equity doesn't match expected
        is_paper_trading = 'paper' in os.getenv('ALPACA_BASE_URL', '').lower()

        if 90000 < equity < 100000 and not is_paper_trading:
            issues.append(f"WARNING: WRONG ACCOUNT DETECTED! Equity ${equity:,.2f} suggests secondary account (not main $956k account)")
        elif 90000 < equity < 100000 and is_paper_trading:
            print(f"  [NOTE] Paper trading account detected (${equity:,.2f}) - this is EXPECTED")

        # Summary
        print(f"\n[VERIFICATION RESULTS]")

        if issues:
            print(f"\n[X] CRITICAL ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")

        if warnings:
            print(f"\n[!] WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"  - {warning}")

        ready = len(issues) == 0

        if ready:
            print(f"\n[OK] ACCOUNT READY FOR {strategy_type} TRADING")
        else:
            print(f"\n[ERROR] ACCOUNT NOT READY - FIX ISSUES BEFORE TRADING")

        print("="*70 + "\n")

        return {
            'ready': ready,
            'account_id': account_id,
            'equity': equity,
            'cash': cash,
            'buying_power': buying_power,
            'options_buying_power': options_buying_power,
            'options_level': options_level,
            'position_count': position_count,
            'issues': issues,
            'warnings': warnings,
            'strategy_type': strategy_type
        }


def test_verification():
    """Test account verification"""

    verifier = AccountVerificationSystem()

    print("\n" + "="*70)
    print("TESTING ACCOUNT VERIFICATION")
    print("="*70)

    # Test different strategies
    strategies = ['BULL_PUT_SPREAD', 'DUAL_OPTIONS', 'STOCK']

    for strategy in strategies:
        print(f"\n\n{'='*70}")
        print(f"TESTING STRATEGY: {strategy}")
        print(f"{'='*70}")

        result = verifier.verify_account_ready(strategy)

        if not result['ready']:
            print(f"\n[WARNING] CANNOT TRADE {strategy} - FIX ISSUES FIRST")


if __name__ == "__main__":
    test_verification()
