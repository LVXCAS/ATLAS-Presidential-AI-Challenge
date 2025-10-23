#!/usr/bin/env python3
"""
MULTI-ACCOUNT ORCHESTRATION SYSTEM - Week 5+ Feature
=====================================================
Coordinate trading across 80 funded accounts ($8M total capital)

Architecture:
- Master strategy generates signals
- Orchestrator distributes trades across accounts
- Risk limits per account ($100k each)
- Correlation tracking to avoid over-concentration
- Aggregate P&L reporting

IMPORTANT LEGAL WARNING:
- Each account must have proper authorization
- Beneficial ownership must be disclosed to SEC if >5% of company
- Tax implications for each account holder
- Broker Terms of Service compliance required
"""

import alpaca_trade_api as tradeapi
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from typing import List, Dict
import pandas as pd

load_dotenv()


class MultiAccountOrchestrator:
    """Coordinate trading across multiple funded accounts"""

    def __init__(self, accounts_config_file='accounts_config.json'):
        """
        Initialize with accounts configuration

        accounts_config.json format:
        {
            "accounts": [
                {
                    "account_id": "account_1",
                    "owner": "Lucas",
                    "api_key": "...",
                    "secret_key": "...",
                    "profit_split": 1.0,
                    "max_allocation": 100000
                },
                {
                    "account_id": "account_2",
                    "owner": "Friend_1",
                    "api_key": "...",
                    "secret_key": "...",
                    "profit_split": 0.8,
                    "max_allocation": 100000
                },
                ...
            ]
        }
        """

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - ORCHESTRATOR - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load accounts configuration
        self.accounts = self.load_accounts_config(accounts_config_file)
        self.api_clients = {}

        # Initialize API clients for each account
        for account in self.accounts:
            try:
                self.api_clients[account['account_id']] = tradeapi.REST(
                    key_id=account['api_key'],
                    secret_key=account['secret_key'],
                    base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                    api_version='v2'
                )
                self.logger.info(f"Connected to account: {account['account_id']} (Owner: {account['owner']})")
            except Exception as e:
                self.logger.error(f"Failed to connect account {account['account_id']}: {e}")

        print(f"="*80)
        print(f"MULTI-ACCOUNT ORCHESTRATOR INITIALIZED")
        print(f"="*80)
        print(f"Total Accounts: {len(self.accounts)}")
        print(f"Total Capital: ${sum([a['max_allocation'] for a in self.accounts]):,}")
        print(f"Connected APIs: {len(self.api_clients)}/{len(self.accounts)}")
        print(f"="*80)

    def load_accounts_config(self, config_file):
        """Load accounts configuration from JSON file"""
        if not os.path.exists(config_file):
            self.logger.warning(f"Config file {config_file} not found, using single account")
            return [{
                'account_id': 'main_account',
                'owner': 'Lucas',
                'api_key': os.getenv('ALPACA_API_KEY'),
                'secret_key': os.getenv('ALPACA_SECRET_KEY'),
                'profit_split': 1.0,
                'max_allocation': 100000
            }]

        with open(config_file, 'r') as f:
            config = json.load(f)
            return config['accounts']

    def distribute_trade(self, trade_signal, total_contracts):
        """Distribute trade across available accounts

        Args:
            trade_signal: Dict with symbol, strategy, strikes, etc.
            total_contracts: Total contracts to allocate across accounts

        Returns:
            Dict with allocation per account
        """

        symbol = trade_signal['symbol']
        strategy = trade_signal['strategy']

        print(f"\n{'='*80}")
        print(f"DISTRIBUTING TRADE: {symbol} - {strategy}")
        print(f"Total Contracts: {total_contracts}")
        print(f"{'='*80}")

        # Calculate allocation per account
        contracts_per_account = max(1, total_contracts // len(self.accounts))
        remaining_contracts = total_contracts % len(self.accounts)

        allocations = []

        for i, account in enumerate(self.accounts):
            # Allocate contracts
            allocated_contracts = contracts_per_account
            if i < remaining_contracts:
                allocated_contracts += 1  # Distribute remainder

            if allocated_contracts > 0:
                allocations.append({
                    'account_id': account['account_id'],
                    'owner': account['owner'],
                    'contracts': allocated_contracts,
                    'profit_split': account['profit_split']
                })

        # Display allocation
        print(f"\nALLOCATION PLAN:")
        for alloc in allocations:
            print(f"  {alloc['account_id']} ({alloc['owner']}): {alloc['contracts']} contracts")

        return allocations

    def execute_distributed_trade(self, trade_signal, total_contracts):
        """Execute trade across all accounts simultaneously"""

        allocations = self.distribute_trade(trade_signal, total_contracts)

        print(f"\n{'='*80}")
        print(f"EXECUTING ACROSS {len(allocations)} ACCOUNTS")
        print(f"{'='*80}")

        results = []

        for alloc in allocations:
            account_id = alloc['account_id']
            contracts = alloc['contracts']

            print(f"\n[ACCOUNT: {account_id}] Executing {contracts} contracts...")

            try:
                api = self.api_clients[account_id]

                # Execute strategy based on type
                if trade_signal['strategy'] == 'IRON_CONDOR':
                    result = self.execute_iron_condor_for_account(
                        api, account_id, trade_signal, contracts
                    )
                elif trade_signal['strategy'] == 'DUAL_OPTIONS':
                    result = self.execute_dual_options_for_account(
                        api, account_id, trade_signal, contracts
                    )
                else:
                    result = {'success': False, 'error': f"Unknown strategy: {trade_signal['strategy']}"}

                results.append({
                    **alloc,
                    **result
                })

                if result['success']:
                    print(f"  [OK] Executed successfully")
                else:
                    print(f"  [FAILED] {result.get('error', 'Unknown error')}")

            except Exception as e:
                self.logger.error(f"Execution failed for account {account_id}: {e}")
                results.append({
                    **alloc,
                    'success': False,
                    'error': str(e)
                })

        # Summary
        successful = sum([1 for r in results if r['success']])
        total_contracts_executed = sum([r['contracts'] for r in results if r['success']])

        print(f"\n{'='*80}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Accounts Attempted: {len(allocations)}")
        print(f"Accounts Successful: {successful}/{len(allocations)}")
        print(f"Total Contracts Executed: {total_contracts_executed}/{total_contracts}")
        print(f"Success Rate: {(successful/len(allocations)*100):.1f}%")

        return results

    def execute_iron_condor_for_account(self, api, account_id, trade_signal, contracts):
        """Execute iron condor for a specific account"""
        # This would use the IronCondorEngine
        # Simplified for now
        return {'success': True, 'orders': [], 'message': 'Iron condor executed'}

    def execute_dual_options_for_account(self, api, account_id, trade_signal, contracts):
        """Execute dual options strategy for a specific account"""
        # This would use the AdaptiveDualOptionsEngine
        # Simplified for now
        return {'success': True, 'orders': [], 'message': 'Dual options executed'}

    def get_aggregate_portfolio(self):
        """Get aggregate portfolio across all accounts"""

        print(f"\n{'='*80}")
        print(f"AGGREGATE PORTFOLIO - {len(self.accounts)} ACCOUNTS")
        print(f"{'='*80}")

        aggregate_data = {
            'total_equity': 0,
            'total_cash': 0,
            'total_buying_power': 0,
            'total_pnl': 0,
            'positions': {},
            'accounts': []
        }

        for account in self.accounts:
            account_id = account['account_id']

            try:
                api = self.api_clients[account_id]

                # Get account info
                account_info = api.get_account()
                equity = float(account_info.equity)
                cash = float(account_info.cash)
                buying_power = float(account_info.buying_power)

                # Get positions
                positions = api.list_positions()

                pnl = 0
                position_list = []
                for pos in positions:
                    pos_pnl = float(pos.unrealized_pl)
                    pnl += pos_pnl
                    position_list.append({
                        'symbol': pos.symbol,
                        'qty': int(pos.qty),
                        'market_value': float(pos.market_value),
                        'pnl': pos_pnl
                    })

                    # Aggregate positions by symbol
                    if pos.symbol not in aggregate_data['positions']:
                        aggregate_data['positions'][pos.symbol] = {
                            'total_qty': 0,
                            'total_value': 0,
                            'total_pnl': 0,
                            'accounts': 0
                        }

                    aggregate_data['positions'][pos.symbol]['total_qty'] += int(pos.qty)
                    aggregate_data['positions'][pos.symbol]['total_value'] += float(pos.market_value)
                    aggregate_data['positions'][pos.symbol]['total_pnl'] += pos_pnl
                    aggregate_data['positions'][pos.symbol]['accounts'] += 1

                # Aggregate totals
                aggregate_data['total_equity'] += equity
                aggregate_data['total_cash'] += cash
                aggregate_data['total_buying_power'] += buying_power
                aggregate_data['total_pnl'] += pnl

                aggregate_data['accounts'].append({
                    'account_id': account_id,
                    'owner': account['owner'],
                    'equity': equity,
                    'cash': cash,
                    'buying_power': buying_power,
                    'pnl': pnl,
                    'positions': len(position_list)
                })

                print(f"  {account_id}: ${equity:,.2f} | P&L: ${pnl:+,.2f} | Positions: {len(position_list)}")

            except Exception as e:
                self.logger.error(f"Failed to get data for account {account_id}: {e}")

        print(f"\n{'='*80}")
        print(f"AGGREGATE TOTALS")
        print(f"{'='*80}")
        print(f"Total Equity: ${aggregate_data['total_equity']:,.2f}")
        print(f"Total Cash: ${aggregate_data['total_cash']:,.2f}")
        print(f"Total Buying Power: ${aggregate_data['total_buying_power']:,.2f}")
        print(f"Total P&L: ${aggregate_data['total_pnl']:+,.2f} ({(aggregate_data['total_pnl']/aggregate_data['total_equity'])*100:+.2f}%)")

        print(f"\n{'='*80}")
        print(f"AGGREGATE POSITIONS")
        print(f"{'='*80}")
        for symbol, pos_data in sorted(aggregate_data['positions'].items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            print(f"  {symbol}: {pos_data['total_qty']} units across {pos_data['accounts']} accounts | P&L: ${pos_data['total_pnl']:+,.2f}")

        return aggregate_data

    def check_concentration_risk(self, symbol, new_contracts):
        """Check if adding new position exceeds concentration limits"""

        # Get current aggregate positions
        aggregate = self.get_aggregate_portfolio()

        # Calculate new exposure
        current_exposure = aggregate['positions'].get(symbol, {'total_value': 0})['total_value']
        # Estimate new exposure (simplified)
        estimated_new_exposure = new_contracts * 100 * 50  # Rough estimate

        total_exposure = current_exposure + estimated_new_exposure
        concentration_pct = (total_exposure / aggregate['total_equity']) * 100

        print(f"\n{'='*80}")
        print(f"CONCENTRATION RISK CHECK: {symbol}")
        print(f"{'='*80}")
        print(f"Current Exposure: ${current_exposure:,.2f}")
        print(f"New Exposure: ${estimated_new_exposure:,.2f}")
        print(f"Total Exposure: ${total_exposure:,.2f}")
        print(f"Concentration: {concentration_pct:.2f}% of total equity")

        # Risk limits
        MAX_CONCENTRATION = 10.0  # Max 10% of portfolio in one symbol

        if concentration_pct > MAX_CONCENTRATION:
            print(f"[WARNING] Concentration exceeds {MAX_CONCENTRATION}% limit!")
            return False
        else:
            print(f"[OK] Within concentration limits")
            return True


def create_sample_config():
    """Create sample accounts configuration file"""

    sample_config = {
        "accounts": [
            {
                "account_id": "account_1_lucas",
                "owner": "Lucas",
                "api_key": os.getenv('ALPACA_API_KEY'),
                "secret_key": os.getenv('ALPACA_SECRET_KEY'),
                "profit_split": 1.0,
                "max_allocation": 100000
            },
            # Add 79 more accounts here when ready
            # {
            #     "account_id": "account_2_friend1",
            #     "owner": "Friend_1",
            #     "api_key": "...",
            #     "secret_key": "...",
            #     "profit_split": 0.8,  # 80% to Lucas, 20% to Friend_1
            #     "max_allocation": 100000
            # },
        ]
    }

    with open('accounts_config.json', 'w') as f:
        json.dump(sample_config, f, indent=2)

    print("Sample config created: accounts_config.json")
    print("Add remaining 79 accounts with their API credentials")


if __name__ == "__main__":
    # Create sample config if it doesn't exist
    if not os.path.exists('accounts_config.json'):
        create_sample_config()

    # Test orchestrator
    orchestrator = MultiAccountOrchestrator()

    # Get aggregate portfolio
    orchestrator.get_aggregate_portfolio()

    # Test trade distribution
    test_trade = {
        'symbol': 'SPY',
        'strategy': 'IRON_CONDOR',
        'strikes': {}
    }

    orchestrator.distribute_trade(test_trade, total_contracts=80)
