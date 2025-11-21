"""
UNIFIED P&L TRACKER
Aggregates profit/loss across all trading accounts and strategies
"""
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import glob

@dataclass
class AccountPnL:
    account_name: str
    account_type: str  # forex, futures, options, stocks
    balance: float
    starting_balance: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    pnl_percent: float
    open_positions: int
    closed_trades_today: int
    win_rate: float
    last_updated: str

@dataclass
class StrategyPnL:
    strategy_name: str
    market_type: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float
    max_drawdown: float

@dataclass
class UnifiedPnL:
    total_balance: float
    total_starting_balance: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    total_pnl_percent: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    accounts: List[AccountPnL]
    strategies: List[StrategyPnL]
    timestamp: str

class UnifiedPnLTracker:
    def __init__(self):
        self.oanda_api_key = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        self.alpaca_base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        # Track starting balances (load from file or set defaults)
        self.starting_balances = self._load_starting_balances()

    def _load_starting_balances(self) -> Dict[str, float]:
        """Load starting balances from file or initialize defaults"""
        if os.path.exists('data/starting_balances.json'):
            with open('data/starting_balances.json') as f:
                return json.load(f)
        return {
            'oanda_practice': 100000.0,  # Default OANDA practice balance
            'alpaca_paper': 100000.0,    # Default Alpaca paper balance
        }

    def _save_starting_balances(self):
        """Save current balances as starting balances"""
        os.makedirs('data', exist_ok=True)
        with open('data/starting_balances.json', 'w') as f:
            json.dump(self.starting_balances, f, indent=2)

    def get_oanda_pnl(self) -> Optional[AccountPnL]:
        """Get OANDA Forex account P&L"""
        try:
            headers = {
                'Authorization': f'Bearer {self.oanda_api_key}',
                'Content-Type': 'application/json'
            }

            # Get account summary
            url = f'https://api-fxpractice.oanda.com/v3/accounts/{self.oanda_account_id}/summary'
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"[OANDA] Error: {response.status_code} - {response.text}")
                return None

            data = response.json()['account']
            balance = float(data.get('balance', 0))
            unrealized_pnl = float(data.get('unrealizedPL', 0))
            open_positions = int(data.get('openPositionCount', 0))

            starting_balance = self.starting_balances.get('oanda_practice', balance)
            realized_pnl = balance - starting_balance
            total_pnl = realized_pnl + unrealized_pnl
            pnl_percent = (total_pnl / starting_balance * 100) if starting_balance > 0 else 0

            # Get trade statistics
            trades_url = f'https://api-fxpractice.oanda.com/v3/accounts/{self.oanda_account_id}/trades'
            trades_response = requests.get(trades_url, headers=headers, timeout=10)
            closed_today = 0
            win_rate = 0.0

            if trades_response.status_code == 200:
                trades = trades_response.json().get('trades', [])
                # Calculate win rate from recent trades
                winning_trades = sum(1 for t in trades if float(t.get('unrealizedPL', 0)) > 0)
                total_trades = len(trades)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            return AccountPnL(
                account_name='OANDA Practice',
                account_type='forex',
                balance=balance,
                starting_balance=starting_balance,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                pnl_percent=pnl_percent,
                open_positions=open_positions,
                closed_trades_today=closed_today,
                win_rate=win_rate,
                last_updated=datetime.now().isoformat()
            )
        except Exception as e:
            print(f"[OANDA] Error fetching P&L: {e}")
            return None

    def get_alpaca_pnl(self) -> Optional[AccountPnL]:
        """Get Alpaca Futures/Options account P&L"""
        try:
            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret
            }

            # Get account info
            url = f'{self.alpaca_base_url}/v2/account'
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"[ALPACA] Error: {response.status_code} - {response.text}")
                return None

            data = response.json()
            equity = float(data.get('equity', 0))

            starting_balance = self.starting_balances.get('alpaca_paper', equity)
            total_pnl = equity - starting_balance
            pnl_percent = (total_pnl / starting_balance * 100) if starting_balance > 0 else 0

            # Get positions
            positions_url = f'{self.alpaca_base_url}/v2/positions'
            pos_response = requests.get(positions_url, headers=headers, timeout=10)

            open_positions = 0
            unrealized_pnl = 0.0

            if pos_response.status_code == 200:
                positions = pos_response.json()
                open_positions = len(positions)
                unrealized_pnl = sum(float(p.get('unrealized_pl', 0)) for p in positions)

            realized_pnl = total_pnl - unrealized_pnl

            # Get recent trades for win rate
            orders_url = f'{self.alpaca_base_url}/v2/orders?status=filled&limit=100'
            orders_response = requests.get(orders_url, headers=headers, timeout=10)

            win_rate = 0.0
            closed_today = 0

            if orders_response.status_code == 200:
                orders = orders_response.json()
                today = datetime.now().date()

                for order in orders:
                    filled_at = datetime.fromisoformat(order['filled_at'].replace('Z', '+00:00'))
                    if filled_at.date() == today:
                        closed_today += 1

            return AccountPnL(
                account_name='Alpaca Paper',
                account_type='futures/options',
                balance=equity,
                starting_balance=starting_balance,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                pnl_percent=pnl_percent,
                open_positions=open_positions,
                closed_trades_today=closed_today,
                win_rate=win_rate,
                last_updated=datetime.now().isoformat()
            )
        except Exception as e:
            print(f"[ALPACA] Error fetching P&L: {e}")
            return None

    def get_strategy_pnl_from_logs(self) -> List[StrategyPnL]:
        """Calculate strategy P&L from execution logs"""
        strategies = {}

        # Parse Forex execution logs
        forex_logs = glob.glob('forex_trades/execution_log_*.json')
        for log_file in forex_logs:
            try:
                with open(log_file) as f:
                    trades = json.load(f)
                    for trade in trades:
                        strategy = trade.get('strategy', 'forex_elite')
                        if strategy not in strategies:
                            strategies[strategy] = {
                                'market_type': 'forex',
                                'trades': [],
                                'wins': 0,
                                'losses': 0
                            }

                        pnl = trade.get('pnl', 0)
                        strategies[strategy]['trades'].append(pnl)
                        if pnl > 0:
                            strategies[strategy]['wins'] += 1
                        elif pnl < 0:
                            strategies[strategy]['losses'] += 1
            except Exception as e:
                print(f"Error parsing {log_file}: {e}")

        # Convert to StrategyPnL objects
        strategy_pnls = []
        for name, data in strategies.items():
            trades = data['trades']
            total_trades = len(trades)

            if total_trades == 0:
                continue

            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t < 0]

            strategy_pnls.append(StrategyPnL(
                strategy_name=name,
                market_type=data['market_type'],
                total_trades=total_trades,
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate=(len(wins) / total_trades * 100) if total_trades > 0 else 0,
                total_pnl=sum(trades),
                avg_win=sum(wins) / len(wins) if wins else 0,
                avg_loss=sum(losses) / len(losses) if losses else 0,
                sharpe_ratio=0.0,  # TODO: Calculate properly
                max_drawdown=0.0   # TODO: Calculate properly
            ))

        return strategy_pnls

    def get_unified_pnl(self) -> UnifiedPnL:
        """Get unified P&L across all accounts"""
        accounts = []

        # Fetch account P&Ls
        oanda_pnl = self.get_oanda_pnl()
        if oanda_pnl:
            accounts.append(oanda_pnl)

        alpaca_pnl = self.get_alpaca_pnl()
        if alpaca_pnl:
            accounts.append(alpaca_pnl)

        # Calculate totals
        total_balance = sum(a.balance for a in accounts)
        total_starting_balance = sum(a.starting_balance for a in accounts)
        total_unrealized_pnl = sum(a.unrealized_pnl for a in accounts)
        total_realized_pnl = sum(a.realized_pnl for a in accounts)
        total_pnl = sum(a.total_pnl for a in accounts)
        total_pnl_percent = (total_pnl / total_starting_balance * 100) if total_starting_balance > 0 else 0

        # Get strategy P&Ls
        strategies = self.get_strategy_pnl_from_logs()

        return UnifiedPnL(
            total_balance=total_balance,
            total_starting_balance=total_starting_balance,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            daily_pnl=total_pnl,  # TODO: Calculate proper daily P&L
            weekly_pnl=total_pnl,  # TODO: Calculate proper weekly P&L
            monthly_pnl=total_pnl,  # TODO: Calculate proper monthly P&L
            accounts=accounts,
            strategies=strategies,
            timestamp=datetime.now().isoformat()
        )

    def format_pnl_summary(self, unified_pnl: UnifiedPnL) -> str:
        """Format P&L for display/Telegram"""
        summary = f"""
=== UNIFIED P&L SUMMARY ===
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TOTAL PORTFOLIO
Balance: ${unified_pnl.total_balance:,.2f}
Starting: ${unified_pnl.total_starting_balance:,.2f}
Total P&L: ${unified_pnl.total_pnl:,.2f} ({unified_pnl.total_pnl_percent:+.2f}%)
Unrealized: ${unified_pnl.total_unrealized_pnl:,.2f}
Realized: ${unified_pnl.total_realized_pnl:,.2f}

ACCOUNTS:
"""
        for account in unified_pnl.accounts:
            summary += f"""
{account.account_name} ({account.account_type.upper()})
  Balance: ${account.balance:,.2f}
  P&L: ${account.total_pnl:,.2f} ({account.pnl_percent:+.2f}%)
  Open Positions: {account.open_positions}
  Win Rate: {account.win_rate:.1f}%
"""

        if unified_pnl.strategies:
            summary += "\nSTRATEGIES:\n"
            for strategy in unified_pnl.strategies:
                summary += f"""
{strategy.strategy_name} ({strategy.market_type.upper()})
  Trades: {strategy.total_trades} (W: {strategy.winning_trades}, L: {strategy.losing_trades})
  Win Rate: {strategy.win_rate:.1f}%
  Total P&L: ${strategy.total_pnl:,.2f}
  Avg Win: ${strategy.avg_win:,.2f} | Avg Loss: ${strategy.avg_loss:,.2f}
"""

        return summary

    def save_pnl_snapshot(self, unified_pnl: UnifiedPnL):
        """Save P&L snapshot to file"""
        os.makedirs('logs/pnl_snapshots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'logs/pnl_snapshots/pnl_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(asdict(unified_pnl), f, indent=2)

        print(f"[PNL TRACKER] Snapshot saved: {filename}")

def main():
    """Test the P&L tracker"""
    tracker = UnifiedPnLTracker()

    print("Fetching unified P&L across all accounts...")
    unified_pnl = tracker.get_unified_pnl()

    # Print summary
    print(tracker.format_pnl_summary(unified_pnl))

    # Save snapshot
    tracker.save_pnl_snapshot(unified_pnl)

    # Return data for programmatic use
    return unified_pnl

if __name__ == '__main__':
    main()
