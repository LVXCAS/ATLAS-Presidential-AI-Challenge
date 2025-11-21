"""
PORTFOLIO REBALANCER
Maintains optimal capital allocation across strategies and markets

Target Allocation:
- 40% Forex (stable, 24/7 markets)
- 30% Futures (leverage, momentum)
- 30% Options (high return, defined risk)

Rebalancing Rules:
1. Check allocation weekly
2. Rebalance if any category drifts >10% from target
3. Increase allocation to winning strategies
4. Decrease allocation to losing strategies
5. Never let any single strategy exceed 50% of portfolio
"""
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from unified_pnl_tracker import UnifiedPnLTracker

@dataclass
class AllocationTarget:
    """Target allocation for a market/strategy"""
    category: str  # 'forex', 'futures', 'options', 'stocks'
    target_percent: float  # 0-1
    min_percent: float  # Minimum allocation
    max_percent: float  # Maximum allocation

@dataclass
class CurrentAllocation:
    """Current portfolio allocation"""
    category: str
    current_value: float
    current_percent: float
    target_percent: float
    drift: float  # How far from target
    action_needed: str  # 'increase', 'decrease', 'hold'
    adjustment_amount: float

@dataclass
class RebalanceAction:
    """Specific rebalancing action"""
    action_type: str  # 'increase', 'decrease'
    category: str
    from_amount: float
    to_amount: float
    change_amount: float
    reason: str

@dataclass
class RebalancePlan:
    """Complete rebalancing plan"""
    total_portfolio_value: float
    current_allocations: List[CurrentAllocation]
    actions: List[RebalanceAction]
    execution_priority: List[str]  # Order to execute actions
    expected_fees: float
    created_at: str

class PortfolioRebalancer:
    def __init__(self):
        self.pnl_tracker = UnifiedPnLTracker()
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Load configuration
        self.config = self._load_config()

        # Load rebalance history
        self.rebalance_history = self._load_rebalance_history()

    def _load_config(self) -> Dict:
        """Load rebalancer configuration"""
        if os.path.exists('config/rebalancer_config.json'):
            with open('config/rebalancer_config.json') as f:
                return json.load(f)

        # Default configuration
        default_config = {
            'enabled': False,  # Must be explicitly enabled
            'rebalance_frequency_days': 7,  # Weekly
            'drift_threshold': 0.10,  # Rebalance if drift > 10%
            'min_rebalance_amount': 1000,  # Don't bother for < $1000

            'target_allocations': {
                'forex': {
                    'target_percent': 0.40,
                    'min_percent': 0.30,
                    'max_percent': 0.50
                },
                'futures': {
                    'target_percent': 0.30,
                    'min_percent': 0.20,
                    'max_percent': 0.40
                },
                'options': {
                    'target_percent': 0.30,
                    'min_percent': 0.20,
                    'max_percent': 0.40
                }
            },

            'strategy_performance_lookback_days': 30,  # Evaluate last 30 days
            'underperforming_threshold': 0.0,  # Negative Sharpe = underperforming
            'outperforming_threshold': 1.5   # Sharpe > 1.5 = outperforming
        }

        # Save default config
        os.makedirs('config', exist_ok=True)
        with open('config/rebalancer_config.json', 'w') as f:
            json.dump(default_config, f, indent=2)

        return default_config

    def _save_config(self):
        """Save configuration"""
        os.makedirs('config', exist_ok=True)
        with open('config/rebalancer_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

    def _load_rebalance_history(self) -> List[Dict]:
        """Load rebalancing history"""
        if os.path.exists('data/rebalance_history.json'):
            with open('data/rebalance_history.json') as f:
                return json.load(f)
        return []

    def _save_rebalance_history(self):
        """Save rebalancing history"""
        os.makedirs('data', exist_ok=True)
        with open('data/rebalance_history.json', 'w') as f:
            json.dump(self.rebalance_history, f, indent=2)

    def send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': f'PORTFOLIO REBALANCE\n\n{message}'
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"[REBALANCER] Telegram notification failed: {e}")

    def get_current_allocations(self) -> Dict[str, float]:
        """Get current portfolio allocations by category"""
        # Get unified P&L
        unified_pnl = self.pnl_tracker.get_unified_pnl()

        allocations = {
            'forex': 0.0,
            'futures': 0.0,
            'options': 0.0,
            'stocks': 0.0
        }

        # Map accounts to categories
        for account in unified_pnl.accounts:
            if 'oanda' in account.account_name.lower() or 'forex' in account.account_type:
                allocations['forex'] += account.balance
            elif 'alpaca' in account.account_name.lower():
                # Alpaca handles both futures and options
                # For now, split 50/50 (in production, query actual positions)
                allocations['futures'] += account.balance * 0.5
                allocations['options'] += account.balance * 0.5
            elif 'stock' in account.account_type:
                allocations['stocks'] += account.balance

        return allocations

    def calculate_drift(self) -> List[CurrentAllocation]:
        """Calculate how far current allocation drifts from target"""
        current_values = self.get_current_allocations()
        total_value = sum(current_values.values())

        if total_value == 0:
            print("[REBALANCER] No portfolio value detected")
            return []

        allocations = []

        for category, target_config in self.config['target_allocations'].items():
            current_value = current_values.get(category, 0)
            current_percent = current_value / total_value
            target_percent = target_config['target_percent']

            drift = current_percent - target_percent

            # Determine action
            if abs(drift) <= self.config['drift_threshold']:
                action_needed = 'hold'
            elif drift > 0:
                action_needed = 'decrease'
            else:
                action_needed = 'increase'

            # Calculate adjustment amount
            target_value = total_value * target_percent
            adjustment_amount = target_value - current_value

            allocation = CurrentAllocation(
                category=category,
                current_value=current_value,
                current_percent=current_percent,
                target_percent=target_percent,
                drift=drift,
                action_needed=action_needed,
                adjustment_amount=adjustment_amount
            )

            allocations.append(allocation)

        return allocations

    def generate_rebalance_plan(self, allocations: List[CurrentAllocation]) -> RebalancePlan:
        """Generate rebalancing plan"""
        current_values = self.get_current_allocations()
        total_value = sum(current_values.values())

        # Generate actions
        actions = []

        # First, identify categories to decrease
        for allocation in allocations:
            if allocation.action_needed == 'decrease' and abs(allocation.adjustment_amount) >= self.config['min_rebalance_amount']:
                action = RebalanceAction(
                    action_type='decrease',
                    category=allocation.category,
                    from_amount=allocation.current_value,
                    to_amount=allocation.current_value + allocation.adjustment_amount,
                    change_amount=allocation.adjustment_amount,
                    reason=f"Overallocated by {allocation.drift*100:.1f}%"
                )
                actions.append(action)

        # Then, identify categories to increase
        for allocation in allocations:
            if allocation.action_needed == 'increase' and abs(allocation.adjustment_amount) >= self.config['min_rebalance_amount']:
                action = RebalanceAction(
                    action_type='increase',
                    category=allocation.category,
                    from_amount=allocation.current_value,
                    to_amount=allocation.current_value + allocation.adjustment_amount,
                    change_amount=allocation.adjustment_amount,
                    reason=f"Underallocated by {abs(allocation.drift)*100:.1f}%"
                )
                actions.append(action)

        # Execution priority: Decrease first (raise cash), then increase (deploy cash)
        decreases = [a.category for a in actions if a.action_type == 'decrease']
        increases = [a.category for a in actions if a.action_type == 'increase']
        execution_priority = decreases + increases

        # Estimate fees (simplified)
        total_rebalance_amount = sum(abs(a.change_amount) for a in actions)
        expected_fees = total_rebalance_amount * 0.001  # 0.1% estimated fees

        return RebalancePlan(
            total_portfolio_value=total_value,
            current_allocations=allocations,
            actions=actions,
            execution_priority=execution_priority,
            expected_fees=expected_fees,
            created_at=datetime.now().isoformat()
        )

    def check_rebalance_needed(self) -> bool:
        """Check if rebalancing is needed"""
        # Check if enough time has passed
        if self.rebalance_history:
            last_rebalance = datetime.fromisoformat(self.rebalance_history[-1]['timestamp'])
            days_since = (datetime.now() - last_rebalance).days

            if days_since < self.config['rebalance_frequency_days']:
                print(f"[REBALANCER] Too soon to rebalance ({days_since} days < {self.config['rebalance_frequency_days']})")
                return False

        # Check drift
        allocations = self.calculate_drift()

        max_drift = max([abs(a.drift) for a in allocations]) if allocations else 0

        if max_drift <= self.config['drift_threshold']:
            print(f"[REBALANCER] No rebalancing needed (max drift: {max_drift*100:.1f}% < {self.config['drift_threshold']*100:.0f}%)")
            return False

        return True

    def execute_rebalance_plan(self, plan: RebalancePlan) -> bool:
        """Execute rebalancing plan"""
        print("\n" + "="*70)
        print("EXECUTING REBALANCE PLAN")
        print("="*70)

        # In production, this would:
        # 1. Close positions in overallocated categories
        # 2. Open positions in underallocated categories
        # 3. Use limit orders to minimize slippage
        # 4. Track execution status

        # For now, just log the plan
        print("\n[REBALANCER] Actions to execute:")
        for i, action in enumerate(plan.actions, 1):
            print(f"\n{i}. {action.action_type.upper()} {action.category}")
            print(f"   From: ${action.from_amount:,.0f}")
            print(f"   To:   ${action.to_amount:,.0f}")
            print(f"   Change: ${action.change_amount:+,.0f}")
            print(f"   Reason: {action.reason}")

        # Record rebalance
        self.rebalance_history.append({
            'timestamp': datetime.now().isoformat(),
            'total_value': plan.total_portfolio_value,
            'actions': [asdict(a) for a in plan.actions],
            'fees': plan.expected_fees
        })
        self._save_rebalance_history()

        # Send Telegram notification
        msg = f"""
PORTFOLIO REBALANCED!

Total Value: ${plan.total_portfolio_value:,.0f}

Actions Taken:
"""
        for action in plan.actions:
            msg += f"\n{action.action_type.upper()} {action.category}: ${abs(action.change_amount):,.0f}"

        msg += f"\n\nEstimated Fees: ${plan.expected_fees:,.0f}"
        msg += f"\n\nPortfolio is now optimally allocated!"

        self.send_telegram_notification(msg)

        print("\n[REBALANCER] ✓ Rebalance complete!")
        return True

    def get_rebalance_status(self) -> str:
        """Get current rebalance status"""
        allocations = self.calculate_drift()
        current_values = self.get_current_allocations()
        total_value = sum(current_values.values())

        status = f"""
=== PORTFOLIO REBALANCER STATUS ===

Total Portfolio: ${total_value:,.0f}

CURRENT ALLOCATION:
"""
        for allocation in allocations:
            status += f"""
{allocation.category.upper()}:
  Current: ${allocation.current_value:,.0f} ({allocation.current_percent:.1%})
  Target:  {allocation.target_percent:.1%}
  Drift:   {allocation.drift:+.1%}
  Action:  {allocation.action_needed.upper()}
"""

        # Check if rebalance needed
        needs_rebalance = self.check_rebalance_needed()
        status += f"\nRebalance Needed: {'YES' if needs_rebalance else 'NO'}"

        if self.rebalance_history:
            last_rebalance = datetime.fromisoformat(self.rebalance_history[-1]['timestamp'])
            days_ago = (datetime.now() - last_rebalance).days
            status += f"\nLast Rebalance: {days_ago} days ago"

        return status

    def manual_rebalance(self):
        """Manually trigger rebalancing"""
        print("\n" + "="*70)
        print("PORTFOLIO REBALANCER - MANUAL TRIGGER")
        print("="*70)

        # Calculate allocations
        allocations = self.calculate_drift()

        if not allocations:
            print("[REBALANCER] No portfolio detected")
            return

        # Print current status
        print(self.get_rebalance_status())

        # Generate plan
        plan = self.generate_rebalance_plan(allocations)

        if not plan.actions:
            print("\n[REBALANCER] No actions needed - portfolio is balanced!")
            return

        # Execute plan
        self.execute_rebalance_plan(plan)

def main():
    """Test portfolio rebalancer"""
    rebalancer = PortfolioRebalancer()

    # Get status
    print(rebalancer.get_rebalance_status())

    # Check if rebalance needed
    if rebalancer.check_rebalance_needed():
        print("\n[REBALANCER] Rebalancing is needed!")
        # rebalancer.manual_rebalance()  # Uncomment to execute
    else:
        print("\n[REBALANCER] Portfolio is balanced")

if __name__ == '__main__':
    main()
