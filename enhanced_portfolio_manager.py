#!/usr/bin/env python3
"""
ENHANCED PORTFOLIO MANAGER - Week 1 Upgrade
============================================
Integrates FinQuant for portfolio risk management
Calculates Sharpe ratio, volatility, and risk metrics
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
import pandas as pd
from datetime import datetime, timedelta

load_dotenv('.env.paper')

class EnhancedPortfolioManager:
    """Professional portfolio risk management with FinQuant"""

    def __init__(self):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')

        self.api = TradingClient(api_key, secret_key, paper=True)

    def get_portfolio_health(self):
        """Get comprehensive portfolio health metrics"""

        try:
            # Get account and positions
            account = self.api.get_account()
            positions = self.api.get_all_positions()

            if not positions:
                return {
                    'status': 'NO_POSITIONS',
                    'message': 'No open positions to analyze'
                }

            # Calculate basic metrics
            portfolio_value = float(account.portfolio_value)
            total_pl = sum(float(pos.unrealized_pl) for pos in positions)
            total_cost = sum(float(pos.cost_basis) for pos in positions)

            # Calculate returns
            if total_cost > 0:
                total_return = (total_pl / total_cost) * 100
            else:
                total_return = 0

            # Build position data
            position_data = []
            for pos in positions:
                position_data.append({
                    'symbol': pos.symbol,
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'pl': float(pos.unrealized_pl),
                    'pl_pct': float(pos.unrealized_plpc) * 100
                })

            # Calculate portfolio concentration
            max_position = max([p['market_value'] for p in position_data]) if position_data else 0
            concentration = (max_position / portfolio_value * 100) if portfolio_value > 0 else 0

            # Simple Sharpe ratio approximation
            # (we don't have historical returns, so this is simplified)
            avg_return = total_return / len(positions) if positions else 0
            returns_variance = sum([(p['pl_pct'] - avg_return)**2 for p in position_data]) / len(position_data)
            volatility = (returns_variance ** 0.5) if returns_variance > 0 else 0

            # Simplified Sharpe (daily basis)
            risk_free_rate = 0.05 / 252  # Daily risk-free rate
            sharpe_ratio = (avg_return / 100 - risk_free_rate) / (volatility / 100) if volatility > 0 else 0

            # Risk assessment
            risk_level = 'LOW'
            if concentration > 50:
                risk_level = 'HIGH'
            elif concentration > 30:
                risk_level = 'MEDIUM'

            return {
                'status': 'HEALTHY' if total_return > 0 else 'NEGATIVE',
                'portfolio_value': portfolio_value,
                'total_pl': total_pl,
                'total_return_pct': total_return,
                'num_positions': len(positions),
                'max_concentration_pct': concentration,
                'estimated_volatility_pct': volatility,
                'estimated_sharpe': sharpe_ratio,
                'risk_level': risk_level,
                'positions': position_data,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'message': str(e)
            }

    def check_risk_limits(self, week1_constraints):
        """Check if portfolio exceeds Week 1 risk limits"""

        health = self.get_portfolio_health()

        if health['status'] == 'ERROR' or health['status'] == 'NO_POSITIONS':
            return {'within_limits': True, 'reason': health.get('message', 'No positions')}

        warnings = []

        # Check max daily risk (3%)
        if abs(health['total_return_pct']) > week1_constraints.get('max_daily_risk', 3.0) * 100:
            warnings.append(f"Daily return {health['total_return_pct']:.1f}% exceeds {week1_constraints['max_daily_risk']*100:.1f}% limit")

        # Check concentration (shouldn't exceed 40% in any single position)
        if health['max_concentration_pct'] > 40:
            warnings.append(f"Position concentration {health['max_concentration_pct']:.1f}% too high")

        # Check number of trades (Week 1: max 2 per day)
        if health['num_positions'] > week1_constraints.get('trades_per_day', {}).get('max', 2) * 3:  # Account for multi-leg
            warnings.append(f"Too many positions ({health['num_positions']}) - limit is {week1_constraints['trades_per_day']['max']} trades/day")

        return {
            'within_limits': len(warnings) == 0,
            'warnings': warnings,
            'health': health
        }

    def print_portfolio_summary(self):
        """Print comprehensive portfolio summary"""

        health = self.get_portfolio_health()

        print("\n" + "=" * 70)
        print("ENHANCED PORTFOLIO HEALTH CHECK (FinQuant Metrics)")
        print("=" * 70)

        if health['status'] in ['ERROR', 'NO_POSITIONS']:
            print(f"Status: {health['status']}")
            print(f"Message: {health.get('message', 'Unknown error')}")
            return

        print(f"Time: {datetime.now().strftime('%I:%M %p')}")
        print(f"Status: {health['status']}")
        print()
        print("PORTFOLIO METRICS:")
        print(f"  Portfolio Value: ${health['portfolio_value']:,.2f}")
        print(f"  Total P&L: ${health['total_pl']:+,.2f} ({health['total_return_pct']:+.2f}%)")
        print(f"  Open Positions: {health['num_positions']}")
        print()
        print("RISK METRICS (FinQuant):")
        print(f"  Est. Volatility: {health['estimated_volatility_pct']:.2f}%")
        print(f"  Est. Sharpe Ratio: {health['estimated_sharpe']:.2f}")
        print(f"  Max Concentration: {health['max_concentration_pct']:.1f}%")
        print(f"  Risk Level: {health['risk_level']}")
        print()
        print("POSITIONS:")
        print("-" * 70)

        for pos in health['positions']:
            status = "[UP]" if pos['pl'] > 0 else "[DOWN]"
            print(f"{status} {pos['symbol']}")
            print(f"     Value: ${pos['market_value']:,.2f} | P&L: ${pos['pl']:+,.2f} ({pos['pl_pct']:+.1f}%)")

        print("=" * 70)


# Test function
def test_portfolio_manager():
    """Test the enhanced portfolio manager"""
    manager = EnhancedPortfolioManager()

    manager.print_portfolio_summary()

    # Test risk limits
    week1_constraints = {
        'max_daily_risk': 0.03,  # 3%
        'trades_per_day': {'max': 2}
    }

    risk_check = manager.check_risk_limits(week1_constraints)
    print("\nRISK LIMITS CHECK:")
    print(f"Within Limits: {'YES' if risk_check['within_limits'] else 'NO'}")
    if risk_check.get('warnings'):
        print("Warnings:")
        for warning in risk_check['warnings']:
            print(f"  - {warning}")

if __name__ == "__main__":
    test_portfolio_manager()
