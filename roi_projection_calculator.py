"""
REALISTIC ROI PROJECTION CALCULATOR
Calculate achievable monthly ROI based on discovered opportunities
"""

import json
import numpy as np

def calculate_realistic_roi():
    """Calculate realistic monthly ROI projection"""

    print("REALISTIC ROI PROJECTION ANALYSIS")
    print("=" * 50)

    # Your portfolio parameters
    portfolio_value = 515000
    max_options_allocation = 0.30  # 30% max in options strategies
    options_capital = portfolio_value * max_options_allocation

    print(f"Portfolio Value: ${portfolio_value:,}")
    print(f"Options Capital Available: ${options_capital:,}")

    # Load discovery data
    try:
        with open('mega_discovery_20250918_2114.json', 'r') as f:
            data = json.load(f)
    except:
        print("No discovery data found")
        return

    best_strategies = data.get('best_strategies', [])
    if not best_strategies:
        print("No strategies found")
        return

    print(f"\nTOP DISCOVERED STRATEGIES:")
    print("-" * 40)

    # Analyze top 10 strategies for realistic deployment
    total_monthly_income = 0
    deployed_capital = 0
    strategies_deployed = 0

    for i, strategy in enumerate(best_strategies[:10]):
        ticker = strategy['ticker']
        strategy_type = strategy['strategy']
        expected_return = strategy['expected_return']
        allocation_required = strategy['allocation_required']

        # Calculate position size (assume 2-5 contracts per strategy)
        if strategy_type == 'covered_call':
            # Need to buy 100 shares per contract
            cost_per_contract = allocation_required * 100
            max_contracts = min(5, int(25000 / cost_per_contract))  # $25K max per strategy
        else:  # cash_secured_put
            # Need cash equal to strike * 100
            cash_per_contract = strategy['strike'] * 100
            max_contracts = min(5, int(25000 / cash_per_contract))

        if max_contracts == 0:
            continue

        # Calculate capital required
        capital_required = cost_per_contract if strategy_type == 'covered_call' else cash_per_contract
        total_capital = capital_required * max_contracts

        if deployed_capital + total_capital > options_capital:
            # Use remaining capital
            remaining_capital = options_capital - deployed_capital
            max_contracts = int(remaining_capital / capital_required)
            total_capital = capital_required * max_contracts

        if max_contracts == 0:
            continue

        # Calculate monthly income (conservative estimate)
        # Use 60% of projected return to account for real-world friction
        realistic_return = expected_return * 0.60
        monthly_return = realistic_return / 12  # Convert annual to monthly

        monthly_income = total_capital * monthly_return
        total_monthly_income += monthly_income
        deployed_capital += total_capital
        strategies_deployed += 1

        print(f"{i+1:2d}. {ticker:5s} {strategy_type:15s} | "
              f"Contracts: {max_contracts:2d} | "
              f"Capital: ${total_capital:7,.0f} | "
              f"Monthly: ${monthly_income:6,.0f}")

        if deployed_capital >= options_capital:
            break

    print("-" * 70)
    print(f"Total Strategies Deployed: {strategies_deployed}")
    print(f"Total Capital Deployed: ${deployed_capital:,.0f}")
    print(f"Total Monthly Options Income: ${total_monthly_income:,.0f}")

    # Calculate total monthly ROI
    monthly_roi = total_monthly_income / portfolio_value
    annualized_roi = (1 + monthly_roi) ** 12 - 1

    print(f"\nROI PROJECTION:")
    print(f"Monthly ROI from Options: {monthly_roi:.2%}")
    print(f"Annualized ROI from Options: {annualized_roi:.1%}")

    # Add base portfolio performance (your existing ETF positions)
    base_monthly_roi = 0.02  # Assume 2% monthly from existing positions
    total_monthly_roi = monthly_roi + base_monthly_roi
    total_annualized_roi = (1 + total_monthly_roi) ** 12 - 1

    print(f"\nCOMBINED WITH EXISTING PORTFOLIO:")
    print(f"Base Monthly ROI (ETFs): {base_monthly_roi:.2%}")
    print(f"Total Monthly ROI: {total_monthly_roi:.2%}")
    print(f"Total Annualized ROI: {total_annualized_roi:.1%}")

    # Compare to target
    target_monthly = 0.527
    achievement_rate = total_monthly_roi / target_monthly

    print(f"\nTARGET COMPARISON:")
    print(f"Target Monthly ROI: {target_monthly:.1%}")
    print(f"Projected Monthly ROI: {total_monthly_roi:.1%}")
    print(f"Achievement Rate: {achievement_rate:.1%}")

    if achievement_rate >= 1.0:
        print(f"🎯 TARGET ACHIEVABLE! Exceeding target by {(achievement_rate-1)*100:.1f}%")
    elif achievement_rate >= 0.8:
        print(f"⚡ CLOSE TO TARGET! Need {(1-achievement_rate)*100:.1f}% more performance")
    else:
        print(f"📈 BUILDING TOWARDS TARGET! {achievement_rate*100:.1f}% of the way there")

    # Calculate what you'd make this month
    this_month_profit = total_monthly_income + (portfolio_value * base_monthly_roi)
    print(f"\nTHIS MONTH PROJECTION:")
    print(f"Options Income: ${total_monthly_income:,.0f}")
    print(f"Base Portfolio Gains: ${portfolio_value * base_monthly_roi:,.0f}")
    print(f"Total Monthly Profit: ${this_month_profit:,.0f}")

    return {
        'monthly_roi': total_monthly_roi,
        'monthly_profit': this_month_profit,
        'achievement_rate': achievement_rate,
        'strategies_deployed': strategies_deployed
    }

if __name__ == "__main__":
    calculate_realistic_roi()