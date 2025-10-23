#!/usr/bin/env python3
"""
CASCADING CAPITAL DEPLOYMENT STRATEGY
Start with Forex ($100K) + Futures ($100K), use profits to fund Options,
use Options profits to fund prop firm challenges
"""

print('='*70)
print('CASCADING CAPITAL DEPLOYMENT STRATEGY')
print('='*70)
print('\nStarting Capital:')
print('  Forex Account:   $100,000')
print('  Futures Account: $100,000')
print('  Total Starting:  $200,000')
print('\nStrategy: Use profits to cascade into higher-leverage vehicles')
print('='*70)

# Base strategies with conservative estimates
forex_monthly_roi = 1.001  # 100.1% from earlier calculation, use 70% = 70%
futures_monthly_roi = 0.867  # 86.7%, use 70% = 60.7%

# Conservative adjustments
forex_conservative = forex_monthly_roi * 0.70  # 70.1%/month
futures_conservative = futures_monthly_roi * 0.70  # 60.7%/month

# Options ROI (only when funded)
options_monthly_roi = 0.524  # 52.4%/month from earlier
options_conservative = options_monthly_roi * 0.70  # 36.7%/month

print('\n' + '='*70)
print('MONTH-BY-MONTH PROJECTION (Conservative 70%):')
print('='*70)

# Initialize
forex_balance = 100000
futures_balance = 100000
options_balance = 0
prop_fund_balance = 0
total_withdrawn = 0

print(f'\n{"Month":<6} {"Forex":<12} {"Futures":<12} {"Options":<12} {"Prop Funds":<12} {"Total":<12} {"Withdrawn":<12}')
print('-'*90)

for month in range(1, 13):
    # Calculate monthly gains
    forex_gain = forex_balance * forex_conservative
    futures_gain = futures_balance * futures_conservative
    options_gain = options_balance * options_conservative if options_balance > 0 else 0

    # Update balances
    forex_balance += forex_gain
    futures_balance += futures_gain
    options_balance += options_gain

    # Cascading strategy

    # Month 1: Let Forex + Futures compound
    if month == 1:
        note = "Let base strategies compound"

    # Month 2: Fund options with 50% of profits
    elif month == 2:
        forex_withdrawal = forex_gain * 0.5
        futures_withdrawal = futures_gain * 0.5
        options_balance += forex_withdrawal + futures_withdrawal
        forex_balance -= forex_withdrawal
        futures_balance -= futures_withdrawal
        note = "Fund Options with 50% profits"

    # Month 3-4: Continue funding options
    elif month in [3, 4]:
        forex_withdrawal = forex_gain * 0.3
        futures_withdrawal = futures_gain * 0.3
        options_balance += forex_withdrawal + futures_withdrawal
        forex_balance -= forex_withdrawal
        futures_balance -= futures_withdrawal
        note = "Fund Options with 30% profits"

    # Month 5: Start prop firm challenges
    elif month == 5:
        # Use 50% of options gains for prop challenges
        if options_gain > 0:
            prop_funding = options_gain * 0.5
            options_balance -= prop_funding
            prop_fund_balance += prop_funding
            note = "Fund Prop Firms from Options"
        else:
            note = "Continue compounding"

    # Month 6+: Optimized allocation
    else:
        # 20% of all gains go to prop firm challenges
        total_gain = forex_gain + futures_gain + options_gain
        prop_funding = total_gain * 0.2

        # Distribute the withdrawal proportionally
        if forex_gain + futures_gain + options_gain > 0:
            forex_withdrawal = (forex_gain / (forex_gain + futures_gain + options_gain)) * prop_funding
            futures_withdrawal = (futures_gain / (forex_gain + futures_gain + options_gain)) * prop_funding
            options_withdrawal = (options_gain / (forex_gain + futures_gain + options_gain)) * prop_funding if options_gain > 0 else 0

            forex_balance -= forex_withdrawal
            futures_balance -= futures_withdrawal
            options_balance -= options_withdrawal
            prop_fund_balance += prop_funding

        note = "20% to Prop Firms"

    total_balance = forex_balance + futures_balance + options_balance + prop_fund_balance

    print(f'{month:<6} ${forex_balance:<11,.0f} ${futures_balance:<11,.0f} ${options_balance:<11,.0f} ${prop_fund_balance:<11,.0f} ${total_balance:<11,.0f} {note}')

print('\n' + '='*70)
print('PROP FIRM CHALLENGE STRATEGY:')
print('='*70)

# Prop firm economics
prop_challenges = [
    {'name': '50K Challenge', 'cost': 500, 'target': 50000, 'split': 0.80},
    {'name': '100K Challenge', 'cost': 1000, 'target': 100000, 'split': 0.80},
    {'name': '200K Challenge', 'cost': 2000, 'target': 200000, 'split': 0.90}
]

print('\nProp Firm Economics:')
for challenge in prop_challenges:
    print(f"  {challenge['name']}: ${challenge['cost']} entry -> ${challenge['target']} funded ({challenge['split']:.0%} profit split)")

# Calculate how many challenges you can fund
prop_capital = prop_fund_balance
print(f'\nCapital available for prop challenges by Month 12: ${prop_capital:,.0f}')

# Strategy: Buy multiple 100K challenges
num_100k_challenges = int(prop_capital / 1000)
print(f'\nCan fund {num_100k_challenges}x $100K challenges (@ $1,000 each)')
print(f'Total funded capital if passed: ${num_100k_challenges * 100000:,.0f}')
print(f'Your 80% split on profits: 80% to you, 20% to firm')

# Potential monthly from prop firms (assuming 10% monthly on funded capital with 80% split)
prop_funded_capital = num_100k_challenges * 100000
prop_monthly_return = 0.10  # Conservative 10%/month on prop capital
prop_monthly_profit = prop_funded_capital * prop_monthly_return
your_split = prop_monthly_profit * 0.80

print(f'\nMonthly income from prop firms (after Month 12):')
print(f'  Funded capital: ${prop_funded_capital:,.0f}')
print(f'  Monthly return @ 10%: ${prop_monthly_profit:,.0f}')
print(f'  Your 80% split: ${your_split:,.0f}/month')

print('\n' + '='*70)
print('YEAR 1 SUMMARY:')
print('='*70)

print(f'\nStarting: $200,000')
print(f'Ending (Month 12):')
print(f'  Forex Account: ${forex_balance:,.0f}')
print(f'  Futures Account: ${futures_balance:,.0f}')
print(f'  Options Account: ${options_balance:,.0f}')
print(f'  Prop Fund Capital: ${prop_fund_balance:,.0f}')
print(f'  TOTAL: ${forex_balance + futures_balance + options_balance + prop_fund_balance:,.0f}')

total_gain = (forex_balance + futures_balance + options_balance + prop_fund_balance) - 200000
total_roi = (total_gain / 200000) * 100

print(f'\nTotal Gain: ${total_gain:,.0f}')
print(f'Total ROI: {total_roi:.1f}%')

print(f'\nProp Firm Leverage (Year 2):')
print(f'  You control: ${prop_funded_capital:,.0f} (Other People\'s Money)')
print(f'  Monthly income potential: ${your_split:,.0f}')
print(f'  Annual income from props: ${your_split * 12:,.0f}')

print('\n' + '='*70)
print('KEY ADVANTAGES OF THIS STRATEGY:')
print('='*70)
print('\n[OK] Start with low-risk strategies (Forex/Futures)')
print('[OK] Use profits (not principal) for higher-risk Options')
print('[OK] Use Options profits for prop firm challenges')
print('[OK] Prop firms = leverage other people\'s capital')
print('[OK] Diversification across multiple income streams')
print('[OK] Risk principal stays in Forex/Futures (lower risk)')
print('[OK] By Year 2, earning from props with firm\'s capital')

print('\n' + '='*70)
print('REALISTIC EXPECTATIONS:')
print('='*70)
print('\n[!] Assumes 70% of theoretical performance (conservative)')
print('[!] Prop firm challenges have 10-20% pass rate')
print('[!] Budget for multiple attempts per challenge')
print(f'[!] With ${prop_capital:,.0f}, can attempt ~{int(prop_capital/5000)} challenges')
print('[!] Passing 5-10 challenges would give substantial leverage')
print('[!] This strategy minimizes risk while maximizing upside')

print('\n' + '='*70)
print('ACTION PLAN:')
print('='*70)
print('\nPhase 1 (Months 1-2): Validate Forex + Futures')
print('  - Paper trade both strategies')
print('  - Achieve consistent profitability')
print('  - Verify win rates match projections')

print('\nPhase 2 (Months 2-4): Add Options')
print('  - Fund options with Forex/Futures profits')
print('  - Start with small positions')
print('  - Scale as you gain confidence')

print('\nPhase 3 (Months 5-8): Prop Firm Challenges')
print('  - Use Options profits for challenge fees')
print('  - Start with $50K challenges (cheaper)')
print('  - Move to $100K/$200K after proving concept')

print('\nPhase 4 (Months 9-12): Scale')
print('  - Pass multiple prop challenges')
print('  - Build portfolio of funded accounts')
print('  - Leverage firm capital for income')

print(f'\nBy end of Year 1: Trading ${prop_funded_capital:,.0f} of firm capital')
print(f'Potential Year 2 income: ${your_split * 12:,.0f}/year from props alone')
print('Plus continued growth in your own Forex/Futures/Options accounts')
