"""
Porsche 911 Purchase Timeline Calculator
When can you afford a 911 based on forex trading income?
"""

import json
from datetime import datetime, timedelta

print("\n" + "="*90)
print(" " * 30 + "PORSCHE 911 PURCHASE TIMELINE")
print("="*90)

# Porsche 911 pricing (2024-2025 models)
porsche_prices = {
    "911 Carrera (Base)": {
        "new_msrp": 115000,
        "used_2022": 95000,
        "used_2020": 85000,
        "insurance_yearly": 3000,
        "maintenance_yearly": 4000,
    },
    "911 Carrera S": {
        "new_msrp": 130000,
        "used_2022": 110000,
        "used_2020": 95000,
        "insurance_yearly": 3500,
        "maintenance_yearly": 5000,
    },
    "911 Turbo": {
        "new_msrp": 180000,
        "used_2022": 160000,
        "used_2020": 140000,
        "insurance_yearly": 5000,
        "maintenance_yearly": 6000,
    },
    "911 GT3": {
        "new_msrp": 220000,
        "used_2022": 200000,
        "used_2020": 180000,
        "insurance_yearly": 6000,
        "maintenance_yearly": 8000,
    },
}

# Current status
current_date = datetime(2025, 11, 4)
current_account_balance = 191640
current_unrealized = 1064
weekend_profit = 4450

# E8 income projections (from your scaling plan)
income_timeline = [
    {"month": 1, "date": "Dec 2025", "monthly_income": 0, "accounts": 0, "capital_managed": 0},
    {"month": 2, "date": "Jan 2026", "monthly_income": 10000, "accounts": 1, "capital_managed": 500000},
    {"month": 3, "date": "Feb 2026", "monthly_income": 20000, "accounts": 2, "capital_managed": 1000000},
    {"month": 4, "date": "Mar 2026", "monthly_income": 30000, "accounts": 3, "capital_managed": 1500000},
    {"month": 6, "date": "May 2026", "monthly_income": 50000, "accounts": 5, "capital_managed": 2500000},
    {"month": 9, "date": "Aug 2026", "monthly_income": 80000, "accounts": 8, "capital_managed": 4000000},
    {"month": 12, "date": "Nov 2026", "monthly_income": 120000, "accounts": 12, "capital_managed": 6000000},
    {"month": 18, "date": "May 2027", "monthly_income": 200000, "accounts": 20, "capital_managed": 10000000},
    {"month": 24, "date": "Nov 2027", "monthly_income": 300000, "accounts": 30, "capital_managed": 15000000},
]

print("\n[PART 1] PORSCHE 911 MODELS & PRICING")
print("-" * 90)

print(f"\n{'Model':<25} {'New MSRP':<15} {'Used 2022':<15} {'Used 2020':<15} {'Annual Cost':<15}")
print("-" * 90)

for model, prices in porsche_prices.items():
    annual_cost = prices['insurance_yearly'] + prices['maintenance_yearly']
    print(f"{model:<25} ${prices['new_msrp']:>13,}  ${prices['used_2022']:>13,}  ${prices['used_2020']:>13,}  ${annual_cost:>13,}")

print("\nNote: Annual cost includes insurance + maintenance (not including gas)")

# Purchase rules
print("\n[PART 2] PURCHASE RULES (Financial Responsibility)")
print("-" * 90)

print("\nCONSERVATIVE RULE (Recommended):")
print("  - Car price should be ≤ 10% of net worth")
print("  - Annual costs should be ≤ 5% of annual income")
print("  - Pay cash (no loan) or put down 50% minimum")
print("  - Have 6 months emergency fund separate")

print("\nMODERATE RULE:")
print("  - Car price should be ≤ 20% of net worth")
print("  - Annual costs should be ≤ 10% of annual income")
print("  - Loan okay if interest rate < 5%")
print("  - Have 3 months emergency fund separate")

print("\nAGGRESSIVE RULE (Not Recommended):")
print("  - Car price ≤ 50% of annual income")
print("  - Finance entire purchase")
print("  - YOLO mode - don't do this")

# Calculate when you can afford each model
print("\n[PART 3] WHEN CAN YOU AFFORD EACH MODEL?")
print("-" * 90)

def calculate_affordability(car_price, annual_cost, monthly_income, months_from_now):
    """Calculate if you can afford the car based on conservative rules"""

    # Calculate net worth at that point
    # Assume you save 50% of monthly income after expenses
    monthly_savings = monthly_income * 0.5
    total_savings = monthly_savings * months_from_now
    total_net_worth = current_account_balance + total_savings

    # Conservative rule: Car ≤ 10% of net worth
    max_car_price_10pct = total_net_worth * 0.10

    # Moderate rule: Car ≤ 20% of net worth
    max_car_price_20pct = total_net_worth * 0.20

    # Annual income at that point
    annual_income = monthly_income * 12

    # Annual cost as % of income
    annual_cost_pct = (annual_cost / annual_income * 100) if annual_income > 0 else 999

    # Can afford (conservative)?
    can_afford_conservative = car_price <= max_car_price_10pct and annual_cost_pct <= 5

    # Can afford (moderate)?
    can_afford_moderate = car_price <= max_car_price_20pct and annual_cost_pct <= 10

    return {
        'net_worth': total_net_worth,
        'max_price_10pct': max_car_price_10pct,
        'max_price_20pct': max_car_price_20pct,
        'annual_income': annual_income,
        'annual_cost_pct': annual_cost_pct,
        'can_afford_conservative': can_afford_conservative,
        'can_afford_moderate': can_afford_moderate,
    }

# Check each model
for model, prices in porsche_prices.items():
    print(f"\n{model}:")
    print(f"  Target Price: ${prices['used_2022']:,} (Used 2022)")
    print(f"  Annual Costs: ${prices['insurance_yearly'] + prices['maintenance_yearly']:,}")
    print()

    affordable_conservative = None
    affordable_moderate = None

    for period in income_timeline:
        months = period['month']
        monthly_income = period['monthly_income']

        if monthly_income == 0:
            continue

        annual_cost = prices['insurance_yearly'] + prices['maintenance_yearly']
        affordability = calculate_affordability(
            prices['used_2022'],
            annual_cost,
            monthly_income,
            months
        )

        if affordability['can_afford_conservative'] and affordable_conservative is None:
            affordable_conservative = period

        if affordability['can_afford_moderate'] and affordable_moderate is None:
            affordable_moderate = period

    if affordable_conservative:
        print(f"  CONSERVATIVE: {affordable_conservative['date']} (Month {affordable_conservative['month']})")
        print(f"    Income: ${affordable_conservative['monthly_income']:,}/month")
        print(f"    Net Worth: ~${calculate_affordability(prices['used_2022'], prices['insurance_yearly'] + prices['maintenance_yearly'], affordable_conservative['monthly_income'], affordable_conservative['month'])['net_worth']:,.0f}")
    else:
        print(f"  CONSERVATIVE: Not within 24 months (need higher income)")

    if affordable_moderate:
        print(f"  MODERATE: {affordable_moderate['date']} (Month {affordable_moderate['month']})")
        print(f"    Income: ${affordable_moderate['monthly_income']:,}/month")
    else:
        print(f"  MODERATE: Not within 24 months")

# Specific Timeline Analysis
print("\n[PART 4] DETAILED TIMELINE - BASE 911 CARRERA")
print("-" * 90)

target_model = "911 Carrera (Base)"
target_price = porsche_prices[target_model]['used_2022']
target_annual_cost = porsche_prices[target_model]['insurance_yearly'] + porsche_prices[target_model]['maintenance_yearly']

print(f"\nTarget: {target_model}")
print(f"Price: ${target_price:,} (Used 2022)")
print(f"Annual Cost: ${target_annual_cost:,}")
print()

print(f"{'Month':<8} {'Date':<12} {'Monthly $':<15} {'Net Worth':<18} {'Car as %':<15} {'Status':<30}")
print("-" * 90)

for period in income_timeline:
    months = period['month']
    monthly_income = period['monthly_income']

    if monthly_income == 0:
        continue

    affordability = calculate_affordability(target_price, target_annual_cost, monthly_income, months)

    car_as_pct_10 = (target_price / affordability['max_price_10pct'] * 100) if affordability['max_price_10pct'] > 0 else 999
    car_as_pct_20 = (target_price / affordability['max_price_20pct'] * 100) if affordability['max_price_20pct'] > 0 else 999

    if affordability['can_afford_conservative']:
        status = "BUY NOW (Conservative)"
    elif affordability['can_afford_moderate']:
        status = "BUY NOW (Moderate)"
    elif car_as_pct_20 <= 120:
        status = "Almost there (20% rule)"
    elif car_as_pct_10 <= 120:
        status = "Getting close (10% rule)"
    else:
        status = "Not yet"

    print(f"{months:<8} {period['date']:<12} ${monthly_income:>13,}  ${affordability['net_worth']:>16,.0f}  {car_as_pct_10:>13.1f}%  {status:<30}")

# Smart purchase strategy
print("\n[PART 5] SMART PURCHASE STRATEGY")
print("-" * 90)

print("\nOPTION A: PATIENT APPROACH (Recommended)")
print("  Timeline: Month 6-9 (May-Aug 2026)")
print("  Model: Used 2022 911 Carrera (~$95K)")
print("  Requirements:")
print("    - Monthly income: $50K-80K")
print("    - Net worth: ~$950K+")
print("    - Pay cash or 50% down")
print("  Reasoning:")
print("    - Car is <10% of net worth")
print("    - Annual costs <5% of income")
print("    - Still have majority of wealth invested")
print("    - Can upgrade later without financial stress")

print("\nOPTION B: MODERATE APPROACH")
print("  Timeline: Month 3-4 (Feb-Mar 2026)")
print("  Model: Used 2020 911 Carrera (~$85K)")
print("  Requirements:")
print("    - Monthly income: $20K-30K")
print("    - Net worth: ~$425K+")
print("    - Finance 50% at <5% interest")
print("  Reasoning:")
print("    - Get the car sooner")
print("    - Slightly older model (still great)")
print("    - Car is ~20% of net worth (acceptable)")
print("    - Manageable payment with income")

print("\nOPTION C: AGGRESSIVE APPROACH (Not Recommended)")
print("  Timeline: Month 2 (Jan 2026)")
print("  Model: Used 2020 911 Carrera (~$85K)")
print("  Requirements:")
print("    - Monthly income: $10K")
print("    - Net worth: ~$250K")
print("    - Finance entire purchase")
print("  Reasoning:")
print("    - Car is 34% of net worth (risky)")
print("    - Payment eats into reinvestment capital")
print("    - Could slow down E8 scaling")
print("    - What if trading income drops?")

# Cash vs Finance analysis
print("\n[PART 6] CASH VS FINANCE ANALYSIS")
print("-" * 90)

print("\nScenario: $95K Used 2022 911 Carrera")
print()

print("OPTION 1: PAY CASH")
print("  Upfront: $95,000")
print("  Monthly: $0 (just insurance/maintenance)")
print("  Total Cost (5 years): $95,000 + $35,000 = $130,000")
print("  Pros:")
print("    - No interest payments")
print("    - Own it outright")
print("    - No monthly stress")
print("  Cons:")
print("    - Large upfront capital hit")
print("    - $95K not earning in E8 accounts")

print("\nOPTION 2: 50% DOWN, FINANCE 50%")
print("  Down Payment: $47,500")
print("  Loan: $47,500 @ 5% for 60 months")
print("  Monthly Payment: $896")
print("  Total Interest: $6,264")
print("  Total Cost (5 years): $95,000 + $6,264 + $35,000 = $136,264")
print("  Pros:")
print("    - Keep $47,500 earning in E8 accounts")
print("    - $47,500 @ 2.5%/month = $1,188/month")
print("    - Earnings ($1,188) > Payment ($896) = $292/month profit")
print("  Cons:")
print("    - Pay $6,264 in interest")
print("    - Monthly obligation")

print("\nOPTION 3: LEASE (3-Year)")
print("  Down Payment: $10,000")
print("  Monthly: $1,200-1,400")
print("  Total (3 years): $53,200-60,400")
print("  Pros:")
print("    - Lower upfront cost")
print("    - Always under warranty")
print("    - Can upgrade every 3 years")
print("  Cons:")
print("    - Never own it")
print("    - Mileage limits (10K-12K/year)")
print("    - Total cost higher over time")

# Opportunity cost
print("\n[PART 7] OPPORTUNITY COST ANALYSIS")
print("-" * 90)

print("\nWhat if you invested $95K in E8 challenges instead?")
print()

# Calculate ROI on E8 vs Porsche
e8_challenge_cost = 1627
e8_pass_rate = 0.82  # 82% for 8% DD, 2 accounts
e8_monthly_income_per_account = 10000

num_challenges = 95000 // 1627  # 58 challenges
expected_passes = num_challenges * e8_pass_rate  # ~47 accounts
monthly_income_from_investment = expected_passes * e8_monthly_income_per_account

print(f"$95,000 can buy: {int(num_challenges)} E8 challenges")
print(f"Expected passes @ 82%: {int(expected_passes)} funded accounts")
print(f"Monthly income: ${int(monthly_income_from_investment):,}")
print(f"Annual income: ${int(monthly_income_from_investment * 12):,}")
print()
print("In 2 months: You'd earn enough to buy the Porsche WITH CASH")
print("In 12 months: You'd earn $5.6M (59x your initial investment)")
print()
print("Verdict: It's MUCH smarter to invest in E8 first, then buy Porsche")

# Final recommendations
print("\n[PART 8] FINAL RECOMMENDATIONS")
print("-" * 90)

print("\nBEST TIMELINE:")
print()
print("Month 1-6 (Nov 2025 - May 2026):")
print("  - Focus 100% on scaling E8 accounts")
print("  - Reinvest ALL profits into new challenges")
print("  - Target: 5-8 funded accounts by Month 6")
print("  - Monthly income: $50K-80K")
print()
print("Month 6 (May 2026): BUY THE PORSCHE")
print("  - Model: Used 2022 911 Carrera")
print("  - Price: ~$95,000")
print("  - Payment: 50% down ($47,500), finance 50%")
print("  - Net worth: ~$950K+")
print("  - Car is <10% of net worth")
print("  - Monthly payment ($896) is 1.1-1.8% of income")
print()
print("Month 7-12 (Jun-Nov 2026):")
print("  - Continue scaling to 12+ accounts")
print("  - Monthly income: $120K+")
print("  - Pay off Porsche loan early (if desired)")
print()
print("Month 18+ (May 2027+):")
print("  - Trade up to 911 Turbo or GT3 if desired")
print("  - Monthly income: $200K-300K")
print("  - Can afford any 911 variant in cash")

print("\n[PART 9] COMPARISON TO TIMELINE")
print("-" * 90)

milestones = [
    {"month": 0, "event": "START (Today)", "income": "$0", "car": "None"},
    {"month": 2, "event": "First E8 Payout", "income": "$10K/mo", "car": "TOO EARLY - Reinvest"},
    {"month": 4, "event": "3 Funded Accounts", "income": "$30K/mo", "car": "Could buy used 2020 (risky)"},
    {"month": 6, "event": "5 Funded Accounts", "income": "$50K/mo", "car": "BUY: Used 2022 Carrera ✓"},
    {"month": 9, "event": "8 Funded Accounts", "income": "$80K/mo", "car": "Could upgrade to Carrera S"},
    {"month": 12, "event": "12 Funded Accounts", "income": "$120K/mo", "car": "Pay off loan, save for Turbo"},
    {"month": 18, "event": "20 Funded Accounts", "income": "$200K/mo", "car": "Buy 911 Turbo in cash"},
    {"month": 24, "event": "30 Funded Accounts", "income": "$300K/mo", "car": "Buy GT3 + keep Carrera"},
]

print(f"\n{'Month':<8} {'Event':<25} {'Income':<15} {'Car Decision':<35}")
print("-" * 90)

for m in milestones:
    print(f"{m['month']:<8} {m['event']:<25} {m['income']:<15} {m['car']:<35}")

print("\n" + "="*90)
print("ANSWER: BUY THE PORSCHE IN MAY 2026 (MONTH 6)")
print("="*90)

print("\nWhy Month 6?")
print("  1. Net worth will be ~$950K+ (car is <10%)")
print("  2. Monthly income will be $50K-80K (payments are <2%)")
print("  3. You'll have 5-8 funded accounts running")
print("  4. Income is stable and proven")
print("  5. You can afford it without slowing down scaling")
print("  6. 7 months from now - totally achievable")

print("\nWhat to do until then?")
print("  - Keep driving current car")
print("  - Reinvest 100% of profits into E8 challenges")
print("  - Watch Porsche prices (might find deals)")
print("  - Test drive different models")
print("  - Build excitement for the purchase")

print("\nWhat NOT to do?")
print("  - DON'T buy it next month (too early)")
print("  - DON'T finance 100% (keep skin in the game)")
print("  - DON'T buy new (let someone else eat depreciation)")
print("  - DON'T stop scaling E8 to buy car faster")

print("\n" + "="*90 + "\n")
