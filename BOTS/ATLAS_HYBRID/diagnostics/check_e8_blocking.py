"""Quick check to see why E8ComplianceAgent is blocking trades."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.e8_compliance_agent import E8ComplianceAgent

# Initialize agent
agent = E8ComplianceAgent(starting_balance=200000, initial_weight=2.0)

# Test with current OANDA balance
market_data = {
    'account_balance': 182788.16,
    'pair': 'EUR_USD',
    'price': 1.15119,
    'date': '2025-11-22'
}

vote, confidence, reasoning = agent.analyze(market_data)

print("=" * 80)
print("E8 COMPLIANCE AGENT ANALYSIS")
print("=" * 80)
print(f"\nVote: {vote}")
print(f"Confidence: {confidence}")
print(f"\nReasoning:")
print(reasoning)
print("\n" + "=" * 80)

# Calculate the actual DD
starting = 200000
current = 182788.16
dd = starting - current
dd_pct = (dd / starting) * 100

print("\nACCOUNT STATUS:")
print(f"  Starting Balance: ${starting:,.2f}")
print(f"  Current Balance:  ${current:,.2f}")
print(f"  Drawdown:         ${dd:,.2f} ({dd_pct:.2f}%)")
print(f"  E8 DD Limit:      6.00%")
print(f"  Over Limit By:    {dd_pct - 6.0:.2f}%")

if dd_pct > 6.0:
    print("\n[BLOCKED] Account exceeds 6% trailing DD limit")
    print("This is why E8ComplianceAgent is blocking all trades.")
else:
    print("\n[OK] Account is within E8 limits")

print("=" * 80)
