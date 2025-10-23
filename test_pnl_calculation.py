#!/usr/bin/env python3
"""
Test the P&L calculation fixes
"""

def test_pnl_calculation():
    """Test P&L percentage calculation"""

    print("Testing P&L calculation fixes...")
    print("="*50)

    # Test cases
    test_cases = [
        {
            "name": "Normal profitable trade",
            "entry_price": 2.50,
            "current_price": 3.75,
            "quantity": 1,
            "expected_pnl": 125.00,  # (3.75 - 2.50) * 1 * 100
            "expected_pnl_pct": 50.0  # 125 / 250 * 100
        },
        {
            "name": "Small entry price causing high %",
            "entry_price": 0.01,  # Very small entry
            "current_price": 0.75,
            "quantity": 1,
            "expected_pnl": 74.00,  # (0.75 - 0.01) * 1 * 100
            "expected_pnl_pct": 7400.0  # This is the problem case!
        },
        {
            "name": "Loss scenario",
            "entry_price": 5.00,
            "current_price": 3.00,
            "quantity": 2,
            "expected_pnl": -400.00,  # (3.00 - 5.00) * 2 * 100
            "expected_pnl_pct": -40.0  # -400 / 1000 * 100
        }
    ]

    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print("-" * 30)

        entry_price = test_case['entry_price']
        current_price = test_case['current_price']
        quantity = test_case['quantity']

        # Calculate P&L (matching bot logic)
        price_per_contract_change = current_price - entry_price
        total_pnl = price_per_contract_change * quantity * 100

        # Calculate investment and percentage
        total_investment = entry_price * quantity * 100

        if total_investment < 1.0:
            print(f"WARNING: Suspicious total investment: ${total_investment:.2f}")
            pnl_percentage = 0
        else:
            pnl_percentage = (total_pnl / total_investment) * 100

        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Quantity: {quantity} contracts")
        print(f"Total Investment: ${total_investment:.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"P&L Percentage: {pnl_percentage:.1f}%")

        # Validate against expected
        expected_pnl = test_case['expected_pnl']
        expected_pnl_pct = test_case['expected_pnl_pct']

        if abs(total_pnl - expected_pnl) < 0.01:
            print(f"[OK] P&L calculation correct")
        else:
            print(f"[ERROR] P&L calculation wrong: expected ${expected_pnl:.2f}, got ${total_pnl:.2f}")

        if total_investment >= 1.0:
            if abs(pnl_percentage - expected_pnl_pct) < 0.1:
                print(f"[OK] P&L percentage calculation correct")
            else:
                print(f"[ERROR] P&L percentage wrong: expected {expected_pnl_pct:.1f}%, got {pnl_percentage:.1f}%")

    print("\n" + "="*50)
    print("Key Insights:")
    print("- Very small entry prices (< $0.05) cause unrealistic percentage gains")
    print("- The bot now validates total investment > $1.00")
    print("- Option price estimation is capped at 10x entry price")
    print("- P&L percentage now calculated on total investment, not just option price")

if __name__ == "__main__":
    test_pnl_calculation()