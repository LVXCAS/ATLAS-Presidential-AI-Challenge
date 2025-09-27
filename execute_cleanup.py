#!/usr/bin/env python3
"""
EXECUTE PORTFOLIO CLEANUP
Close losing positions to free capital for concentrated strategy
"""

from simple_portfolio_cleaner import SimplePortfolioCleaner

def main():
    print("EXECUTING PORTFOLIO CLEANUP!")
    print("=" * 50)
    print("Closing losing positions to free capital for concentrated strategy...")

    cleaner = SimplePortfolioCleaner()

    # Execute the cleanup
    result = cleaner.close_all_positions(execute=True)

    print(f"\nCLEANUP COMPLETE! Processed {len(result)} position closures")
    print("Capital freed for Intel-puts-style concentrated deployment!")

    return result

if __name__ == "__main__":
    main()