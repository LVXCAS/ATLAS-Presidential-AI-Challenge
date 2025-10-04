#!/usr/bin/env python3
"""
TERMINAL LOGGER - Real-Time System Visibility
==============================================
Beautiful terminal output showing exactly what the system is doing
"""

from datetime import datetime
from colorama import init, Fore, Back, Style
import sys

# Initialize colorama for Windows
init(autoreset=True)

class TerminalLogger:
    """Professional terminal logging with colors and structure"""

    def __init__(self, system_name="WEEK 1 TRADING SYSTEM"):
        self.system_name = system_name
        self.scan_count = 0
        self.trade_count = 0

    def header(self):
        """Print system header"""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{Style.BRIGHT}{self.system_name.center(80)}")
        print("=" * 80)
        print(f"{Fore.WHITE}Started: {datetime.now().strftime('%I:%M:%S %p PDT')}")
        print(f"Mode: Paper Trading | Threshold: 4.0+ confidence | Target: 5-8% weekly ROI")
        print("=" * 80 + "\n")

    def scan_start(self, scan_number):
        """Log scan start"""
        self.scan_count = scan_number
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"\n{Fore.YELLOW}[{timestamp}] SCAN #{scan_number} STARTING...")
        print(f"{Fore.WHITE}{'-' * 80}")

    def opportunity_found(self, symbol, strategy, price, score, qualified=True):
        """Log opportunity discovery"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')

        if qualified:
            print(f"{Fore.GREEN}[OK] [{timestamp}] OPPORTUNITY FOUND: {symbol}")
            print(f"  {Fore.WHITE}Strategy: {strategy} | Price: ${price:.2f} | Score: {Fore.GREEN}{score:.2f}")
        else:
            print(f"{Fore.RED}[X] [{timestamp}] {symbol}: Score {score:.2f} (below 4.0 threshold)")

    def greeks_analysis(self, symbol, delta, theta, vega, boost):
        """Log Black-Scholes Greeks analysis"""
        print(f"{Fore.CYAN}  [GREEKS] {symbol}:")
        print(f"    Delta: {delta:.3f} | Theta: {theta:.3f} | Vega: {vega:.3f}")
        if boost > 0:
            print(f"    {Fore.GREEN}Score boost: +{boost:.2f}")

    def execution_start(self, symbol, strategy):
        """Log trade execution start"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}>>> [{timestamp}] EXECUTING TRADE: {symbol}")
        print(f"{Fore.WHITE}    Strategy: {strategy}")

    def order_submitted(self, order_type, symbol, qty, status="SUBMITTED"):
        """Log order submission"""
        if status == "FILLED":
            print(f"{Fore.GREEN}    [OK] {order_type}: {symbol} x{qty} - {status}")
        else:
            print(f"{Fore.YELLOW}    [->] {order_type}: {symbol} x{qty} - {status}")

    def execution_complete(self, symbol, num_orders, total_cost):
        """Log execution completion"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        self.trade_count += 1
        print(f"{Fore.GREEN}{Style.BRIGHT}[SUCCESS] [{timestamp}] EXECUTION COMPLETE: {symbol}")
        print(f"{Fore.WHITE}    Orders filled: {num_orders} | Total cost: ${total_cost:,.2f}")
        print(f"{Fore.WHITE}    Trades today: {self.trade_count}/2")

    def execution_error(self, symbol, error):
        """Log execution error"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"{Fore.RED}[FAIL] [{timestamp}] EXECUTION FAILED: {symbol}")
        print(f"{Fore.RED}    Error: {error}")

    def portfolio_update(self, value, pl, pl_pct, positions):
        """Log portfolio status"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')

        pl_color = Fore.GREEN if pl >= 0 else Fore.RED
        print(f"\n{Fore.CYAN}[{timestamp}] PORTFOLIO UPDATE:")
        print(f"{Fore.WHITE}  Value: ${value:,.2f} | P&L: {pl_color}${pl:+,.2f} ({pl_pct:+.2f}%)")
        print(f"{Fore.WHITE}  Open positions: {positions}")

    def risk_check(self, status, warnings=None):
        """Log risk management check"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')

        if status == "OK":
            print(f"{Fore.GREEN}[{timestamp}] RISK CHECK: All limits OK")
        else:
            print(f"{Fore.RED}[{timestamp}] RISK CHECK: WARNING!")
            if warnings:
                for warning in warnings:
                    print(f"{Fore.RED}  ! {warning}")

    def scan_complete(self, opportunities_found, trades_executed):
        """Log scan completion"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"\n{Fore.WHITE}{'-' * 80}")
        print(f"{Fore.YELLOW}[{timestamp}] SCAN #{self.scan_count} COMPLETE")
        print(f"{Fore.WHITE}  Opportunities found: {opportunities_found}")
        print(f"{Fore.WHITE}  Trades executed: {trades_executed}")
        print(f"{Fore.WHITE}  Next scan in 5 minutes...")

    def market_status(self, is_open):
        """Log market status"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')

        if is_open:
            print(f"{Fore.GREEN}[{timestamp}] MARKET: OPEN - Scanning active")
        else:
            print(f"{Fore.RED}[{timestamp}] MARKET: CLOSED - Scanner paused")

    def rd_discovery(self, strategies_found, best_score):
        """Log R&D discovery results"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"\n{Fore.MAGENTA}[{timestamp}] R&D DISCOVERY:")
        print(f"{Fore.WHITE}  New strategies discovered: {strategies_found}")
        if best_score:
            print(f"{Fore.WHITE}  Best score: {Fore.GREEN}{best_score:.2f}")

    def day_summary(self, scans, opportunities, trades, pl, pl_pct):
        """Print end of day summary"""
        pl_color = Fore.GREEN if pl >= 0 else Fore.RED

        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{Style.BRIGHT}{'END OF DAY SUMMARY'.center(80)}")
        print("=" * 80)
        print(f"{Fore.WHITE}Total scans: {scans}")
        print(f"{Fore.WHITE}Opportunities found: {opportunities}")
        print(f"{Fore.WHITE}Trades executed: {trades}")
        print(f"{Fore.WHITE}Daily P&L: {pl_color}${pl:+,.2f} ({pl_pct:+.2f}%)")
        print("=" * 80)

    def info(self, message):
        """Log informational message"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"{Fore.BLUE}[{timestamp}] INFO: {message}")

    def warning(self, message):
        """Log warning message"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"{Fore.YELLOW}[{timestamp}] WARNING: {message}")

    def error(self, message):
        """Log error message"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"{Fore.RED}[{timestamp}] ERROR: {message}")

    def success(self, message):
        """Log success message"""
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        print(f"{Fore.GREEN}[{timestamp}] SUCCESS: {message}")


# Demo function
def demo_logger():
    """Demonstrate the terminal logger"""

    logger = TerminalLogger()

    # System startup
    logger.header()

    # Scan 1
    logger.scan_start(1)
    logger.market_status(True)

    # Found opportunity
    logger.opportunity_found("INTC", "Intel Dual Strategy", 23.45, 4.2, qualified=True)
    logger.greeks_analysis("INTC", -0.268, -0.014, 0.15, 0.8)

    # Execution
    logger.execution_start("INTC", "Intel Dual Strategy")
    logger.order_submitted("CASH SECURED PUT", "INTC251024P00033000", 2, "FILLED")
    logger.order_submitted("LONG CALL", "INTC251024C00036000", 2, "FILLED")
    logger.execution_complete("INTC", 2, 720.50)

    # Portfolio update
    logger.portfolio_update(100250.00, 250.00, 0.25, 2)

    # Risk check
    logger.risk_check("OK")

    # Scan complete
    logger.scan_complete(1, 1)

    # Scan 2 - no opportunities
    logger.scan_start(2)
    logger.opportunity_found("AMD", "Intel Dual Strategy", 145.20, 3.8, qualified=False)
    logger.opportunity_found("NVDA", "Intel Dual Strategy", 485.50, 3.9, qualified=False)
    logger.scan_complete(0, 0)

    # R&D update
    logger.rd_discovery(3, 4.5)

    # Day summary
    logger.day_summary(12, 3, 1, 250.00, 0.25)


if __name__ == "__main__":
    # Install colorama if needed
    try:
        from colorama import init
    except ImportError:
        print("Installing colorama for colored terminal output...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
        from colorama import init, Fore, Back, Style
        init(autoreset=True)

    demo_logger()
