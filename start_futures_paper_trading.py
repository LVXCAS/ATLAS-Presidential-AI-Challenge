#!/usr/bin/env python3
"""
START FUTURES PAPER TRADING
Conservative paper trading with VERY small size

Skips validation, goes straight to paper trading with maximum safety limits.
Perfect for: Testing the system, building confidence, learning futures mechanics.

SAFETY LIMITS:
- Only 1 contract per trade (MES=$5/point, MNQ=$2/point)
- Max 2 simultaneous positions
- Max $100 risk per trade
- Auto-stop after 3 consecutive losses
- Max $500 total risk across all positions

Usage:
    python start_futures_paper_trading.py           # Conservative mode
    python start_futures_paper_trading.py --max-risk 200  # Higher risk
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanners.futures_scanner import AIEnhancedFuturesScanner
from data.futures_data_fetcher import FuturesDataFetcher
from datetime import datetime, timedelta
import json
import time
import argparse
from typing import List, Dict


class ConservativeFuturesTrader:
    """
    Ultra-conservative futures paper trading system

    Safety First Approach:
    - Very small position sizes
    - Strict risk limits
    - Auto-shutoff on losing streaks
    - Detailed logging for learning
    """

    def __init__(self, max_risk_per_trade: float = 100.0, max_positions: int = 2, max_total_risk: float = 500.0):
        """
        Initialize conservative futures trader

        Args:
            max_risk_per_trade: Max $ risk per trade (default $100)
            max_positions: Max simultaneous positions (default 2)
            max_total_risk: Max total risk across all positions (default $500)
        """
        print("\n" + "="*70)
        print("CONSERVATIVE FUTURES PAPER TRADING")
        print("="*70)
        print(f"Max Risk Per Trade: ${max_risk_per_trade:.2f}")
        print(f"Max Positions: {max_positions}")
        print(f"Max Total Risk: ${max_total_risk:.2f}")
        print(f"Mode: PAPER TRADING (Alpaca Paper Account)")
        print("="*70 + "\n")

        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.max_total_risk = max_total_risk

        self.scanner = AIEnhancedFuturesScanner(paper_trading=True)
        self.data_fetcher = FuturesDataFetcher(paper_trading=True)

        self.active_positions = {}
        self.trade_history = []
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3

        self.start_time = datetime.now()
        self.total_risk_deployed = 0.0

        print("[SYSTEM READY] Conservative futures trading initialized")
        print("Scanning for opportunities with strict safety limits...\n")

    def calculate_position_size(self, opportunity: Dict) -> int:
        """
        Calculate conservative position size

        Returns:
            Number of contracts (always 1 in conservative mode)
        """
        # Always 1 contract in conservative mode
        risk_per_contract = opportunity['risk_per_contract']

        if risk_per_contract > self.max_risk_per_trade:
            print(f"⚠ Risk per contract ${risk_per_contract:.2f} exceeds max ${self.max_risk_per_trade:.2f}")
            return 0

        return 1  # Always 1 contract

    def can_take_trade(self, opportunity: Dict) -> tuple[bool, str]:
        """Check if we can take this trade given safety limits"""

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Max consecutive losses ({self.max_consecutive_losses}) reached - Stop trading"

        # Check max positions
        if len(self.active_positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        # Check if already have position in this symbol
        if opportunity['symbol'] in self.active_positions:
            return False, f"Already have position in {opportunity['symbol']}"

        # Check risk per contract
        risk = opportunity['risk_per_contract']
        if risk > self.max_risk_per_trade:
            return False, f"Risk ${risk:.2f} exceeds max ${self.max_risk_per_trade:.2f}"

        # Check total risk deployed
        if self.total_risk_deployed + risk > self.max_total_risk:
            return False, f"Total risk would exceed ${self.max_total_risk:.2f}"

        return True, "OK"

    def execute_paper_trade(self, opportunity: Dict):
        """Execute paper trade (simulated)"""

        # Calculate position size
        contracts = self.calculate_position_size(opportunity)

        if contracts == 0:
            print(f"[SKIP] {opportunity['symbol']} - Risk too high")
            return

        # Create position
        position = {
            'symbol': opportunity['symbol'],
            'direction': opportunity['direction'],
            'contracts': contracts,
            'entry_price': opportunity['entry_price'],
            'entry_time': datetime.now(),
            'stop_loss': opportunity['stop_loss'],
            'take_profit': opportunity['take_profit'],
            'risk_per_contract': opportunity['risk_per_contract'],
            'total_risk': opportunity['risk_per_contract'] * contracts,
            'status': 'ACTIVE',
            'score': opportunity['final_score'],
            'confidence': opportunity['confidence']
        }

        # Add to active positions
        self.active_positions[opportunity['symbol']] = position
        self.total_risk_deployed += position['total_risk']

        print(f"\n✓ [PAPER TRADE EXECUTED] {position['symbol']} {position['direction']}")
        print(f"  Contracts: {position['contracts']}")
        print(f"  Entry: ${position['entry_price']:.2f}")
        print(f"  Stop: ${position['stop_loss']:.2f}")
        print(f"  Target: ${position['take_profit']:.2f}")
        print(f"  Risk: ${position['total_risk']:.2f}")
        print(f"  Score: {position['score']:.2f} | Confidence: {position['confidence']:.0%}")
        print(f"  Total Deployed Risk: ${self.total_risk_deployed:.2f} / ${self.max_total_risk:.2f}")

    def update_positions(self):
        """Update all active positions"""

        closed_positions = []

        for symbol, position in self.active_positions.items():
            if position['status'] != 'ACTIVE':
                continue

            # Get current price
            current_price = self.data_fetcher.get_current_price(symbol)
            if not current_price:
                continue

            position['current_price'] = current_price

            # Calculate current P&L
            if position['direction'] == 'LONG':
                pnl_per_contract = current_price - position['entry_price']
            else:  # SHORT
                pnl_per_contract = position['entry_price'] - current_price

            # Apply point value
            point_value = 5.0 if 'MES' in symbol else 2.0  # MES=$5/pt, MNQ=$2/pt
            pnl = pnl_per_contract * point_value * position['contracts']
            position['current_pnl'] = pnl

            # Check exit conditions
            exit_reason = None

            # Check LONG exits
            if position['direction'] == 'LONG':
                if current_price <= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    position['exit_price'] = position['stop_loss']
                elif current_price >= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                    position['exit_price'] = position['take_profit']

            # Check SHORT exits
            elif position['direction'] == 'SHORT':
                if current_price >= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    position['exit_price'] = position['stop_loss']
                elif current_price <= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                    position['exit_price'] = position['take_profit']

            # Close position if exit triggered
            if exit_reason:
                self.close_position(symbol, exit_reason)
                closed_positions.append(symbol)

        # Remove closed positions from active
        for symbol in closed_positions:
            del self.active_positions[symbol]

    def close_position(self, symbol: str, reason: str):
        """Close position and record outcome"""

        position = self.active_positions[symbol]
        position['status'] = 'CLOSED'
        position['exit_time'] = datetime.now()
        position['exit_reason'] = reason

        # Calculate final P&L
        if position['direction'] == 'LONG':
            pnl_per_contract = position['exit_price'] - position['entry_price']
        else:
            pnl_per_contract = position['entry_price'] - position['exit_price']

        point_value = 5.0 if 'MES' in symbol else 2.0
        final_pnl = pnl_per_contract * point_value * position['contracts']
        position['final_pnl'] = final_pnl

        # Calculate duration
        duration = (position['exit_time'] - position['entry_time']).total_seconds() / 3600
        position['duration_hours'] = duration

        # Update consecutive losses
        if final_pnl > 0:
            self.consecutive_losses = 0
            position['outcome'] = 'WIN'
            emoji = "✓"
        else:
            self.consecutive_losses += 1
            position['outcome'] = 'LOSS'
            emoji = "✗"

        # Update total risk
        self.total_risk_deployed -= position['total_risk']

        # Log
        print(f"\n{emoji} [POSITION CLOSED] {symbol} - {reason}")
        print(f"  Entry: ${position['entry_price']:.2f} → Exit: ${position['exit_price']:.2f}")
        print(f"  P&L: ${final_pnl:.2f} ({position['outcome']})")
        print(f"  Duration: {duration:.1f} hours")
        print(f"  Consecutive Losses: {self.consecutive_losses}")

        # Add to history
        self.trade_history.append(position)

        # Save trade log
        self.save_trade_log()

    def get_statistics(self) -> Dict:
        """Get trading statistics"""

        if not self.trade_history:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }

        wins = [t for t in self.trade_history if t['outcome'] == 'WIN']
        losses = [t for t in self.trade_history if t['outcome'] == 'LOSS']

        total_pnl = sum(t['final_pnl'] for t in self.trade_history)
        win_rate = len(wins) / len(self.trade_history)

        avg_win = sum(t['final_pnl'] for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(abs(t['final_pnl']) for t in losses) / len(losses) if losses else 0.0

        return {
            'total_trades': len(self.trade_history),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'active_positions': len(self.active_positions),
            'consecutive_losses': self.consecutive_losses
        }

    def display_status(self):
        """Display current trading status"""

        stats = self.get_statistics()
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600

        print("\n" + "="*70)
        print("CONSERVATIVE FUTURES PAPER TRADING - STATUS")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: {runtime:.1f} hours")
        print()
        print(f"Active Positions: {stats['active_positions']} / {self.max_positions}")
        print(f"Risk Deployed: ${self.total_risk_deployed:.2f} / ${self.max_total_risk:.2f}")
        print()
        print(f"Completed Trades: {stats['total_trades']}")

        if stats['total_trades'] > 0:
            print(f"  Wins: {stats['wins']}")
            print(f"  Losses: {stats['losses']}")
            print(f"  Win Rate: {stats['win_rate']:.1%}")
            print(f"  Total P&L: ${stats['total_pnl']:.2f}")
            print(f"  Avg Win: ${stats['avg_win']:.2f}")
            print(f"  Avg Loss: ${stats['avg_loss']:.2f}")

        print()
        print(f"Consecutive Losses: {stats['consecutive_losses']} / {self.max_consecutive_losses}")

        if stats['consecutive_losses'] >= self.max_consecutive_losses:
            print("\n⚠ MAX CONSECUTIVE LOSSES REACHED - TRADING STOPPED")

        print("="*70)

    def save_trade_log(self):
        """Save trade history to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'futures_paper_trades_{timestamp}.json'

        data = {
            'start_time': self.start_time.isoformat(),
            'current_time': datetime.now().isoformat(),
            'configuration': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_positions': self.max_positions,
                'max_total_risk': self.max_total_risk,
                'max_consecutive_losses': self.max_consecutive_losses
            },
            'statistics': self.get_statistics(),
            'trade_history': self.trade_history,
            'active_positions': list(self.active_positions.values())
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def run_trading_session(self, duration_hours: int = 8):
        """Run paper trading session"""

        print(f"\nStarting {duration_hours}-hour conservative futures paper trading session...")
        print("System will scan every 15 minutes and execute high-quality setups\n")

        end_time = datetime.now() + timedelta(hours=duration_hours)
        scan_interval = 900  # 15 minutes
        status_interval = 3600  # 1 hour
        last_status_time = time.time()

        try:
            while datetime.now() < end_time:
                # Check if trading should stop
                if self.consecutive_losses >= self.max_consecutive_losses:
                    print("\n⚠ TRADING STOPPED - Max consecutive losses reached")
                    break

                # Update existing positions
                self.update_positions()

                # Scan for new opportunities (if we have capacity)
                if len(self.active_positions) < self.max_positions:
                    try:
                        opportunities = self.scanner.scan_all_futures()

                        for opp in opportunities:
                            # Check if we can take this trade
                            can_trade, reason = self.can_take_trade(opp)

                            if can_trade:
                                self.execute_paper_trade(opp)
                            elif "already have position" not in reason.lower() and len(self.active_positions) < self.max_positions:
                                print(f"[SKIP] {opp['symbol']} - {reason}")

                            # Only one new trade per scan cycle
                            if opp['symbol'] in self.active_positions:
                                break

                    except Exception as e:
                        print(f"[ERROR] Scan failed: {e}")

                # Display status every hour
                if time.time() - last_status_time >= status_interval:
                    self.display_status()
                    last_status_time = time.time()

                # Wait before next cycle
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Trading session stopped by user")

        # Close all positions at end of session
        print("\n[SESSION ENDING] Closing all positions...")
        for symbol in list(self.active_positions.keys()):
            self.close_position(symbol, "SESSION_END")

        # Final summary
        self.display_final_summary()

    def display_final_summary(self):
        """Display final trading summary"""

        stats = self.get_statistics()
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600

        print("\n\n" + "="*70)
        print("CONSERVATIVE FUTURES PAPER TRADING - SESSION COMPLETE")
        print("="*70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Runtime: {runtime:.1f} hours")
        print()
        print("FINAL RESULTS:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")

        if stats['total_trades'] > 0:
            print(f"  Avg Win: ${stats['avg_win']:.2f}")
            print(f"  Avg Loss: ${stats['avg_loss']:.2f}")

            if stats['avg_loss'] > 0:
                print(f"  Win/Loss Ratio: {stats['avg_win']/stats['avg_loss']:.2f}")

        print()

        if stats['total_pnl'] > 0:
            print("✓ PROFITABLE SESSION")
        elif stats['total_pnl'] < 0:
            print("✗ LOSING SESSION")
        else:
            print("⚬ BREAKEVEN SESSION")

        print("="*70)

        # Save final log
        self.save_trade_log()
        print(f"\n[TRADE LOG SAVED] Check futures_paper_trades_*.json for details\n")


def main():
    """Run conservative futures paper trading"""

    parser = argparse.ArgumentParser(description='Conservative Futures Paper Trading')
    parser.add_argument('--max-risk', type=float, default=100.0, help='Max risk per trade (default: $100)')
    parser.add_argument('--max-positions', type=int, default=2, help='Max simultaneous positions (default: 2)')
    parser.add_argument('--max-total-risk', type=float, default=500.0, help='Max total risk (default: $500)')
    parser.add_argument('--duration', type=int, default=8, help='Session duration in hours (default: 8)')
    parser.add_argument('--quick-test', action='store_true', help='Quick 1-hour test')

    args = parser.parse_args()

    if args.quick_test:
        args.duration = 1
        print("\n[QUICK TEST MODE] Running 1-hour test session\n")

    # Initialize trader
    trader = ConservativeFuturesTrader(
        max_risk_per_trade=args.max_risk,
        max_positions=args.max_positions,
        max_total_risk=args.max_total_risk
    )

    # Run trading session
    trader.run_trading_session(duration_hours=args.duration)


if __name__ == "__main__":
    main()
