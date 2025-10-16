#!/usr/bin/env python3
"""
Profit/Loss Monitor - Daily Trading Limits System
Monitors daily P&L and triggers sell-all when limits are reached:
- Profit Target: +5.75% (take profits)
- Loss Limit: -4.9% (cut losses)
"""

import asyncio
import time
import json
import logging
from datetime import datetime, date
from typing import Dict, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Import broker integration
from agents.broker_integration import AlpacaBrokerIntegration

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfitTargetMonitor:
    """Monitor daily profit/loss and trigger sell-all at targets"""

    def __init__(self):
        self.broker = None
        self.initial_equity = None
        self.current_equity = None
        self.daily_profit_pct = 0.0
        self.profit_target_pct = 5.75  # 5.75% daily profit target
        self.loss_limit_pct = -4.9     # 4.9% daily loss limit
        self.target_hit = False
        self.loss_limit_hit = False
        self.monitoring_active = True

        # Track starting equity for the day
        self.trading_date = date.today()
        self.starting_equity_file = "daily_starting_equity.json"

        logger.info(f"Profit/Loss Monitor initialized - Profit Target: {self.profit_target_pct}% | Loss Limit: {self.loss_limit_pct}%")

    async def initialize_broker(self):
        """Initialize broker connection"""
        try:
            self.broker = AlpacaBrokerIntegration(paper_trading=True)
            # Test connection by getting account info
            account_info = await self.broker.get_account_info()
            if account_info:
                logger.info("Broker connection established")
                return True
            else:
                logger.error("Failed to get account info from broker")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize broker: {e}")
            return False

    def load_starting_equity(self) -> Optional[float]:
        """Load or set starting equity for the day"""
        try:
            if os.path.exists(self.starting_equity_file):
                with open(self.starting_equity_file, 'r') as f:
                    data = json.load(f)

                # Check if it's the same trading day
                if data.get('date') == str(self.trading_date):
                    starting_equity = data.get('starting_equity')
                    logger.info(f"Using saved starting equity for {self.trading_date}: ${starting_equity:,.2f}")
                    return starting_equity

            # If no file or different date, get current portfolio value as starting point
            account = self.broker.api.get_account()

            # FIXED: Use portfolio_value (same as we use for current_equity)
            current_equity = float(account.portfolio_value)

            # Save new starting equity
            with open(self.starting_equity_file, 'w') as f:
                json.dump({
                    'date': str(self.trading_date),
                    'starting_equity': current_equity,
                    'timestamp': datetime.now().isoformat(),
                    'account_equity': float(account.equity),
                    'cash': float(account.cash)
                }, f)

            logger.info(f"New trading day - Starting equity: ${current_equity:,.2f}")
            return current_equity

        except Exception as e:
            logger.error(f"Error loading starting equity: {e}")
            return None

    async def check_daily_profit(self) -> Tuple[float, float, bool, bool]:
        """
        Check current daily profit/loss percentage
        Returns: (current_equity, profit_pct, profit_target_hit, loss_limit_hit)
        """
        try:
            # Get current account info
            account = self.broker.api.get_account()

            # FIXED: Use portfolio_value which is more accurate than equity
            # portfolio_value = actual market value of all holdings + cash
            # equity = portfolio_value but can lag in some cases
            self.current_equity = float(account.portfolio_value)

            # Also get cash and last_equity for comparison
            current_cash = float(account.cash)
            last_equity = float(account.last_equity)
            account_equity = float(account.equity)

            # Debug logging to understand discrepancies
            logger.debug(f"Account values - Equity: ${account_equity:.2f}, "
                        f"Portfolio: ${self.current_equity:.2f}, "
                        f"Cash: ${current_cash:.2f}, "
                        f"Last Equity: ${last_equity:.2f}")

            # Get or set starting equity for the day
            if self.initial_equity is None:
                self.initial_equity = self.load_starting_equity()

            if self.initial_equity is None:
                logger.error("Could not determine starting equity")
                return self.current_equity, 0.0, False, False

            # Calculate daily profit percentage
            daily_profit = self.current_equity - self.initial_equity
            self.daily_profit_pct = (daily_profit / self.initial_equity) * 100

            # Check if profit target is hit
            profit_target_hit = self.daily_profit_pct >= self.profit_target_pct

            # Check if loss limit is hit
            loss_limit_hit = self.daily_profit_pct <= self.loss_limit_pct

            return self.current_equity, self.daily_profit_pct, profit_target_hit, loss_limit_hit

        except Exception as e:
            logger.error(f"Error checking daily profit: {e}")
            return 0.0, 0.0, False, False

    async def sell_all_positions(self, reason: str = "Target/limit reached"):
        """Sell all positions and cancel all orders"""
        try:
            if "loss" in reason.lower() or "limit" in reason.lower():
                logger.info(f"🛑 LOSS LIMIT HIT! Selling all positions: {reason}")
            else:
                logger.info(f"🎯 PROFIT TARGET HIT! Selling all positions: {reason}")

            # Cancel all pending orders first
            try:
                orders = self.broker.api.list_orders(status='open')
                for order in orders:
                    self.broker.api.cancel_order(order.id)
                    logger.info(f"Cancelled order: {order.id}")
            except Exception as e:
                logger.warning(f"Error cancelling orders: {e}")

            # Close all positions
            success = await self.broker.close_all_positions()

            if success:
                logger.info("✅ All positions closed successfully")

                # Log the event
                self.log_trade_event(reason)

                # Set appropriate flags
                if "loss" in reason.lower() or "limit" in reason.lower():
                    self.loss_limit_hit = True
                else:
                    self.target_hit = True

                return True
            else:
                logger.error("❌ Failed to close all positions")
                return False

        except Exception as e:
            logger.error(f"Error in sell_all_positions: {e}")
            return False

    def log_trade_event(self, reason: str):
        """Log the trading event (profit target or loss limit) for record keeping"""
        try:
            is_loss_event = "loss" in reason.lower() or "limit" in reason.lower()

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'date': str(self.trading_date),
                'event_type': 'loss_limit' if is_loss_event else 'profit_target',
                'initial_equity': self.initial_equity,
                'final_equity': self.current_equity,
                'daily_profit_pct': self.daily_profit_pct,
                'profit_target_pct': self.profit_target_pct,
                'loss_limit_pct': self.loss_limit_pct,
                'pnl_amount': self.current_equity - self.initial_equity,
                'reason': reason
            }

            # Append to trading events log
            log_file = "trading_events.json"
            events = []

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    events = json.load(f)

            events.append(log_entry)

            with open(log_file, 'w') as f:
                json.dump(events, f, indent=2)

            if is_loss_event:
                logger.info(f"Loss limit event logged: {self.daily_profit_pct:.2f}%")
            else:
                logger.info(f"Profit target event logged: +{self.daily_profit_pct:.2f}%")

        except Exception as e:
            logger.error(f"Error logging trading event: {e}")

    async def monitor_profit_target(self, check_interval: int = 60):
        """
        Main monitoring loop - checks profit and loss limits every minute
        Args:
            check_interval: seconds between checks (default 60 seconds)
        """
        logger.info(f"Starting profit/loss monitoring (checking every {check_interval}s)")
        logger.info(f"Profit target: {self.profit_target_pct}% | Loss limit: {self.loss_limit_pct}%")

        while self.monitoring_active and not self.target_hit and not self.loss_limit_hit:
            try:
                current_equity, profit_pct, profit_target_hit, loss_limit_hit = await self.check_daily_profit()

                # Log current status
                profit_amount = current_equity - (self.initial_equity or 0)
                logger.info(f"💰 Current: ${current_equity:,.2f} | "
                          f"Daily P&L: ${profit_amount:+,.2f} ({profit_pct:+.2f}%) | "
                          f"Target: {self.profit_target_pct}% | Limit: {self.loss_limit_pct}%")

                # Check if profit target is hit
                if profit_target_hit and not self.target_hit:
                    logger.info(f"🎯 PROFIT TARGET REACHED! {profit_pct:.2f}% >= {self.profit_target_pct}%")

                    # Trigger sell-all for profit
                    success = await self.sell_all_positions(
                        f"Daily profit target {self.profit_target_pct}% reached ({profit_pct:.2f}%)"
                    )

                    if success:
                        logger.info("✅ Profit-taking completed successfully")
                        break
                    else:
                        logger.error("❌ Profit-taking failed, continuing monitoring")

                # Check if loss limit is hit
                elif loss_limit_hit and not self.loss_limit_hit:
                    logger.warning(f"🛑 LOSS LIMIT REACHED! {profit_pct:.2f}% <= {self.loss_limit_pct}%")

                    # Trigger sell-all for loss protection
                    success = await self.sell_all_positions(
                        f"Daily loss limit {self.loss_limit_pct}% reached ({profit_pct:.2f}%)"
                    )

                    if success:
                        logger.info("✅ Loss protection completed successfully")
                        break
                    else:
                        logger.error("❌ Loss protection failed, continuing monitoring")

                # Wait before next check
                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(check_interval)

        logger.info("Profit/loss monitoring stopped")

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring_active = False
        logger.info("Monitoring stopped by user request")

    def get_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'target_hit': self.target_hit,
            'loss_limit_hit': self.loss_limit_hit,
            'profit_target_pct': self.profit_target_pct,
            'loss_limit_pct': self.loss_limit_pct,
            'current_profit_pct': self.daily_profit_pct,
            'initial_equity': self.initial_equity,
            'current_equity': self.current_equity,
            'trading_date': str(self.trading_date)
        }

# Convenience function for other modules
async def start_profit_monitoring(check_interval: int = 60):
    """Start profit monitoring in a background task"""
    monitor = ProfitTargetMonitor()

    # Initialize broker
    if not await monitor.initialize_broker():
        logger.error("Failed to initialize broker for profit monitoring")
        return None

    # Start monitoring
    task = asyncio.create_task(monitor.monitor_profit_target(check_interval))
    return monitor, task

if __name__ == "__main__":
    async def main():
        # Create and start monitor
        monitor = ProfitTargetMonitor()

        if await monitor.initialize_broker():
            await monitor.monitor_profit_target()
        else:
            print("Failed to initialize broker connection")

    # Run the monitor
    asyncio.run(main())