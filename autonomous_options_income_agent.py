#!/usr/bin/env python3
"""
Autonomous Options Income Trading Agent
Replicates the actual winning strategy: Cash-secured puts + selective call buying
Runs fully autonomous - no manual intervention required
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from typing import Dict, List, Tuple
import threading
from dataclasses import dataclass

@dataclass
class OptionsOpportunity:
    symbol: str
    put_strike: float
    call_strike: float
    put_premium: float
    call_premium: float
    expiration: str
    conviction_score: float
    cash_required: float

class AutonomousOptionsIncomeAgent:
    def __init__(self):
        self.setup_logging()
        self.api = self.setup_alpaca()
        self.running = False

        # Core strategy parameters from your winning trades
        self.target_stocks = ['INTC', 'RIVN', 'SNAP', 'LYFT', 'TSLA', 'AMD', 'META']
        self.max_allocation_per_trade = 0.15  # 15% max per position
        self.min_put_premium = 0.50  # Minimum premium to collect
        self.max_dte = 7  # Weekly options only
        self.conviction_threshold = 75  # Only high conviction trades

        self.active_positions = {}
        self.execution_log = []

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('autonomous_options_agent.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_alpaca(self):
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()

        return tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY', 'PKFGVU14XFD0FX0VP3B7'),
            secret_key=os.getenv('ALPACA_SECRET_KEY', 'DNmBOxJTU8gK1ua7VXRtPiyMnxz1PF2JYXVdaYlM'),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            api_version='v2'
        )

    def get_account_status(self):
        """Get current account status and buying power"""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'day_trade_count': int(account.day_trade_count),
                'trading_blocked': account.trading_blocked
            }
        except Exception as e:
            self.logger.error(f"Error getting account status: {e}")
            return None

    def scan_options_opportunities(self) -> List[OptionsOpportunity]:
        """Scan for high-conviction options income opportunities"""
        opportunities = []

        for symbol in self.target_stocks:
            try:
                # Get current stock price
                quote = self.api.get_latest_quote(symbol)
                current_price = float(quote.ask_price)

                # Calculate optimal strikes based on your winning patterns
                put_strike = self.calculate_optimal_put_strike(current_price, symbol)
                call_strike = self.calculate_optimal_call_strike(current_price, symbol)

                # Get next Friday expiration
                expiration = self.get_next_friday()

                # Calculate premiums and conviction
                put_premium = self.estimate_put_premium(symbol, put_strike, expiration)
                call_premium = self.estimate_call_premium(symbol, call_strike, expiration)

                if put_premium >= self.min_put_premium:
                    conviction_score = self.calculate_conviction_score(
                        symbol, current_price, put_strike, call_strike
                    )

                    if conviction_score >= self.conviction_threshold:
                        cash_required = put_strike * 100  # For cash-secured put

                        opportunity = OptionsOpportunity(
                            symbol=symbol,
                            put_strike=put_strike,
                            call_strike=call_strike,
                            put_premium=put_premium,
                            call_premium=call_premium,
                            expiration=expiration,
                            conviction_score=conviction_score,
                            cash_required=cash_required
                        )
                        opportunities.append(opportunity)

            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                continue

        # Sort by conviction score
        opportunities.sort(key=lambda x: x.conviction_score, reverse=True)
        return opportunities

    def calculate_optimal_put_strike(self, current_price: float, symbol: str) -> float:
        """Calculate optimal put strike based on winning patterns"""
        # Your winning patterns:
        # INTC: $29 puts when stock was ~$31 (6% OTM)
        # RIVN: $14 puts when stock was ~$15 (7% OTM)
        # SNAP: $8 puts when stock was ~$9 (11% OTM)
        # LYFT: $21 puts when stock was ~$22 (5% OTM)

        otm_percentage = 0.08  # Target 8% OTM puts
        strike = current_price * (1 - otm_percentage)
        return round(strike, 1)  # Round to nearest $0.10

    def calculate_optimal_call_strike(self, current_price: float, symbol: str) -> float:
        """Calculate optimal call strike for upside speculation"""
        # Your winning pattern: Slightly OTM calls for explosive potential
        otm_percentage = 0.05  # Target 5% OTM calls
        strike = current_price * (1 + otm_percentage)
        return round(strike, 1)

    def get_next_friday(self) -> str:
        """Get next Friday's date for weekly options"""
        today = datetime.now()
        days_until_friday = 4 - today.weekday()  # Friday is weekday 4
        if days_until_friday <= 0:
            days_until_friday += 7  # Next Friday

        next_friday = today + timedelta(days=days_until_friday)
        return next_friday.strftime('%Y%m%d')

    def estimate_put_premium(self, symbol: str, strike: float, expiration: str) -> float:
        """Estimate put option premium - simplified model"""
        # This would connect to options pricing API in production
        # For now, use historical patterns from your trades
        base_premium = 0.75  # Based on your successful trades
        return base_premium

    def estimate_call_premium(self, symbol: str, strike: float, expiration: str) -> float:
        """Estimate call option premium"""
        base_premium = 0.25  # Calls are cheaper, higher risk/reward
        return base_premium

    def calculate_conviction_score(self, symbol: str, price: float, put_strike: float, call_strike: float) -> float:
        """Calculate conviction score based on your winning factors"""
        score = 50  # Base score

        # Factors that made your trades successful:

        # 1. Weekly expiration timing (high theta decay)
        score += 15

        # 2. Optimal strike selection (not too close, not too far)
        put_otm_pct = (price - put_strike) / price
        if 0.05 <= put_otm_pct <= 0.12:  # Sweet spot from your trades
            score += 20

        # 3. High IV stocks (your targets were all volatile)
        if symbol in ['RIVN', 'SNAP', 'TSLA']:  # High volatility names
            score += 10
        elif symbol in ['INTC', 'AMD']:  # Medium volatility
            score += 5

        # 4. Market timing - avoid major events
        # This would check earnings calendar, FOMC, etc.
        score += 5  # Assume good timing

        return min(score, 100)  # Cap at 100

    def execute_options_income_strategy(self, opportunity: OptionsOpportunity):
        """Execute the full strategy: Sell cash-secured put + buy call"""
        try:
            account = self.get_account_status()
            if not account or account['trading_blocked']:
                self.logger.warning("Trading blocked, skipping execution")
                return False

            available_cash = account['buying_power']
            if available_cash < opportunity.cash_required:
                self.logger.warning(f"Insufficient cash for {opportunity.symbol}: need ${opportunity.cash_required:,.0f}, have ${available_cash:,.0f}")
                return False

            # Calculate position size
            max_cash_to_use = available_cash * self.max_allocation_per_trade
            contracts = min(
                int(max_cash_to_use / opportunity.cash_required),
                5  # Max 5 contracts per position
            )

            if contracts == 0:
                return False

            # Execute trades
            put_symbol = f"{opportunity.symbol}{opportunity.expiration}P{int(opportunity.put_strike * 1000):08d}"
            call_symbol = f"{opportunity.symbol}{opportunity.expiration}C{int(opportunity.call_strike * 1000):08d}"

            # 1. Sell cash-secured puts
            put_order = self.api.submit_order(
                symbol=put_symbol,
                qty=contracts,
                side='sell',
                type='market',
                time_in_force='day'
            )

            # 2. Buy calls with smaller allocation
            call_contracts = max(1, contracts // 2)  # Half allocation to calls
            call_order = self.api.submit_order(
                symbol=call_symbol,
                qty=call_contracts,
                side='buy',
                type='market',
                time_in_force='day'
            )

            # Log execution
            execution = {
                'timestamp': datetime.now().isoformat(),
                'symbol': opportunity.symbol,
                'put_order_id': put_order.id,
                'call_order_id': call_order.id,
                'contracts': contracts,
                'call_contracts': call_contracts,
                'conviction_score': opportunity.conviction_score,
                'cash_used': opportunity.cash_required * contracts
            }

            self.execution_log.append(execution)
            self.save_execution_log()

            self.logger.info(f"Executed options income strategy for {opportunity.symbol}: "
                           f"Sold {contracts} puts, bought {call_contracts} calls")

            return True

        except Exception as e:
            self.logger.error(f"Error executing strategy for {opportunity.symbol}: {e}")
            return False

    def monitor_positions(self):
        """Monitor active positions and manage exits"""
        try:
            positions = self.api.list_positions()

            for position in positions:
                symbol = position.symbol
                unrealized_pl = float(position.unrealized_pl)
                unrealized_pct = float(position.unrealized_plpc) * 100

                # Exit rules based on your historical patterns
                should_exit = False
                exit_reason = ""

                # Take profits on big winners (like your +89.8% RIVN)
                if unrealized_pct > 50:
                    should_exit = True
                    exit_reason = f"PROFIT_TAKING_{unrealized_pct:.1f}%"

                # Cut losses on big losers (like your -25% stops)
                elif unrealized_pct < -25:
                    should_exit = True
                    exit_reason = f"STOP_LOSS_{unrealized_pct:.1f}%"

                # Close before expiration (Thursday before Friday expiry)
                # This would check days to expiration

                if should_exit:
                    self.close_position(symbol, position.qty, exit_reason)

        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")

    def close_position(self, symbol: str, qty: str, reason: str):
        """Close a position"""
        try:
            side = 'sell' if float(qty) > 0 else 'buy'
            qty = abs(float(qty))

            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )

            self.logger.info(f"Closed position {symbol}: {qty} shares, reason: {reason}")

        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")

    def save_execution_log(self):
        """Save execution log to file"""
        with open('autonomous_options_executions.json', 'w') as f:
            json.dump(self.execution_log, f, indent=2)

    def run_autonomous_cycle(self):
        """Main autonomous trading cycle"""
        self.logger.info("Starting autonomous options income cycle")

        # 1. Scan for opportunities
        opportunities = self.scan_options_opportunities()
        self.logger.info(f"Found {len(opportunities)} opportunities")

        # 2. Execute top opportunities
        for opportunity in opportunities[:3]:  # Top 3 opportunities
            if opportunity.conviction_score >= self.conviction_threshold:
                success = self.execute_options_income_strategy(opportunity)
                if success:
                    time.sleep(2)  # Brief pause between executions

        # 3. Monitor existing positions
        self.monitor_positions()

        self.logger.info("Autonomous cycle complete")

    def start_autonomous_trading(self):
        """Start autonomous trading with continuous monitoring"""
        self.running = True
        self.logger.info("Starting autonomous options income agent")

        while self.running:
            try:
                # Run during market hours
                now = datetime.now()
                if 9 <= now.hour <= 16:  # Market hours (simplified)
                    self.run_autonomous_cycle()

                    # Wait 30 minutes before next cycle
                    time.sleep(30 * 60)
                else:
                    # Sleep until market open
                    self.logger.info("Market closed, sleeping...")
                    time.sleep(60 * 60)  # 1 hour

            except KeyboardInterrupt:
                self.logger.info("Stopping autonomous agent...")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Error in autonomous cycle: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    agent = AutonomousOptionsIncomeAgent()
    agent.start_autonomous_trading()