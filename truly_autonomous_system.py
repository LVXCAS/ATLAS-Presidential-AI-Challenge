"""
TRULY AUTONOMOUS SELF-DETERMINING SYSTEM
Real-time performance monitoring with immediate scaling decisions
No schedules - pure event-driven intelligence
"""

import asyncio
import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('truly_autonomous.log'),
        logging.StreamHandler()
    ]
)

class TrulyAutonomousSystem:
    """Self-determining autonomous trading system with real-time intelligence"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Self-determining thresholds
        self.scaling_thresholds = {
            'winner_threshold': 3.0,      # Scale up at +3%
            'strong_winner_threshold': 5.0,   # Aggressive scale at +5%
            'loser_threshold': -2.0,      # Scale down at -2%
            'heavy_loser_threshold': -3.0,    # Aggressive scale down at -3%
            'min_scale_amount': 1000,     # Minimum $1000 trades
            'max_position_scale': 0.15,   # Max 15% position increase
            'risk_scale_down': 0.25       # 25% reduction for losers
        }

        # Intelligence parameters
        self.intelligence_settings = {
            'monitoring_interval': 300,    # Check every 5 minutes
            'rapid_check_interval': 60,    # Rapid checks every 1 minute during volatility
            'volatility_threshold': 2.0,   # 2% moves trigger rapid mode
            'momentum_confirmation': 3,    # 3 consecutive checks to confirm trend
            'max_daily_trades': 20,        # Prevent overtrading
            'buying_power_reserve': 0.1    # Keep 10% cash reserve
        }

        # State tracking
        self.last_portfolio_value = 0
        self.position_performance_history = {}
        self.scaling_actions_today = 0
        self.rapid_monitoring_mode = False
        self.last_check_time = datetime.now()

        logging.info("TRULY AUTONOMOUS SYSTEM INITIALIZED")
        logging.info("Real-time self-determining intelligence ACTIVE")

    async def get_current_state(self):
        """Get comprehensive current state for decision making"""

        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.list_positions()

            portfolio_value = float(account.portfolio_value)
            buying_power = float(account.buying_power)
            day_change = portfolio_value - self.last_portfolio_value if self.last_portfolio_value > 0 else 0

            # Analyze each position's performance
            position_analysis = []
            total_unrealized_pl = 0

            for pos in positions:
                unrealized_pl = float(pos.unrealized_pl)
                market_value = float(pos.market_value)
                pnl_pct = (unrealized_pl / market_value) * 100 if market_value > 0 else 0

                position_data = {
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'value': market_value,
                    'pnl': unrealized_pl,
                    'pnl_pct': pnl_pct,
                    'avg_entry_price': float(pos.avg_entry_price)
                }

                position_analysis.append(position_data)
                total_unrealized_pl += unrealized_pl

                # Track performance history for momentum detection
                if pos.symbol not in self.position_performance_history:
                    self.position_performance_history[pos.symbol] = []

                self.position_performance_history[pos.symbol].append({
                    'timestamp': datetime.now(),
                    'pnl_pct': pnl_pct,
                    'value': market_value
                })

                # Keep only last 20 data points
                if len(self.position_performance_history[pos.symbol]) > 20:
                    self.position_performance_history[pos.symbol] = self.position_performance_history[pos.symbol][-20:]

            portfolio_pnl_pct = (total_unrealized_pl / portfolio_value) * 100 if portfolio_value > 0 else 0

            return {
                'portfolio_value': portfolio_value,
                'buying_power': buying_power,
                'day_change': day_change,
                'portfolio_pnl_pct': portfolio_pnl_pct,
                'positions': position_analysis,
                'market_open': self.is_market_open()
            }

        except Exception as e:
            logging.error(f"Error getting current state: {e}")
            return None

    def is_market_open(self):
        """Check if market is open"""

        try:
            clock = self.alpaca.get_clock()
            return clock.is_open
        except:
            # Fallback check
            now = datetime.now()
            is_weekday = now.weekday() < 5
            market_start = now.replace(hour=6, minute=30, second=0)
            market_end = now.replace(hour=13, minute=0, second=0)
            return is_weekday and market_start <= now <= market_end

    async def analyze_scaling_opportunities(self, state):
        """Self-determine scaling opportunities based on real-time analysis"""

        if not state or not state['market_open']:
            return []

        opportunities = []
        positions = state['positions']

        # Detect high-volatility situations for rapid monitoring
        portfolio_change_rate = abs(state['portfolio_pnl_pct'])
        if portfolio_change_rate > self.intelligence_settings['volatility_threshold']:
            self.rapid_monitoring_mode = True
            logging.info(f"🔥 RAPID MONITORING MODE: Portfolio volatility {portfolio_change_rate:.1f}%")

        for position in positions:
            symbol = position['symbol']
            pnl_pct = position['pnl_pct']
            current_value = position['value']

            # Analyze momentum from history
            momentum_strength = self.calculate_momentum_strength(symbol)

            # SCALE UP OPPORTUNITIES
            if pnl_pct >= self.scaling_thresholds['strong_winner_threshold']:
                # Strong winner with momentum
                if momentum_strength > 0.5:  # Confirmed upward momentum
                    scale_amount = min(
                        current_value * self.scaling_thresholds['max_position_scale'] * 1.5,  # Aggressive scaling
                        state['buying_power'] * 0.4,
                        15000  # Max $15K for strong winners
                    )

                    opportunities.append({
                        'type': 'AGGRESSIVE_SCALE_UP',
                        'symbol': symbol,
                        'amount': scale_amount,
                        'reason': f"Strong winner +{pnl_pct:.1f}% with momentum {momentum_strength:.2f}",
                        'priority': 'HIGH'
                    })

            elif pnl_pct >= self.scaling_thresholds['winner_threshold']:
                # Regular winner
                scale_amount = min(
                    current_value * self.scaling_thresholds['max_position_scale'],
                    state['buying_power'] * 0.3,
                    10000
                )

                opportunities.append({
                    'type': 'SCALE_UP',
                    'symbol': symbol,
                    'amount': scale_amount,
                    'reason': f"Winner +{pnl_pct:.1f}%",
                    'priority': 'MEDIUM'
                })

            # SCALE DOWN OPPORTUNITIES
            elif pnl_pct <= self.scaling_thresholds['heavy_loser_threshold']:
                # Heavy loser - immediate risk management
                shares_to_sell = int(position['qty'] * self.scaling_thresholds['risk_scale_down'] * 1.5)  # Aggressive

                opportunities.append({
                    'type': 'AGGRESSIVE_SCALE_DOWN',
                    'symbol': symbol,
                    'shares': shares_to_sell,
                    'reason': f"Heavy loser {pnl_pct:.1f}% - risk management",
                    'priority': 'URGENT'
                })

            elif pnl_pct <= self.scaling_thresholds['loser_threshold']:
                # Regular loser
                shares_to_sell = int(position['qty'] * self.scaling_thresholds['risk_scale_down'])

                opportunities.append({
                    'type': 'SCALE_DOWN',
                    'symbol': symbol,
                    'shares': shares_to_sell,
                    'reason': f"Loser {pnl_pct:.1f}%",
                    'priority': 'MEDIUM'
                })

        # Filter opportunities by available resources and daily limits
        filtered_opportunities = []

        for opp in opportunities:
            # Check daily trade limit
            if self.scaling_actions_today >= self.intelligence_settings['max_daily_trades']:
                logging.warning("Daily trade limit reached - filtering opportunities")
                break

            # Check minimum amounts
            if opp['type'] in ['SCALE_UP', 'AGGRESSIVE_SCALE_UP']:
                if opp['amount'] >= self.scaling_thresholds['min_scale_amount']:
                    filtered_opportunities.append(opp)
            else:
                if opp.get('shares', 0) > 0:
                    filtered_opportunities.append(opp)

        # Sort by priority
        priority_order = {'URGENT': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        filtered_opportunities.sort(key=lambda x: priority_order.get(x['priority'], 4))

        return filtered_opportunities

    def calculate_momentum_strength(self, symbol):
        """Calculate momentum strength from performance history"""

        if symbol not in self.position_performance_history:
            return 0

        history = self.position_performance_history[symbol]
        if len(history) < 3:
            return 0

        # Calculate recent trend
        recent_pnl = [h['pnl_pct'] for h in history[-5:]]  # Last 5 data points

        if len(recent_pnl) < 2:
            return 0

        # Simple momentum calculation
        trend = np.polyfit(range(len(recent_pnl)), recent_pnl, 1)[0]  # Slope
        consistency = 1 - (np.std(recent_pnl) / (abs(np.mean(recent_pnl)) + 0.1))  # Consistency score

        momentum = min(1.0, max(-1.0, trend * consistency))
        return momentum

    async def execute_scaling_opportunities(self, opportunities):
        """Execute the self-determined scaling opportunities"""

        if not opportunities:
            return

        logging.info(f"🎯 EXECUTING {len(opportunities)} SELF-DETERMINED OPPORTUNITIES")

        executed_actions = []

        for opp in opportunities:
            try:
                symbol = opp['symbol']

                if opp['type'] in ['SCALE_UP', 'AGGRESSIVE_SCALE_UP']:
                    # Scale up position
                    amount = opp['amount']
                    ticker_data = self.alpaca.get_latest_trade(symbol)
                    current_price = float(ticker_data.price)
                    shares_to_buy = int(amount / current_price)

                    if shares_to_buy > 0:
                        order = self.alpaca.submit_order(
                            symbol=symbol,
                            qty=shares_to_buy,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )

                        executed_actions.append(f"🚀 {opp['type']}: {symbol} +{shares_to_buy} shares - {opp['reason']}")
                        self.scaling_actions_today += 1

                elif opp['type'] in ['SCALE_DOWN', 'AGGRESSIVE_SCALE_DOWN']:
                    # Scale down position
                    shares_to_sell = opp['shares']

                    order = self.alpaca.submit_order(
                        symbol=symbol,
                        qty=shares_to_sell,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )

                    executed_actions.append(f"⚠️ {opp['type']}: {symbol} -{shares_to_sell} shares - {opp['reason']}")
                    self.scaling_actions_today += 1

            except Exception as e:
                logging.error(f"Failed to execute {opp['type']} for {opp['symbol']}: {e}")

        # Report executed actions
        for action in executed_actions:
            logging.info(f"  {action}")

        if executed_actions:
            logging.info(f"✅ SELF-DETERMINED EXECUTION COMPLETED: {len(executed_actions)} actions")

    async def continuous_intelligence_loop(self):
        """Main intelligence loop - truly autonomous decision making"""

        logging.info("🧠 STARTING CONTINUOUS INTELLIGENCE LOOP")
        logging.info("System will self-determine scaling opportunities in real-time")

        while True:
            try:
                # Get current state
                state = await self.get_current_state()

                if state and state['market_open']:
                    # Analyze opportunities
                    opportunities = await self.analyze_scaling_opportunities(state)

                    if opportunities:
                        logging.info(f"🎯 SELF-DETERMINED {len(opportunities)} SCALING OPPORTUNITIES")
                        await self.execute_scaling_opportunities(opportunities)
                    else:
                        # Only log every 10th check to avoid spam
                        if int(time.time()) % 10 == 0:
                            logging.info(f"📊 Monitoring: Portfolio {state['portfolio_pnl_pct']:+.2f}% - No scaling needed")

                    # Update state
                    self.last_portfolio_value = state['portfolio_value']
                    self.last_check_time = datetime.now()

                # Determine next check interval based on conditions
                if self.rapid_monitoring_mode and state and state['market_open']:
                    sleep_time = self.intelligence_settings['rapid_check_interval']
                    # Exit rapid mode if volatility subsides
                    if abs(state.get('portfolio_pnl_pct', 0)) < 1.0:
                        self.rapid_monitoring_mode = False
                        logging.info("📈 Exiting rapid monitoring mode - volatility subsided")
                else:
                    sleep_time = self.intelligence_settings['monitoring_interval']

                # Reset daily counter at market open
                now = datetime.now()
                if now.hour == 6 and now.minute == 30:
                    self.scaling_actions_today = 0
                    logging.info("🌅 Daily counter reset - new trading day")

                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                logging.info("🛑 Autonomous system stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in intelligence loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

async def main():
    """Start the truly autonomous self-determining system"""

    print("TRULY AUTONOMOUS SELF-DETERMINING SYSTEM")
    print("=" * 60)
    print("Real-time intelligence with event-driven scaling")
    print("No schedules - pure performance-based decisions")
    print("System will self-determine when to scale positions")
    print("=" * 60)

    system = TrulyAutonomousSystem()
    await system.continuous_intelligence_loop()

if __name__ == "__main__":
    asyncio.run(main())