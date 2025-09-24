"""
AUTONOMOUS MARKET OPEN SYSTEM
Fully automated trading system that executes at 6:30 AM PT market open
No manual intervention required - system does everything autonomously
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime, timedelta
import yfinance as yf
import alpaca_trade_api as tradeapi
import numpy as np
import json
import logging
import os
from dotenv import load_dotenv
import sys

# Add agents directory to path
sys.path.append('./agents')
from autonomous_brain import AutonomousTradingBrain
from momentum_trading_agent import MomentumTradingAgent
from options_volatility_agent import OptionsVolatilityAgent
from risk_manager_agent import RiskManagerAgent
from portfolio_allocator_agent import PortfolioAllocatorAgent

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_trading.log'),
        logging.StreamHandler()
    ]
)

class AutonomousMarketOpenSystem:
    """Fully autonomous trading system for market open execution"""

    def __init__(self):
        # Initialize Alpaca API
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # System parameters
        self.portfolio_value = 0
        self.buying_power = 0
        self.positions = {}

        # Initialize AI Agents
        try:
            self.autonomous_brain = AutonomousTradingBrain()
            self.momentum_agent = MomentumTradingAgent()
            self.options_agent = OptionsVolatilityAgent()
            self.risk_agent = RiskManagerAgent()
            self.portfolio_agent = PortfolioAllocatorAgent()
            self.agents_loaded = True
            logging.info("AI Agents loaded successfully")
        except Exception as e:
            logging.warning(f"AI Agents failed to load: {e}")
            logging.warning("Continuing with basic autonomous system")
            self.agents_loaded = False

        # Autonomous settings
        self.autonomous_settings = {
            'max_daily_risk': 0.02,  # 2% max daily loss
            'position_size_limit': 0.20,  # 20% max per position
            'regime_confidence_threshold': 0.6,  # 60% minimum confidence
            'momentum_threshold': 0.005,  # 0.5% momentum threshold
            'volatility_limit': 0.25,  # 25% max volatility
            'market_open_time': '06:30',  # 6:30 AM PT
            'stop_loss_percent': 0.15,  # 15% stop loss
            'take_profit_percent': 0.05  # 5% take profit
        }

        # Trading state
        self.is_trading_active = False
        self.daily_pnl = 0
        self.trade_count = 0

        logging.info("Autonomous Market Open System Initialized")
        logging.info(f"Market Open Execution: {self.autonomous_settings['market_open_time']} PT")

    async def get_current_portfolio_status(self):
        """Get current portfolio status"""

        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.list_positions()

            self.portfolio_value = float(account.portfolio_value)
            self.buying_power = float(account.buying_power)
            self.daily_pnl = float(account.unrealized_pl)

            logging.info(f"Portfolio Value: ${self.portfolio_value:,.0f}")
            logging.info(f"Buying Power: ${self.buying_power:,.0f}")
            logging.info(f"Unrealized P&L: ${self.daily_pnl:+,.0f}")

            # Update positions
            self.positions = {}
            for pos in positions:
                self.positions[pos.symbol] = {
                    'qty': int(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'avg_entry_price': float(pos.avg_entry_price)
                }

            return True

        except Exception as e:
            logging.error(f"Error getting portfolio status: {e}")
            return False

    async def analyze_market_regime(self):
        """Enhanced market regime analysis using AI agents"""

        logging.info("Starting AI-enhanced market regime analysis...")

        try:
            # Get basic market data
            spy_data = yf.download('SPY', period='3mo', interval='1d', progress=False)
            spy_close = spy_data['Close']

            # Calculate indicators
            current_price = spy_close.iloc[-1]
            sma_20 = spy_close.rolling(20).mean().iloc[-1]
            sma_50 = spy_close.rolling(50).mean().iloc[-1]

            # Momentum analysis
            momentum_1d = (current_price / spy_close.iloc[-2] - 1)
            momentum_5d = (current_price / spy_close.iloc[-6] - 1)

            # Volatility
            returns = spy_close.pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            current_vol = volatility.iloc[-1]

            # Use AI agents if available
            if self.agents_loaded:
                try:
                    # Get momentum agent analysis
                    momentum_signal = await self.momentum_agent.analyze_momentum('SPY')

                    # Get options volatility analysis
                    vol_signal = await self.options_agent.analyze_volatility('SPY')

                    # Combine signals with autonomous brain
                    market_data = {
                        'price': current_price,
                        'momentum_1d': momentum_1d,
                        'momentum_5d': momentum_5d,
                        'volatility': current_vol,
                        'sma_20': sma_20,
                        'sma_50': sma_50
                    }

                    # Get AI recommendation
                    ai_recommendation = await self.autonomous_brain.analyze_market_state(market_data)

                    if ai_recommendation:
                        logging.info("Using AI-enhanced regime analysis")
                        return ai_recommendation

                except Exception as e:
                    logging.warning(f"AI analysis failed, using fallback: {e}")

            # Fallback to basic analysis
            is_bull_trend = current_price > sma_20 > sma_50
            is_strong_momentum = momentum_5d > 0.01
            is_low_volatility = current_vol < 0.20

            if is_bull_trend and is_strong_momentum and is_low_volatility:
                regime = "BULLISH_HIGH_CONFIDENCE"
                confidence = 0.85
                expected_move = 0.012
                action = "AGGRESSIVE_LONG"
            elif is_bull_trend and is_strong_momentum:
                regime = "BULLISH_MEDIUM_CONFIDENCE"
                confidence = 0.70
                expected_move = 0.008
                action = "MODERATE_LONG"
            elif is_bull_trend:
                regime = "BULLISH_LOW_CONFIDENCE"
                confidence = 0.60
                expected_move = 0.004
                action = "CONSERVATIVE_LONG"
            elif is_strong_momentum:
                regime = "NEUTRAL_MOMENTUM"
                confidence = 0.55
                expected_move = 0.002
                action = "HOLD"
            else:
                regime = "BEARISH_DEFENSIVE"
                confidence = 0.40
                expected_move = -0.005
                action = "DEFENSIVE"

            regime_data = {
                'regime': regime,
                'confidence': confidence,
                'expected_move': expected_move,
                'action': action,
                'momentum_1d': momentum_1d,
                'momentum_5d': momentum_5d,
                'volatility': current_vol,
                'bull_trend': is_bull_trend
            }

            logging.info(f"Market Regime: {regime}")
            logging.info(f"Confidence: {confidence*100:.0f}%")
            logging.info(f"Recommended Action: {action}")
            logging.info(f"Expected Move: {expected_move*100:+.1f}%")

            return regime_data

        except Exception as e:
            logging.error(f"Error in regime analysis: {e}")
            return {
                'regime': 'ERROR',
                'confidence': 0.0,
                'expected_move': 0.0,
                'action': 'HOLD',
                'momentum_1d': 0.0,
                'momentum_5d': 0.0,
                'volatility': 0.15,
                'bull_trend': False
            }

    async def calculate_autonomous_position_sizing(self, regime_data):
        """Calculate position sizes autonomously based on regime"""

        logging.info("Calculating autonomous position sizing...")

        action = regime_data['action']
        confidence = regime_data['confidence']
        expected_move = regime_data['expected_move']

        # Base position sizing based on confidence
        if confidence >= 0.80:
            base_allocation = 0.90  # 90% of portfolio in high confidence
        elif confidence >= 0.70:
            base_allocation = 0.75  # 75% of portfolio in medium confidence
        elif confidence >= 0.60:
            base_allocation = 0.60  # 60% of portfolio in low confidence
        else:
            base_allocation = 0.30  # 30% of portfolio in uncertain times

        # Position adjustments
        position_targets = {}

        # Define target positions based on action
        if action in ["AGGRESSIVE_LONG", "MODERATE_LONG"]:
            # Focus on leveraged growth positions
            position_targets = {
                'TQQQ': base_allocation * 0.30,  # 30% in tech leverage
                'SOXL': base_allocation * 0.20,  # 20% in semiconductor leverage
                'IWM': base_allocation * 0.25,   # 25% in small cap leverage
                'QQQ': base_allocation * 0.15,   # 15% in tech
                'SPY': base_allocation * 0.10    # 10% in market
            }
        elif action == "CONSERVATIVE_LONG":
            # Focus on less leveraged positions
            position_targets = {
                'SPY': base_allocation * 0.40,   # 40% in market
                'QQQ': base_allocation * 0.30,   # 30% in tech
                'IWM': base_allocation * 0.20,   # 20% in small cap leverage
                'TQQQ': base_allocation * 0.10   # 10% in tech leverage
            }
        elif action == "DEFENSIVE":
            # Reduce all positions
            position_targets = {
                'SPY': base_allocation * 0.50,   # 50% in defensive market
                'QQQ': base_allocation * 0.30,   # 30% in stable tech
                'IWM': base_allocation * 0.20    # 20% in small cap
            }
        else:  # HOLD
            # Maintain current allocation
            for symbol, pos_data in self.positions.items():
                current_weight = pos_data['market_value'] / self.portfolio_value
                position_targets[symbol] = current_weight

        logging.info(f"Base Allocation: {base_allocation*100:.0f}% of portfolio")
        logging.info(f"Position Targets: {position_targets}")

        return position_targets

    async def execute_autonomous_trades(self, position_targets, regime_data):
        """Execute trades autonomously at market open"""

        logging.info("Starting autonomous trade execution...")

        if not self.is_market_open():
            logging.warning("Market is not open - skipping trade execution")
            return []

        executed_trades = []

        try:
            for symbol, target_weight in position_targets.items():
                target_value = self.portfolio_value * target_weight
                current_value = self.positions.get(symbol, {}).get('market_value', 0)

                # Calculate position change needed
                value_difference = target_value - current_value

                # Minimum trade size filter
                if abs(value_difference) < 1000:  # Skip trades under $1000
                    continue

                # Get current price
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period='1d', interval='1m').iloc[-1]['Close']

                # Calculate shares to trade
                shares_difference = int(value_difference / current_price)

                if shares_difference == 0:
                    continue

                # Execute trade
                side = 'buy' if shares_difference > 0 else 'sell'
                qty = abs(shares_difference)

                logging.info(f"Executing {side.upper()} order: {qty} shares of {symbol} at ~${current_price:.2f}")

                try:
                    order = self.alpaca.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )

                    executed_trades.append({
                        'symbol': symbol,
                        'side': side,
                        'qty': qty,
                        'price': current_price,
                        'value': qty * current_price,
                        'order_id': order.id,
                        'timestamp': datetime.now().isoformat()
                    })

                    logging.info(f"Order submitted: {order.id}")
                    self.trade_count += 1

                except Exception as e:
                    logging.error(f"Failed to execute {side} order for {symbol}: {e}")

        except Exception as e:
            logging.error(f"Error in trade execution: {e}")

        logging.info(f"Executed {len(executed_trades)} trades autonomously")
        return executed_trades

    async def set_autonomous_risk_management(self):
        """Set up autonomous risk management for all positions"""

        logging.info("Setting up autonomous risk management...")

        try:
            # Set stop losses for all positions
            for symbol, pos_data in self.positions.items():
                qty = pos_data['qty']
                avg_price = pos_data['avg_entry_price']

                if qty > 0:  # Long position
                    stop_price = avg_price * (1 - self.autonomous_settings['stop_loss_percent'])

                    # Submit stop loss order
                    stop_order = self.alpaca.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='stop',
                        stop_price=stop_price,
                        time_in_force='gtc'  # Good till canceled
                    )

                    logging.info(f"Stop loss set for {symbol}: {qty} shares at ${stop_price:.2f}")

        except Exception as e:
            logging.error(f"Error setting risk management: {e}")

    def is_market_open(self):
        """Check if market is currently open"""

        try:
            clock = self.alpaca.get_clock()
            return clock.is_open
        except:
            # Fallback: check if it's a weekday during market hours
            now = datetime.now()
            is_weekday = now.weekday() < 5
            market_start = now.replace(hour=6, minute=30, second=0)  # 6:30 AM PT
            market_end = now.replace(hour=13, minute=0, second=0)    # 1:00 PM PT
            is_market_hours = market_start <= now <= market_end

            return is_weekday and is_market_hours

    async def autonomous_market_open_routine(self):
        """Main autonomous routine that runs at market open"""

        logging.info("="*60)
        logging.info("AUTONOMOUS MARKET OPEN ROUTINE STARTING")
        logging.info("="*60)

        try:
            # Step 1: Get portfolio status
            if not await self.get_current_portfolio_status():
                logging.error("Failed to get portfolio status - aborting")
                return

            # Step 2: Check daily risk limits
            daily_loss_limit = self.portfolio_value * self.autonomous_settings['max_daily_risk']
            if self.daily_pnl < -daily_loss_limit:
                logging.warning(f"Daily loss limit hit: ${self.daily_pnl:+,.0f} < ${-daily_loss_limit:,.0f}")
                logging.warning("Stopping autonomous trading for today")
                return

            # Step 3: Analyze market regime
            regime_data = await self.analyze_market_regime()

            # Step 4: Check confidence threshold
            if regime_data['confidence'] < self.autonomous_settings['regime_confidence_threshold']:
                logging.warning(f"Regime confidence too low: {regime_data['confidence']*100:.0f}% < {self.autonomous_settings['regime_confidence_threshold']*100:.0f}%")
                logging.info("Maintaining current positions")
                return

            # Step 5: Calculate position targets
            position_targets = await self.calculate_autonomous_position_sizing(regime_data)

            # Step 6: Execute trades
            executed_trades = await self.execute_autonomous_trades(position_targets, regime_data)

            # Step 7: Set risk management
            await self.set_autonomous_risk_management()

            # Step 8: Save trading log
            trading_session = {
                'timestamp': datetime.now().isoformat(),
                'regime_data': regime_data,
                'position_targets': position_targets,
                'executed_trades': executed_trades,
                'portfolio_value': self.portfolio_value,
                'daily_pnl': self.daily_pnl,
                'trade_count': self.trade_count
            }

            with open(f"autonomous_trading_session_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
                json.dump(trading_session, f, indent=2, default=str)

            logging.info("Autonomous market open routine completed successfully")
            logging.info(f"Trades executed: {len(executed_trades)}")
            logging.info(f"Portfolio value: ${self.portfolio_value:,.0f}")

        except Exception as e:
            logging.error(f"Error in autonomous routine: {e}")

    def schedule_autonomous_trading(self):
        """Schedule autonomous trading for market open"""

        logging.info("Scheduling autonomous trading system...")

        # Schedule for Monday through Friday at 6:30 AM PT
        schedule.every().monday.at("06:30").do(lambda: asyncio.run(self.autonomous_market_open_routine()))
        schedule.every().tuesday.at("06:30").do(lambda: asyncio.run(self.autonomous_market_open_routine()))
        schedule.every().wednesday.at("06:30").do(lambda: asyncio.run(self.autonomous_market_open_routine()))
        schedule.every().thursday.at("06:30").do(lambda: asyncio.run(self.autonomous_market_open_routine()))
        schedule.every().friday.at("06:30").do(lambda: asyncio.run(self.autonomous_market_open_routine()))

        # Schedule daily monitoring at 10 AM PT
        schedule.every().day.at("10:00").do(lambda: asyncio.run(self.daily_monitoring_routine()))

        # Schedule autonomous scaling every 2 hours during market hours
        schedule.every().day.at("08:30").do(lambda: asyncio.run(self.autonomous_scaling_routine()))
        schedule.every().day.at("10:30").do(lambda: asyncio.run(self.autonomous_scaling_routine()))
        schedule.every().day.at("12:30").do(lambda: asyncio.run(self.autonomous_scaling_routine()))

        # Schedule end-of-day review at 2 PM PT
        schedule.every().day.at("14:00").do(lambda: asyncio.run(self.end_of_day_routine()))

        logging.info("Autonomous trading scheduled:")
        logging.info("  Market Open Execution: Mon-Fri 6:30 AM PT")
        logging.info("  Autonomous Scaling: 8:30 AM, 10:30 AM, 12:30 PM PT")
        logging.info("  Daily Monitoring: Daily 10:00 AM PT")
        logging.info("  End of Day Review: Daily 2:00 PM PT")

    async def autonomous_scaling_routine(self):
        """Autonomously scale up/down based on real-time performance"""

        logging.info("🤖 AUTONOMOUS SCALING ROUTINE ACTIVATED")

        try:
            await self.get_current_portfolio_status()

            # Don't trade if market is closed
            if not self.is_market_open():
                logging.info("Market closed - skipping autonomous scaling")
                return

            # Get positions and analyze performance
            positions = self.alpaca.list_positions()
            winners = []
            losers = []

            for pos in positions:
                unrealized_pl = float(pos.unrealized_pl)
                market_value = float(pos.market_value)
                pnl_pct = (unrealized_pl / market_value) * 100 if market_value > 0 else 0

                pos_data = {
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'value': market_value,
                    'pnl': unrealized_pl,
                    'pnl_pct': pnl_pct
                }

                if pnl_pct > 3.0:  # Strong winners (>3%)
                    winners.append(pos_data)
                elif pnl_pct < -2.0:  # Clear losers (<-2%)
                    losers.append(pos_data)

            logging.info(f"Performance Analysis: {len(winners)} winners, {len(losers)} losers")

            # AUTONOMOUS SCALING LOGIC
            trades_executed = []
            account = self.alpaca.get_account()
            buying_power = float(account.buying_power)

            # Scale up strong winners automatically
            if winners and buying_power > 5000:
                for winner in winners[:2]:  # Top 2 winners only
                    symbol = winner['symbol']
                    current_value = winner['value']

                    # Scale up by 10% of position, max $10K
                    scale_amount = min(current_value * 0.10, buying_power * 0.3, 10000)

                    if scale_amount > 1000:
                        try:
                            # Get latest price
                            ticker_data = self.alpaca.get_latest_trade(symbol)
                            current_price = float(ticker_data.price)
                            shares_to_buy = int(scale_amount / current_price)

                            if shares_to_buy > 0:
                                order = self.alpaca.submit_order(
                                    symbol=symbol,
                                    qty=shares_to_buy,
                                    side='buy',
                                    type='market',
                                    time_in_force='day'
                                )

                                trades_executed.append(f"SCALED UP {symbol}: +{shares_to_buy} shares (+{winner['pnl_pct']:.1f}%)")
                                logging.info(f"🚀 AUTONOMOUS SCALE UP: {symbol} +{shares_to_buy} shares (${scale_amount:,.0f})")

                        except Exception as e:
                            logging.error(f"Scale up failed for {symbol}: {e}")

            # Scale down losers automatically (risk management)
            for loser in losers:
                symbol = loser['symbol']
                current_qty = loser['qty']

                if loser['pnl_pct'] < -3.0:  # Heavy losers get reduced
                    shares_to_sell = int(current_qty * 0.25)  # Reduce by 25%

                    if shares_to_sell > 0:
                        try:
                            order = self.alpaca.submit_order(
                                symbol=symbol,
                                qty=shares_to_sell,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )

                            trades_executed.append(f"SCALED DOWN {symbol}: -{shares_to_sell} shares ({loser['pnl_pct']:.1f}%)")
                            logging.info(f"⚠️ AUTONOMOUS SCALE DOWN: {symbol} -{shares_to_sell} shares (risk management)")

                        except Exception as e:
                            logging.error(f"Scale down failed for {symbol}: {e}")

            # Report autonomous actions
            if trades_executed:
                logging.info(f"✅ AUTONOMOUS SCALING COMPLETED: {len(trades_executed)} trades")
                for trade in trades_executed:
                    logging.info(f"  {trade}")
            else:
                logging.info("📊 No autonomous scaling needed - portfolio performing within targets")

        except Exception as e:
            logging.error(f"Error in autonomous scaling: {e}")

    async def daily_monitoring_routine(self):
        """Monitor positions and adjust if needed"""

        logging.info("Running daily monitoring routine...")

        try:
            await self.get_current_portfolio_status()

            # Check if daily loss limit is hit
            daily_loss_limit = self.portfolio_value * self.autonomous_settings['max_daily_risk']
            if self.daily_pnl < -daily_loss_limit:
                logging.critical(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:+,.0f}")
                logging.critical("Liquidating all positions")

                # Emergency liquidation
                positions = self.alpaca.list_positions()
                for pos in positions:
                    self.alpaca.submit_order(
                        symbol=pos.symbol,
                        qty=abs(int(pos.qty)),
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )

        except Exception as e:
            logging.error(f"Error in daily monitoring: {e}")

    async def end_of_day_routine(self):
        """End of day analysis and logging"""

        logging.info("Running end of day routine...")

        try:
            await self.get_current_portfolio_status()

            logging.info("="*60)
            logging.info("END OF DAY SUMMARY")
            logging.info("="*60)
            logging.info(f"Portfolio Value: ${self.portfolio_value:,.0f}")
            logging.info(f"Daily P&L: ${self.daily_pnl:+,.0f}")
            logging.info(f"Trades Today: {self.trade_count}")
            logging.info("="*60)

        except Exception as e:
            logging.error(f"Error in end of day routine: {e}")

    def run_autonomous_system(self):
        """Run the autonomous trading system continuously"""

        logging.info("STARTING AUTONOMOUS TRADING SYSTEM")
        logging.info("="*60)
        logging.info("System will trade autonomously at market open")
        logging.info("No manual intervention required")
        logging.info("Check autonomous_trading.log for updates")
        logging.info("="*60)

        # Schedule all routines
        self.schedule_autonomous_trading()

        # Run continuously
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logging.info("Autonomous system stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in autonomous system: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

def main():
    """Start autonomous trading system"""

    system = AutonomousMarketOpenSystem()

    # Run test routine immediately if market is open
    if system.is_market_open():
        logging.info("Market is open - running test routine")
        asyncio.run(system.autonomous_market_open_routine())

    # Start autonomous system
    system.run_autonomous_system()

if __name__ == "__main__":
    main()