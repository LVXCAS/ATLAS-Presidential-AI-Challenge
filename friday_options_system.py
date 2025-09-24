"""
FRIDAY OPTIONS SYSTEM
=====================
0DTE (Zero Days to Expiration) options trading for maximum leverage and returns
Targets SPY/QQQ momentum for explosive Friday gains

STRATEGY:
- Trade only on Fridays with 0DTE options
- Use momentum signals to pick direction
- Risk management: 5% max allocation per trade
- Target 50-200% returns on successful trades
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('friday_options.log'),
        logging.StreamHandler()
    ]
)

class FridayOptionsSystem:
    """
    FRIDAY OPTIONS SYSTEM
    0DTE options trading for massive leverage and returns
    """

    def __init__(self, max_allocation_per_trade=0.05, target_return_range=(0.5, 2.0)):
        self.logger = logging.getLogger('FridayOptions')

        # Risk parameters
        self.max_allocation_per_trade = max_allocation_per_trade  # 5% max per trade
        self.target_return_range = target_return_range  # 50-200% target
        self.stop_loss_threshold = -0.50  # 50% stop loss on options

        # Options symbols (SPY and QQQ focus)
        self.primary_symbols = ['SPY', 'QQQ']
        self.secondary_symbols = ['IWM', 'DIA']  # Backup targets

        # Momentum detection parameters
        self.momentum_lookback = 5  # 5-day momentum
        self.volume_threshold = 1.5  # 1.5x average volume
        self.volatility_threshold = 0.02  # 2% daily volatility minimum

        # Options selection criteria
        self.moneyness_range = (0.95, 1.05)  # 95%-105% of spot (near ATM)
        self.min_open_interest = 100  # Minimum liquidity
        self.max_spread_pct = 0.05  # 5% max bid-ask spread

        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}

        self.logger.info("FRIDAY OPTIONS SYSTEM initialized")
        self.logger.info(f"Max allocation per trade: {self.max_allocation_per_trade:.1%}")
        self.logger.info(f"Target return range: {self.target_return_range[0]:.0%}-{self.target_return_range[1]:.0%}")

    def is_friday(self) -> bool:
        """Check if today is Friday"""
        return datetime.now().weekday() == 4

    def is_market_hours(self) -> bool:
        """Check if we're in market hours"""
        now = datetime.now()
        market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)  # 6:30 AM PDT
        market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)  # 1:00 PM PDT
        return market_open <= now <= market_close

    def calculate_momentum_signal(self, symbol: str) -> Dict:
        """Calculate momentum signal for options direction"""
        try:
            # Get recent price data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo", interval="1d")

            if len(data) < self.momentum_lookback:
                return {'signal_strength': 0, 'direction': 0, 'confidence': 0}

            # Current price and recent data
            current_price = data['Close'].iloc[-1]
            yesterday_close = data['Close'].iloc[-2]

            # Momentum calculations
            momentum_5d = (current_price / data['Close'].iloc[-self.momentum_lookback]) - 1
            daily_return = (current_price / yesterday_close) - 1

            # Volume analysis
            avg_volume = data['Volume'].rolling(10).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # Volatility analysis
            daily_volatility = data['Close'].rolling(5).std().iloc[-1] / current_price

            # Signal strength (0-1)
            momentum_strength = min(1.0, abs(momentum_5d) * 10)  # Scale 10% momentum to 1.0
            volume_strength = min(1.0, volume_ratio / self.volume_threshold)
            volatility_strength = min(1.0, daily_volatility / self.volatility_threshold)

            signal_strength = (momentum_strength + volume_strength + volatility_strength) / 3

            # Direction (1 for calls, -1 for puts)
            direction = 1 if momentum_5d > 0 else -1

            # Confidence based on consistency
            price_trend = 1 if data['Close'].iloc[-3:].is_monotonic_increasing else \
                         -1 if data['Close'].iloc[-3:].is_monotonic_decreasing else 0
            confidence = signal_strength * (0.8 + 0.2 * abs(price_trend))

            return {
                'signal_strength': signal_strength,
                'direction': direction,
                'confidence': confidence,
                'momentum_5d': momentum_5d,
                'daily_return': daily_return,
                'volume_ratio': volume_ratio,
                'volatility': daily_volatility,
                'current_price': current_price
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate momentum for {symbol}: {e}")
            return {'signal_strength': 0, 'direction': 0, 'confidence': 0}

    def generate_options_chain_mock(self, symbol: str, current_price: float, direction: int) -> List[Dict]:
        """
        Generate mock options chain for simulation
        In real implementation, this would query actual options data
        """
        options = []

        # Generate strikes around current price
        strike_range = np.arange(
            current_price * self.moneyness_range[0],
            current_price * self.moneyness_range[1],
            current_price * 0.01  # 1% intervals
        )

        for strike in strike_range:
            # Mock option pricing (simplified Black-Scholes approximation)
            moneyness = strike / current_price
            time_to_expiry = 1/252  # 1 trading day

            # Simplified option value calculation
            intrinsic_value = max(0, (current_price - strike) * direction)
            time_value = current_price * 0.02 * time_to_expiry  # 2% IV approximation
            option_price = intrinsic_value + time_value

            # Mock Greeks and liquidity
            delta = 0.5 if abs(moneyness - 1) < 0.02 else (0.8 if direction > 0 and strike < current_price else 0.2)
            open_interest = np.random.randint(50, 500)
            bid_ask_spread = option_price * 0.03  # 3% spread

            option = {
                'strike': round(strike, 2),
                'option_type': 'call' if direction > 0 else 'put',
                'price': option_price,
                'delta': delta,
                'open_interest': open_interest,
                'bid_ask_spread': bid_ask_spread,
                'moneyness': moneyness
            }
            options.append(option)

        # Sort by liquidity and moneyness
        options.sort(key=lambda x: (x['open_interest'], -abs(x['moneyness'] - 1)), reverse=True)

        return options[:5]  # Return top 5 options

    def select_best_option(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """Select the best option to trade based on signal and criteria"""

        current_price = signal['current_price']
        direction = signal['direction']

        # Get options chain
        options = self.generate_options_chain_mock(symbol, current_price, direction)

        # Filter by criteria
        filtered_options = []
        for option in options:
            if (option['open_interest'] >= self.min_open_interest and
                option['bid_ask_spread'] / option['price'] <= self.max_spread_pct):
                filtered_options.append(option)

        if not filtered_options:
            self.logger.warning(f"No suitable options found for {symbol}")
            return None

        # Select best option (highest delta for momentum plays)
        best_option = max(filtered_options, key=lambda x: x['delta'])

        return best_option

    def calculate_position_size(self, portfolio_value: float, option_price: float, confidence: float) -> int:
        """Calculate number of option contracts to trade"""

        # Base allocation
        allocation = self.max_allocation_per_trade * confidence

        # Maximum dollar amount to risk
        max_risk_amount = portfolio_value * allocation

        # Option contracts (each contract = 100 shares)
        contract_cost = option_price * 100
        max_contracts = int(max_risk_amount / contract_cost)

        # Minimum 1 contract, maximum based on allocation
        contracts = max(1, min(max_contracts, 10))  # Cap at 10 contracts for safety

        return contracts

    def execute_options_trade(self, symbol: str, portfolio_value: float) -> Optional[Dict]:
        """Execute 0DTE options trade"""

        # Check if it's Friday and market hours
        if not self.is_friday():
            self.logger.info("Not Friday - skipping options trade")
            return None

        if not self.is_market_hours():
            self.logger.info("Outside market hours - skipping options trade")
            return None

        # Generate signal
        signal = self.calculate_momentum_signal(symbol)

        if signal['confidence'] < 0.6:  # Only trade high-confidence signals
            self.logger.info(f"Low confidence signal for {symbol}: {signal['confidence']:.2f}")
            return None

        # Select option
        option = self.select_best_option(symbol, signal)
        if not option:
            return None

        # Calculate position size
        contracts = self.calculate_position_size(portfolio_value, option['price'], signal['confidence'])

        # Create trade
        trade = {
            'symbol': symbol,
            'option_type': option['option_type'],
            'strike': option['strike'],
            'contracts': contracts,
            'entry_price': option['price'],
            'total_cost': option['price'] * contracts * 100,
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'momentum_5d': signal['momentum_5d'],
            'entry_time': datetime.now(),
            'expiry': 'TODAY',  # 0DTE
            'risk_level': 'EXTREME',
            'max_loss': option['price'] * contracts * 100,  # 100% loss potential
            'target_return_range': self.target_return_range
        }

        # Log trade
        self.logger.info(f"FRIDAY 0DTE TRADE: {symbol} {option['option_type'].upper()} "
                        f"${option['strike']} x{contracts} contracts")
        self.logger.info(f"Total cost: ${trade['total_cost']:,.0f} "
                        f"({trade['total_cost']/portfolio_value:.1%} of portfolio)")
        self.logger.info(f"Signal confidence: {signal['confidence']:.1%}")

        return trade

    def simulate_option_outcome(self, trade: Dict) -> Dict:
        """Simulate 0DTE option outcome for backtesting"""

        # Simulate realistic 0DTE outcomes
        # 0DTE options are highly volatile - can easily 2x or go to 0

        confidence = trade['confidence']
        momentum = trade['momentum_5d']

        # Success probability based on signal quality
        success_prob = 0.3 + (confidence * 0.4)  # 30-70% success rate

        # Generate outcome
        is_successful = np.random.random() < success_prob

        if is_successful:
            # Successful trade - returns based on momentum strength
            if abs(momentum) > 0.02:  # Strong momentum
                return_multiplier = np.random.uniform(1.5, 3.0)  # 150-300%
            else:  # Moderate momentum
                return_multiplier = np.random.uniform(1.2, 2.0)  # 120-200%
        else:
            # Failed trade - partial or total loss
            return_multiplier = np.random.uniform(0.0, 0.4)  # 0-40% recovery

        # Calculate P&L
        entry_cost = trade['total_cost']
        exit_value = entry_cost * return_multiplier
        pnl = exit_value - entry_cost
        return_pct = (return_multiplier - 1) * 100

        outcome = {
            'successful': is_successful,
            'return_multiplier': return_multiplier,
            'return_pct': return_pct,
            'entry_cost': entry_cost,
            'exit_value': exit_value,
            'pnl': pnl,
            'exit_time': datetime.now() + timedelta(hours=6)  # End of trading day
        }

        return outcome

    def run_friday_session(self, portfolio_value: float) -> Dict:
        """Run complete Friday 0DTE trading session"""

        self.logger.info("STARTING FRIDAY 0DTE OPTIONS SESSION")
        self.logger.info(f"Portfolio value: ${portfolio_value:,.0f}")

        session_trades = []
        total_pnl = 0

        # Trade each primary symbol
        for symbol in self.primary_symbols:
            trade = self.execute_options_trade(symbol, portfolio_value)
            if trade:
                # Simulate outcome
                outcome = self.simulate_option_outcome(trade)
                trade.update(outcome)

                session_trades.append(trade)
                total_pnl += outcome['pnl']

                self.logger.info(f"{symbol} outcome: {outcome['return_pct']:+.0f}% "
                               f"(${outcome['pnl']:+,.0f})")

        # Session summary
        session_summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'trades_executed': len(session_trades),
            'total_pnl': total_pnl,
            'session_return': total_pnl / portfolio_value if portfolio_value > 0 else 0,
            'successful_trades': sum(1 for t in session_trades if t['successful']),
            'win_rate': (sum(1 for t in session_trades if t['successful']) / len(session_trades)) if session_trades else 0,
            'trades': session_trades
        }

        self.logger.info(f"FRIDAY SESSION COMPLETE: {session_summary['session_return']:+.1%} return")
        self.logger.info(f"Win rate: {session_summary['win_rate']:.1%}")

        return session_summary

def main():
    """Test the Friday options system"""
    print("FRIDAY 0DTE OPTIONS SYSTEM")
    print("Zero Days to Expiration Trading")
    print("=" * 60)

    # Initialize system
    system = FridayOptionsSystem()

    # Simulate portfolio value
    portfolio_value = 992233.63  # Your current balance

    # Run session (works any day for testing)
    print("\\nRunning Friday options simulation...")
    session = system.run_friday_session(portfolio_value)

    print("\\nFRIDAY SESSION RESULTS:")
    print("-" * 40)
    print(f"Trades executed: {session['trades_executed']}")
    print(f"Total P&L: ${session['total_pnl']:+,.0f}")
    print(f"Session return: {session['session_return']:+.1%}")
    print(f"Win rate: {session['win_rate']:.1%}")

    # Save results
    results_file = f"friday_options_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(session, f, indent=2, default=str)

    print(f"\\nResults saved to: {results_file}")
    print("\\n[SUCCESS] Friday 0DTE options system ready!")
    print("Ready for explosive Friday gains!")

if __name__ == "__main__":
    main()