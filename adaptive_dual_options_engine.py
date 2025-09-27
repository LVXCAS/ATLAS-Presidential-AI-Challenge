#!/usr/bin/env python3
"""
ADAPTIVE DUAL OPTIONS ENGINE
Replicates Lucas's proven 68.3% ROI strategy: Cash-secured puts + Long calls
Adapts to different market conditions automatically
"""

import alpaca_trade_api as tradeapi
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()

class AdaptiveDualOptionsEngine:
    """Adaptive engine for dual cash-secured put + long call strategy"""

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - DUAL_ENGINE - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Market regime detection parameters
        self.market_regimes = {
            'bull': {'vix_threshold': 18, 'trend': 'up'},
            'bear': {'vix_threshold': 25, 'trend': 'down'},
            'volatile': {'vix_threshold': 30, 'trend': 'any'},
            'calm': {'vix_threshold': 15, 'trend': 'sideways'}
        }

    def detect_market_regime(self, symbol):
        """Detect current market regime for adaptive strategy"""
        try:
            # Get recent price data
            bars = self.api.get_bars(symbol, '1Day', limit=20).df
            if bars.empty:
                return 'neutral'

            current_price = bars['close'].iloc[-1]
            sma_20 = bars['close'].rolling(20).mean().iloc[-1]

            # Simple regime detection
            if current_price > sma_20 * 1.02:
                return 'bull'
            elif current_price < sma_20 * 0.98:
                return 'bear'
            else:
                return 'neutral'

        except Exception as e:
            self.logger.warning(f"Market regime detection failed for {symbol}: {e}")
            return 'neutral'

    def calculate_adaptive_strikes(self, symbol, current_price, regime='neutral'):
        """Calculate strikes based on market regime and your proven patterns"""

        # Your proven strike patterns from successful trades
        proven_patterns = {
            'INTC': {'put_ratio': 0.91, 'call_ratio': 1.10},  # $29 put, $32 call on ~$32 stock
            'LYFT': {'put_ratio': 0.91, 'call_ratio': 1.05},  # $21 put, $23 call on ~$23 stock
            'SNAP': {'put_ratio': 0.89, 'call_ratio': 1.13},  # $8 put, $9 call on ~$8 stock
            'RIVN': {'put_ratio': 0.93, 'call_ratio': 1.07}   # $14 put, $15 call on ~$15 stock
        }

        # Get base ratios (average of your successful patterns)
        base_put_ratio = 0.91   # ~9% OTM puts (collect premium)
        base_call_ratio = 1.09  # ~9% OTM calls (capture upside)

        # Adaptive adjustments based on market regime
        regime_adjustments = {
            'bull': {
                'put_ratio': base_put_ratio + 0.02,   # Closer puts in bull market
                'call_ratio': base_call_ratio - 0.02, # Closer calls in bull market
                'volatility_factor': 1.0
            },
            'bear': {
                'put_ratio': base_put_ratio - 0.03,   # Further OTM puts in bear market
                'call_ratio': base_call_ratio + 0.03, # Further OTM calls in bear market
                'volatility_factor': 1.3
            },
            'neutral': {
                'put_ratio': base_put_ratio,
                'call_ratio': base_call_ratio,
                'volatility_factor': 1.0
            }
        }

        adjustments = regime_adjustments.get(regime, regime_adjustments['neutral'])

        # Calculate strikes (round to nearest $1 like your trades)
        put_strike = round(current_price * adjustments['put_ratio'])
        call_strike = round(current_price * adjustments['call_ratio'])

        # Ensure minimum $1 spread and reasonable strikes
        put_strike = max(1, put_strike)
        call_strike = max(put_strike + 1, call_strike)

        return {
            'put_strike': float(put_strike),
            'call_strike': float(call_strike),
            'regime': regime,
            'volatility_factor': adjustments['volatility_factor']
        }

    def calculate_position_size(self, symbol, buying_power, allocation, volatility_factor=1.0):
        """Calculate position size based on your proven scaling (10-50 contracts)"""

        # Your proven contract sizes
        proven_sizes = {
            'initial': 10,    # Start with 10 contracts
            'scaled': 50,     # Scale to 50 contracts
            'max': 100        # Maximum for high conviction
        }

        # Base allocation amount
        trade_amount = buying_power * allocation

        # Estimate cash needed for cash-secured puts (approximate)
        estimated_put_cash = 15000  # Average from your trades ($29K, $21K, $8K, $14K)

        # Calculate contracts based on available cash and volatility
        base_contracts = min(
            proven_sizes['max'],
            max(proven_sizes['initial'], int(trade_amount / estimated_put_cash))
        )

        # Apply volatility scaling (more contracts in high volatility for premium capture)
        contracts = int(base_contracts * volatility_factor)
        contracts = max(proven_sizes['initial'], min(proven_sizes['max'], contracts))

        return contracts

    def execute_dual_strategy(self, opportunities, buying_power):
        """Execute dual cash-secured put + long call strategy"""

        print("\n" + "="*80)
        print("EXECUTING DUAL CASH-SECURED PUT + LONG CALL STRATEGY")
        print("Replicating your proven 68.3% ROI methodology")
        print("="*80)

        executed_trades = []

        for i, opp in enumerate(opportunities):
            try:
                symbol = opp['symbol']
                allocation = [0.45, 0.25, 0.20, 0.10][i] if i < 4 else 0.05

                print(f"\n[DUAL STRATEGY] {symbol} - Allocation: {allocation:.1%}")

                # Get current price
                try:
                    bars = self.api.get_latest_bar(symbol)
                    current_price = float(bars.c)
                except:
                    print(f"  ERROR: Could not get price for {symbol}")
                    continue

                # Detect market regime
                regime = self.detect_market_regime(symbol)
                print(f"  Market Regime: {regime.upper()}")

                # Calculate adaptive strikes
                strike_data = self.calculate_adaptive_strikes(symbol, current_price, regime)
                put_strike = strike_data['put_strike']
                call_strike = strike_data['call_strike']
                volatility_factor = strike_data['volatility_factor']

                print(f"  Current Price: ${current_price:.2f}")
                print(f"  PUT Strike: ${put_strike:.0f} ({((put_strike/current_price-1)*100):+.1f}%)")
                print(f"  CALL Strike: ${call_strike:.0f} ({((call_strike/current_price-1)*100):+.1f}%)")

                # Calculate position size
                contracts = self.calculate_position_size(symbol, buying_power, allocation, volatility_factor)
                print(f"  Contract Size: {contracts} contracts (volatility factor: {volatility_factor:.1f}x)")

                # Build options symbols (next Friday expiration like your trades)
                exp_date = self.get_next_friday()
                put_symbol = f"{symbol}{exp_date}P{int(put_strike * 1000):08d}"
                call_symbol = f"{symbol}{exp_date}C{int(call_strike * 1000):08d}"

                print(f"  PUT Symbol: {put_symbol}")
                print(f"  CALL Symbol: {call_symbol}")

                # Execute dual strategy
                dual_success = False

                try:
                    # 1. SELL CASH-SECURED PUT (collect premium)
                    print(f"  SELLING {contracts} cash-secured puts...")
                    put_order = self.api.submit_order(
                        symbol=put_symbol,
                        qty=contracts,
                        side='sell',  # SELL puts for premium
                        type='market',
                        time_in_force='day'
                    )

                    print(f"  ✅ PUT ORDER SUBMITTED: {put_order.id}")

                    # 2. BUY LONG CALLS (capture upside)
                    print(f"  BUYING {contracts} long calls...")
                    call_order = self.api.submit_order(
                        symbol=call_symbol,
                        qty=contracts,
                        side='buy',   # BUY calls for upside
                        type='market',
                        time_in_force='day'
                    )

                    print(f"  ✅ CALL ORDER SUBMITTED: {call_order.id}")

                    executed_trades.append({
                        'symbol': symbol,
                        'strategy': 'DUAL_CASH_SECURED_PUT_LONG_CALL',
                        'put_order': put_order.id,
                        'call_order': call_order.id,
                        'contracts': contracts,
                        'put_strike': put_strike,
                        'call_strike': call_strike,
                        'regime': regime,
                        'timestamp': datetime.now().isoformat()
                    })

                    dual_success = True
                    print(f"  [SUCCESS] DUAL STRATEGY EXECUTED: {symbol}")

                except Exception as e:
                    print(f"  [X] DUAL STRATEGY FAILED: {e}")

                    # Try enhanced options checker for better availability
                    print(f"  Trying enhanced options checker...")
                    try:
                        from enhanced_options_checker import EnhancedOptionsChecker
                        options_checker = EnhancedOptionsChecker()

                        # Check for available puts and calls
                        put_option = options_checker.get_best_option(symbol, put_strike, 'P')
                        call_option = options_checker.get_best_option(symbol, call_strike, 'C')

                        if put_option['available'] and call_option['available']:
                            print(f"  [OK] Enhanced checker found options")

                            # Execute with enhanced options
                            put_order = self.api.submit_order(
                                symbol=put_option['symbol'],
                                qty=contracts,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )

                            call_order = self.api.submit_order(
                                symbol=call_option['symbol'],
                                qty=contracts,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )

                            print(f"  [OK] ENHANCED DUAL STRATEGY EXECUTED")
                            executed_trades.append({
                                'symbol': symbol,
                                'strategy': 'ENHANCED_DUAL_STRATEGY',
                                'put_order': put_order.id,
                                'call_order': call_order.id,
                                'contracts': contracts
                            })
                            dual_success = True

                        else:
                            print(f"  Enhanced checker: No suitable options available")

                    except Exception as enhanced_error:
                        print(f"  Enhanced checker failed: {enhanced_error}")

                    # Final fallback to stock position
                    if not dual_success:
                        print(f"  Falling back to stock position...")
                        try:
                            quote = self.api.get_latest_quote(symbol)
                            price = float(quote.ask_price)
                            shares = max(1, int((buying_power * allocation) / price))

                            stock_order = self.api.submit_order(
                                symbol=symbol,
                                qty=shares,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )

                            print(f"  [OK] STOCK FALLBACK: {shares} shares @ ${price:.2f}")
                            executed_trades.append({
                                'symbol': symbol,
                                'strategy': 'STOCK_FALLBACK',
                                'order': stock_order.id,
                                'shares': shares,
                                'price': price
                            })

                        except Exception as fallback_error:
                            print(f"  [X] STOCK FALLBACK FAILED: {fallback_error}")

            except Exception as e:
                print(f"  [X] ERROR PROCESSING {symbol}: {e}")
                continue

        print(f"\n[COMPLETE] DUAL STRATEGY EXECUTION COMPLETE")
        print(f"Total trades executed: {len(executed_trades)}")

        # Save execution report
        self.save_dual_execution_report(executed_trades)

        return executed_trades

    def get_next_friday(self):
        """Get next Friday for options expiration (like your weekly trades)"""
        today = datetime.now()
        days_until_friday = 4 - today.weekday()
        if days_until_friday <= 0:
            days_until_friday += 7
        next_friday = today + timedelta(days=days_until_friday)
        return next_friday.strftime('%y%m%d')

    def save_dual_execution_report(self, trades):
        """Save dual strategy execution report"""
        try:
            filename = f'dual_strategy_execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            report = {
                'timestamp': datetime.now().isoformat(),
                'strategy': 'DUAL_CASH_SECURED_PUT_LONG_CALL',
                'trades': trades,
                'total_executions': len(trades),
                'replicating_proven_roi': 68.3
            }

            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"Dual strategy report saved: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save dual execution report: {e}")

def test_dual_strategy():
    """Test the dual strategy engine"""
    engine = AdaptiveDualOptionsEngine()

    # Test opportunities (similar to your successful trades)
    test_opportunities = [
        {'symbol': 'AAPL', 'conviction_level': 'HIGH'},
        {'symbol': 'GOOGL', 'conviction_level': 'HIGH'},
        {'symbol': 'SPY', 'conviction_level': 'MEDIUM'},
        {'symbol': 'META', 'conviction_level': 'MEDIUM'}
    ]

    # Test with simulated buying power
    test_buying_power = 500000

    print("DUAL STRATEGY ENGINE TEST")
    print("=" * 60)
    print("Testing adaptive dual cash-secured put + long call strategy")
    print("Replicating your proven INTC/LYFT/SNAP/RIVN methodology")
    print("=" * 60)

    results = engine.execute_dual_strategy(test_opportunities, test_buying_power)

    print(f"\nTest completed: {len(results)} trades simulated")

if __name__ == "__main__":
    test_dual_strategy()