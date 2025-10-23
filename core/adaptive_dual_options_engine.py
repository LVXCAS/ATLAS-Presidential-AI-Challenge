#!/usr/bin/env python3
"""
ADAPTIVE DUAL OPTIONS ENGINE
Replicates Lucas's proven 68.3% ROI strategy: Cash-secured puts + Long calls
Adapts to different market conditions automatically
"""

import alpaca_trade_api as tradeapi
import os
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import sys

# Add agents path for QuantLib pricer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.quantlib_pricing import QuantLibPricer


class AdaptiveDualOptionsEngine:
    """Adaptive engine for dual cash-secured put + long call strategy"""

    def __init__(self, api_key=None, secret_key=None, base_url=None):
        """
        Initialize adaptive dual options engine

        Args:
            api_key: Optional Alpaca API key (if not provided, loads from environment)
            secret_key: Optional Alpaca secret key
            base_url: Optional Alpaca base URL
        """
        # Only load environment if credentials not provided
        # This prevents overriding credentials set by parent systems
        if not api_key:
            # Use override=True to ensure we load the CORRECT credentials
            load_dotenv(override=True)
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL')

        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - DUAL_ENGINE - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize QuantLib pricer for Greeks
        try:
            self.quantlib_pricer = QuantLibPricer()
            self.use_greeks = True
            self.logger.info("QuantLib Greeks integration ACTIVE")
        except Exception as e:
            self.logger.warning(f"QuantLib not available, using simple pricing: {e}")
            self.quantlib_pricer = None
            self.use_greeks = False

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

    def calculate_greeks_based_strikes(self, symbol, current_price, regime='neutral'):
        """Calculate strikes using QuantLib Greeks and delta targeting

        Target deltas:
        - Put: -0.35 delta (35% probability of profit)
        - Call: 0.35 delta (35% probability of profit)

        Lower deltas = further OTM = less collateral needed
        """
        try:
            # Get next Friday expiration
            exp_date_str = self.get_next_friday()
            # Convert to datetime for QuantLib
            exp_year = int('20' + exp_date_str[:2])
            exp_month = int(exp_date_str[2:4])
            exp_day = int(exp_date_str[4:6])
            expiry_date = datetime(exp_year, exp_month, exp_day)

            # Target deltas based on regime
            regime_deltas = {
                'bull': {'put_delta': -0.30, 'call_delta': 0.40},  # More aggressive in bull
                'bear': {'put_delta': -0.40, 'call_delta': 0.30},  # More conservative in bear
                'neutral': {'put_delta': -0.35, 'call_delta': 0.35}  # Balanced
            }

            targets = regime_deltas.get(regime, regime_deltas['neutral'])
            target_put_delta = targets['put_delta']
            target_call_delta = targets['call_delta']

            # Determine strike increment
            increment = 1.0 if current_price >= 25 else 0.5

            # Search for optimal put strike (iterate from 0.85x to 0.95x of current price)
            best_put_strike = None
            best_put_delta_diff = float('inf')
            put_greeks = None

            for ratio in [0.85, 0.87, 0.89, 0.91, 0.93, 0.95]:
                test_strike = self.round_to_valid_strike(current_price * ratio, current_price)

                try:
                    greeks = self.quantlib_pricer.price_european_option(
                        'PUT', current_price, test_strike, expiry_date, symbol
                    )

                    delta_diff = abs(greeks['delta'] - target_put_delta)
                    if delta_diff < best_put_delta_diff:
                        best_put_delta_diff = delta_diff
                        best_put_strike = test_strike
                        put_greeks = greeks

                except Exception as e:
                    continue

            # Search for optimal call strike (iterate from 1.05x to 1.15x of current price)
            best_call_strike = None
            best_call_delta_diff = float('inf')
            call_greeks = None

            for ratio in [1.05, 1.07, 1.09, 1.11, 1.13, 1.15]:
                test_strike = self.round_to_valid_strike(current_price * ratio, current_price)

                try:
                    greeks = self.quantlib_pricer.price_european_option(
                        'CALL', current_price, test_strike, expiry_date, symbol
                    )

                    delta_diff = abs(greeks['delta'] - target_call_delta)
                    if delta_diff < best_call_delta_diff:
                        best_call_delta_diff = delta_diff
                        best_call_strike = test_strike
                        call_greeks = greeks

                except Exception as e:
                    continue

            # Return Greeks-optimized strikes
            if best_put_strike and best_call_strike:
                return {
                    'put_strike': float(best_put_strike),
                    'call_strike': float(best_call_strike),
                    'regime': regime,
                    'volatility_factor': 1.0,
                    'put_greeks': put_greeks,
                    'call_greeks': call_greeks,
                    'method': 'GREEKS_DELTA_TARGETING'
                }
            else:
                raise Exception("Could not find optimal strikes with Greeks")

        except Exception as e:
            self.logger.warning(f"Greeks calculation failed for {symbol}: {e}, falling back to simple method")
            return None

    def calculate_adaptive_strikes(self, symbol, current_price, regime='neutral'):
        """Calculate strikes based on market regime and your proven patterns

        Uses QuantLib Greeks if available, otherwise falls back to proven percentages
        """

        # Try Greeks-based calculation first if enabled
        if self.use_greeks:
            greeks_result = self.calculate_greeks_based_strikes(symbol, current_price, regime)
            if greeks_result:
                return greeks_result

        # Fallback to proven percentage-based method
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
            'volatility_factor': adjustments['volatility_factor'],
            'method': 'PERCENTAGE_BASED'
        }

    def calculate_position_size(self, symbol, buying_power, allocation, volatility_factor=1.0):
        """Calculate realistic position size for $100k account

        For cash-secured puts, we need: strike × 100 × contracts
        Reserve 50% of allocation for calls, 50% for puts
        """

        # Realistic contract sizes for $100k account
        realistic_sizes = {
            'min': 1,     # Minimum 1 contract
            'max': 5      # Max 5 contracts (conservative)
        }

        # Allocate 50% to puts, 50% to calls
        trade_amount = buying_power * allocation * 0.5  # Half for each leg

        # Get current stock price to estimate put collateral
        try:
            bars = self.api.get_latest_bar(symbol)
            current_price = float(bars.c)

            # Cash-secured put needs: strike × 100
            # Use 90% of current price as estimate for strike
            estimated_strike = current_price * 0.9
            collateral_per_contract = estimated_strike * 100

            # Calculate how many contracts we can afford
            contracts = int(trade_amount / collateral_per_contract)

            # Apply bounds
            contracts = max(realistic_sizes['min'], min(realistic_sizes['max'], contracts))

        except:
            # Fallback to 1 contract if price fetch fails
            contracts = 1

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
                # Realistic allocations for $100k account (10%, 8%, 5%)
                allocation = [0.10, 0.08, 0.05][i] if i < 3 else 0.03

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
                method = strike_data.get('method', 'UNKNOWN')

                print(f"  Current Price: ${current_price:.2f}")
                print(f"  Strike Method: {method}")
                print(f"  PUT Strike: ${put_strike:.0f} ({((put_strike/current_price-1)*100):+.1f}%)")
                print(f"  CALL Strike: ${call_strike:.0f} ({((call_strike/current_price-1)*100):+.1f}%)")

                # Display Greeks if available
                if 'put_greeks' in strike_data and strike_data['put_greeks']:
                    put_greeks = strike_data['put_greeks']
                    print(f"  PUT Greeks:")
                    print(f"    Delta: {put_greeks['delta']:.3f} (prob of profit: {abs(put_greeks['delta'])*100:.1f}%)")
                    print(f"    Theta: ${put_greeks['theta']:.2f}/day (time decay)")
                    print(f"    Vega: ${put_greeks['vega']:.2f} (IV sensitivity)")
                    print(f"    Premium: ${put_greeks['price']:.2f}")

                if 'call_greeks' in strike_data and strike_data['call_greeks']:
                    call_greeks = strike_data['call_greeks']
                    print(f"  CALL Greeks:")
                    print(f"    Delta: {call_greeks['delta']:.3f} (prob of profit: {call_greeks['delta']*100:.1f}%)")
                    print(f"    Theta: ${call_greeks['theta']:.2f}/day (time decay)")
                    print(f"    Vega: ${call_greeks['vega']:.2f} (IV sensitivity)")
                    print(f"    Premium: ${call_greeks['price']:.2f}")

                # Calculate position size
                contracts = self.calculate_position_size(symbol, buying_power, allocation, volatility_factor)
                print(f"  Contract Size: {contracts} contracts (volatility factor: {volatility_factor:.1f}x)")

                # Round strikes to valid increments
                put_strike = self.round_to_valid_strike(put_strike, current_price)
                call_strike = self.round_to_valid_strike(call_strike, current_price)

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
                    print(f"  [VERIFY] PRE-SUBMIT: symbol={put_symbol}, qty={contracts}, side=sell")

                    put_order = self.api.submit_order(
                        symbol=put_symbol,
                        qty=contracts,
                        side='sell',  # SELL puts for premium
                        type='market',
                        time_in_force='day'
                    )

                    print(f"  [OK] PUT ORDER SUBMITTED: {put_order.id}")
                    print(f"  [VERIFY] POST-SUBMIT: order_id={put_order.id}, requested_qty={contracts}")

                    # Verify order filled correctly (wait 2 seconds for fill)
                    time.sleep(2)
                    try:
                        filled_order = self.api.get_order_by_id(put_order.id)
                        filled_qty = abs(int(filled_order.filled_qty)) if filled_order.filled_qty else 0
                        if filled_qty != contracts:
                            print(f"  [WARNING] PUT QTY MISMATCH! Requested: {contracts}, Filled: {filled_qty}")
                            self.logger.warning(f"PUT order {put_order.id} quantity mismatch: requested {contracts}, filled {filled_qty}")
                        else:
                            print(f"  [VERIFY] PUT filled correctly: {filled_qty} contracts")
                    except Exception as verify_error:
                        print(f"  [WARN] Could not verify PUT fill: {verify_error}")

                    # 2. BUY LONG CALLS (capture upside)
                    print(f"  BUYING {contracts} long calls...")
                    print(f"  [VERIFY] PRE-SUBMIT: symbol={call_symbol}, qty={contracts}, side=buy")

                    call_order = self.api.submit_order(
                        symbol=call_symbol,
                        qty=contracts,
                        side='buy',   # BUY calls for upside
                        type='market',
                        time_in_force='day'
                    )

                    print(f"  [OK] CALL ORDER SUBMITTED: {call_order.id}")
                    print(f"  [VERIFY] POST-SUBMIT: order_id={call_order.id}, requested_qty={contracts}")

                    # Verify order filled correctly (wait 2 seconds for fill)
                    time.sleep(2)
                    try:
                        filled_order = self.api.get_order_by_id(call_order.id)
                        filled_qty = int(filled_order.filled_qty) if filled_order.filled_qty else 0
                        if filled_qty != contracts:
                            print(f"  [WARNING] CALL QTY MISMATCH! Requested: {contracts}, Filled: {filled_qty}")
                            self.logger.warning(f"CALL order {call_order.id} quantity mismatch: requested {contracts}, filled {filled_qty}")
                        else:
                            print(f"  [VERIFY] CALL filled correctly: {filled_qty} contracts")
                    except Exception as verify_error:
                        print(f"  [WARN] Could not verify CALL fill: {verify_error}")

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

                    # DISABLED: Stock fallback was causing massive losses
                    # Created 5977 AMD shares ($1.4M position) and other huge positions
                    # Better to skip trade than buy massive stock positions
                    if not dual_success:
                        print(f"  [SKIP] Options not available - no fallback to stock")
                        print(f"  [REASON] Stock fallback disabled - caused 66% losing rate")
                        print(f"  [SAFE] Better to miss opportunity than take massive stock risk")
                        # COMMENTED OUT DANGEROUS STOCK FALLBACK:
                        # try:
                        #     quote = self.api.get_latest_quote(symbol)
                        #     price = float(quote.ask_price)
                        #     shares = max(1, int((buying_power * allocation) / price))
                        #
                        #     stock_order = self.api.submit_order(
                        #         symbol=symbol,
                        #         qty=shares,
                        #         side='buy',
                        #         type='market',
                        #         time_in_force='day'
                        #     )
                        #
                        #     print(f"  [OK] STOCK FALLBACK: {shares} shares @ ${price:.2f}")
                        #     executed_trades.append({
                        #         'symbol': symbol,
                        #         'strategy': 'STOCK_FALLBACK',
                        #         'order': stock_order.id,
                        #         'shares': shares,
                        #         'price': price
                        #     })
                        #
                        # except Exception as fallback_error:
                        #     print(f"  [X] STOCK FALLBACK FAILED: {fallback_error}")

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

    def round_to_valid_strike(self, strike, stock_price):
        """Round strike to valid options increment based on stock price

        Standard options strike intervals:
        - $0.50 for stocks under $25
        - $1.00 for stocks $25-$200
        - $2.50 for stocks $200-$500
        - $5.00 for stocks $500+
        """
        if stock_price < 25:
            increment = 0.5
        elif stock_price < 200:
            increment = 1.0
        elif stock_price < 500:
            increment = 2.5
        else:
            increment = 5.0

        # Round to nearest increment
        return round(strike / increment) * increment

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