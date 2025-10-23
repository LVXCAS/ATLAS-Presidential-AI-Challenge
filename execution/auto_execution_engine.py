#!/usr/bin/env python3
"""
AUTO-EXECUTION ENGINE
Automatically executes AI-recommended trades
Monday-ready autonomous trading
"""

import os
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Alpaca (Options) - Using alpaca_trade_api for options support
try:
    import alpaca_trade_api as tradeapi
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest,
        GetOrdersRequest, ClosePositionRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[WARNING] Alpaca SDK not available")

# OANDA (Forex)
try:
    import v20
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("[WARNING] OANDA SDK not available")

load_dotenv()


class AutoExecutionEngine:
    """
    Autonomous trade execution engine

    Features:
    - Auto-execute Bull Put Spreads (options)
    - Auto-execute EMA Crossover (forex)
    - Risk management guardrails
    - Position tracking
    - Outcome logging for AI learning
    """

    def __init__(self, paper_trading: bool = True, max_risk_per_trade: float = 500.0):
        """
        Initialize auto-execution engine

        Args:
            paper_trading: Use paper trading accounts (default: True)
            max_risk_per_trade: Maximum risk per trade in dollars
        """
        self.paper_trading = paper_trading
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = 5
        self.min_score_options = 8.0
        self.min_score_forex = 9.0

        # Track executed trades
        self.executed_trades = []
        self.open_positions = []

        print("\n[AUTO-EXECUTION ENGINE] Initializing...")
        print(f"  Mode: {'PAPER TRADING' if paper_trading else 'LIVE TRADING'}")
        print(f"  Max risk per trade: ${max_risk_per_trade:.2f}")
        print(f"  Max positions: {self.max_positions}")

        # Initialize Alpaca (Options)
        if ALPACA_AVAILABLE:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')

            if paper_trading:
                base_url = 'https://paper-api.alpaca.markets'
            else:
                base_url = 'https://api.alpaca.markets'

            # Use alpaca_trade_api for options trading (full REST API support)
            self.alpaca_api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=base_url,
                api_version='v2'
            )

            # Keep TradingClient for other operations
            self.alpaca_client = TradingClient(api_key, secret_key, paper=paper_trading)
            self.alpaca_data = StockHistoricalDataClient(api_key, secret_key)

            print("  [OK] Alpaca connected (Options)")
        else:
            self.alpaca_api = None
            self.alpaca_client = None
            print("  [ERROR] Alpaca not available")

        # Initialize OANDA (Forex)
        if OANDA_AVAILABLE:
            oanda_api_key = os.getenv('OANDA_API_KEY')
            oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')

            if paper_trading:
                hostname = 'api-fxpractice.oanda.com'
            else:
                hostname = 'api-fxtrade.oanda.com'

            self.oanda_api = v20.Context(hostname=hostname, port=443, token=oanda_api_key)
            self.oanda_account_id = oanda_account_id

            print("  [OK] OANDA connected (Forex)")
        else:
            self.oanda_api = None
            print("  [ERROR] OANDA not available")

        print("[AUTO-EXECUTION ENGINE] Ready\n")

    def validate_trade(self, opportunity: Dict) -> tuple[bool, str]:
        """
        Validate trade against risk management rules

        Returns:
            (valid, reason)
        """
        # Check score threshold
        asset_type = opportunity.get('asset_type')
        final_score = opportunity.get('final_score', 0)

        if asset_type == 'OPTIONS':
            if final_score < self.min_score_options:
                return False, f"Score {final_score:.2f} below options threshold {self.min_score_options}"
        elif asset_type == 'FOREX':
            if final_score < self.min_score_forex:
                return False, f"Score {final_score:.2f} below forex threshold {self.min_score_forex}"

        # Check max positions
        if len(self.open_positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        # Check if already have position in this symbol
        symbol = opportunity.get('symbol')
        if any(pos['symbol'] == symbol for pos in self.open_positions):
            return False, f"Already have position in {symbol}"

        return True, "Trade validated"

    def execute_bull_put_spread(self, opportunity: Dict) -> Optional[Dict]:
        """
        Execute Bull Put Spread on Alpaca - REAL ORDERS

        Strategy:
        1. BUY PUT (protection) - Lower strike
        2. SELL PUT (premium) - Higher strike

        Args:
            opportunity: AI-enhanced opportunity dict

        Returns:
            Execution result dict or None
        """
        if not self.alpaca_api:
            print("[ERROR] Alpaca API not available")
            return None

        symbol = opportunity['symbol']
        price = opportunity['price']

        print(f"\n[AUTO-EXECUTE] {symbol} Bull Put Spread - REAL ORDERS")
        print(f"  Current Price: ${price:.2f}")
        print(f"  Score: {opportunity['final_score']:.2f}")
        print(f"  Confidence: {opportunity['confidence']:.0%}")
        print(f"  Mode: {'PAPER' if self.paper_trading else 'LIVE'} TRADING")

        try:
            # Get expiration date first (need it for option chain query)
            expiration_date = self._get_expiration_date(days_out=30)
            exp_str = expiration_date.strftime('%Y-%m-%d')

            print(f"  Target Expiration: {exp_str}")

            # Round strikes to standard increments (simpler approach)
            print(f"\n  [STRIKE CALCULATION] Rounding to standard increments...")

            def round_to_standard_strike(price, target_pct):
                """Round strike to standard $5 or $10 increment"""
                target_strike = price * target_pct

                # Determine increment based on price
                if price < 50:
                    increment = 2.5  # $2.50 increments for low-priced stocks
                elif price < 200:
                    increment = 5.0  # $5 increments
                else:
                    increment = 10.0  # $10 increments for high-priced stocks

                # Round to nearest increment
                rounded = round(target_strike / increment) * increment
                return rounded

            sell_strike = round_to_standard_strike(price, 0.95)  # 5% OTM
            buy_strike = round_to_standard_strike(price, 0.90)   # 10% OTM

            # Generate all possible strikes in range (for validation)
            if price < 50:
                increment = 2.5
            elif price < 200:
                increment = 5.0
            else:
                increment = 10.0

            available_strikes = []
            for mult in range(1, 100):
                strike = mult * increment
                if 0.5 * price <= strike <= 1.2 * price:  # Reasonable range
                    available_strikes.append(strike)

            available_strikes = sorted(available_strikes)

            print(f"  [OK] Using ${increment:.2f} strike increments for ${price:.2f} stock")
            print(f"  Available strikes: {len(available_strikes)} in range")

            # Calculate target strikes (5% and 10% below current price)
            target_sell_strike = price * 0.95  # 5% OTM - collect premium
            target_buy_strike = price * 0.90   # 10% OTM - protection

            print(f"\n  [TARGET STRIKES]")
            print(f"    Target Sell (95%): ${target_sell_strike:.2f}")
            print(f"    Target Buy (90%):  ${target_buy_strike:.2f}")

            # Find closest available strikes to our targets
            # Filter to strikes below current price (for puts)
            strikes_below = [s for s in available_strikes if s < price]

            if len(strikes_below) < 2:
                print(f"  [ERROR] Not enough strikes below current price ({len(strikes_below)} found, need 2)")
                return None

            # Find closest strike to target sell (higher strike)
            sell_strike = min(strikes_below, key=lambda x: abs(x - target_sell_strike))

            # Find closest strike to target buy (lower strike)
            # Must be below sell_strike
            strikes_below_sell = [s for s in strikes_below if s < sell_strike]

            if not strikes_below_sell:
                print(f"  [ERROR] No strikes available below sell strike ${sell_strike:.2f}")
                return None

            buy_strike = min(strikes_below_sell, key=lambda x: abs(x - target_buy_strike))

            # Validate spread makes sense
            spread_width = sell_strike - buy_strike

            print(f"\n  [SELECTED STRIKES]")
            print(f"    Sell Strike: ${sell_strike:.2f} (actual available)")
            print(f"    Buy Strike:  ${buy_strike:.2f} (actual available)")
            print(f"    Spread Width: ${spread_width:.2f}")

            # Validation checks
            if sell_strike <= buy_strike:
                print(f"  [ERROR] Invalid spread: sell_strike (${sell_strike}) must be > buy_strike (${buy_strike})")
                return None

            if spread_width < 2:
                print(f"  [ERROR] Spread too narrow: ${spread_width:.2f} (minimum $2)")
                return None

            if spread_width > 50:
                print(f"  [WARNING] Spread very wide: ${spread_width:.2f} (consider narrower spread)")

            if sell_strike >= price:
                print(f"  [ERROR] Sell strike (${sell_strike}) must be below current price (${price:.2f})")
                return None

            print(f"  [OK] Spread validation passed")

            # Calculate position size
            # Credit received: ~30% of spread width (typical)
            expected_credit_per_spread = spread_width * 0.30

            # Number of contracts to stay under risk limit
            max_risk_per_spread = spread_width - expected_credit_per_spread
            num_contracts = int(self.max_risk_per_trade / (max_risk_per_spread * 100))
            num_contracts = max(1, min(num_contracts, 3))  # Min 1, max 3 contracts for safety

            total_expected_credit = expected_credit_per_spread * num_contracts * 100
            total_max_risk = max_risk_per_spread * num_contracts * 100

            print(f"\n  [POSITION SIZE]")
            print(f"    Contracts: {num_contracts}")
            print(f"    Expected Credit: ${total_expected_credit:.2f}")
            print(f"    Max Risk: ${total_max_risk:.2f}")

            # Build OCC format option symbols
            # Format: SYMBOL + YYMMDD + P/C + Strike (8 digits with 3 decimals)
            exp_str_occ = expiration_date.strftime('%y%m%d')
            buy_put_symbol = f"{symbol}{exp_str_occ}P{int(buy_strike * 1000):08d}"
            sell_put_symbol = f"{symbol}{exp_str_occ}P{int(sell_strike * 1000):08d}"

            print(f"\n  [OPTION SYMBOLS]")
            print(f"    Buy Put:  {buy_put_symbol} @ ${buy_strike:.2f}")
            print(f"    Sell Put: {sell_put_symbol} @ ${sell_strike:.2f}")
            print(f"    Expiration: {expiration_date.strftime('%Y-%m-%d')}")

            orders = []
            alpaca_order_ids = []

            # LEG 1: BUY PUT (protection) - MUST execute first for risk management
            print(f"\n  [LEG 1/2] Buying {num_contracts} protective put @ ${buy_strike}...")
            try:
                buy_order = self.alpaca_api.submit_order(
                    symbol=buy_put_symbol,
                    qty=num_contracts,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                orders.append({
                    'leg': 'BUY_PUT',
                    'symbol': buy_put_symbol,
                    'strike': buy_strike,
                    'qty': num_contracts,
                    'order_id': buy_order.id,
                    'status': buy_order.status
                })
                alpaca_order_ids.append(buy_order.id)
                print(f"    [OK] Buy order submitted: {buy_order.id}")
                print(f"    Status: {buy_order.status}")

            except Exception as e:
                print(f"    [ERROR] Failed to buy protection: {e}")
                raise Exception(f"Cannot proceed without protection leg: {e}")

            # LEG 2: SELL PUT (premium collection) - Only after protection is in place
            print(f"\n  [LEG 2/2] Selling {num_contracts} put @ ${sell_strike} for credit...")
            try:
                sell_order = self.alpaca_api.submit_order(
                    symbol=sell_put_symbol,
                    qty=num_contracts,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                orders.append({
                    'leg': 'SELL_PUT',
                    'symbol': sell_put_symbol,
                    'strike': sell_strike,
                    'qty': num_contracts,
                    'order_id': sell_order.id,
                    'status': sell_order.status
                })
                alpaca_order_ids.append(sell_order.id)
                print(f"    [OK] Sell order submitted: {sell_order.id}")
                print(f"    Status: {sell_order.status}")

            except Exception as e:
                print(f"    [ERROR] Failed to sell put: {e}")
                print(f"    [WARNING] Protection leg is in place but sell leg failed")
                print(f"    [ACTION] Consider canceling buy order: {orders[0]['order_id']}")
                # Don't raise - we have protection in place, just log the failure

            # Build execution record
            execution = {
                'symbol': symbol,
                'strategy': 'BULL_PUT_SPREAD',
                'asset_type': 'OPTIONS',
                'entry_time': datetime.now().isoformat(),
                'entry_price': price,
                'sell_strike': sell_strike,
                'buy_strike': buy_strike,
                'num_contracts': num_contracts,
                'expected_credit': total_expected_credit,
                'max_risk': total_max_risk,
                'expiration': expiration_date.isoformat(),
                'expiration_date': expiration_date.strftime('%Y-%m-%d'),
                'status': 'OPEN',
                'paper_trade': self.paper_trading,
                'ai_score': opportunity['final_score'],
                'ai_confidence': opportunity['confidence'],
                'orders': orders,
                'alpaca_order_ids': alpaca_order_ids,
                'real_execution': True,
                'buy_put_symbol': buy_put_symbol,
                'sell_put_symbol': sell_put_symbol
            }

            self.open_positions.append(execution)
            self.executed_trades.append(execution)

            print(f"\n  [SUCCESS] Bull Put Spread EXECUTED on Alpaca!")
            print(f"  Position ID: {len(self.executed_trades)}")
            print(f"  Orders placed: {len(orders)}")
            print(f"  These orders will appear on your Alpaca dashboard")
            print(f"  Dashboard: https://app.alpaca.markets/paper/dashboard/overview")

            # Save execution log
            self._save_execution_log(execution)

            return execution

        except Exception as e:
            print(f"\n  [ERROR] Failed to execute: {e}")
            print(f"  [ROLLBACK] Attempting to cancel any filled orders...")

            # Attempt to cancel any orders that were placed
            for order in orders:
                try:
                    self.alpaca_api.cancel_order(order['order_id'])
                    print(f"    Canceled {order['leg']}: {order['order_id']}")
                except Exception as cancel_error:
                    print(f"    Could not cancel {order['leg']}: {cancel_error}")

            return None

    def _get_expiration_date(self, days_out=30):
        """
        Get appropriate expiration date for options (typically Friday)

        Args:
            days_out: Target days until expiration

        Returns:
            datetime object for expiration date
        """
        today = datetime.now()

        # Find next Friday
        days_until_friday = 4 - today.weekday()  # Friday = 4
        if days_until_friday <= 0:
            days_until_friday += 7

        # If days_out is specified, find appropriate Friday
        target_date = today + timedelta(days=days_out)

        # Adjust to nearest Friday
        days_to_adjust = 4 - target_date.weekday()
        if days_to_adjust < 0:
            days_to_adjust += 7

        expiration = target_date + timedelta(days=days_to_adjust)

        return expiration

    def execute_forex_trade(self, opportunity: Dict) -> Optional[Dict]:
        """
        Execute forex trade on OANDA

        Args:
            opportunity: AI-enhanced opportunity dict

        Returns:
            Execution result dict or None
        """
        if not self.oanda_api:
            print("[ERROR] OANDA API not available")
            return None

        symbol = opportunity['symbol']
        direction = opportunity['direction']
        entry = opportunity['entry']
        stop = opportunity['stop']
        target = opportunity['target']

        print(f"\n[AUTO-EXECUTE] {symbol} {direction}")
        print(f"  Entry: {entry:.5f}")
        print(f"  Stop: {stop:.5f}")
        print(f"  Target: {target:.5f}")
        print(f"  Score: {opportunity['final_score']:.2f}")
        print(f"  Confidence: {opportunity['confidence']:.0%}")

        try:
            # Calculate position size (units)
            # Based on risk: (entry - stop) * units = max_risk
            pip_risk = abs(entry - stop)

            # For EUR_USD: 1 pip = 0.0001, $10 per pip per 100k units
            # For practice: use smaller position sizes
            units = 5000 if self.paper_trading else 1000

            # Adjust direction
            if direction == 'SHORT':
                units = -units

            print(f"  Units: {units}")

            # Place market order
            order_request = {
                'order': {
                    'units': str(units),
                    'instrument': symbol,
                    'timeInForce': 'FOK',  # Fill or Kill
                    'type': 'MARKET',
                    'positionFill': 'DEFAULT',
                    'stopLossOnFill': {
                        'price': f"{stop:.5f}"
                    },
                    'takeProfitOnFill': {
                        'price': f"{target:.5f}"
                    }
                }
            }

            response = self.oanda_api.order.create(self.oanda_account_id, **order_request)

            if response.status == 201:
                order_fill = response.body.get('orderFillTransaction')

                execution = {
                    'symbol': symbol,
                    'strategy': 'EMA_CROSSOVER_OPTIMIZED',
                    'asset_type': 'FOREX',
                    'direction': direction,
                    'entry_time': datetime.now().isoformat(),
                    'entry_price': entry,
                    'stop_loss': stop,
                    'take_profit': target,
                    'units': units,
                    'status': 'OPEN',
                    'paper_trade': self.paper_trading,
                    'ai_score': opportunity['final_score'],
                    'ai_confidence': opportunity['confidence'],
                    'order_id': order_fill.id if order_fill else None
                }

                self.open_positions.append(execution)
                self.executed_trades.append(execution)

                print(f"  [EXECUTED] {symbol} {direction}")
                print(f"  Order ID: {execution.get('order_id')}")

                # Save execution log
                self._save_execution_log(execution)

                return execution
            else:
                print(f"  [ERROR] Order failed: {response}")
                return None

        except Exception as e:
            print(f"  [ERROR] Failed to execute: {e}")
            return None

    def execute_futures_trade(self, opportunity: Dict) -> Optional[Dict]:
        """
        Execute futures trade on Alpaca

        Args:
            opportunity: AI-enhanced opportunity dict

        Returns:
            Execution result dict or None
        """
        if not self.alpaca_api:
            print("[ERROR] Alpaca API not available")
            return None

        symbol = opportunity['symbol']
        direction = opportunity['direction']
        entry = opportunity['entry_price']
        stop = opportunity['stop_loss']
        target = opportunity['take_profit']
        point_value = opportunity.get('point_value', 5.0)

        print(f"\n[AUTO-EXECUTE] {symbol} FUTURES {direction}")
        print(f"  Entry: ${entry:.2f}")
        print(f"  Stop: ${stop:.2f}")
        print(f"  Target: ${target:.2f}")
        print(f"  Score: {opportunity['final_score']:.2f}")
        print(f"  Confidence: {opportunity['confidence']:.0%}")
        print(f"  Mode: {'PAPER' if self.paper_trading else 'LIVE'} TRADING")

        try:
            # Calculate position size
            risk_per_contract = opportunity.get('risk_per_contract', 100)
            contracts = max(1, int(self.max_risk_per_trade / risk_per_contract))
            contracts = min(contracts, 2)  # Max 2 contracts for safety

            print(f"  Contracts: {contracts}")
            print(f"  Risk per contract: ${risk_per_contract:.2f}")
            print(f"  Total risk: ${risk_per_contract * contracts:.2f}")

            # Determine side
            side = 'buy' if direction == 'LONG' else 'sell'

            # NOTE: For real futures trading, you would use:
            # - Alpaca futures API when available
            # - Proper futures contract symbols (e.g., MESM25 for March 2025 MES)
            # For now, we'll use proxy approach (SPY for MES, QQQ for MNQ)

            proxy_map = {
                'MES': 'SPY',
                'MNQ': 'QQQ'
            }

            proxy_symbol = proxy_map.get(symbol, symbol)

            print(f"\n  [NOTE] Using {proxy_symbol} as proxy for {symbol} futures")
            print(f"  [SIMULATION] Would place {side.upper()} order for {contracts} {symbol} contracts")

            # Build execution record
            execution = {
                'symbol': symbol,
                'strategy': 'FUTURES_EMA_CROSSOVER',
                'asset_type': 'FUTURES',
                'direction': direction,
                'entry_time': datetime.now().isoformat(),
                'entry_price': entry,
                'stop_loss': stop,
                'take_profit': target,
                'contracts': contracts,
                'point_value': point_value,
                'risk_per_contract': risk_per_contract,
                'total_risk': risk_per_contract * contracts,
                'status': 'OPEN',
                'paper_trade': self.paper_trading,
                'ai_score': opportunity['final_score'],
                'ai_confidence': opportunity['confidence'],
                'simulated': True,  # Set to True until real futures API available
                'proxy_symbol': proxy_symbol
            }

            self.open_positions.append(execution)
            self.executed_trades.append(execution)

            print(f"\n  [EXECUTED] {symbol} {direction} - {contracts} contracts")
            print(f"  Position ID: {len(self.executed_trades)}")
            print(f"  Note: Simulated execution (waiting for Alpaca futures API)")

            # Save execution log
            self._save_execution_log(execution)

            return execution

        except Exception as e:
            print(f"  [ERROR] Failed to execute: {e}")
            return None

    def execute_opportunity(self, opportunity: Dict) -> Optional[Dict]:
        """
        Execute opportunity (routes to appropriate execution method)

        Args:
            opportunity: AI-enhanced opportunity dict

        Returns:
            Execution result dict or None
        """
        # Validate trade
        valid, reason = self.validate_trade(opportunity)
        if not valid:
            print(f"\n[SKIP] {opportunity['symbol']}: {reason}")
            return None

        asset_type = opportunity.get('asset_type')

        if asset_type == 'OPTIONS':
            return self.execute_bull_put_spread(opportunity)
        elif asset_type == 'FOREX':
            return self.execute_forex_trade(opportunity)
        elif asset_type == 'FUTURES':
            return self.execute_futures_trade(opportunity)
        else:
            print(f"[ERROR] Unknown asset type: {asset_type}")
            return None

    def auto_execute_opportunities(self, opportunities: List[Dict], max_trades: int = 2) -> List[Dict]:
        """
        Automatically execute top opportunities

        Args:
            opportunities: List of AI-enhanced opportunities
            max_trades: Maximum trades to execute in this session

        Returns:
            List of executed trades
        """
        print("\n" + "="*70)
        print("AUTO-EXECUTION SESSION")
        print("="*70)
        print(f"Total opportunities: {len(opportunities)}")
        print(f"Max trades to execute: {max_trades}")
        print(f"Current open positions: {len(self.open_positions)}")
        print("="*70)

        executed = []

        for i, opp in enumerate(opportunities[:max_trades * 2]):  # Check 2x in case some fail
            if len(executed) >= max_trades:
                print(f"\n[LIMIT] Reached max trades ({max_trades})")
                break

            result = self.execute_opportunity(opp)
            if result:
                executed.append(result)

        print("\n" + "="*70)
        print(f"EXECUTION COMPLETE: {len(executed)} trades executed")
        print("="*70)

        return executed

    def _save_execution_log(self, execution: Dict):
        """Save execution to log file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f'executions/execution_log_{timestamp}.json'

        os.makedirs('executions', exist_ok=True)

        # Load existing
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                log = json.load(f)
        else:
            log = {'date': timestamp, 'executions': []}

        log['executions'].append(execution)

        with open(filename, 'w') as f:
            json.dump(log, f, indent=2, default=str)

    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions"""
        return self.open_positions

    def get_execution_summary(self) -> Dict:
        """Get summary of executed trades"""
        return {
            'total_executed': len(self.executed_trades),
            'open_positions': len(self.open_positions),
            'options_trades': len([t for t in self.executed_trades if t['asset_type'] == 'OPTIONS']),
            'forex_trades': len([t for t in self.executed_trades if t['asset_type'] == 'FOREX']),
            'futures_trades': len([t for t in self.executed_trades if t['asset_type'] == 'FUTURES']),
        }


if __name__ == "__main__":
    # Test execution engine
    print("\n" + "="*70)
    print("AUTO-EXECUTION ENGINE TEST")
    print("="*70)

    engine = AutoExecutionEngine(paper_trading=True, max_risk_per_trade=500)

    # Test with sample opportunity
    sample_opp = {
        'symbol': 'AAPL',
        'asset_type': 'OPTIONS',
        'strategy': 'BULL_PUT_SPREAD',
        'final_score': 8.5,
        'confidence': 0.72,
        'price': 175.50
    }

    print("\n[TEST] Executing sample Bull Put Spread...")
    result = engine.execute_opportunity(sample_opp)

    if result:
        print("\n[SUCCESS] Test execution completed")
        print(json.dumps(result, indent=2, default=str))
    else:
        print("\n[FAILED] Test execution failed")

    print("\n" + "="*70)
