"""
ATLAS Live Trading Loop

Connects to OANDA and trades real market data in paper mode.
"""

import os
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import logging
import json
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

from adapters.oanda_adapter import OandaAdapter
from core.trade_logger import TradeLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_kelly_position_size(balance: float, kelly_fraction: float, stop_loss_pips: float,
                                  symbol: str, max_lots: float = None, min_lots: float = None) -> int:
    """
    Calculate position size using Kelly Criterion.

    Args:
        balance: Current account balance
        kelly_fraction: Kelly fraction to use (0.10 = 1/10 Kelly)
        stop_loss_pips: Stop loss distance in pips
        symbol: Trading pair (for pip value calculation)
        max_lots: Maximum lot size (optional cap)
        min_lots: Minimum lot size (optional floor)

    Returns:
        Position size in units (100,000 units = 1 standard lot)

    Formula:
        Risk Amount = Balance × Kelly Fraction
        Lot Size = Risk Amount / (Stop Loss Pips × Pip Value)

    Example:
        Balance: $182,999
        Kelly Fraction: 0.10 (1/10 Kelly = 10% optimal)
        Stop Loss: 15 pips
        Pip Value: $10/pip/lot (EUR/USD)

        Risk Amount = $182,999 × 0.10 = $18,299
        Lot Size = $18,299 / (15 pips × $10/pip) = 122 lots
        Units = 122 × 100,000 = 12,200,000 units
    """
    # Pip value per standard lot (100,000 units)
    if 'JPY' in symbol:
        pip_value = 10.0  # For JPY pairs (0.01 pip = $10)
    else:
        pip_value = 10.0  # For major pairs (0.0001 pip = $10)

    # Calculate Kelly risk amount
    risk_amount = balance * kelly_fraction

    # Calculate lot size
    lot_size = risk_amount / (stop_loss_pips * pip_value)

    # Apply caps if specified
    if max_lots is not None:
        lot_size = min(lot_size, max_lots)
    if min_lots is not None:
        lot_size = max(lot_size, min_lots)

    # Convert to units (round to nearest lot)
    units = int(round(lot_size)) * 100000

    # Safety: Minimum 1 lot (100,000 units)
    units = max(units, 100000)

    logger.info(f"[KELLY] Balance: ${balance:,.2f}, Risk: ${risk_amount:,.2f} ({kelly_fraction*100:.1f}%)")
    logger.info(f"[KELLY] SL: {stop_loss_pips} pips, Lot Size: {lot_size:.2f}, Units: {units:,}")

    return units


def run_live_trading(coordinator, learning_engine, days: int = 20, fast_scan: bool = False):
    """
    Run live paper trading with real OANDA market data.

    Args:
        coordinator: ATLAS coordinator
        learning_engine: Learning engine
        days: Number of days to run
        fast_scan: If True, use 1-minute scans for testing (default: False)
    """
    try:
        print(f"\n{'='*80}")
        print(f"STARTING LIVE PAPER TRADING - {days} DAYS")
        print(f"{'='*80}\n")

        # Load config for Kelly Criterion parameters
        config_path = Path(__file__).parent / "config" / "hybrid_optimized.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Extract Kelly parameters
            kelly_fraction = config['risk_management'].get('kelly_fraction', 0.10)
            max_lots = config['trading_parameters'].get('max_lots', 25.0)
            min_lots = config['trading_parameters'].get('min_lots', 3.0)

            print(f"[CONFIG] Kelly Fraction: {kelly_fraction*100:.1f}% (1/{int(1/kelly_fraction)} Kelly)")
            print(f"[CONFIG] Lot Range: {min_lots:.1f} - {max_lots:.1f} lots")
            print(f"[CONFIG] Position Sizing: Kelly Criterion (Dynamic)\n")

        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
            kelly_fraction = 0.10
            max_lots = 25.0
            min_lots = 3.0

        # Initialize OANDA
        try:
            oanda = OandaAdapter()
            balance_data = oanda.get_account_balance()

            if not balance_data:
                print("[ERROR] Could not connect to OANDA")
                return

            print(f"[OK] Connected to OANDA")
            print(f"[OK] Account Balance: ${balance_data['balance']:,.2f}")
            print(f"[OK] Unrealized P/L: ${balance_data.get('unrealized_pnl', 0):,.2f}\n")

        except Exception as e:
            print(f"[ERROR] OANDA initialization failed: {e}")
            return

        # Initialize Trade Logger
        trade_logger = TradeLogger()
        print(f"[OK] Trade Logger initialized")
        print(f"[OK] Logging to: {trade_logger.log_dir}\n")

        # Trading pairs
        pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]

        # Scan interval (seconds)
        if fast_scan:
            scan_interval = 60  # 1 minute for rapid testing
        elif coordinator.mode == "paper":
            scan_interval = 300  # 5 minutes for paper trading
        else:
            scan_interval = 900  # 15 minutes for live trading

        # Calculate end time
        start_time = datetime.now()
        end_time = start_time + timedelta(days=days)

        print(f"[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[END]   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[SCAN INTERVAL] Every {scan_interval // 60} minutes\n")

        scan_count = 0

        # Main trading loop
        while datetime.now() < end_time:
            scan_count += 1
            scan_start = datetime.now()

            print(f"\n{'='*80}")
            print(f"SCAN #{scan_count} - {scan_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            # Update account balance
            try:
                balance_data = oanda.get_account_balance()
                current_balance = balance_data['balance']
                unrealized_pnl = balance_data.get('unrealized_pnl', 0)

                print(f"Account: ${current_balance:,.2f} (Unrealized P/L: ${unrealized_pnl:+,.2f})")

            except Exception as e:
                logger.error(f"Could not fetch account balance: {e}")
                current_balance = 200000  # Default

            # Check open positions
            try:
                positions = oanda.get_open_positions()
                if positions:
                    print(f"\nOpen Positions: {len(positions)}")
                    for pos in positions:
                        print(f"  {pos['instrument']}: {pos['units']} units @ {pos['avg_price']:.5f}, "
                              f"P/L: ${pos.get('unrealized_pnl', 0):+,.2f}")

                        # Update trailing stop for profitable positions
                        unrealized_pnl = pos.get('unrealized_pnl', 0)
                        if unrealized_pnl > 0:
                            # Only trail if we're in profit
                            trailing_distance = 14  # Trail 14 pips behind current price
                            trade_id = pos.get('trade_id')

                            if trade_id:
                                try:
                                    oanda.update_trailing_stop(
                                        trade_id=trade_id,
                                        trailing_distance_pips=trailing_distance,
                                        symbol=pos['instrument']
                                    )
                                except Exception as trail_err:
                                    logger.warning(f"Could not update trailing stop for {pos['instrument']}: {trail_err}")
                else:
                    print(f"\nOpen Positions: 0")
            except Exception as e:
                logger.error(f"Could not fetch positions: {e}")
                positions = []

            print(f"\n{'-'*80}")
            print(f"Scanning {len(pairs)} pairs...")
            print(f"{'-'*80}")

            # Scan each pair
            opportunities_found = 0
            trades_executed = 0

            for pair in pairs:
                try:
                    # Get current market data
                    market_data_raw = oanda.get_market_data(pair)

                    if not market_data_raw:
                        logger.warning(f"No market data for {pair}")
                        continue

                    # Get candles for indicators
                    candles = oanda.get_candles(pair, 'H1', count=201)

                    if not candles or len(candles) < 199:
                        logger.warning(f"Insufficient candle data for {pair} ({len(candles) if candles else 0} candles)")
                        continue

                    # Calculate indicators
                    closes = np.array([c['close'] for c in candles])
                    highs = np.array([c['high'] for c in candles])
                    lows = np.array([c['low'] for c in candles])

                    # Helper function for EMA
                    def calc_ema(data, period):
                        multiplier = 2 / (period + 1)
                        ema = [data[0]]
                        for price in data[1:]:
                            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
                        return ema[-1]

                    # RSI (proper calculation)
                    deltas = np.diff(closes)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    avg_gain = np.mean(gains[-14:])
                    avg_loss = np.mean(losses[-14:]) or 0.00001
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                    # MACD (proper EMA calculation)
                    ema12 = calc_ema(closes, 12)
                    ema26 = calc_ema(closes, 26)
                    macd = ema12 - ema26

                    # MACD signal line (9-period EMA of MACD)
                    # Calculate MACD history for signal
                    macd_values = []
                    for i in range(len(closes) - 26):
                        ema12_i = calc_ema(closes[:26+i], 12)
                        ema26_i = calc_ema(closes[:26+i], 26)
                        macd_values.append(ema12_i - ema26_i)
                    macd_signal = calc_ema(np.array(macd_values[-9:]), 9) if len(macd_values) >= 9 else macd
                    macd_hist = macd - macd_signal

                    # EMAs (proper exponential)
                    ema50 = calc_ema(closes, 50)
                    ema200 = calc_ema(closes, 200)

                    # Bollinger Bands
                    bb_middle = closes[-20:].mean()
                    bb_std = closes[-20:].std()
                    bb_upper = bb_middle + (2 * bb_std)
                    bb_lower = bb_middle - (2 * bb_std)

                    # ADX (Average Directional Index - proper calculation)
                    def calc_adx(highs, lows, closes, period=14):
                        # Calculate True Range
                        tr = np.maximum(
                            highs[1:] - lows[1:],
                            np.maximum(
                                np.abs(highs[1:] - closes[:-1]),
                                np.abs(lows[1:] - closes[:-1])
                            )
                        )

                        # Calculate directional movement
                        up_move = highs[1:] - highs[:-1]
                        down_move = lows[:-1] - lows[1:]

                        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

                        # Smooth with EMA
                        atr_smooth = calc_ema(tr, period)
                        plus_di = 100 * calc_ema(plus_dm, period) / atr_smooth
                        minus_di = 100 * calc_ema(minus_dm, period) / atr_smooth

                        # Calculate DX and ADX
                        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 0.00001)
                        return dx  # Simplified - proper ADX needs 14-period EMA of DX

                    adx = calc_adx(highs, lows, closes, 14)

                    # ATR (Average True Range - proper calculation)
                    tr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(
                        np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:] - closes[:-1])
                    )
                    )
                    atr = calc_ema(tr, 14)


                    # Build market data for ATLAS
                    enriched_data = {
                    "pair": pair,
                    "price": market_data_raw['bid'],
                    "time": datetime.now(),
                    "session": get_session(),
                    "indicators": {
                        "rsi": rsi,
                        "macd": macd,
                        "macd_signal": macd_signal,
                        "macd_hist": macd_hist,
                        "ema50": ema50,
                        "ema200": ema200,
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower,
                        "bb_middle": bb_middle,
                        "adx": adx,
                        "atr": atr,
                    },
                    "account_balance": current_balance,
                        "date": datetime.now().date(),
                    }

                    # Analyze opportunity
                    decision = coordinator.analyze_opportunity(enriched_data)

                    # Log decision
                    print(f"\n{pair}:")
                    print(f"  Price: {market_data_raw['bid']:.5f}, RSI: {rsi:.1f}, "
                          f"MACD: {macd:.6f}, ADX: {adx:.1f}")
                    print(f"  Score: {decision.get('score', decision.get('weighted_score', 0)):.2f} / {coordinator.score_threshold}")
                    print(f"  Decision: {decision['decision']}")

                    if decision['decision'] in ['BUY', 'SELL']:
                        opportunities_found += 1

                        # Execute trade with Kelly Criterion position sizing
                        try:
                            # Convert BUY/SELL to long/short for OANDA adapter
                            decision_direction = decision['decision']

                            # DEBUG: Log conversion
                            print(f"  [DEBUG] ATLAS Decision: '{decision_direction}' (type: {type(decision_direction)})")
                            print(f"  [DEBUG] decision_direction == 'BUY': {decision_direction == 'BUY'}")
                            print(f"  [DEBUG] decision_direction == 'SELL': {decision_direction == 'SELL'}")

                            direction = 'long' if decision_direction == 'BUY' else 'short'

                            print(f"  [DEBUG] Converting to: '{direction}'")

                            stop_loss_pips = 25  # Widened from 14 to survive 20-30 pip volatility
                            take_profit_pips = 50  # Widened from 21 for 1:2 RR ratio

                            # Calculate position size using Kelly Criterion
                            units = calculate_kelly_position_size(
                                balance=current_balance,
                                kelly_fraction=kelly_fraction,
                                stop_loss_pips=stop_loss_pips,
                                symbol=pair,
                                max_lots=max_lots,
                                min_lots=min_lots
                            )

                            # Calculate SL/TP prices for logging
                            pip_value = 0.0001 if 'JPY' not in pair else 0.01
                            current_price = market_data_raw['bid'] if decision_direction == 'SELL' else market_data_raw['ask']

                            if decision_direction == 'BUY':
                                sl_price = current_price - (stop_loss_pips * pip_value)
                                tp_price = current_price + (take_profit_pips * pip_value)
                            else:
                                sl_price = current_price + (stop_loss_pips * pip_value)
                                tp_price = current_price - (take_profit_pips * pip_value)

                            # Prepare Kelly calculation details for logging
                            kelly_calc = {
                                'balance': current_balance,
                                'kelly_fraction': kelly_fraction,
                                'risk_amount': current_balance * kelly_fraction,
                                'stop_loss_pips': stop_loss_pips,
                                'take_profit_pips': take_profit_pips,
                                'stop_loss_price': sl_price,
                                'take_profit_price': tp_price,
                                'calculated_lots': units / 100000,
                                'units': units,
                                'max_lots': max_lots,
                                'min_lots': min_lots
                            }

                            # Log trade decision
                            trade_id = trade_logger.log_trade_decision(
                                decision_data=decision,
                                market_data=enriched_data,
                                kelly_calc=kelly_calc,
                                account_balance=current_balance
                            )

                            # Execute trade
                            result = oanda.open_position(
                                symbol=pair,
                                direction=direction,
                                units=units,
                                stop_loss_pips=stop_loss_pips,
                                take_profit_pips=take_profit_pips
                            )

                            if result and isinstance(result, dict):
                                trades_executed += 1
                                print(f"  [TRADE EXECUTED] {decision_direction} {units:,} units ({units/100000:.1f} lots)")
                                print(f"    Entry: {result.get('price', 'N/A')}")
                                print(f"    SL: {stop_loss_pips} pips, TP: {take_profit_pips} pips")
                                print(f"    Trade ID: {trade_id}")
                                logger.info(f"[KELLY] Trade executed with {units:,} units")

                                # Log successful entry
                                trade_logger.log_trade_entry(trade_id, result)
                            else:
                                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Trade rejected by broker'
                                print(f"  [ERROR] Trade execution failed: {error_msg}")
                                logger.warning(f"Trade failed for {pair}: {error_msg}")

                                # Log failure
                                trade_logger.log_trade_failure(trade_id, error_msg)

                        except Exception as e:
                            logger.error(f"Trade execution error for {pair}: {e}")
                            print(f"  [ERROR] Exception during trade execution: {str(e)}")

                except Exception as e:
                    logger.error(f"Error analyzing {pair}: {e}")
                    continue

            # Scan summary
            print(f"\n{'-'*80}")
            print(f"[SCAN #{scan_count} COMPLETE]")
            print(f"  Opportunities Found: {opportunities_found}")
            print(f"  Trades Executed: {trades_executed}")

            stats = coordinator.get_statistics()
            print(f"  Total Decisions: {stats['total_decisions']}")
            print(f"  Total Trades: {stats['trades_executed']}")
            print(f"  Execution Rate: {stats['execution_rate']:.1f}%")

            # Calculate time to next scan
            scan_duration = (datetime.now() - scan_start).total_seconds()
            sleep_time = max(0, scan_interval - scan_duration)

            if sleep_time > 0:
                next_scan = datetime.now() + timedelta(seconds=sleep_time)
                print(f"  Next scan: {next_scan.strftime('%H:%M:%S')} ({sleep_time // 60:.0f} min)")
                print(f"{'='*80}\n")

                # Save state periodically
                if scan_count % 6 == 0:  # Every 6 scans (6 hours in live, 30 min in paper)
                    state_dir = Path(__file__).parent / "learning" / "state"
                    coordinator.save_state(str(state_dir))
                    learning_engine.save_learning_data(str(state_dir / "learning_data.json"))
                    print(f"[AUTOSAVE] State saved at {datetime.now().strftime('%H:%M:%S')}")

                # Show progress during wait to prevent "hang" perception
                if sleep_time > 60:
                    print(f"[WAITING] Sleeping {sleep_time // 60:.0f} minutes until next scan...")
                    # Update every minute during wait
                    for i in range(int(sleep_time // 60)):
                        try:
                            time.sleep(60)
                        except KeyboardInterrupt:
                            raise  # Re-raise to be caught by outer handler
                        remaining = int(sleep_time // 60) - i - 1
                        if remaining > 0 and remaining % 5 == 0:
                            print(f"[WAITING] {remaining} minutes until next scan...")
                    # Sleep remaining seconds
                    try:
                        time.sleep(sleep_time % 60)
                    except KeyboardInterrupt:
                        raise  # Re-raise to be caught by outer handler
                else:
                    try:
                        time.sleep(sleep_time)
                    except KeyboardInterrupt:
                        raise  # Re-raise to be caught by outer handler
            else:
                print(f"  [WARNING] Scan took {scan_duration:.0f}s (longer than {scan_interval}s interval)")
                print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("ATLAS TRADING INTERRUPTED BY USER (Ctrl+C)")
        print(f"{'='*80}\n")
        print("[SAVING STATE] Please wait...")
        
        # Save state before exiting
        try:
            state_dir = Path(__file__).parent / "learning" / "state"
            coordinator.save_state(str(state_dir))
            learning_engine.save_learning_data(str(state_dir / "learning_data.json"))
            print("[OK] State saved successfully")
        except Exception as e:
            print(f"[WARNING] Error saving state: {e}")
        
        # Show summary
        try:
            stats = coordinator.get_statistics()
            print(f"\n[SESSION SUMMARY]")
            print(f"  Total Scans: {scan_count}")
            print(f"  Total Decisions: {stats['total_decisions']}")
            print(f"  Total Trades: {stats['trades_executed']}")
            print(f"  Execution Rate: {stats['execution_rate']:.1f}%")
            
            final_balance = oanda.get_account_balance()
            if isinstance(final_balance, dict):
                print(f"\n  Final Balance: ${final_balance.get('balance', 0):,.2f}")
        except:
            pass
        
        print(f"\n{'='*80}")
        print("ATLAS stopped. To resume, run the same command again.")
        print(f"{'='*80}\n")
        return
    
    # Final report
    print(f"\n{'='*80}")
    print(f"LIVE TRADING COMPLETE - {days} DAYS")
    print(f"{'='*80}\n")

    stats = coordinator.get_statistics()
    print(f"Total Scans: {scan_count}")
    print(f"Total Decisions: {stats['total_decisions']}")
    print(f"Total Trades: {stats['trades_executed']}")
    print(f"Execution Rate: {stats['execution_rate']:.1f}%")

    # Final balance
    try:
        final_balance = oanda.get_account_balance()
        if isinstance(final_balance, dict):
            print(f"\nFinal Balance: ${final_balance.get('balance', 0):,.2f}")
            print(f"Starting Balance: $200,000.00")
            print(f"P/L: ${final_balance.get('balance', 0) - 200000:+,.2f}")
    except:
        pass

    print(f"\n{'='*80}\n")


def get_session() -> str:
    """Determine current trading session based on time."""
    now = datetime.now()
    hour = now.hour

    if 2 <= hour < 8:
        return "asian"
    elif 8 <= hour < 13:
        return "london"
    elif 13 <= hour < 18:
        return "ny"
    else:
        return "off_hours"
