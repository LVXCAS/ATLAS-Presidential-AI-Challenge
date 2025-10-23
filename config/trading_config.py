#!/usr/bin/env python3
"""
Trading System Configuration
Centralized configuration for all trading parameters and constants.

This module contains all magic numbers and configuration values
extracted from the core trading system for maintainability.
"""

from typing import Final

# ============================================================================
# OPTIONS SCANNER CONFIGURATION
# ============================================================================

# Scanning Parameters
OPTIONS_SCAN_INTERVAL_HOURS: Final[int] = 4
"""Hours between options scans (default: 4)"""

OPTIONS_MAX_TRADES_PER_DAY: Final[int] = 4
"""Maximum number of options trades per day"""

OPTIONS_MIN_SCORE_THRESHOLD: Final[float] = 8.0
"""Minimum AI score required to execute options trade"""

OPTIONS_MAX_RISK_PER_TRADE: Final[float] = 500.0
"""Maximum risk per options trade in dollars"""

# Market Hours (Pacific Time)
OPTIONS_MARKET_OPEN_HOUR: Final[int] = 6
"""Market open hour in PT (6:30 AM PT = 9:30 AM ET)"""

OPTIONS_MARKET_CLOSE_HOUR: Final[int] = 13
"""Market close hour in PT (1:00 PM PT = 4:00 PM ET)"""

# Bull Put Spread Parameters
OPTIONS_SELL_STRIKE_PCT: Final[float] = 0.95
"""Sell strike percentage of current price (5% OTM)"""

OPTIONS_BUY_STRIKE_PCT: Final[float] = 0.90
"""Buy strike percentage of current price (10% OTM)"""

OPTIONS_EXPECTED_CREDIT_PCT: Final[float] = 0.30
"""Expected credit as percentage of spread width"""

OPTIONS_MIN_SPREAD_WIDTH: Final[float] = 2.0
"""Minimum spread width in dollars"""

OPTIONS_MAX_SPREAD_WIDTH: Final[float] = 50.0
"""Maximum spread width in dollars"""

OPTIONS_MIN_CONTRACTS: Final[int] = 1
"""Minimum number of contracts"""

OPTIONS_MAX_CONTRACTS: Final[int] = 3
"""Maximum number of contracts for safety"""

OPTIONS_EXPIRATION_DAYS: Final[int] = 30
"""Target days until options expiration"""

# ============================================================================
# FOREX TRADING CONFIGURATION
# ============================================================================

# Scanning Parameters
FOREX_SCAN_INTERVAL_MINUTES: Final[int] = 15
"""Minutes between forex scans"""

FOREX_TARGET_WIN_RATE: Final[float] = 0.636
"""Target win rate for forex strategy (63.6%)"""

FOREX_PAIRS: Final[list[str]] = ['EUR_USD']
"""Default forex pairs to scan"""

FOREX_MIN_SCORE_THRESHOLD: Final[float] = 9.0
"""Minimum AI score required to execute forex trade"""

# Strategy Parameters (EMA v3.0)
FOREX_EMA_FAST: Final[int] = 8
"""Fast EMA period (Fibonacci number)"""

FOREX_EMA_SLOW: Final[int] = 21
"""Slow EMA period (Fibonacci number)"""

FOREX_EMA_TREND: Final[int] = 200
"""Trend EMA period"""

FOREX_RSI_PERIOD: Final[int] = 14
"""RSI calculation period"""

# Entry Conditions
FOREX_MIN_EMA_SEPARATION_PCT: Final[float] = 0.00015
"""Minimum EMA separation as percentage (0.015%)"""

FOREX_RSI_LONG_LOWER: Final[int] = 51
"""Minimum RSI for long entries"""

FOREX_RSI_LONG_UPPER: Final[int] = 79
"""Maximum RSI for long entries"""

FOREX_RSI_SHORT_LOWER: Final[int] = 21
"""Minimum RSI for short entries"""

FOREX_RSI_SHORT_UPPER: Final[int] = 49
"""Maximum RSI for short entries"""

FOREX_VOLUME_FILTER_PCT: Final[float] = 0.55
"""Volume filter threshold (55% of average)"""

FOREX_SCORE_THRESHOLD: Final[float] = 7.2
"""Minimum strategy score for entry"""

# Risk Management
FOREX_ATR_STOP_MULTIPLIER: Final[float] = 2.0
"""Stop loss distance in ATR multiples"""

FOREX_ATR_TARGET_MULTIPLIER: Final[float] = 1.5
"""Take profit distance in ATR multiples"""

FOREX_MIN_RISK_REWARD: Final[float] = 1.5
"""Minimum risk/reward ratio"""

# Position Sizing
FOREX_PAPER_TRADING_UNITS: Final[int] = 5000
"""Position size for paper trading"""

FOREX_LIVE_TRADING_UNITS: Final[int] = 1000
"""Position size for live trading"""

# ============================================================================
# FUTURES TRADING CONFIGURATION
# ============================================================================

# Observation Mode
FUTURES_OBSERVATION_DURATION_HOURS: Final[int] = 48
"""Hours to observe before enabling live trading"""

FUTURES_MIN_WIN_RATE: Final[float] = 0.60
"""Minimum win rate required to approve strategy (60%)"""

FUTURES_MIN_COMPLETED_SIGNALS: Final[int] = 10
"""Minimum completed signals needed for validation"""

# Scanning Parameters
FUTURES_SCAN_INTERVAL_SECONDS: Final[int] = 900
"""Seconds between futures scans (15 minutes)"""

FUTURES_STATUS_INTERVAL_SECONDS: Final[int] = 3600
"""Seconds between status updates (1 hour)"""

# Contracts
FUTURES_CONTRACTS: Final[list[str]] = ['MES', 'MNQ']
"""Micro E-mini futures contracts to trade"""

# Strategy Parameters (EMA Crossover)
FUTURES_EMA_FAST: Final[int] = 10
"""Fast EMA period"""

FUTURES_EMA_SLOW: Final[int] = 20
"""Slow EMA period"""

FUTURES_EMA_TREND: Final[int] = 200
"""Trend EMA period"""

FUTURES_RSI_PERIOD: Final[int] = 14
"""RSI calculation period"""

# Entry Conditions
FUTURES_MIN_EMA_SEPARATION: Final[float] = 0.5
"""Minimum points separation between fast/slow EMA"""

FUTURES_MIN_TREND_DISTANCE: Final[float] = 2.0
"""Minimum distance from trend EMA in points"""

FUTURES_RSI_LONG_THRESHOLD: Final[int] = 55
"""RSI threshold for long entries"""

FUTURES_RSI_SHORT_THRESHOLD: Final[int] = 45
"""RSI threshold for short entries"""

FUTURES_SCORE_THRESHOLD: Final[float] = 9.0
"""Minimum score for entry"""

# Risk Management
FUTURES_ATR_STOP_MULTIPLIER: Final[float] = 2.0
"""Stop loss distance in ATR multiples"""

FUTURES_ATR_TARGET_MULTIPLIER: Final[float] = 3.0
"""Take profit distance in ATR multiples"""

FUTURES_MIN_RISK_REWARD: Final[float] = 1.5
"""Minimum risk/reward ratio"""

# Contract Specifications
FUTURES_MES_POINT_VALUE: Final[float] = 5.0
"""MES point value in dollars"""

FUTURES_MNQ_POINT_VALUE: Final[float] = 2.0
"""MNQ point value in dollars"""

# Conservative Mode
FUTURES_MAX_RISK_PER_TRADE: Final[float] = 100.0
"""Maximum risk per futures trade in dollars"""

FUTURES_MAX_POSITIONS: Final[int] = 2
"""Maximum number of concurrent futures positions"""

FUTURES_MAX_TOTAL_RISK: Final[float] = 500.0
"""Maximum total risk across all futures positions"""

FUTURES_MAX_CONTRACTS: Final[int] = 2
"""Maximum contracts per trade for safety"""

# ============================================================================
# POSITION MONITORING CONFIGURATION
# ============================================================================

# Refresh Settings
MONITOR_DEFAULT_REFRESH_INTERVAL: Final[int] = 30
"""Default refresh interval in seconds for watch mode"""

# Display Settings
MONITOR_MAX_REASONS_DISPLAYED: Final[int] = 3
"""Maximum AI reasoning items to display per trade"""

# ============================================================================
# RISK MANAGEMENT (GLOBAL)
# ============================================================================

MAX_POSITIONS: Final[int] = 5
"""Maximum total positions across all asset types"""

MAX_DAILY_LOSS: Final[float] = 1000.0
"""Maximum daily loss in dollars (circuit breaker)"""

POSITION_SIZE_PERCENT: Final[float] = 0.02
"""Position size as percentage of account (2%)"""

# ============================================================================
# AI ENHANCEMENT CONFIGURATION
# ============================================================================

AI_CONFIDENCE_THRESHOLD: Final[float] = 0.65
"""Minimum AI confidence level"""

AI_MIN_LEARNING_OUTCOMES: Final[int] = 10
"""Minimum outcomes needed before AI adjusts strategy"""

# ============================================================================
# FILE PATHS AND LOGGING
# ============================================================================

LOG_DIRECTORY: Final[str] = "logs"
"""Directory for log files"""

EXECUTION_DIRECTORY: Final[str] = "executions"
"""Directory for execution logs"""

DATA_DIRECTORY: Final[str] = "data"
"""Directory for market data cache"""

STATUS_FILE: Final[str] = "auto_scanner_status.json"
"""Status file for auto scanner"""

# ============================================================================
# TIMEFRAMES
# ============================================================================

TIMEFRAME_1HOUR: Final[str] = "H1"
"""1-hour timeframe code"""

TIMEFRAME_4HOUR: Final[str] = "H4"
"""4-hour timeframe code"""

TIMEFRAME_1DAY: Final[str] = "1Day"
"""Daily timeframe code"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_point_value(symbol: str) -> float:
    """
    Get point value for futures contract.

    Args:
        symbol: Futures symbol (e.g., 'MES', 'MNQ')

    Returns:
        Point value in dollars
    """
    if symbol == 'MES':
        return FUTURES_MES_POINT_VALUE
    elif symbol == 'MNQ':
        return FUTURES_MNQ_POINT_VALUE
    else:
        return 1.0


def is_market_hours(hour: int, minute: int) -> bool:
    """
    Check if given time is within market hours.

    Args:
        hour: Hour in 24-hour format (Pacific Time)
        minute: Minute

    Returns:
        True if within market hours
    """
    if hour < OPTIONS_MARKET_OPEN_HOUR:
        return False
    if hour == OPTIONS_MARKET_OPEN_HOUR and minute < 30:
        return False
    if hour >= OPTIONS_MARKET_CLOSE_HOUR:
        return False
    return True


def validate_trade_limits(
    current_positions: int,
    daily_loss: float,
    trade_risk: float
) -> tuple[bool, str]:
    """
    Validate if trade is within risk limits.

    Args:
        current_positions: Number of open positions
        daily_loss: Current daily loss
        trade_risk: Risk amount for new trade

    Returns:
        Tuple of (valid, reason)
    """
    if current_positions >= MAX_POSITIONS:
        return False, f"Max positions reached ({MAX_POSITIONS})"

    if abs(daily_loss) >= MAX_DAILY_LOSS:
        return False, f"Daily loss limit reached (${MAX_DAILY_LOSS})"

    if abs(daily_loss) + trade_risk > MAX_DAILY_LOSS:
        return False, f"Trade would exceed daily loss limit"

    return True, "Trade within limits"


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_configuration_summary() -> None:
    """Print summary of current configuration."""
    print("\n" + "=" * 70)
    print("TRADING SYSTEM CONFIGURATION SUMMARY")
    print("=" * 70)

    print("\nOPTIONS:")
    print(f"  Scan Interval: {OPTIONS_SCAN_INTERVAL_HOURS}h")
    print(f"  Max Trades/Day: {OPTIONS_MAX_TRADES_PER_DAY}")
    print(f"  Min Score: {OPTIONS_MIN_SCORE_THRESHOLD}")
    print(f"  Max Risk: ${OPTIONS_MAX_RISK_PER_TRADE}")

    print("\nFOREX:")
    print(f"  Scan Interval: {FOREX_SCAN_INTERVAL_MINUTES}m")
    print(f"  Target Win Rate: {FOREX_TARGET_WIN_RATE:.1%}")
    print(f"  Pairs: {', '.join(FOREX_PAIRS)}")
    print(f"  Min Score: {FOREX_MIN_SCORE_THRESHOLD}")

    print("\nFUTURES:")
    print(f"  Observation: {FUTURES_OBSERVATION_DURATION_HOURS}h")
    print(f"  Min Win Rate: {FUTURES_MIN_WIN_RATE:.0%}")
    print(f"  Contracts: {', '.join(FUTURES_CONTRACTS)}")
    print(f"  Max Risk: ${FUTURES_MAX_RISK_PER_TRADE}")

    print("\nRISK MANAGEMENT:")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"  Max Daily Loss: ${MAX_DAILY_LOSS}")
    print(f"  Position Size: {POSITION_SIZE_PERCENT:.1%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_configuration_summary()
