@echo off
REM ============================================================================
REM CODEBASE CLEANUP SCRIPT
REM Organizes 167+ files into logical folder structure
REM Based on comprehensive cleanup analysis
REM ============================================================================

echo.
echo ================================================================================
echo                         CODEBASE CLEANUP UTILITY
echo ================================================================================
echo.
echo This will organize your trading bot codebase:
echo   - Move old bot versions to DEPRECATED/
echo   - Move scan outputs to ARCHIVE/
echo   - Move utilities to UTILITIES/
echo   - Keep production files in root
echo.
echo Files will be MOVED (not deleted) - you can always undo
echo.
pause

REM ============================================================================
REM CREATE FOLDER STRUCTURE
REM ============================================================================

echo.
echo [1/6] Creating folder structure...

if not exist DEPRECATED mkdir DEPRECATED
if not exist DEPRECATED\old_bots mkdir DEPRECATED\old_bots
if not exist DEPRECATED\backtesting mkdir DEPRECATED\backtesting
if not exist DEPRECATED\empire_versions mkdir DEPRECATED\empire_versions
if not exist DEPRECATED\testing mkdir DEPRECATED\testing

if not exist UTILITIES mkdir UTILITIES

if not exist ARCHIVE mkdir ARCHIVE
if not exist ARCHIVE\scan_outputs mkdir ARCHIVE\scan_outputs
if not exist ARCHIVE\old_configs mkdir ARCHIVE\old_configs
if not exist ARCHIVE\logs mkdir ARCHIVE\logs

echo   Done.

REM ============================================================================
REM MOVE UTILITIES
REM ============================================================================

echo.
echo [2/6] Moving utility scripts to UTILITIES/...

if exist check_oanda_positions.py move check_oanda_positions.py UTILITIES\
if exist monitor_new_bot.py move monitor_new_bot.py UTILITIES\
if exist quick_status.py move quick_status.py UTILITIES\
if exist analyze_current_trades.py move analyze_current_trades.py UTILITIES\
if exist performance_analytics.py move performance_analytics.py UTILITIES\
if exist trade_journal.py move trade_journal.py UTILITIES\
if exist forex_calendar.py move forex_calendar.py UTILITIES\
if exist force_scan_now.py move force_scan_now.py UTILITIES\
if exist check_trade_peak.py move check_trade_peak.py UTILITIES\
if exist check_why_no_trades.py move check_why_no_trades.py UTILITIES\
if exist close_all_positions.py move close_all_positions.py UTILITIES\
if exist close_position_now.py move close_position_now.py UTILITIES\
if exist calculate_pnl.py move calculate_pnl.py UTILITIES\
if exist calculate_correct_pnl.py move calculate_correct_pnl.py UTILITIES\

echo   Done.

REM ============================================================================
REM ARCHIVE SCAN OUTPUTS
REM ============================================================================

echo.
echo [3/6] Archiving old scan outputs (150+ JSON files)...

move rd_validated_strategies_*.json ARCHIVE\scan_outputs\ 2>nul
move agentic_options_*.json ARCHIVE\scan_outputs\ 2>nul
move monday_ai_scan_*.json ARCHIVE\scan_outputs\ 2>nul
move signal_*.json ARCHIVE\scan_outputs\ 2>nul
move empire_log_2025*.json ARCHIVE\scan_outputs\ 2>nul

echo   Done.

REM ============================================================================
REM MOVE OLD BOT VERSIONS
REM ============================================================================

echo.
echo [4/6] Moving old bot versions to DEPRECATED/old_bots/...

if exist ACTUALLY_WORKING_TRADER.py move ACTUALLY_WORKING_TRADER.py DEPRECATED\old_bots\
if exist SIMPLE_WORKING_TRADER.py move SIMPLE_WORKING_TRADER.py DEPRECATED\old_bots\
if exist SIMPLE_OPTIONS_TRADER.py move SIMPLE_OPTIONS_TRADER.py DEPRECATED\old_bots\
if exist PROP_FIRM_TRADER.py move PROP_FIRM_TRADER.py DEPRECATED\old_bots\
if exist OPENBB_OPTIONS_TRADER.py move OPENBB_OPTIONS_TRADER.py DEPRECATED\old_bots\
if exist OPTIONS_STYLE_STOCK_TRADER.py move OPTIONS_STYLE_STOCK_TRADER.py DEPRECATED\old_bots\
if exist REAL_OPTIONS_TRADER.py move REAL_OPTIONS_TRADER.py DEPRECATED\old_bots\
if exist SP500_EFFICIENT_SCANNER.py move SP500_EFFICIENT_SCANNER.py DEPRECATED\old_bots\
if exist AGENTIC_OPTIONS_SCANNER_SP500.py move AGENTIC_OPTIONS_SCANNER_SP500.py DEPRECATED\old_bots\
if exist ai_enhanced_forex_scanner.py move ai_enhanced_forex_scanner.py DEPRECATED\old_bots\
if exist ai_enhanced_options_scanner.py move ai_enhanced_options_scanner.py DEPRECATED\old_bots\
if exist auto_options_scanner.py move auto_options_scanner.py DEPRECATED\old_bots\
if exist multi_strategy_options_scanner.py move multi_strategy_options_scanner.py DEPRECATED\old_bots\

echo   Done.

REM ============================================================================
REM MOVE BACKTESTING/OPTIMIZATION SCRIPTS
REM ============================================================================

echo.
echo [5/6] Moving backtesting scripts to DEPRECATED/backtesting/...

if exist forex_v4_optimized.py move forex_v4_optimized.py DEPRECATED\backtesting\
if exist forex_v4_backtest.py move forex_v4_backtest.py DEPRECATED\backtesting\
if exist forex_v3_enhanced_backtest.py move forex_v3_enhanced_backtest.py DEPRECATED\backtesting\
if exist forex_quick_optimizer.py move forex_quick_optimizer.py DEPRECATED\backtesting\
if exist forex_parameter_optimizer.py move forex_parameter_optimizer.py DEPRECATED\backtesting\
if exist forex_comprehensive_optimization.py move forex_comprehensive_optimization.py DEPRECATED\backtesting\
if exist forex_optimization_backtest.py move forex_optimization_backtest.py DEPRECATED\backtesting\
if exist quick_forex_backtest.py move quick_forex_backtest.py DEPRECATED\backtesting\
if exist FOREX_PARAMETER_TESTER.py move FOREX_PARAMETER_TESTER.py DEPRECATED\backtesting\
if exist FOREX_PARAMETER_COMPARISON.py move FOREX_PARAMETER_COMPARISON.py DEPRECATED\backtesting\
if exist prop_firm_eval_simulator.py move prop_firm_eval_simulator.py DEPRECATED\backtesting\

echo   Done.

REM ============================================================================
REM MOVE EMPIRE/LAUNCHER VERSIONS
REM ============================================================================

echo.
echo [6/6] Moving old empire versions to DEPRECATED/empire_versions/...

if exist FIXED_AUTONOMOUS_EMPIRE.py move FIXED_AUTONOMOUS_EMPIRE.py DEPRECATED\empire_versions\
if exist fixed_forex_execution_engine.py move fixed_forex_execution_engine.py DEPRECATED\empire_versions\
if exist EMPIRE_LAUNCHER_V2.py move EMPIRE_LAUNCHER_V2.py DEPRECATED\empire_versions\
if exist LAUNCH_TRADING_EMPIRE_NOW.py move LAUNCH_TRADING_EMPIRE_NOW.py DEPRECATED\empire_versions\
if exist START_ALL_WEATHER_TRADING.py move START_ALL_WEATHER_TRADING.py DEPRECATED\empire_versions\
if exist TRADING_EMPIRE_MASTER.py move TRADING_EMPIRE_MASTER.py DEPRECATED\empire_versions\
if exist autonomous_trading_empire.py move autonomous_trading_empire.py DEPRECATED\empire_versions\
if exist MEGA_CLEANUP_SCRIPT.py move MEGA_CLEANUP_SCRIPT.py DEPRECATED\empire_versions\

echo   Done.

REM ============================================================================
REM SUMMARY
REM ============================================================================

echo.
echo ================================================================================
echo                           CLEANUP COMPLETE
echo ================================================================================
echo.
echo Organized files into:
echo   - DEPRECATED/old_bots/        (old trading bot versions)
echo   - DEPRECATED/backtesting/     (optimization scripts)
echo   - DEPRECATED/empire_versions/ (old system versions)
echo   - UTILITIES/                  (monitoring tools)
echo   - ARCHIVE/scan_outputs/       (old JSON scan files)
echo.
echo Production files remain in root:
echo   - WORKING_FOREX_OANDA.py      (current production bot)
echo   - MULTI_MARKET_TRADER.py      (new hybrid system)
echo   - SHARED/                     (shared libraries)
echo   - FOREX/, FUTURES/, CRYPTO/   (market-specific bots)
echo.
echo To undo: Simply move files back from DEPRECATED/ and ARCHIVE/
echo ================================================================================
echo.

pause
