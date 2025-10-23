@echo off
REM ========================================
REM Quick Start Script for Trading Bot
REM Last Updated: October 20, 2025
REM ========================================

echo.
echo ================================================================
echo TRADING SYSTEM STARTUP
echo ================================================================
echo.
echo This script will start your enhanced trading system with:
echo   - 80-stock S&P 500 watchlist
echo   - 7 advanced AI agents
echo   - Real-time data from Alpaca/Polygon/OpenBB/Yahoo
echo   - Paper trading mode (safe testing)
echo.
echo ================================================================
echo.

REM Change to the trading directory
cd /d C:\Users\kkdo\PC-HIVE-TRADING

REM Run quick pre-flight check
echo [1/3] Running quick pre-flight check...
python test_account_status.py
if errorlevel 1 (
    echo.
    echo [ERROR] Account status check failed!
    echo Please verify your API credentials in .env file
    pause
    exit /b 1
)

echo.
echo [2/3] Verifying data connectivity (quick test)...
python test_80_stock_watchlist.py --quick
if errorlevel 1 (
    echo.
    echo [WARNING] Some stocks failed data fetch
    echo Bot will continue but may have reduced opportunities
    echo.
)

echo.
echo [3/3] Starting enhanced trading system...
echo.
echo ================================================================
echo STARTING TRADING BOT
echo ================================================================
echo.
echo Press Ctrl+C to stop the bot at any time
echo Logs will be displayed below...
echo.

REM Start the enhanced trading system
python start_enhanced_trading.py

REM If the bot exits, pause so user can see any errors
echo.
echo ================================================================
echo Bot has stopped
echo ================================================================
pause
